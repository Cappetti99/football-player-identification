#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from ft.evaluation.identity_benchmark import (
    annotation_rows,
    assign_second_review,
    canonical_hash,
    group_identity_units,
    merge_identity_units,
    anchor_payload,
    read_csv,
    read_json,
    representative_anchors,
    sha256_file,
    write_csv,
    write_json,
)


ANNOTATION_FIELDS = [
    "item_id", "item_type", "video_id", "split", "reviewer",
    "second_review_required", "annotation_status", "gt_player_id",
    "gt_team_id", "gt_jersey_number", "jersey_visibility", "pair_label",
    "uncertainty_reason", "notes",
]


def main():
    parser = argparse.ArgumentParser(description="Build frozen Identity Evaluation V1 annotation artifacts.")
    parser.add_argument("--spec", default="evaluation/identity_benchmark_v1.yaml")
    parser.add_argument("--artifacts-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true", help="Replace an existing frozen manifest")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    artifacts_root = Path(args.artifacts_root or spec.get("artifacts_root", "artifacts/costume-video"))
    output = Path(args.output_dir or spec.get("output_dir", "evaluation_outputs/identity_benchmark_v1"))
    output.mkdir(parents=True, exist_ok=True)

    manifest = build_benchmark(spec, spec_path, artifacts_root)
    frozen_path = output / "benchmark_manifest.json"
    if frozen_path.is_file() and not args.force:
        frozen = read_json(frozen_path)
        if frozen.get("benchmark_sha256") != manifest.get("benchmark_sha256"):
            raise SystemExit(
                "benchmark_manifest.json is frozen and inputs changed; "
                "use a new benchmark version or pass --force explicitly"
            )
    build_visual_assets(manifest, output)
    write_json(manifest, frozen_path)
    identity_items = manifest["identity_units"]
    pair_items = manifest["pairs"]
    all_items = identity_items + pair_items
    write_csv(annotation_rows(all_items, "A"), output / "annotations" / "reviewer_a.csv", ANNOTATION_FIELDS)
    write_csv(annotation_rows(all_items, "B", second_only=True), output / "annotations" / "reviewer_b.csv", ANNOTATION_FIELDS)
    write_csv([], output / "annotations" / "adjudication.csv", ANNOTATION_FIELDS)
    write_csv([], output / "ground_truth.csv", ANNOTATION_FIELDS)
    write_annotation_html(manifest, output, reviewer="A", second_only=False)
    write_annotation_html(manifest, output, reviewer="B", second_only=True)
    print(
        f"identity_units={len(identity_items)} pairs={len(pair_items)} "
        f"second_review={sum(item['second_review_required'] for item in all_items)} output={output}"
    )


def build_benchmark(spec, spec_path, artifacts_root):
    max_frames = int(spec.get("max_frames", 1200))
    anchors_per_unit = int(spec.get("anchors_per_unit", 8))
    all_source_units = []
    files = {}
    rosters = {}
    source_rows = {}
    split_by_video = {}

    for video in spec.get("videos", []):
        video_id = str(video["video_id"])
        split = str(video["split"])
        if video_id in split_by_video and split_by_video[video_id] != split:
            raise ValueError(f"video {video_id} appears in multiple splits")
        split_by_video[video_id] = split
        roster_path = Path(video["roster_path"])
        require_file(roster_path)
        files[str(roster_path)] = sha256_file(roster_path)
        rosters[video_id] = read_json(roster_path)
        for run in video.get("source_runs", []):
            metadata = artifacts_root / run / "metadata"
            tracklets = metadata / f"{video_id}_tracklets.csv"
            manifest = metadata / f"{video_id}_run_manifest.json"
            require_file(tracklets)
            require_file(manifest)
            files[str(tracklets)] = sha256_file(tracklets)
            files[str(manifest)] = sha256_file(manifest)
            rows = read_csv(tracklets)
            source_rows[(video_id, run)] = rows
            all_source_units.extend(group_identity_units(rows, run, video_id, split, max_frames=max_frames))

    identity_units = merge_identity_units(
        all_source_units,
        anchors_per_unit=anchors_per_unit,
        iou_threshold=float(spec.get("merge_iou_threshold", 0.5)),
        min_shared_frames=int(spec.get("merge_min_shared_frames", 3)),
        min_overlap_fraction=float(spec.get("merge_min_overlap_fraction", 0.2)),
    )
    display_to_unit = member_display_lookup(identity_units)
    pairs = build_pairs(spec, artifacts_root, source_rows, display_to_unit, split_by_video, files)
    all_items = assign_second_review(
        identity_units + pairs,
        fraction=float(spec.get("second_review_fraction", 0.2)),
        seed=int(spec.get("seed", 20260702)),
    )
    identity_units = [item for item in all_items if item["item_type"] == "identity"]
    pairs = [item for item in all_items if item["item_type"] == "pair"]
    resolved = {
        "benchmark_version": str(spec.get("version", "1")),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "seed": int(spec.get("seed", 20260702)),
        "max_frames": max_frames,
        "anchors_per_unit": anchors_per_unit,
        "second_review_fraction": float(spec.get("second_review_fraction", 0.2)),
        "artifact_hashes": dict(sorted(files.items())),
        "rosters": rosters,
        "pair_source_specs": [
            {
                "video_id": str(video["video_id"]),
                "run": source["run"],
                "mechanism": source["mechanism"],
                "artifact": source.get("artifact") or default_pair_artifact(
                    str(video["video_id"]), source["mechanism"]
                ),
            }
            for video in spec.get("videos", [])
            for source in video.get("pair_sources", [])
        ],
        "identity_units": identity_units,
        "pairs": pairs,
    }
    resolved["benchmark_sha256"] = canonical_hash({
        key: value for key, value in resolved.items() if key != "benchmark_sha256"
    })
    return resolved


def member_display_lookup(units):
    lookup = {}
    for unit in units:
        for member in unit["members"]:
            for display_id in member["display_track_ids"]:
                lookup[(unit["video_id"], member["run"], str(display_id))] = unit["item_id"]
    return lookup


def build_pairs(spec, artifacts_root, source_rows, display_to_unit, split_by_video, files):
    accepted = []
    rejected_by_stratum = defaultdict(list)
    for video in spec.get("videos", []):
        video_id = str(video["video_id"])
        split = split_by_video[video_id]
        for source in video.get("pair_sources", []):
            run = source["run"]
            mechanism = source["mechanism"]
            filename = source.get("artifact") or default_pair_artifact(video_id, mechanism)
            path = artifacts_root / run / "metadata" / filename
            require_file(path)
            files[str(path)] = sha256_file(path)
            payload = read_json(path)
            accepted_rows, rejected_rows = pair_records(payload, mechanism)
            for status, rows in (("accepted", accepted_rows), ("rejected", rejected_rows)):
                for row in rows:
                    normalized = normalize_pair(
                        row,
                        video_id,
                        split,
                        run,
                        mechanism,
                        status,
                        display_to_unit,
                        source_rows.get((video_id, run), []),
                    )
                    if normalized is None:
                        continue
                    if status == "accepted":
                        accepted.append(normalized)
                    else:
                        rejected_by_stratum[(video_id, mechanism, normalized["link_type"])].append(normalized)

    limit = int(spec.get("rejected_near_miss_per_stratum", 20))
    randomizer = random.Random(int(spec.get("seed", 20260702)))
    selected_rejected = []
    for _stratum, rows in sorted(rejected_by_stratum.items()):
        unique = {row["item_id"]: row for row in rows}
        rows = sorted(
            unique.values(),
            key=lambda row: (
                -float(row.get("visual_similarity") or -1.0),
                -float(row.get("similarity_margin") or -1.0),
                row["item_id"],
            ),
        )
        pool_size = len(rows)
        if limit > 0 and pool_size > limit:
            boundary = rows[: max(limit * 2, limit)]
            randomizer.shuffle(boundary)
            rows = sorted(boundary[:limit], key=lambda row: row["item_id"])
        probability = len(rows) / max(1, pool_size)
        for row in rows:
            row["sampling_probability"] = probability
            row["strata"].append("rejected_near_miss")
        selected_rejected.extend(rows)
    return sorted(
        {row["item_id"]: row for row in accepted + selected_rejected}.values(),
        key=lambda row: (row["video_id"], row["mechanism"], row["item_id"]),
    )


def default_pair_artifact(video_id, mechanism):
    suffix = {
        "prtreid_linking": "prtreid_linking.json",
        "prtreid_identity_bridge": "prtreid_identity_bridge.json",
        "jersey_identity_linking": "jersey_identity_linking.json",
        "identity_propagation": "identity_propagation.json",
    }[mechanism]
    return f"{video_id}_{suffix}"


def pair_records(payload, mechanism):
    if mechanism == "prtreid_linking":
        accepted = payload.get("accepted_links", [])
        rejected = payload.get("rejected_links") or [
            row for row in payload.get("candidates", []) if row not in accepted
        ]
    elif mechanism == "prtreid_identity_bridge":
        accepted = payload.get("applied_links", [])
        rejected = payload.get("rejected_links") or payload.get("candidates", [])
    elif mechanism == "jersey_identity_linking":
        accepted = payload.get("accepted_links", [])
        rejected = payload.get("rejected_links", [])
    else:
        accepted = payload.get("propagations", [])
        rejected = payload.get("rejected_propagations", [])
    return accepted, rejected


def normalize_pair(row, video_id, split, run, mechanism, status, display_to_unit, source_rows):
    left = first_value(row, "from_display_track_id", "anchor_display_track_id", "source_display_id")
    right = first_value(row, "to_display_track_id", "target_display_track_id")
    if left is None or right is None:
        return None
    left_unit = display_to_unit.get((video_id, run, str(left)))
    right_unit = display_to_unit.get((video_id, run, str(right)))
    grouped = pair_rows_by_display(source_rows, mechanism)
    left_rows = grouped.get(str(left), [])
    right_rows = grouped.get(str(right), [])
    if not left_rows or not right_rows:
        return None
    stable = {
        "video_id": video_id,
        "run": run,
        "mechanism": mechanism,
        "left_display_id": str(left),
        "right_display_id": str(right),
    }
    return {
        "item_id": f"{video_id}_pair_{canonical_hash(stable)[:12]}",
        "item_type": "pair",
        "video_id": video_id,
        "split": split,
        "run": run,
        "mechanism": mechanism,
        "status": status,
        "link_type": str(row.get("link_type") or mechanism),
        "left_display_id": str(left),
        "right_display_id": str(right),
        "left_unit_id": left_unit,
        "right_unit_id": right_unit,
        "left_anchors": [anchor_payload(item) for item in representative_anchors(left_rows, 4)],
        "right_anchors": [anchor_payload(item) for item in representative_anchors(right_rows, 4)],
        "visual_similarity": row.get("visual_similarity"),
        "similarity_margin": row.get("similarity_margin"),
        "sampling_probability": 1.0,
        "strata": [f"{mechanism}_{status}"],
    }


def pair_rows_by_display(rows, mechanism):
    grouped = defaultdict(list)
    previous_field = {
        "prtreid_linking": "prtreid_link_previous_display_track_id",
        "jersey_identity_linking": "jersey_link_previous_display_track_id",
    }.get(mechanism)
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        display = row.get(previous_field) if previous_field else None
        if display in (None, "", "None"):
            display = row.get("display_track_id")
        if display not in (None, "", "None"):
            grouped[str(display)].append(row)
    return grouped


def first_value(row, *keys):
    for key in keys:
        if row.get(key) not in (None, "", "None"):
            return row[key]
    return None


def build_visual_assets(manifest, output):
    sheets = output / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    units = {row["item_id"]: row for row in manifest["identity_units"]}
    for unit in units.values():
        path = sheets / f"{unit['item_id']}.jpg"
        write_identity_sheet(unit["anchors"], path)
        unit["sheet_path"] = str(path.relative_to(output))
    for pair in manifest["pairs"]:
        path = sheets / f"{pair['item_id']}.jpg"
        write_pair_sheet(pair["left_anchors"], pair["right_anchors"], path)
        pair["sheet_path"] = str(path.relative_to(output))


def write_identity_sheet(anchors, path):
    images = [labeled_crop(anchor, "f") for anchor in anchors]
    if not images:
        images = [np.zeros((256, 128, 3), dtype=np.uint8)]
    cv2.imwrite(str(path), np.hstack(images))


def write_pair_sheet(left, right, path):
    left_strip = np.hstack([labeled_crop(row, "A f") for row in left[:4]])
    right_strip = np.hstack([labeled_crop(row, "B f") for row in right[:4]])
    width = max(left_strip.shape[1], right_strip.shape[1])
    left_strip = cv2.copyMakeBorder(left_strip, 0, 0, 0, width - left_strip.shape[1], cv2.BORDER_CONSTANT)
    right_strip = cv2.copyMakeBorder(right_strip, 0, 0, 0, width - right_strip.shape[1], cv2.BORDER_CONSTANT)
    cv2.imwrite(str(path), np.vstack([left_strip, right_strip]))


def labeled_crop(anchor, prefix):
    image = cv2.imread(str(anchor.get("crop_path"))) if anchor.get("crop_path") else None
    if image is None or image.size == 0:
        image = np.zeros((256, 128, 3), dtype=np.uint8)
    image = letterbox(image, 128, 256)
    cv2.putText(
        image, f"{prefix}{anchor.get('frame')}", (3, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return image


def letterbox(image, width, height):
    scale = min(width / max(1, image.shape[1]), height / max(1, image.shape[0]))
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def write_annotation_html(manifest, output, reviewer, second_only):
    items = manifest["identity_units"] + manifest["pairs"]
    if second_only:
        items = [item for item in items if item["second_review_required"]]
    public = [
        {
            "item_id": item["item_id"],
            "item_type": item["item_type"],
            "video_id": item["video_id"],
            "split": item["split"],
            "sheet_path": item["sheet_path"],
            "roster": manifest["rosters"].get(item["video_id"], []),
            "second_review_required": item["second_review_required"],
        }
        for item in items
    ]
    payload = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    html = ANNOTATION_HTML.replace("__ROWS__", payload).replace("__REVIEWER__", reviewer)
    (output / f"annotate_reviewer_{reviewer.lower()}.html").write_text(html, encoding="utf-8")


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"required benchmark input not found: {path}")


ANNOTATION_HTML = r"""<!doctype html><html lang="it"><head><meta charset="utf-8"><title>Identity benchmark</title>
<style>body{font-family:system-ui;background:#111;color:#eee;margin:0}header{position:sticky;top:0;background:#222;padding:10px}main{max-width:1050px;margin:18px auto;text-align:center}img{max-width:100%;max-height:62vh;border:1px solid #555}button,select,input,textarea{font-size:15px;margin:4px;padding:8px}textarea{width:90%;min-height:55px}.hidden{display:none}.active{outline:3px solid white}.good{background:#176b35;color:white}.bad{background:#9b2525;color:white}.neutral{background:#725d14;color:white}</style></head><body>
<header><b id="progress"></b><button id="prev">←</button><button id="next">→</button><button id="incomplete">Prossimo incompleto</button><button id="import">Importa CSV</button><input class="hidden" type="file" id="importFile" accept=".csv,text/csv"><button id="download">Scarica CSV</button></header>
<main><h3 id="title"></h3><img id="sheet"><section id="identity"><select id="status"><option value="">stato…</option><option>determinate</option><option>not_determinable</option><option>exclude</option></select><select id="player"><option value="">giocatore…</option></select><select id="visibility"><option value="">visibilità jersey…</option><option>full</option><option>partial</option><option>not_visible</option></select></section>
<section id="pair"><button class="good" data-pair="same">1 stesso</button><button class="bad" data-pair="different">2 diverso</button><button class="neutral" data-pair="uncertain">3 incerto</button><button data-pair="exclude">4 escludi</button></section>
<input id="reason" placeholder="motivo incertezza"><br><textarea id="notes" placeholder="note"></textarea></main>
<script>const rows=__ROWS__,reviewer='__REVIEWER__',key='identity-benchmark-'+reviewer;let labels=JSON.parse(localStorage.getItem(key)||'{}'),i=0;const $=x=>document.getElementById(x);
function current(){return rows[i]} function save(){localStorage.setItem(key,JSON.stringify(labels))}
function complete(r,v){if(r.item_type==='pair')return ['same','different','uncertain','exclude'].includes(v.pair_label);if(v.annotation_status==='determinate')return !!v.gt_player_id&&['full','partial','not_visible'].includes(v.jersey_visibility);if(v.annotation_status==='not_determinable')return !!v.uncertainty_reason&&['full','partial','not_visible'].includes(v.jersey_visibility);if(v.annotation_status==='exclude')return !!v.uncertainty_reason;return false}
function capture(){const r=current(),old=labels[r.item_id]||{};labels[r.item_id]={...old,item_id:r.item_id,item_type:r.item_type,video_id:r.video_id,split:r.split,reviewer,second_review_required:r.second_review_required,annotation_status:$('status').value,gt_player_id:$('player').value,gt_team_id:$('player').selectedOptions[0]?.dataset.team||'',gt_jersey_number:$('player').selectedOptions[0]?.dataset.jersey||'',jersey_visibility:$('visibility').value,pair_label:old.pair_label||'',uncertainty_reason:$('reason').value,notes:$('notes').value};save()}
function render(){const r=current(),v=labels[r.item_id]||{};$('title').textContent=`${r.item_id} · ${i+1}/${rows.length}`;$('sheet').src=r.sheet_path;$('identity').classList.toggle('hidden',r.item_type!=='identity');$('pair').classList.toggle('hidden',r.item_type!=='pair');$('player').innerHTML='<option value=\"\">giocatore…</option>'+r.roster.map(p=>`<option value=\"${p.player_id}\" data-team=\"${p.team_id??''}\" data-jersey=\"${p.jersey_number??''}\">${p.name||p.player_id} #${p.jersey_number??'?'}</option>`).join('');$('status').value=v.annotation_status||'';$('player').value=v.gt_player_id||'';$('visibility').value=v.jersey_visibility||'';$('reason').value=v.uncertainty_reason||'';$('notes').value=v.notes||'';document.querySelectorAll('[data-pair]').forEach(b=>b.classList.toggle('active',b.dataset.pair===v.pair_label));$('progress').textContent=`Complete ${rows.filter(x=>complete(x,labels[x.item_id]||{})).length}/${rows.length}`}
function move(d){capture();i=Math.max(0,Math.min(rows.length-1,i+d));render()} $('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);document.querySelectorAll('[data-pair]').forEach(b=>b.onclick=()=>{capture();labels[current().item_id].pair_label=b.dataset.pair;save();move(1)});
$('incomplete').onclick=()=>{capture();for(let step=1;step<=rows.length;step++){const candidate=(i+step)%rows.length;if(!complete(rows[candidate],labels[rows[candidate].item_id]||{})){i=candidate;render();return}}alert('Tutte le annotazioni sono complete')};
document.onkeydown=e=>{if(e.target.matches('input,textarea,select'))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(current().item_type==='pair'&&'1234'.includes(e.key))document.querySelectorAll('[data-pair]')[Number(e.key)-1].click()};
function parseCsv(text){const table=[],row=[];let field='',quoted=false;for(let p=0;p<text.length;p++){const c=text[p];if(quoted){if(c==='"'&&text[p+1]==='"'){field+='"';p++}else if(c==='"'){quoted=false}else field+=c}else if(c==='"'){quoted=true}else if(c===','){row.push(field);field=''}else if(c==='\n'){row.push(field);table.push(row.splice(0));field=''}else if(c!=='\r'){field+=c}}if(field||row.length){row.push(field);table.push(row)}const fields=table.shift()||[];return table.filter(r=>r.some(Boolean)).map(r=>Object.fromEntries(fields.map((f,j)=>[f,r[j]??''])))}
$('import').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{const imported=parseCsv(await e.target.files[0].text()),known=new Set(rows.map(r=>r.item_id));let count=0;for(const row of imported){if(!known.has(row.item_id))continue;labels[row.item_id]=row;count++}save();render();alert(`Importate ${count} annotazioni`)};
function cell(v){v=v==null?'':String(v);return /[",\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v}$('download').onclick=()=>{capture();const incomplete=rows.filter(r=>!complete(r,labels[r.item_id]||{}));if(incomplete.length){alert(`Restano ${incomplete.length} annotazioni incomplete`);return}const fields=['item_id','item_type','video_id','split','reviewer','second_review_required','annotation_status','gt_player_id','gt_team_id','gt_jersey_number','jersey_visibility','pair_label','uncertainty_reason','notes'],lines=[fields.join(',')];for(const r of rows){const o=labels[r.item_id]||{item_id:r.item_id,item_type:r.item_type,video_id:r.video_id,split:r.split,reviewer,second_review_required:r.second_review_required};lines.push(fields.map(f=>cell(o[f])).join(','))}const a=document.createElement('a'),blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'});a.href=URL.createObjectURL(blob);a.download=`reviewer_${reviewer.toLowerCase()}.csv`;a.click();URL.revokeObjectURL(a.href)};render();</script></body></html>"""


if __name__ == "__main__":
    main()
