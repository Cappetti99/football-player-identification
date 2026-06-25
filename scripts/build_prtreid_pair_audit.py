#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Build a guided visual audit for PRTReID linker pairs.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--frames-per-tracklet", type=int, default=4)
    args = parser.parse_args()

    metadata = Path(args.artifacts_root) / args.run / "metadata"
    diagnostics = read_json(metadata / f"{args.video_id}_prtreid_linking.json")
    rows = read_csv(metadata / f"{args.video_id}_tracklets.csv")
    candidates = select_candidates(diagnostics.get("candidates", []), args.max_pairs)
    output_dir = Path(args.output_dir or Path("evaluation_outputs/prtreid_pair_audit") / args.run)
    sheets_dir = output_dir / "pairs"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    rows_by_display = group_rows_by_display(rows)

    manifest = []
    for index, candidate in enumerate(candidates, start=1):
        pair_id = f"{args.video_id}_{index:04d}"
        left_id = int(candidate["from_display_track_id"])
        right_id = int(candidate["to_display_track_id"])
        left = representative_rows(rows_by_display.get(left_id, []), args.frames_per_tracklet)
        right = representative_rows(rows_by_display.get(right_id, []), args.frames_per_tracklet)
        sheet_path = sheets_dir / f"{pair_id}.jpg"
        write_pair_sheet(left, right, sheet_path, candidate)
        row = {
            "pair_id": pair_id,
            "video_id": args.video_id,
            "run": args.run,
            "from_display_track_id": left_id,
            "to_display_track_id": right_id,
            "link_type": candidate.get("link_type"),
            "visual_similarity": candidate.get("visual_similarity"),
            "similarity_margin": candidate.get("similarity_margin"),
            "mutual_nearest": candidate.get("mutual_nearest"),
            "gap": candidate.get("gap"),
            "distance": candidate.get("distance"),
            "sheet_path": str(sheet_path),
            "label": "uncertain",
            "notes": "",
        }
        manifest.append(row)

    write_csv(manifest, output_dir / "labels.csv")
    write_html(manifest, output_dir / "index.html")
    print(f"audit pairs={len(manifest)} output={output_dir}")


def select_candidates(rows, limit):
    unique = {}
    for row in rows or []:
        key = (int(row["from_display_track_id"]), int(row["to_display_track_id"]), row.get("link_type"))
        unique[key] = row
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            0 if row.get("mutual_nearest") else 1,
            -float(row.get("visual_similarity") or -1.0),
            -float(row.get("similarity_margin") or 0.0),
            int(row.get("gap") or 0),
        ),
    )
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    same = [row for row in ordered if row.get("link_type") == "same_scene"]
    cross = [row for row in ordered if row.get("link_type") == "cross_scene"]
    minimum_cross = min(len(cross), max(1, limit // 4))
    selected = same[: max(0, limit - minimum_cross)] + cross[:minimum_cross]
    selected_keys = {candidate_key(row) for row in selected}
    for row in ordered:
        if len(selected) >= limit:
            break
        if candidate_key(row) in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(candidate_key(row))
    return selected


def candidate_key(row):
    return (
        int(row["from_display_track_id"]),
        int(row["to_display_track_id"]),
        row.get("link_type"),
    )


def group_rows_by_display(rows):
    output = {}
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        try:
            display_id = int(row.get("prtreid_link_previous_display_track_id") or row.get("display_track_id"))
        except (TypeError, ValueError):
            continue
        output.setdefault(display_id, []).append(row)
    return output


def representative_rows(rows, limit):
    rows = sorted(rows, key=lambda row: int(row.get("frame") or 0))
    if not rows:
        return []
    if len(rows) <= limit:
        return rows
    indices = np.linspace(0, len(rows) - 1, int(limit)).round().astype(int)
    return [rows[int(index)] for index in indices]


def write_pair_sheet(left_rows, right_rows, path, candidate):
    left = row_strip(left_rows, "FROM")
    right = row_strip(right_rows, "TO")
    width = max(left.shape[1], right.shape[1])
    left = pad_width(left, width)
    right = pad_width(right, width)
    header = np.zeros((42, width, 3), dtype=np.uint8)
    label = (
        f"{candidate.get('link_type')} sim={float(candidate.get('visual_similarity') or 0):.4f} "
        f"margin={float(candidate.get('similarity_margin') or 0):.4f} gap={candidate.get('gap')}"
    )
    cv2.putText(header, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([header, left, right]))


def row_strip(rows, prefix):
    images = []
    for row in rows:
        image = cv2.imread(str(row.get("crop_path"))) if row.get("crop_path") else None
        if image is None or image.size == 0:
            image = np.zeros((256, 128, 3), dtype=np.uint8)
        image = letterbox(image, 128, 256)
        cv2.putText(
            image,
            f"{prefix} f{row.get('frame')}",
            (3, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        images.append(image)
    if not images:
        images = [np.zeros((256, 128, 3), dtype=np.uint8)]
    return np.hstack(images)


def letterbox(image, width, height):
    scale = min(width / max(1, image.shape[1]), height / max(1, image.shape[0]))
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def pad_width(image, width):
    if image.shape[1] >= width:
        return image
    return cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT)


def write_html(rows, path):
    page_rows = []
    for row in rows:
        page_row = dict(row)
        sheet_path = Path(row["sheet_path"])
        try:
            relative_sheet = sheet_path.relative_to(path.parent)
        except ValueError:
            relative_sheet = Path("pairs") / sheet_path.name
        page_row["sheet_path"] = str(relative_sheet)
        page_rows.append(page_row)
    payload = json.dumps(page_rows, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(
        """<!doctype html><html lang="it"><head><meta charset="utf-8"><title>PRTReID pair audit</title>
<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0}header{position:sticky;top:0;background:#1b1b1b;padding:12px;z-index:2;border-bottom:1px solid #444}main{max-width:850px;margin:20px auto;padding:0 16px;text-align:center}img{max-width:100%;max-height:68vh;border:1px solid #555}button{font-size:16px;padding:10px 15px;margin:4px;border:1px solid #777;border-radius:6px;cursor:pointer}.same{background:#176b35;color:white}.different{background:#9b2525;color:white}.uncertain{background:#725d14;color:white}.active{outline:3px solid white}textarea{width:100%;min-height:55px;background:#222;color:#eee;border:1px solid #666;padding:8px}.meta{color:#bbb}.nav{background:#333;color:white}#progress{font-weight:700;margin-right:20px}</style></head><body>
<header><span id="progress"></span><button class="nav" id="prev">← precedente</button><button class="nav" id="next">successiva →</button><button id="download">Scarica labels.csv</button></header>
<main><h2 id="title"></h2><p class="meta" id="meta"></p><img id="sheet" alt="coppia PRTReID"><div><button class="same" data-label="same">1 · stesso</button><button class="different" data-label="different">2 · diverso</button><button class="uncertain" data-label="uncertain">3 · incerto</button></div><textarea id="notes" placeholder="Note facoltative (per esempio: numeri 8 e 23)"></textarea></main>
<script>const rows = """ + payload + r""";
const storageKey='prtreid-audit-'+(rows[0]?.run||'audit');let labels=JSON.parse(localStorage.getItem(storageKey)||'{}');let index=0;const el=id=>document.getElementById(id);
function save(){localStorage.setItem(storageKey,JSON.stringify(labels));}
function render(){const row=rows[index],saved=labels[row.pair_id]||{};el('title').textContent=`${row.pair_id} (${index+1}/${rows.length})`;el('meta').textContent=`${row.link_type} · similarity ${Number(row.visual_similarity).toFixed(4)} · margin ${Number(row.similarity_margin).toFixed(4)}`;el('sheet').src=row.sheet_path;el('notes').value=saved.notes||'';document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('active',b.dataset.label===saved.label));el('progress').textContent=`Classificate ${Object.keys(labels).length}/${rows.length}`;}
function setLabel(label){const id=rows[index].pair_id;labels[id]={label,notes:el('notes').value};save();if(index<rows.length-1)index++;render();}
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>setLabel(b.dataset.label));el('notes').onchange=()=>{const id=rows[index].pair_id;if(labels[id]){labels[id].notes=el('notes').value;save();}};el('prev').onclick=()=>{index=Math.max(0,index-1);render();};el('next').onclick=()=>{index=Math.min(rows.length-1,index+1);render();};document.onkeydown=e=>{if(e.target.matches('textarea,input'))return;if(e.key==='1')setLabel('same');if(e.key==='2')setLabel('different');if(e.key==='3')setLabel('uncertain');if(e.key==='ArrowLeft')el('prev').click();if(e.key==='ArrowRight')el('next').click();};
function csvCell(value){const s=value==null?'':String(value);return /[",\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;}
el('download').onclick=()=>{const fields=Object.keys(rows[0]),lines=[fields.map(csvCell).join(',')];for(const row of rows){const saved=labels[row.pair_id]||{},out={...row,label:saved.label||'uncertain',notes:saved.notes||''};lines.push(fields.map(f=>csvCell(out[f])).join(','));}const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='labels.csv';a.click();URL.revokeObjectURL(a.href);};render();
</script></body></html>""",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["pair_id", "label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
