#!/usr/bin/env python3
"""Build a blind visual audit for GSR jersey crop legibility."""

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-correct-tracklets", action="store_true")
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    tracklets = json.loads((oracle_dir / "tracklets.json").read_text())
    crops = read_csv(oracle_dir / "crops.csv")
    crops_by_track = defaultdict(list)
    for crop in crops:
        crop = normalize_crop(crop)
        crops_by_track[crop["gt_track_id"]].append(crop)

    selected = []
    for track in tracklets:
        if track.get("current_correct") and not args.include_correct_tracklets:
            continue
        track_crops = crops_by_track.get(str(track["gt_track_id"]), [])
        selected.extend(select_track_crops(track, track_crops, args.top_k))

    manifest = []
    for index, item in enumerate(selected, start=1):
        source = Path(item["crop_path"])
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        asset_name = f"crop_{index:04d}_track_{item['gt_track_id']}_frame_{item['frame']}{suffix}"
        destination = assets_dir / asset_name
        status = "copied"
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            status = "missing"
        manifest.append({
            "crop_id": f"crop_{index:04d}",
            "asset": f"assets/{asset_name}" if status == "copied" else "",
            "asset_status": status,
            **item,
        })

    write_csv(output_dir / "manifest.csv", manifest)
    (output_dir / "index.html").write_text(render_html(manifest), encoding="utf-8")
    print(json.dumps({
        "tracklets": len({row["gt_track_id"] for row in manifest}),
        "crops": len(manifest),
        "copied": sum(row["asset_status"] == "copied" for row in manifest),
        "missing": sum(row["asset_status"] == "missing" for row in manifest),
        "html": str(output_dir / "index.html"),
        "manifest": str(output_dir / "manifest.csv"),
    }, indent=2))


def select_track_crops(track, crops, top_k):
    gt_jersey = integer(track.get("gt_jersey"))
    ranked = sorted(crops, key=lambda row: row["selection_score"], reverse=True)
    choices = []

    def add(row, reason):
        if row is None:
            return
        key = (row["crop_path"], row["frame"])
        existing = next((item for item in choices if item["_key"] == key), None)
        if existing:
            if reason not in existing["selection_reason"]:
                existing["selection_reason"] += f"+{reason}"
            return
        choices.append({
            "_key": key,
            "gt_track_id": str(track["gt_track_id"]),
            "gt_jersey": gt_jersey,
            "current_prediction": integer(track.get("current_prediction")),
            "current_correct": bool(track.get("current_correct")),
            "recoverable_current_error": bool(track.get("recoverable_current_error")),
            "selection_reason": reason,
            **row,
        })

    add(ranked[0] if ranked else None, "selector_top1")
    add(next((row for row in ranked if row["winner"] == gt_jersey), None), "oracle_best_correct")
    add(next((row for row in ranked if row["winner"] != gt_jersey), None), "best_wrong")
    first_correct = sorted(
        (row for row in crops if row["winner"] == gt_jersey),
        key=lambda row: row["frame"],
    )
    add(first_correct[0] if first_correct else None, "first_correct")
    for rank, row in enumerate(ranked[:max(0, top_k)], start=1):
        add(row, f"top{rank}")

    for row in choices:
        row.pop("_key", None)
    return choices


def normalize_crop(row):
    return {
        **row,
        "gt_track_id": str(row.get("gt_track_id")),
        "frame": integer(row.get("frame")) or 0,
        "winner": integer(row.get("winner")),
        "winner_confidence": floating(row.get("winner_confidence")),
        "agreement": integer(row.get("agreement")) or 0,
        "source_agreement": integer(row.get("source_agreement")) or 0,
        "crop_quality": floating(row.get("crop_quality")),
        "selection_score": floating(row.get("selection_score")),
        "crop_path": str(row.get("crop_path") or ""),
    }


def render_html(rows):
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GSR legibility audit</title>
<style>
body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}button,select{{font:inherit;padding:6px}}
.toolbar{{position:sticky;top:0;background:#111;padding:10px 0;z-index:2}}.track{{margin:24px 0;border-top:1px solid #555}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}.card{{background:#222;padding:10px;border-radius:8px}}
img{{width:100%;height:260px;object-fit:contain;background:#000}}.meta{{font-size:12px;color:#bbb;white-space:pre-line}}
.truth{{display:none;color:#ffcc66}}body.reveal .truth{{display:block}}.missing{{height:260px;display:grid;place-items:center;background:#400}}
</style></head><body>
<div class="toolbar"><button onclick="document.body.classList.toggle('reveal')">Toggle GT</button>
<button onclick="exportCsv()">Export labels.csv</button> <span id="progress"></span></div><div id="root"></div>
<script>
const rows={payload}; const labels=JSON.parse(localStorage.getItem('gsr-legibility-labels')||'{{}}');
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function render(){{const groups={{}}; rows.forEach(r=>(groups[r.gt_track_id]??=[]).push(r));
document.getElementById('root').innerHTML=Object.entries(groups).map(([track,items])=>`<section class="track"><h2>GT track ${{esc(track)}}</h2><div class="grid">${{items.map(card).join('')}}</div></section>`).join(''); progress();}}
function card(r){{const media=r.asset?`<img loading="lazy" src="${{esc(r.asset)}}">`:'<div class="missing">missing crop</div>';
return `<article class="card">${{media}}<div class="truth">GT jersey #${{esc(r.gt_jersey)}} | OCR #${{esc(r.winner)}}</div>
<div class="meta">frame ${{r.frame}} | ${{esc(r.selection_reason)}}\nscore ${{Number(r.selection_score).toFixed(3)}} | agreement ${{r.agreement}} | sources ${{r.source_agreement}}</div>
<select onchange="setLabel('${{r.crop_id}}',this.value)"><option value="">unlabeled</option>${{['readable','partial','not_readable','uncertain'].map(v=>`<option ${{labels[r.crop_id]===v?'selected':''}}>${{v}}</option>`).join('')}}</select></article>`;}}
function setLabel(id,v){{if(v)labels[id]=v;else delete labels[id];localStorage.setItem('gsr-legibility-labels',JSON.stringify(labels));progress();}}
function progress(){{document.getElementById('progress').textContent=`${{Object.keys(labels).length}}/${{rows.length}} labeled`;}}
function exportCsv(){{let out='crop_id,label,gt_track_id,frame,crop_path\\n';rows.forEach(r=>out+=`"${{r.crop_id}}","${{labels[r.crop_id]||''}}","${{r.gt_track_id}}","${{r.frame}}","${{String(r.crop_path).replaceAll('"','""')}}"\\n`);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out],{{type:'text/csv'}}));a.download='labels.csv';a.click();}}
render();</script></body></html>"""


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def integer(value):
    try: return int(float(value))
    except (TypeError, ValueError): return None


def floating(value):
    try: return float(value)
    except (TypeError, ValueError): return 0.0


if __name__ == "__main__":
    main()
