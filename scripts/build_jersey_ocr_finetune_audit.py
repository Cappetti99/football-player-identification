#!/usr/bin/env python3
"""Create a sequence-safe manual audit for jersey-number recognition crops."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--x-min", type=float, default=0.10)
    parser.add_argument("--x-max", type=float, default=0.90)
    parser.add_argument("--y-min", type=float, default=0.08)
    parser.add_argument("--y-max", type=float, default=0.72)
    args = parser.parse_args()
    bounds = (args.x_min, args.y_min, args.x_max, args.y_max)
    if args.limit <= 0 or not valid_bounds(bounds):
        raise ValueError("invalid limit or crop bounds")

    manifest = json.loads(Path(args.sequence_manifest).read_text(encoding="utf-8"))
    if manifest.get("split") != "train":
        raise ValueError("fine-tuning audit requires the GSR train manifest")
    allowed = set(manifest.get("train_sequences") or [])
    frozen = set(manifest.get("validation_sequences") or [])
    rows = read_csv(args.crops)
    rows = [row for row in rows if row.get("sequence") in allowed]
    observed = {row.get("sequence") for row in rows}
    if observed & frozen:
        raise ValueError(f"audit contains frozen sequences: {sorted(observed & frozen)}")
    selected = sequence_round_robin(rows, args.limit, args.seed)

    output = Path(args.output_dir).resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    missing = 0
    for index, row in enumerate(selected, start=1):
        source = Path(str(row.get("crop_path") or ""))
        name = f"number_{index:04d}_{row['sequence']}_f{integer(row.get('frame')):06d}.jpg"
        destination = assets / name
        status, pixels = materialize_number_crop(source, destination, bounds)
        missing += status != "ok"
        audit_rows.append({
            "audit_id": f"number_{index:04d}",
            "asset": f"assets/{name}" if status == "ok" else "",
            "number_crop_path": str(destination) if status == "ok" else "",
            "asset_status": status,
            "crop_bounds_normalized": json.dumps(bounds),
            "crop_box_pixels": json.dumps(pixels) if pixels else "",
            **row,
        })
    write_csv(output / "manifest.csv", audit_rows)
    namespace = "jersey-ocr-finetune-" + hashlib.sha256(
        "\n".join(row["crop_path"] for row in audit_rows).encode()
    ).hexdigest()[:16]
    (output / "index.html").write_text(render_html(audit_rows, namespace), encoding="utf-8")
    summary = {
        "rows": len(audit_rows),
        "sequences": len({row["sequence"] for row in audit_rows}),
        "missing": missing,
        "frozen_sequences_observed": sorted(observed & frozen),
        "bounds": bounds,
        "html": str(output / "index.html"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def materialize_number_crop(source, destination, bounds):
    try:
        from PIL import Image
        with Image.open(source) as image:
            width, height = image.size
            x0, y0, x1, y1 = bounds
            box = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
            crop = image.convert("RGB").crop(box)
            if crop.width < 8 or crop.height < 8:
                return "crop_too_small", box
            crop.save(destination, quality=95)
        return "ok", box
    except FileNotFoundError:
        return "missing_source", None
    except Exception as exc:
        return f"error:{type(exc).__name__}", None


def sequence_round_robin(rows, limit, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        values.sort(key=lambda row: (priority(row), integer(row.get("frame")), row.get("crop_path", "")))
        # Randomize only equal-priority examples deterministically.
        rng.shuffle(values)
        values.sort(key=priority)
    sequences = sorted(grouped)
    rng.shuffle(sequences)
    output = []
    while len(output) < limit:
        changed = False
        for sequence in sequences:
            if grouped[sequence]:
                output.append(grouped[sequence].pop(0))
                changed = True
                if len(output) == limit:
                    break
        if not changed:
            break
    return output


def priority(row):
    return {"positive": 0, "ignore": 1, "hard_negative": 2}.get(row.get("pseudo_label"), 3)


def render_html(rows, namespace):
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Jersey OCR fine-tune audit</title>
<style>body{{font:14px system-ui;background:#111;color:#eee;margin:18px}}.bar{{position:sticky;top:0;background:#111;padding:10px;z-index:3}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}}.card{{background:#222;padding:10px;border-radius:8px}}img{{width:100%;height:230px;object-fit:contain;background:#000}}select,input{{width:100%;box-sizing:border-box;padding:7px;margin-top:5px}}.truth{{display:none;color:#fc6}}body.reveal .truth{{display:block}}</style></head><body>
<div class="bar"><button onclick="document.body.classList.toggle('reveal')">Toggle GT</button> <button onclick="download()">Export review.csv</button> <span id="progress"></span></div><div class="grid" id="root"></div>
<script>const rows={payload},key={json.dumps(namespace)},state=JSON.parse(localStorage.getItem(key)||'{{}}');
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function card(r){{let v=state[r.audit_id]||{{}};return `<div class="card">${{r.asset?`<img loading="lazy" src="${{esc(r.asset)}}">`:'missing'}}<div>${{r.sequence}} · frame ${{r.frame}}</div><div class="truth">GSR GT: ${{r.gt_jersey}}</div><select onchange="setLabel('${{r.audit_id}}',this.value)"><option value="">unreviewed</option>${{['readable','unreadable','uncertain'].map(x=>`<option ${{v.label===x?'selected':''}}>${{x}}</option>`).join('')}}</select><input inputmode="numeric" maxlength="2" placeholder="trascrizione visibile" value="${{esc(v.text||'')}}" onchange="setText('${{r.audit_id}}',this.value)"></div>`}}
function save(){{localStorage.setItem(key,JSON.stringify(state));render()}}function setLabel(id,v){{state[id]=state[id]||{{}};state[id].label=v;save()}}function setText(id,v){{state[id]=state[id]||{{}};state[id].text=v.trim();save()}}function render(){{root.innerHTML=rows.map(card).join('');progress.textContent=Object.values(state).filter(v=>v.label).length+'/'+rows.length+' reviewed'}}
function q(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}function download(){{let cols=['audit_id','review_label','transcription','sequence','gt_track_id','gt_jersey','frame','crop_path','number_crop_path'];let s=cols.join(',')+'\\n';for(const r of rows){{let v=state[r.audit_id]||{{}};s+=cols.map(c=>q(c==='review_label'?v.label||'':c==='transcription'?v.text||'':r[c])).join(',')+'\\n'}}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s]));a.download='review_jersey_ocr.csv';a.click()}}render();</script></body></html>'''


def valid_bounds(values):
    x0, y0, x1, y1 = values
    return 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
