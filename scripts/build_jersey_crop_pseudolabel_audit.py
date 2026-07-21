#!/usr/bin/env python3
"""Build a balanced blind contact sheet for crop pseudo-label verification."""

import argparse
import csv
import html
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudolabels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=("positive", "hard_negative", "ignore"),
        default=("positive", "hard_negative"),
        help="Pseudo-label groups to sample (default: positive hard_negative).",
    )
    args = parser.parse_args()
    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")
    rows = read_csv(Path(args.pseudolabels))
    selected = []
    for label in args.labels:
        selected.extend(sequence_round_robin(
            [row for row in rows if row.get("pseudo_label") == label],
            args.per_class,
            args.seed,
        ))
    selected.sort(key=lambda row: (row["pseudo_label"], row["sequence"], row["gt_track_id"], int(row["frame"])))

    output = Path(args.output_dir)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(selected, start=1):
        source = Path(row["crop_path"])
        suffix = source.suffix if source.suffix else ".jpg"
        asset_name = f"crop_{index:04d}_{row['pseudo_label']}{suffix}"
        destination = assets / asset_name
        status = "missing"
        if source.is_file():
            shutil.copy2(source, destination)
            status = "copied"
        manifest.append({
            "audit_id": f"crop_{index:04d}",
            "asset": f"assets/{asset_name}" if status == "copied" else "",
            "asset_status": status,
            **row,
        })
    write_csv(output / "manifest.csv", manifest)
    namespace = "jersey-crop-audit-" + hashlib.sha256(
        "\n".join(row["crop_path"] for row in manifest).encode("utf-8")
    ).hexdigest()[:16]
    (output / "index.html").write_text(render_html(manifest, namespace), encoding="utf-8")
    label_counts = {
        label: sum(row["pseudo_label"] == label for row in manifest)
        for label in args.labels
    }
    print(json.dumps({
        "rows": len(manifest),
        "labels": label_counts,
        "sequences": len({row["sequence"] for row in manifest}),
        "copied": sum(row["asset_status"] == "copied" for row in manifest),
        "missing": sum(row["asset_status"] == "missing" for row in manifest),
        "html": str(output / "index.html"),
        "review_storage_namespace": namespace,
    }, indent=2))


def sequence_round_robin(rows, limit, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    rng = random.Random(seed)
    for sequence_rows in grouped.values():
        rng.shuffle(sequence_rows)
    sequences = sorted(grouped)
    rng.shuffle(sequences)
    output = []
    while len(output) < limit:
        added = False
        for sequence in sequences:
            if grouped[sequence]:
                output.append(grouped[sequence].pop())
                added = True
                if len(output) >= limit:
                    break
        if not added:
            break
    return output


def render_html(rows, storage_namespace="jersey-crop-audit"):
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    storage_key = json.dumps(storage_namespace)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Jersey crop pseudo-label audit</title>
<style>body{{font:14px system-ui;background:#111;color:#eee;margin:20px}}.toolbar{{position:sticky;top:0;background:#111;padding:10px;z-index:2}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}.card{{background:#222;padding:10px;border-radius:8px}}img{{width:100%;height:280px;object-fit:contain;background:#000}}.truth{{display:none;color:#fc6}}body.reveal .truth{{display:block}}select{{width:100%;padding:6px}}</style></head><body>
<div class="toolbar"><button onclick="document.body.classList.toggle('reveal')">Toggle pseudo/GT</button> <button onclick="download()">Export review.csv</button> <span id="p"></span></div><div class="grid" id="root"></div>
<script>const rows={payload},storageKey={storage_key},labels=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function card(r){{return `<div class="card">${{r.asset?`<img loading="lazy" src="${{esc(r.asset)}}">`:'missing'}}<div>frame ${{r.frame}} · q=${{Number(r.crop_quality).toFixed(3)}}</div><div class="truth">pseudo=${{r.pseudo_label}} #${{r.pseudo_number}} · GT #${{r.gt_jersey}}<br>${{esc(r.source_winners)}}</div><select onchange="setv('${{r.audit_id}}',this.value)"><option value="">unreviewed</option>${{['clean_back','usable_not_back','not_clean','wrong_pseudolabel','uncertain'].map(v=>`<option ${{labels[r.audit_id]===v?'selected':''}}>${{v}}</option>`).join('')}}</select></div>`}};
function render(){{root.innerHTML=rows.map(card).join('');p.textContent=Object.keys(labels).length+'/'+rows.length+' reviewed'}}function setv(k,v){{if(v)labels[k]=v;else delete labels[k];localStorage.setItem(storageKey,JSON.stringify(labels));render()}}
function download(){{let s='audit_id,review_label,pseudo_label,sequence,gt_track_id,frame,crop_path\\n';for(const r of rows)s+=`"${{r.audit_id}}","${{labels[r.audit_id]||''}}","${{r.pseudo_label}}","${{r.sequence}}","${{r.gt_track_id}}","${{r.frame}}","${{String(r.crop_path).replaceAll('"','""')}}"\\n`;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s]));a.download='review.csv';a.click()}}render();</script></body></html>"""


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
