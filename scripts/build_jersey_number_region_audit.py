#!/usr/bin/env python3
"""Create a browser audit for manual jersey number-region boxes."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()
    rows = sequence_round_robin(load_rows(args.reviews), args.limit)
    output = Path(args.output_dir).resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    exported = []
    for index, row in enumerate(rows, start=1):
        source = Path(row["crop_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        name = f"region_{index:04d}_{row['sequence']}_f{row['frame']:06d}{source.suffix.lower()}"
        shutil.copy2(source, assets / name)
        exported.append({**row, "audit_id": f"region_{index:04d}", "asset": f"assets/{name}"})
    write_csv(output / "manifest.csv", exported)
    (output / "index.html").write_text(render_html(exported), encoding="utf-8")
    summary = {
        "rows": len(exported),
        "sequences": len({row["sequence"] for row in exported}),
        "instructions": "draw tight digit box; mark absent when no number region is visible",
        "html": str(output / "index.html"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        values = list(csv.DictReader(handle))
    output = []
    for row in values:
        label = str(row.get("review_label") or "").strip().lower()
        if label not in {"readable", "unreadable"}:
            continue
        output.append({
            "sequence": str(row.get("sequence") or ""),
            "gt_track_id": str(row.get("gt_track_id") or ""),
            "gt_jersey": str(int(float(row.get("gt_jersey") or 0))),
            "frame": int(float(row.get("frame") or 0)),
            "crop_path": str(row.get("crop_path") or ""),
            "readability_label": label,
        })
    output.sort(key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"]))
    return output


def sequence_round_robin(rows, limit):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    output = []
    sequences = sorted(grouped)
    while len(output) < limit:
        changed = False
        for sequence in sequences:
            if grouped[sequence]:
                output.append(grouped[sequence].pop(0))
                changed = True
                if len(output) >= limit:
                    break
        if not changed:
            break
    return output


def render_html(rows):
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Number region audit</title>
<style>body{{font:14px system-ui;background:#111;color:#eee;margin:16px}}.bar{{position:sticky;top:0;background:#111;padding:8px;z-index:2}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}.card{{background:#222;padding:9px;border-radius:8px}}canvas{{width:100%;height:260px;background:#000;cursor:crosshair}}select{{width:100%;padding:6px;margin-top:5px}}small{{color:#bbb}}</style></head><body>
<div class="bar"><button onclick="download()">Export number_region_review.csv</button> <span id="progress"></span></div><div class="grid" id="root"></div>
<script>const rows={payload},key='jersey-number-region-v1',state=JSON.parse(localStorage.getItem(key)||'{{}}');
function save(){{localStorage.setItem(key,JSON.stringify(state));progress.textContent=Object.values(state).filter(v=>v.label).length+'/'+rows.length+' reviewed'}}
function build(){{for(const r of rows){{let c=document.createElement('div');c.className='card';c.innerHTML=`<canvas></canvas><div>${{r.audit_id}} · ${{r.sequence}} · frame ${{r.frame}}</div><small>readability: ${{r.readability_label}}</small><select><option value="">unreviewed</option><option value="present">present</option><option value="absent">absent</option><option value="uncertain">uncertain</option></select>`;root.appendChild(c);let canvas=c.querySelector('canvas'),sel=c.querySelector('select'),img=new Image(),v=state[r.audit_id]||{{}};sel.value=v.label||'';sel.onchange=()=>{{state[r.audit_id]=state[r.audit_id]||{{}};state[r.audit_id].label=sel.value;if(sel.value!=='present')delete state[r.audit_id].box;save();draw()}};img.onload=()=>{{canvas.width=img.width;canvas.height=img.height;draw()}};img.src=r.asset;let start=null;canvas.onpointerdown=e=>{{let p=point(e,canvas);start=p}};canvas.onpointerup=e=>{{if(!start)return;let p=point(e,canvas),x0=Math.min(start.x,p.x),y0=Math.min(start.y,p.y),x1=Math.max(start.x,p.x),y1=Math.max(start.y,p.y);start=null;if(x1-x0<3||y1-y0<3)return;state[r.audit_id]={{label:'present',box:[x0/canvas.width,y0/canvas.height,x1/canvas.width,y1/canvas.height]}};sel.value='present';save();draw()}};canvas.ondblclick=()=>{{state[r.audit_id]={{}};sel.value='';save();draw()}};function draw(){{let x=canvas.getContext('2d');x.clearRect(0,0,canvas.width,canvas.height);x.drawImage(img,0,0);let b=(state[r.audit_id]||{{}}).box;if(b){{x.strokeStyle='#00ff66';x.lineWidth=Math.max(2,canvas.width/250);x.strokeRect(b[0]*canvas.width,b[1]*canvas.height,(b[2]-b[0])*canvas.width,(b[3]-b[1])*canvas.height)}}}}}}save()}}
function point(e,c){{let r=c.getBoundingClientRect();return{{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height}}}}
function q(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}function download(){{let cols=['audit_id','region_label','xmin','ymin','xmax','ymax','sequence','gt_track_id','gt_jersey','frame','crop_path','asset'];let out=cols.join(',')+'\\n';for(const r of rows){{let v=state[r.audit_id]||{{}},b=v.box||[];let row={{...r,region_label:v.label||'',xmin:b[0]??'',ymin:b[1]??'',xmax:b[2]??'',ymax:b[3]??''}};out+=cols.map(k=>q(row[k])).join(',')+'\\n'}}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out]));a.download='number_region_review.csv';a.click()}}build();</script></body></html>'''


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
