#!/usr/bin/env python3
"""Build a portable HTML audit of paired CTC recoveries and regressions."""

import argparse
import csv
import html
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualitative-cases", required=True)
    parser.add_argument("--ocr-diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crops-per-track", type=int, default=3)
    args = parser.parse_args()
    if args.crops_per_track < 1:
        raise ValueError("--crops-per-track must be positive")

    cases = read_csv(args.qualitative_cases)
    diagnostics = json.loads(Path(args.ocr_diagnostics).read_text())
    crops = crops_by_eval_id(diagnostics)
    output = Path(args.output_dir).resolve()
    assets = output / "assets"
    output.mkdir(parents=True, exist_ok=False)
    assets.mkdir()

    copied, missing = 0, 0
    cards = []
    for case_index, case in enumerate(cases):
        selected = crops.get(str(case.get("eval_track_id") or ""), [])
        images = []
        for crop_index, crop in enumerate(selected[:args.crops_per_track]):
            source = Path(str(crop.get("crop_path") or ""))
            if not source.is_file():
                missing += 1
                continue
            name = f"case_{case_index:03d}_crop_{crop_index:02d}{source.suffix.lower()}"
            shutil.copy2(source, assets / name)
            images.append(f'<img loading="lazy" src="assets/{name}">')
            copied += 1
        cards.append(card(case, images))

    counts = Counter(case["transition"] for case in cases)
    summary = {
        "cases": len(cases),
        "transitions": dict(sorted(counts.items())),
        "crops_copied": copied,
        "missing_crops": missing,
        "crops_per_track": args.crops_per_track,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "index.html").write_text(page(summary, cards), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def crops_by_eval_id(payload):
    tracklets = payload.get("tracklets") if isinstance(payload, dict) else None
    records = tracklets.values() if isinstance(tracklets, dict) else []
    output = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        eval_id = str(row.get("display_track_id", row.get("eval_track_id", "")))
        output[eval_id] = list(row.get("selected_crops") or [])
    return output


def card(case, images):
    values = {key: html.escape(str(value)) for key, value in case.items()}
    return f'''<article class="{values['transition']}">
<h2>{values['transition']}: {values['sequence']} / track {values['gt_track_id']}</h2>
<p>GT <b>{values['gt_jersey']}</b> · baseline <b>{values['baseline_prediction']}</b>
· candidate <b>{values['candidate_prediction']}</b></p>
<div class="images">{''.join(images) or '<em>Nessun crop disponibile</em>'}</div>
</article>'''


def page(summary, cards):
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>CTC SJN transfer qualitative audit</title>
<style>body{{font-family:system-ui;margin:2rem;background:#f5f5f5}}article{{background:white;padding:1rem;margin:1rem 0;border-left:8px solid #27864b}}article.correct_to_wrong{{border-color:#b83b3b}}.images{{display:flex;gap:.6rem;flex-wrap:wrap}}img{{height:220px;max-width:300px;object-fit:contain;background:#222}}</style>
</head><body><h1>CTC SJN→GSR qualitative audit</h1>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>{''.join(cards)}</body></html>'''


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
