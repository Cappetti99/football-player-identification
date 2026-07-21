#!/usr/bin/env python3
"""Create a crop audit for frozen correct-to-wrong fusion regressions."""

import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-provenance", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-crops-per-method", type=int, default=5)
    args = parser.parse_args()
    if args.max_crops_per_method < 1:
        parser.error("--max-crops-per-method must be positive")

    provenance_path = Path(args.fusion_provenance).resolve()
    baseline_root = Path(args.baseline_run).resolve()
    candidate_root = Path(args.candidate_run).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    regressions = [
        row for row in read_csv(provenance_path)
        if row.get("transition") == "correct_to_wrong"
    ]
    run_data = {
        "baseline": load_run(baseline_root),
        "candidate": load_run(candidate_root),
    }
    audit_rows = []
    sections = []
    for case_index, regression in enumerate(regressions, start=1):
        key = str(regression["sequence"]), str(regression["gt_track_id"])
        case_name = f"{case_index:02d}_{safe(key[0])}_track_{safe(key[1])}"
        case_dir = output / "cases" / case_name
        method_blocks = []
        for method, data in run_data.items():
            eval_id = data["predictions"].get(key)
            crops = selected_crop_paths(data["diagnostics"], eval_id)
            copied = []
            for crop_index, source in enumerate(crops[:args.max_crops_per_method], start=1):
                source_path = Path(source)
                if not source_path.is_file():
                    continue
                destination = case_dir / method / f"{crop_index:02d}_{source_path.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
                copied.append(destination.relative_to(output))
                audit_rows.append({
                    **regression,
                    "method": method,
                    "eval_track_id": eval_id or "",
                    "source_crop": str(source_path),
                    "audit_crop": str(destination),
                })
            images = "".join(
                f'<img src="{html.escape(str(path))}" loading="lazy">'
                for path in copied
            ) or "<p>Nessun crop disponibile.</p>"
            method_blocks.append(f"<h3>{html.escape(method)}</h3><div class='crops'>{images}</div>")
        sections.append(
            f"<section><h2>{html.escape(key[0])} — track {html.escape(key[1])}</h2>"
            f"<p>GT {html.escape(str(regression['gt_jersey_number']))}; "
            f"baseline {html.escape(str(regression['baseline_prediction']))}; "
            f"candidate/fusion {html.escape(str(regression['fused_prediction']))}</p>"
            + "".join(method_blocks) + "</section>"
        )

    if audit_rows:
        write_csv(output / "regression_crops.csv", audit_rows)
    write_csv(output / "regressions.csv", regressions)
    index = """<!doctype html><meta charset="utf-8"><title>Frozen jersey regressions</title>
<style>body{font-family:sans-serif;max-width:1200px;margin:auto}section{border-top:1px solid #aaa;padding:1rem}.crops{display:flex;gap:.5rem;flex-wrap:wrap}.crops img{height:240px;max-width:220px;object-fit:contain;background:#222}</style>
<h1>Frozen correct-to-wrong regressions</h1>""" + "".join(sections)
    (output / "index.html").write_text(index, encoding="utf-8")
    summary = {
        "regressions": len(regressions),
        "copied_crops": len(audit_rows),
        "fusion_provenance": record(provenance_path),
        "baseline_run": str(baseline_root),
        "candidate_run": str(candidate_root),
        "html": str(output / "index.html"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def load_run(root):
    predictions = {
        (str(row["sequence"]), str(row["gt_track_id"])): str(row["eval_track_id"])
        for row in read_csv(root / "predictions.csv")
    }
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    return {"predictions": predictions, "diagnostics": diagnostics}


def selected_crop_paths(diagnostics, eval_id):
    if eval_id is None:
        return []
    tracklets = diagnostics.get("tracklets") or {}
    for diagnostic in tracklets.values():
        display = diagnostic.get("display_track_id", diagnostic.get("eval_track_id"))
        if str(display) != str(eval_id):
            continue
        return [
            str(row.get("crop_path"))
            for row in diagnostic.get("selected_crops", [])
            if row.get("crop_path")
        ]
    return []


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def record(path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def safe(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


if __name__ == "__main__":
    main()
