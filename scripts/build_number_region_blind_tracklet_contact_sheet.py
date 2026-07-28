#!/usr/bin/env python3
"""Build a single self-contained HTML contact sheet for a small set of tracklets.

Purpose: a quick qualitative look at the crops the number-region detector
already selected but found nothing in (the "blind" tracklets from the
coverage-gap audit), to spot a shared visual pattern (team kit, camera angle,
blur) before deciding whether the detector needs more training data. This is
a one-time visual check, not a manual annotation campaign: it embeds the
images already selected by the pipeline, no new crops, no labels collected.

Input: `region_detection_coverage.csv` produced by
`audit_jersey_number_region_detector_coverage.py` (has crop_path, sequence,
gt_track_id, frame, crop_quality, detector_confidence, detected).

Output: one HTML file with images embedded as base64 data URIs, so it can be
copied off the remote machine and opened anywhere without the original crop
paths.
"""
import argparse
import base64
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="region_detection_coverage.csv path")
    parser.add_argument(
        "--tracks",
        required=True,
        help="comma-separated sequence::gt_track_id pairs to include",
    )
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--max-side", type=int, default=240, help="thumbnail max side in px")
    args = parser.parse_args()

    requested = {item.strip() for item in args.tracks.split(",") if item.strip()}
    rows = read_csv(args.csv)
    grouped = defaultdict(list)
    for row in rows:
        label = f"{row.get('sequence', '')}::{row.get('gt_track_id', '')}"
        if label in requested:
            grouped[label].append(row)

    missing = requested - set(grouped)
    if missing:
        print(f"WARNING: no rows found for: {sorted(missing)}", file=sys.stderr)

    sections = []
    for label in sorted(grouped):
        items = sorted(grouped[label], key=lambda row: int(row.get("frame") or 0))
        cards = [render_card(row, args.max_side) for row in items]
        sections.append(
            f'<section><h2>{escape(label)}</h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    html = HTML_TEMPLATE.format(body="\n".join(sections), count=len(grouped))
    Path(args.output_html).write_text(html, encoding="utf-8")
    print(f"tracklets_rendered={len(grouped)}")
    print(f"html={args.output_html}")


def render_card(row, max_side):
    crop_path = row.get("crop_path")
    data_uri, error = encode_thumbnail(crop_path, max_side)
    detected = str(row.get("detected", "")).lower() == "true"
    confidence = row.get("detector_confidence") or ""
    caption = (
        f"frame {escape(str(row.get('frame', '')))} &middot; "
        f"quality {escape(str(row.get('crop_quality', '')))[:5]} &middot; "
        f"{'detected conf ' + escape(str(confidence))[:5] if detected else 'no detection'}"
    )
    if error:
        body = f'<div class="missing">unreadable: {escape(error)}</div>'
    else:
        body = f'<img src="{data_uri}" alt="{escape(str(crop_path))}">'
    return f'<div class="card">{body}<div class="caption">{caption}</div></div>'


def encode_thumbnail(path, max_side):
    try:
        from PIL import Image
    except ImportError as exc:
        return None, str(exc)
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}", None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Blind tracklet contact sheet ({count} tracklets)</title>
<style>
body {{ font-family: sans-serif; background: #111; color: #eee; margin: 1.5rem; }}
h2 {{ border-bottom: 1px solid #444; padding-bottom: 0.25rem; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 2rem; }}
.card {{ background: #1c1c1c; border-radius: 6px; padding: 0.4rem; width: 260px; }}
.card img {{ max-width: 100%; display: block; border-radius: 4px; }}
.missing {{ padding: 2rem 0.5rem; text-align: center; color: #f66; font-size: 0.85rem; }}
.caption {{ font-size: 0.75rem; color: #aaa; margin-top: 0.35rem; }}
</style>
</head>
<body>
<h1>Blind tracklet contact sheet</h1>
<p>Crops already selected by the pipeline for tracklets with zero detected number-region
across the full confidence sweep. Qualitative look only, no new labels collected.</p>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
