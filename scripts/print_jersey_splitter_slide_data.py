#!/usr/bin/env python3
"""Parse audit_tracklet_identity_splitter.py stdout logs into the slide table.

The splitter script has no --output-dir: it only prints to stdout. This parses
the two summary lines it prints per run
("tracklets_evaluated=N skipped_too_few_samples=M", "flagged_as_mixed_identity=K")
plus each following JSON line for a flagged tracklet, from a log saved with
`| tee`.

    python scripts/print_jersey_splitter_slide_data.py \
        --run "Int-Ata=logs/splitter_audit_int_ata.log" \
        --run "Inter-Juve=logs/splitter_audit_inter_juve.log" \
        --run "Inter-Atalanta=logs/splitter_audit_inter_atalanta.log"
"""

import argparse
import json
import re
from pathlib import Path

SUMMARY_RE = re.compile(r"tracklets_evaluated=(\d+) skipped_too_few_samples=(\d+)")
FLAG_COUNT_RE = re.compile(r"flagged_as_mixed_identity=(\d+)")


def parse_log(path):
    text = Path(path).read_text()
    summary_match = SUMMARY_RE.search(text)
    flag_count_match = FLAG_COUNT_RE.search(text)
    if not summary_match or not flag_count_match:
        raise SystemExit(f"{path} does not contain the expected summary lines")

    evaluated = int(summary_match.group(1))
    flagged_count = int(flag_count_match.group(1))

    flagged = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('{"display_track_id"'):
            flagged.append(json.loads(line))
    if len(flagged) != flagged_count:
        raise SystemExit(
            f"{path}: flagged_as_mixed_identity={flagged_count} but parsed "
            f"{len(flagged)} JSON lines -- log truncated or format changed"
        )
    return evaluated, flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=LOGFILE")
    args = parser.parse_args()

    print(f"{'video':16s}{'tracklet':>10s}{'flag':>6s}   dettaglio")
    for entry in args.run:
        label, path = entry.split("=", 1)
        evaluated, flagged = parse_log(path)
        detail = ", ".join(
            f"track {f['display_track_id']} ({'+'.join(str(v) for v in f['cluster_sizes'].values())})"
            for f in flagged
        ) or "-"
        print(f"{label:16s}{evaluated:>10d}{len(flagged):>6d}   {detail}")


if __name__ == "__main__":
    main()
