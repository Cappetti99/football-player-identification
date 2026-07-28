#!/usr/bin/env python3
"""Merge verified pseudo-labeled region crops into the existing number-region
training set, keeping the validation partition byte-for-byte untouched.

Takes the original sequence-disjoint dataset (gsr_train_manual_v1, built by
build_yolo_jersey_number_region_dataset.py) and adds new images+labels (from
scripts/harvest_verified_region_crops.py --output-dir, one directory per
custom video: Int-Ata/Inter-Juve/Inter-Atalanta) into images/train and
labels/train only. These custom videos are not part of the GSR train/val/
frozen sequence split, so there is no leakage risk to check against that
split -- they are a disjoint domain, added purely to increase train-set size
and diversity. The validation partition (GSR sequences) stays exactly as
before, so the coverage/hard-miss comparison against the three prior
retraining attempts (smoke/full/sjn_to_gsr, all evaluated on the same frozen
OCR run) remains apples-to-apples.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", required=True, help="e.g. path to gsr_train_manual_v1")
    parser.add_argument(
        "--pseudolabel-dir", action="append", required=True,
        help="directory with images/ and labels/ subfolders (repeatable, one per video)",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = Path(args.base_dataset).resolve()
    base_manifest = json.loads((base / "manifest.json").read_text())
    if base_manifest.get("target") != "jersey_number_region":
        raise ValueError("--base-dataset does not look like a jersey_number_region dataset")
    if base_manifest.get("frozen_sequences_observed"):
        raise ValueError("base dataset already reports frozen-sequence leakage; refusing to build on it")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "images" / "train").mkdir(parents=True)
    (output / "images" / "val").mkdir(parents=True)
    (output / "labels" / "train").mkdir(parents=True)
    (output / "labels" / "val").mkdir(parents=True)

    base_train_images = copy_tree(base / "images" / "train", output / "images" / "train")
    base_train_labels = copy_tree(base / "labels" / "train", output / "labels" / "train")
    val_images = copy_tree(base / "images" / "val", output / "images" / "val")
    val_labels = copy_tree(base / "labels" / "val", output / "labels" / "val")
    if base_train_images != base_train_labels or val_images != val_labels:
        raise RuntimeError("base dataset images/labels count mismatch")

    pseudolabel_added = 0
    pseudolabel_sources = []
    for pseudo_dir in args.pseudolabel_dir:
        pseudo = Path(pseudo_dir).resolve()
        images_added = copy_tree(pseudo / "images", output / "images" / "train", prefix=pseudo.name)
        labels_added = copy_tree(pseudo / "labels", output / "labels" / "train", prefix=pseudo.name)
        if images_added != labels_added:
            raise RuntimeError(f"{pseudo}: images/labels count mismatch ({images_added} vs {labels_added})")
        pseudolabel_added += images_added
        pseudolabel_sources.append({"dir": str(pseudo), "images_added": images_added})

    yaml_content = (
        f"path: {output}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: number_region\n"
    )
    (output / "data.yaml").write_text(yaml_content)

    manifest = {
        **base_manifest,
        "format": "jersey_number_region_plus_verified_pseudolabels_v1",
        "base_dataset": str(base),
        "base_dataset_manifest_sha256": sha256(base / "manifest.json"),
        "base_train_images": base_train_images,
        "pseudolabel_sources": pseudolabel_sources,
        "pseudolabel_images_added": pseudolabel_added,
        "train": {"images": base_train_images + pseudolabel_added},
        "val": {"images": val_images},
        "note": (
            "validation partition is byte-identical to base_dataset; only "
            "images/train and labels/train were extended, with verified "
            "pseudo-labeled crops from custom videos outside the GSR "
            "train/val/frozen sequence split (disjoint domain, no leakage "
            "check applicable)."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def copy_tree(source, destination, prefix=None):
    count = 0
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        name = f"{prefix}_{path.name}" if prefix else path.name
        shutil.copy2(path, destination / name)
        count += 1
    return count


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
