"""Rebuild MP4 clips from SoccerNet `img1` frame folders.

The pipeline expects a video file as input. SoccerNet Tracking and SoccerNet-GSR
are stored as frame folders, so this script creates MP4s without touching the
original dataset.

Brought over from the legacy Football-Tracking tree unchanged, so that FT can
build its own inputs instead of depending on another checkout. Works for GSR as
is: `seqinfo.ini` is optional, and without it the FPS falls back to 25 and the
frame size is read from the first image.

    python scripts/make_soccernet_videos.py \
        --soccernet-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
        --output-dir input_videos/soccernet_gsr \
        --sequence SNGS-089 --sequence SNGS-091
"""

import argparse
import configparser
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild MP4 videos from SoccerNet MOT frame folders."
    )
    parser.add_argument(
        "--soccernet-dir",
        default="/media/data-lie/cappetti/dataset/SoccerNet/tracking-2023",
    )
    parser.add_argument("--output-dir", default="input_videos/soccernet")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence name to export, for example SNMOT-150. Can be repeated.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split filter: train, test, challenge2023. Can be repeated.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override FPS from seqinfo.ini.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.soccernet_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"SoccerNet folder not found: {source_dir}")

    sequences = discover_sequences(source_dir, set(args.sequence), set(args.split))
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    if not sequences:
        raise FileNotFoundError(f"No SoccerNet sequences with img1 frames found under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        output_path = output_dir / f"{sequence.name}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing video: {output_path}")
            continue

        fps, width, height = read_seqinfo(sequence, fps_override=args.fps)
        # Frame file order matters. Lexicographic sorting would be wrong for
        # names like 1.jpg, 10.jpg, 2.jpg, so use numeric ids when possible.
        frame_paths = sorted(
            [
                path
                for path in (sequence / "img1").iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=frame_sort_key,
        )
        if not frame_paths:
            print(f"Skipping {sequence}: no frames found")
            continue

        if width is None or height is None:
            width, height = read_image_size(frame_paths[0])

        print(f"Writing {output_path} from {len(frame_paths)} frames at {fps:g} FPS")
        write_video(frame_paths, output_path, fps, width, height)


def discover_sequences(source_dir, names, splits):
    """Return sequence directories that contain an `img1` folder."""
    sequences = []
    for img_dir in sorted(source_dir.rglob("img1")):
        sequence = img_dir.parent
        if names and sequence.name not in names:
            continue
        if splits and not sequence_matches_split(sequence, splits):
            continue
        sequences.append(sequence)
    return sequences


def sequence_matches_split(sequence, splits):
    parts = {part.lower() for part in sequence.parts}
    normalized_splits = {split.lower() for split in splits}
    return bool(parts & normalized_splits)


def read_seqinfo(sequence, fps_override=None):
    """Read FPS and dimensions from MOT seqinfo.ini when available."""
    fps = fps_override or 25.0
    width = None
    height = None
    seqinfo = sequence / "seqinfo.ini"
    if not seqinfo.exists():
        return fps, width, height

    parser = configparser.ConfigParser()
    parser.read(seqinfo)
    if parser.has_section("Sequence"):
        if fps_override is None:
            fps = parser.getfloat("Sequence", "frameRate", fallback=fps)
        width = parser.getint("Sequence", "imWidth", fallback=None)
        height = parser.getint("Sequence", "imHeight", fallback=None)
    return fps, width, height


def frame_sort_key(path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else path.name


def read_image_size(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    height, width = image.shape[:2]
    return width, height


def write_video(frame_paths, output_path, fps, width, height):
    """Encode frames into an MP4 using OpenCV's mp4v writer."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
