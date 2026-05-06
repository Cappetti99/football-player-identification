import argparse
from copy import deepcopy
from pathlib import Path

from ft.config import apply_overrides, load_config


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def build_parser():
    parser = argparse.ArgumentParser(description="FT football tracking pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the end-to-end FT pipeline")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--video-path", default=None)
    run.add_argument("--input-dir", default=None, help="Directory containing videos to process")
    run.add_argument("--output-dir", default=None, help="Output directory used with --input-dir")
    run.add_argument("--artifacts-root", default=None, help="Artifacts root directory used with --input-dir")
    run.add_argument("--limit", type=int, default=None, help="Maximum number of videos to process from --input-dir")
    run.add_argument("--model-path", default=None)
    run.add_argument("--output-path", default=None)
    run.add_argument("--artifacts-dir", default=None)
    run.add_argument("--roster-path", default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    run.add_argument("--wandb-project", default=None)
    run.add_argument("--wandb-name", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from ft.pipeline import run_pipeline

        config = load_config(args.config)
        config = apply_overrides(config, args)
        if args.input_dir:
            run_batch(config, args)
        else:
            run_pipeline(config)


def run_batch(config, args):
    videos = discover_videos(args.input_dir, limit=args.limit)
    if not videos:
        raise RuntimeError(f"No video files found in input directory: {args.input_dir}")

    output_dir = Path(args.output_dir or "output_videos/costume-video")
    artifacts_root = Path(args.artifacts_root or "artifacts/costume-video")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    print(f"FT input dir: {args.input_dir}")
    print(f"FT batch videos: {len(videos)}")
    for index, video_path in enumerate(videos, start=1):
        video_id = video_path.stem
        print(f"FT batch {index}/{len(videos)}: {video_path}")
        run_config = deepcopy(config)
        run_config["video_path"] = str(video_path)
        run_config["output_path"] = str(output_dir / f"{video_id}_ft.mp4")
        run_config["artifacts_dir"] = str(artifacts_root / f"{video_id}_ft")
        run_pipeline(run_config)


def discover_videos(input_dir, limit=None):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    videos = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if limit is not None:
        videos = videos[: int(limit)]
    return videos


if __name__ == "__main__":
    main()
