# Costume Video Input

Put the videos you want to analyse in this folder.

Supported extensions:

```text
.mp4
.mov
.avi
.mkv
.m4v
```

Example:

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --input-dir costume-video \
  --model-path best_yolo26x_gsr_light.pt \
  --roster-path configs/roster.example.json \
  --max-frames 1800
```

Outputs are written to:

```text
output_videos/costume-video/
artifacts/costume-video/
```
