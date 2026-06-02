import json
import re
from pathlib import Path

import numpy as np


IMAGE_ID_KEYS = ("image_id", "image_ids", "sample_id", "frame_id", "filename", "file_name")


def load_tvcalib_homography(
    path,
    image_id=None,
    frame=None,
    coordinate_system="tvcalib_centered",
    invert=False,
    pitch_length=105.0,
    pitch_width=68.0,
    temporal_index=0,
):
    """Load one TVCalib homography and convert it to FT image-to-pitch coordinates."""
    records = load_tvcalib_records(path)
    if not records:
        raise ValueError(f"No TVCalib records found in {path}")
    record = _select_record(records, image_id=image_id, frame=frame)
    if "homography" not in record:
        raise ValueError(f"Selected TVCalib record has no homography field: {path}")
    return tvcalib_homography_to_pitch(
        record["homography"],
        coordinate_system=coordinate_system,
        invert=invert,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        temporal_index=temporal_index,
    )


def load_tvcalib_homography_map(
    path,
    coordinate_system="tvcalib_centered",
    invert=False,
    pitch_length=105.0,
    pitch_width=68.0,
    temporal_index=0,
    frame_offset=0,
):
    """Return {frame_index: homography} from a TVCalib per_sample_output file."""
    records = load_tvcalib_records(path)
    homographies = {}
    for index, record in enumerate(records):
        if "homography" not in record:
            continue
        frame = _record_frame(record)
        if frame is None:
            frame = index
        frame = int(frame) - int(frame_offset or 0)
        homographies[frame] = tvcalib_homography_to_pitch(
            record["homography"],
            coordinate_system=coordinate_system,
            invert=invert,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            temporal_index=temporal_index,
        )
    if not homographies:
        raise ValueError(f"No TVCalib homographies found in {path}")
    return homographies


def load_tvcalib_records(path):
    """Load TVCalib JSON, JSONL, or keyed-record output."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "homography" in payload:
            return [payload]
        for key in ("records", "samples", "per_sample_output"):
            if isinstance(payload.get(key), list):
                return payload[key]
        keyed_records = []
        for key, value in payload.items():
            if isinstance(value, dict) and "homography" in value:
                record = dict(value)
                record.setdefault("image_id", key)
                keyed_records.append(record)
        if keyed_records:
            return keyed_records
    raise ValueError(f"Unsupported TVCalib output format: {path}")


def tvcalib_homography_to_pitch(
    homography,
    coordinate_system="tvcalib_centered",
    invert=False,
    pitch_length=105.0,
    pitch_width=68.0,
    temporal_index=0,
):
    """Convert a TVCalib homography into FT's [0, L] x [0, W] pitch meters."""
    h = _as_homography(homography, temporal_index=temporal_index)
    if invert:
        h = np.linalg.inv(h)

    coordinate_system = (coordinate_system or "tvcalib_centered").lower()
    if coordinate_system in {"tvcalib_centered", "centered", "soccer_pitch_centered"}:
        translate_to_ft_pitch = np.asarray(
            [
                [1.0, 0.0, float(pitch_length) / 2.0],
                [0.0, 1.0, float(pitch_width) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        h = translate_to_ft_pitch @ h
    elif coordinate_system in {"ft", "pitch", "top_left", "top_left_pitch", "none"}:
        pass
    else:
        raise ValueError(
            "Unsupported TVCalib coordinate system "
            f"{coordinate_system!r}; expected tvcalib_centered or ft"
        )

    if not np.isfinite(h).all():
        raise ValueError("TVCalib homography contains non-finite values")
    if abs(float(h[2, 2])) > 1e-12:
        h = h / h[2, 2]
    return h.astype(np.float32)


def _select_record(records, image_id=None, frame=None):
    if image_id is not None:
        wanted = _normalize_image_id(image_id)
        for record in records:
            if wanted in {_normalize_image_id(value) for value in _record_image_values(record)}:
                return record
        raise ValueError(
            f"No TVCalib record found for image_id={image_id!r}; "
            f"available={_available_ids(records)}"
        )

    if frame is not None:
        frame = int(frame)
        for record in records:
            if _record_frame(record) == frame:
                return record
        raise ValueError(
            f"No TVCalib record found for frame={frame}; available={_available_ids(records)}"
        )

    return records[0]


def _record_image_values(record):
    values = []
    for key in IMAGE_ID_KEYS:
        value = record.get(key)
        if value is None:
            continue
        values.extend(_flatten_values(value))
    return values


def _flatten_values(value):
    if isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            flat.extend(_flatten_values(item))
        return flat
    return [value]


def _normalize_image_id(value):
    if value is None:
        return None
    value = str(value)
    stem = Path(value).stem
    return stem or value


def _record_frame(record):
    for key in ("frame", "frame_index", "frame_idx"):
        if record.get(key) is not None:
            return int(record[key])
    for value in _record_image_values(record):
        number = _numeric_suffix(_normalize_image_id(value))
        if number is not None:
            return number
    return None


def _numeric_suffix(value):
    if value is None:
        return None
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def _available_ids(records, limit=8):
    ids = []
    for record in records[:limit]:
        values = _record_image_values(record)
        ids.append(values[0] if values else record.get("frame", record.get("frame_index")))
    return ids


def _as_homography(value, temporal_index=0):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 3):
        h = arr
    elif arr.ndim >= 3 and arr.shape[-2:] == (3, 3):
        matrices = arr.reshape(-1, 3, 3)
        index = int(temporal_index)
        if index < 0 or index >= len(matrices):
            raise ValueError(
                f"temporal_index={temporal_index} outside homography stack "
                f"of size {len(matrices)}"
            )
        h = matrices[index]
    else:
        raise ValueError(
            f"TVCalib homography must be shaped 3x3 or (..., 3, 3), got {arr.shape}"
        )
    if not np.isfinite(h).all():
        raise ValueError("TVCalib homography contains non-finite values")
    return h
