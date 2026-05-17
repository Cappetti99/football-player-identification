DEFAULT_OVERLAY_CONFIG = {
    "show_display_id": True,
    "show_jersey": True,
    "show_jersey_winner": False,
    "show_jersey_min_confidence": 0.65,
    "show_jersey_min_votes": 3,
    "show_jersey_min_stable_votes": 20,
    "show_jersey_min_winner_margin": 0.20,
    "show_jersey_min_head_votes": 8,
    "show_jersey_min_head_confidence": 0.70,
    "require_ocr_jersey_evidence": True,
    "show_player_id": False,
    "show_player_id_min_confidence": 0.80,
    "show_identity_confidence": False,
}


def draw_overlay(frames, tracks, config=None):
    import cv2

    config = overlay_config(config)
    output = []
    for frame_num, frame in enumerate(frames):
        image = frame.copy()
        for raw_id, track in tracks.get("players", [])[frame_num].items():
            bbox = [int(v) for v in track["bbox"]]
            if track.get("role_detection") == "referee_candidate":
                color = tuple(int(v) for v in track.get("semantic_group_color", (0, 255, 255)))
                label = referee_label(track, prefix="ref?")
            else:
                color = tuple(
                    int(v)
                    for v in track.get("semantic_group_color", track.get("team_color", (160, 160, 160)))
                )
                label = player_label(raw_id, track, config)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            draw_label(image, bbox[0], max(18, bbox[1] - 6), label)
        for _, track in tracks.get("referees", [])[frame_num].items():
            bbox = [int(v) for v in track["bbox"]]
            color = tuple(int(v) for v in track.get("semantic_group_color", (0, 255, 255)))
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = referee_label(track, prefix="ref")
            draw_label(image, bbox[0], max(18, bbox[1] - 6), label)
        for _, ball in tracks.get("ball", [])[frame_num].items():
            bbox = [int(v) for v in ball["bbox"]]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(image, (cx, cy), 8, (0, 0, 0), 1)
        output.append(image)
    return output


def player_label(raw_id, track, config=None):
    config = overlay_config(config)
    display_id = track.get("display_track_id", raw_id)
    team = track.get("team")
    team_label = "?" if team in (None, 0) else str(team)
    group = group_short_label(track)
    player_id = track.get("player_id", "unknown")
    jersey = visible_jersey(track, config)
    conf = float(track.get("identity_confidence", 0.0))
    parts = [group or "T" + team_label]
    if jersey is not None:
        parts.append(f"#{jersey}")
    if should_show_player_id(player_id, conf, config):
        parts.append(str(player_id))
        if config["show_identity_confidence"]:
            parts.append(f"{conf:.2f}")
    elif config["show_display_id"]:
        parts.append(f"ID{display_id}")
    return " ".join(parts)


def visible_jersey(track, config):
    if not config["show_jersey"]:
        return None
    evidence = track.get("jersey_evidence") or {}
    if config["require_ocr_jersey_evidence"] and not evidence:
        return None
    if config["show_jersey_winner"] and evidence:
        return track.get("jersey_number")
    confidence = float(evidence.get("confidence", track.get("jersey_confidence", 0.0)) or 0.0)
    head_confidence = float(evidence.get("head_confidence", 0.0) or 0.0)
    votes = int(evidence.get("votes", track.get("jersey_votes", 0)) or 0)
    winner_margin = float(evidence.get("winner_margin", 0.0) or 0.0)
    confidence_pass = (
        confidence >= config["show_jersey_min_confidence"]
        and votes >= config["show_jersey_min_votes"]
    )
    stable_pass = (
        votes >= config["show_jersey_min_stable_votes"]
        and winner_margin >= config["show_jersey_min_winner_margin"]
    )
    head_pass = (
        votes >= config["show_jersey_min_head_votes"]
        and head_confidence >= config["show_jersey_min_head_confidence"]
    )
    if not confidence_pass and not stable_pass and not head_pass:
        return None
    return track.get("jersey_number")


def should_show_player_id(player_id, confidence, config):
    return (
        config["show_player_id"]
        and player_id not in (None, "unknown")
        and confidence >= config["show_player_id_min_confidence"]
    )


def overlay_config(config):
    merged = dict(DEFAULT_OVERLAY_CONFIG)
    merged.update(config or {})
    return merged


def group_short_label(track):
    group_id = track.get("semantic_group_id")
    mapping = {
        1: "T1",
        2: "T2",
        3: "GK1",
        4: "GK2",
        5: "REF",
    }
    return mapping.get(group_id)


def referee_label(track, prefix="ref"):
    color = track.get("referee_like_color")
    score = float(track.get("referee_like_score", 0.0))
    if color:
        return f"{prefix} {color} {score:.2f}"
    return prefix


def draw_label(image, x, y, text):
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - h - 5), (x + w + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
