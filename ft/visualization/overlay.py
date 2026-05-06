import cv2


def draw_overlay(frames, tracks):
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
                label = player_label(raw_id, track)
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


def player_label(raw_id, track):
    display_id = track.get("display_track_id", raw_id)
    team = track.get("team")
    team_label = "?" if team in (None, 0) else str(team)
    group = group_short_label(track)
    player_id = track.get("player_id", "unknown")
    jersey = track.get("jersey_number")
    conf = float(track.get("identity_confidence", 0.0))
    if player_id == "unknown":
        if jersey is not None:
            return f"{group or 'T' + team_label} #{jersey} ID{display_id}"
        return f"{group or 'T' + team_label} ID{display_id}"
    if jersey is not None:
        return f"{group or 'T' + team_label} #{jersey} {player_id} {conf:.2f}"
    return f"{group or 'T' + team_label} {player_id} {conf:.2f}"


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
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - h - 5), (x + w + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
