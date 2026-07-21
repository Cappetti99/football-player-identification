"""Numeric CTC recognizer and probabilistic multi-frame jersey aggregation."""

import math


DIGITS = "0123456789"
BLANK_INDEX = 10
CANDIDATES = tuple(str(value) for value in range(100))


def build_numeric_crnn(pretrained=True, hidden_size=256):
    try:
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:
        raise RuntimeError("torch and torchvision are required for numeric CTC") from exc

    class NumericCRNN(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            self.features = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            )
            self.sequence = nn.LSTM(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.2,
                bidirectional=True,
            )
            self.classifier = nn.Linear(hidden_size * 2, len(DIGITS) + 1)

        def forward(self, images):
            features = self.features(images).mean(dim=2)
            features = features.permute(2, 0, 1)
            encoded, _ = self.sequence(features)
            return self.classifier(encoded)

    return NumericCRNN()


def encode_text(text):
    text = str(text)
    if not text or len(text) > 2 or any(character not in DIGITS for character in text):
        raise ValueError(f"jersey text must contain one or two digits: {text!r}")
    return [DIGITS.index(character) for character in text]


def greedy_decode(logits):
    """Decode T x C logits, returning text and mean retained probability."""
    import torch

    probabilities = torch.softmax(logits, dim=-1)
    values, indices = probabilities.max(dim=-1)
    output = []
    confidence = []
    previous = None
    for index, value in zip(indices.tolist(), values.tolist()):
        if index != BLANK_INDEX and index != previous:
            output.append(DIGITS[index])
            confidence.append(value)
        previous = index
    text = "".join(output)
    return text, (sum(confidence) / len(confidence) if confidence else 0.0)


def ctc_log_probability(log_probs, text):
    """Exact CTC log probability for one 1–2 digit candidate."""
    import torch

    targets = encode_text(text)
    extended = [BLANK_INDEX]
    for target in targets:
        extended.extend([target, BLANK_INDEX])
    states = len(extended)
    previous = log_probs.new_full((states,), float("-inf"))
    previous[0] = log_probs[0, BLANK_INDEX]
    if states > 1:
        previous[1] = log_probs[0, extended[1]]
    for time in range(1, log_probs.shape[0]):
        current = log_probs.new_full((states,), float("-inf"))
        for state, label in enumerate(extended):
            sources = [previous[state]]
            if state > 0:
                sources.append(previous[state - 1])
            if state > 1 and label != BLANK_INDEX and label != extended[state - 2]:
                sources.append(previous[state - 2])
            current[state] = torch.logsumexp(torch.stack(sources), dim=0) + log_probs[time, label]
        previous = current
    return torch.logsumexp(previous[-2:] if states > 1 else previous[-1:], dim=0)


def candidate_log_probabilities(logits, candidates=CANDIDATES):
    import torch

    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.stack([ctc_log_probability(log_probs, value) for value in candidates])
    scores = scores - torch.logsumexp(scores, dim=0)
    return {candidate: float(score) for candidate, score in zip(candidates, scores)}


def aggregate_frames(frame_scores, weights=None, minimum_weight=0.05):
    """Aggregate normalized candidate log probabilities across distinct frames."""
    if not frame_scores:
        return {"prediction": None, "confidence": 0.0, "margin": 0.0, "scores": {}}
    weights = weights or [1.0] * len(frame_scores)
    if len(weights) != len(frame_scores):
        raise ValueError("weights and frame_scores must have equal length")
    candidates = sorted(set.intersection(*(set(scores) for scores in frame_scores)), key=jersey_sort_key)
    totals = {
        candidate: sum(
            max(minimum_weight, float(weight)) * float(scores[candidate])
            for scores, weight in zip(frame_scores, weights)
        )
        for candidate in candidates
    }
    normalizer = logsumexp(totals.values())
    probabilities = {candidate: math.exp(value - normalizer) for candidate, value in totals.items()}
    ranking = sorted(probabilities.items(), key=lambda item: (-item[1], jersey_sort_key(item[0])))
    winner, confidence = ranking[0]
    runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
    return {
        "prediction": int(winner),
        "confidence": confidence,
        "margin": confidence - runner_up,
        "scores": dict(ranking),
    }


def digit_factorized_log_scores(candidate_scores):
    """Factor a normalized 0–99 distribution into length/tens/units marginals."""
    if set(candidate_scores) != set(CANDIDATES):
        raise ValueError("candidate_scores must contain every jersey number from 0 to 99")
    one_digit = logsumexp(candidate_scores[str(value)] for value in range(10))
    two_digit = logsumexp(candidate_scores[str(value)] for value in range(10, 100))
    units = {
        digit: logsumexp(
            candidate_scores[str(value)] for value in range(100) if value % 10 == digit
        )
        for digit in range(10)
    }
    tens = {
        digit: logsumexp(candidate_scores[str(value)] for value in range(digit * 10, digit * 10 + 10))
        - two_digit
        for digit in range(1, 10)
    }
    values = {}
    for number in range(100):
        if number < 10:
            values[str(number)] = one_digit + units[number]
        else:
            values[str(number)] = two_digit + tens[number // 10] + units[number % 10]
    normalizer = logsumexp(values.values())
    return {key: value - normalizer for key, value in values.items()}


def interpolate_track_scores(frame_scores, digit_weight):
    """Fuse whole-number CTC evidence with digit-factorized evidence."""
    if not 0 <= digit_weight <= 1:
        raise ValueError("digit_weight must be between 0 and 1")
    if not frame_scores:
        return {"prediction": None, "confidence": 0.0, "margin": 0.0, "scores": {}}
    whole = {
        candidate: sum(scores[candidate] for scores in frame_scores)
        for candidate in CANDIDATES
    }
    digit_frames = [digit_factorized_log_scores(scores) for scores in frame_scores]
    digits = {
        candidate: sum(scores[candidate] for scores in digit_frames)
        for candidate in CANDIDATES
    }
    combined = {
        candidate: (1.0 - digit_weight) * whole[candidate] + digit_weight * digits[candidate]
        for candidate in CANDIDATES
    }
    normalizer = logsumexp(combined.values())
    probabilities = {candidate: math.exp(value - normalizer) for candidate, value in combined.items()}
    ranking = sorted(probabilities.items(), key=lambda item: (-item[1], jersey_sort_key(item[0])))
    winner, confidence = ranking[0]
    runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
    return {
        "prediction": int(winner),
        "confidence": confidence,
        "margin": confidence - runner_up,
        "scores": dict(ranking),
    }


def logsumexp(values):
    values = list(values)
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def jersey_sort_key(value):
    return int(value)
