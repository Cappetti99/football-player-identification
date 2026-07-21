"""Supervised multi-frame jersey recognizer with interpretable digit heads."""


ARCHITECTURE = "resnet18_attention_digit_heads_v1"


def build_multiframe_digit_recognizer(pretrained=True):
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:
        raise RuntimeError("torch and torchvision are required") from exc

    class MultiFrameDigitRecognizer(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            self.features = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            )
            self.attention = nn.Linear(512, 1)
            self.length_head = nn.Linear(512, 2)
            self.tens_head = nn.Linear(512, 10)
            self.units_head = nn.Linear(512, 10)

        def forward(self, images, mask):
            if images.ndim != 5 or mask.ndim != 2:
                raise ValueError("expected BxFxCxHxW images and BxF mask")
            batch, frames = images.shape[:2]
            encoded = self.features(images.flatten(0, 1)).mean(dim=(2, 3))
            encoded = encoded.reshape(batch, frames, -1)
            attention_logits = self.attention(encoded).squeeze(-1)
            attention = masked_softmax(attention_logits, mask)
            pooled = (encoded * attention.unsqueeze(-1)).sum(dim=1)
            return {
                "length_logits": self.length_head(pooled),
                "tens_logits": self.tens_head(pooled),
                "units_logits": self.units_head(pooled),
                "attention_logits": attention_logits,
                "attention": attention,
                "frame_length_logits": self.length_head(encoded),
                "frame_tens_logits": self.tens_head(encoded),
                "frame_units_logits": self.units_head(encoded),
            }

    return MultiFrameDigitRecognizer()


def masked_softmax(logits, mask):
    import torch

    mask = mask.bool()
    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have equal shape")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every bag must contain at least one frame")
    values = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(values, dim=1).masked_fill(~mask, 0.0)


def number_targets(values, device=None):
    import torch

    numbers = torch.as_tensor(values, dtype=torch.long, device=device)
    if torch.any((numbers < 0) | (numbers > 99)):
        raise ValueError("jersey numbers must be in [0, 99]")
    two_digits = numbers >= 10
    lengths = two_digits.long()
    tens = torch.where(two_digits, numbers // 10, torch.full_like(numbers, -100))
    units = numbers % 10
    return lengths, tens, units


def number_log_probabilities(outputs, temperature=1.0):
    """Return normalized Bx100 log probabilities from the three heads."""
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    length = torch.log_softmax(outputs["length_logits"] / temperature, dim=-1)
    tens = torch.log_softmax(outputs["tens_logits"][:, 1:] / temperature, dim=-1)
    units = torch.log_softmax(outputs["units_logits"] / temperature, dim=-1)
    one_digit = length[:, 0:1] + units
    two_digits = (
        length[:, 1:2, None]
        + tens[:, :, None]
        + units[:, None, :]
    ).reshape(-1, 90)
    scores = torch.cat([one_digit, two_digits], dim=1)
    return scores - torch.logsumexp(scores, dim=1, keepdim=True)


def load_ctc_encoder(model, checkpoint):
    """Initialize the shared ResNet encoder from a numeric CTC checkpoint."""
    state = checkpoint.get("state_dict", checkpoint)
    source = {
        key[len("features."):]: value
        for key, value in state.items()
        if key.startswith("features.")
    }
    missing, unexpected = model.features.load_state_dict(source, strict=False)
    if missing or unexpected or not source:
        raise ValueError(
            f"incompatible CTC encoder: missing={missing} unexpected={unexpected}"
        )
    return len(source)
