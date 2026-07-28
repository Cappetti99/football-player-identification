"""Decision layer: turns evidence into the assignments the pipeline applies."""

from ft.decision.jersey_policy import (
    DEFAULT_JERSEY_POLICY,
    resolve_jersey_assignments,
)

__all__ = ["DEFAULT_JERSEY_POLICY", "resolve_jersey_assignments"]
