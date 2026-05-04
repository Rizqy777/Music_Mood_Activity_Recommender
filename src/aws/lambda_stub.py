from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LambdaTriggerEvent:
    layer: str
    dataset: str
    location: str


def build_lambda_event(layer: str, dataset: str, location: str) -> dict[str, str]:
    """Prepare the event shape expected by a future AWS Lambda trigger."""
    event = LambdaTriggerEvent(layer=layer, dataset=dataset, location=location)
    return {
        "layer": event.layer,
        "dataset": event.dataset,
        "location": event.location,
    }
