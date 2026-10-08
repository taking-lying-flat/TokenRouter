import math
from typing import Optional


def _exponential_warmup(step, warmup_steps, start_value, end_value):
    if step < warmup_steps:
        value = start_value * (end_value / start_value) ** (step / warmup_steps)
    else:
        value = end_value
    return value


def _cosine_warmup(step, warmup_steps, start_value, end_value):
    if step < warmup_steps:
        value = end_value + (start_value - end_value) / 2 * (
            math.cos(math.pi * step / warmup_steps) + 1
        )
    else:
        value = end_value
    return value


def _linear_warmup(step, warmup_steps, start_value, end_value):
    if step < warmup_steps:
        value = start_value + (end_value - start_value) * step / warmup_steps
    else:
        value = end_value
    return value


def create_deferral_threshold_schedule(
    warmup_schedule: Optional[str],
    warmup_steps: int,
    start_value: float,
    end_value: float,
):
    if warmup_schedule is None:
        warmup_schedule = "constant"

    if warmup_schedule == "exponential":
        warmup_func = _exponential_warmup
    elif warmup_schedule == "cosine":
        warmup_func = _cosine_warmup
    elif warmup_schedule == "linear":
        warmup_func = _linear_warmup
    elif warmup_schedule == "constant":
        warmup_func = lambda step, warmup_steps, start_value, end_value: end_value
    else:
        raise ValueError(f"Unknown warmup schedule: {warmup_schedule}")

    return [
        warmup_func(step, warmup_steps, start_value, end_value)
        for step in range(warmup_steps)
    ]
