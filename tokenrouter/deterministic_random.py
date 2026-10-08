"""Stable pseudo-random helpers for reproducible routing experiments."""

from __future__ import annotations

import hashlib
from typing import Union


MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


def splitmix64(value: int) -> int:
    """Mix an integer into a deterministic unsigned 64-bit value."""
    value = (value + _SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def stable_hash64(*parts: Union[str, int, bytes]) -> int:
    """Hash heterogeneous values without Python's per-process hash salt."""
    digest = hashlib.blake2b(digest_size=8, person=b"TokenRouter")
    for part in parts:
        if isinstance(part, bytes):
            encoded = part
        else:
            encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def uint64_to_unit_float(value: int) -> float:
    """Map an unsigned 64-bit integer to a float in ``[0, 1)``."""
    return (value >> 11) * (1.0 / (1 << 53))


def request_seed(routing_seed: int, request_id: str) -> int:
    """Return the stable base seed shared by both schedulers for a request."""
    return stable_hash64("request", routing_seed, request_id)


def request_routing_probability(
    routing_seed: int,
    request_id: str,
    min_routing_p: float,
    max_routing_p: float,
) -> float:
    """Assign one stable peer-switching probability to a request."""
    base = request_seed(routing_seed, request_id)
    unit = uint64_to_unit_float(splitmix64(base ^ 0xD1B54A32D192ED03))
    return min_routing_p + (max_routing_p - min_routing_p) * unit


# Kept as a source-compatible alias for configurations/tests created during
# development before the probability semantics were named precisely.
request_reference_probability = request_routing_probability


def token_route_uniform(
    routing_seed: int,
    request_id: str,
    token_position: int,
) -> float:
    """Return a stable per-request, per-token routing draw in ``[0, 1)``."""
    base = request_seed(routing_seed, request_id)
    token_value = (base + (token_position + 1) * _SPLITMIX_GAMMA) & MASK64
    return uint64_to_unit_float(splitmix64(token_value ^ 0x94D049BB133111EB))
