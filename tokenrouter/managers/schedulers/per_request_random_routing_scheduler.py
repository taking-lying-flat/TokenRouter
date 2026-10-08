"""Per-request reproducible random routing."""

from __future__ import annotations

from typing import Optional

import torch
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.deterministic_random import (
    request_routing_probability,
    token_route_uniform,
)
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class PerRequestRandomRoutingScheduler(BaseTokenRoutingScheduler):
    """Assign each request a peer-switching probability and reuse it.

    A routing decision denotes whether the request should move to the peer
    after the current forward pass.  Each request gets one switching
    probability in the configured range; both models use that request-specific
    probability, matching ``RandomRoutingScheduler`` semantics.

    The probability and routing draws are deterministic functions of the
    request id, a monotonically increasing per-request routing step, and the
    configured seed.  The step is transferred with the request so retries at
    an unchanged output length cannot repeat the same draw and livelock.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta,
        )
        kwargs = self.server_args.scheduler_kwargs
        self.min_routing_p = float(
            kwargs.get("min_routing_p", kwargs.get("min_reference_p", 0.0))
        )
        self.max_routing_p = float(
            kwargs.get("max_routing_p", kwargs.get("max_reference_p", 1.0))
        )
        self.routing_seed = int(kwargs.get("routing_seed", 0))
        self.min_local_steps_after_switch = int(
            kwargs.get("min_local_steps_after_switch", 0)
        )
        self.finish_guard_tokens = int(kwargs.get("finish_guard_tokens", 0))
        self.entry_server_name = str(kwargs.get("entry_server_name", "quick"))
        self._validate_routing_args()

    def _validate_routing_args(self) -> None:
        if not 0.0 <= self.min_routing_p <= self.max_routing_p <= 1.0:
            raise ValueError(
                "Per-request routing probabilities must satisfy "
                "0 <= min_routing_p <= max_routing_p <= 1; got "
                f"{self.min_routing_p=} {self.max_routing_p=}"
            )
        if self.min_local_steps_after_switch < 0:
            raise ValueError("min_local_steps_after_switch must be nonnegative")
        if self.finish_guard_tokens < 0:
            raise ValueError("finish_guard_tokens must be nonnegative")
        if self.entry_server_name not in self.server_args.all_server_names:
            raise ValueError(
                f"entry_server_name={self.entry_server_name!r} is not in "
                f"all_server_names={self.server_args.all_server_names!r}"
            )

    def routing_probability(self, request_id: str) -> float:
        return request_routing_probability(
            self.routing_seed,
            request_id,
            self.min_routing_p,
            self.max_routing_p,
        )

    def route(self, batch, result) -> torch.Tensor:
        decisions = []
        for req in batch.reqs:
            probability = self.routing_probability(req.rid)
            routing_step = getattr(req, "_tokenrouter_random_routing_step", 0)
            draw = token_route_uniform(
                self.routing_seed,
                req.rid,
                routing_step,
            )
            req._tokenrouter_random_routing_step = routing_step + 1
            max_new_tokens = getattr(
                getattr(req, "sampling_params", None), "max_new_tokens", None
            )
            remaining_after_current = (
                max_new_tokens - len(req.output_ids) - 1
                if max_new_tokens is not None
                else None
            )
            cooldown = getattr(req, "_tokenrouter_random_routing_cooldown", 0)
            if (
                self.finish_guard_tokens > 0
                and remaining_after_current is not None
                and remaining_after_current <= self.finish_guard_tokens
            ):
                req._tokenrouter_random_routing_cooldown = 0
                # Make the HTTP entry scheduler own request termination.  The
                # cooperative peer's finished-message path can lose the final
                # stream event, so return before the request reaches it.
                decisions.append(
                    self.server_args.server_name != self.entry_server_name
                )
            elif cooldown > 0:
                req._tokenrouter_random_routing_cooldown = cooldown - 1
                decisions.append(False)
            else:
                send_to_peer = draw < probability
                req._tokenrouter_random_routing_cooldown = (
                    self.min_local_steps_after_switch if send_to_peer else 0
                )
                decisions.append(send_to_peer)

        return torch.tensor(decisions, dtype=torch.bool)
