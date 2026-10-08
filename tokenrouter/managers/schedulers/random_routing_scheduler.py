import hashlib
from bisect import bisect_left
from typing import Optional

from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class RandomRoutingScheduler(BaseTokenRoutingScheduler):
    """Route each next-token step using a fixed global model distribution."""

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
        self.routing_seed = int(kwargs.get("routing_seed", 0))
        configured = kwargs.get("routing_probabilities")
        model_names = self.server_args.all_server_names

        if configured is None:
            if len(model_names) != 2:
                raise ValueError(
                    "routing_probabilities is required for multi-model random routing"
                )
            send_peer_p = float(kwargs.get("send_peer_p", 0.5))
            probabilities = {
                self.server_args.server_name: 1.0 - send_peer_p,
                self.peer_name: send_peer_p,
            }
        elif isinstance(configured, dict):
            missing = set(model_names) - set(configured)
            extra = set(configured) - set(model_names)
            if missing or extra:
                raise ValueError(
                    "routing_probabilities keys must exactly match model names; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            probabilities = {
                name: float(configured[name]) for name in model_names
            }
        else:
            if len(configured) != len(model_names):
                raise ValueError(
                    "routing_probabilities list length must match model count"
                )
            probabilities = {
                name: float(probability)
                for name, probability in zip(model_names, configured)
            }

        if any(probability < 0 for probability in probabilities.values()):
            raise ValueError("routing probabilities must be non-negative")
        probability_sum = sum(probabilities.values())
        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError(
                f"routing probabilities must sum to 1, got {probability_sum}"
            )

        self.routing_model_names = model_names
        cumulative = 0.0
        self.routing_cdf = []
        for name in model_names:
            cumulative += probabilities[name]
            self.routing_cdf.append(cumulative)
        self.routing_cdf[-1] = 1.0

    def route(self, batch, result) -> list[str]:
        destinations = []
        for req in batch.reqs:
            # Stable per request/token step, independent of batching order.
            token_step = len(req.output_ids)
            digest = hashlib.sha256(
                f"{self.routing_seed}:{req.rid}:{token_step}".encode()
            ).digest()
            sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
            index = min(bisect_left(self.routing_cdf, sample), len(self.routing_cdf) - 1)
            destinations.append(self.routing_model_names[index])
        return destinations
