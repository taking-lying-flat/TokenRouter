import math
from typing import Optional

import torch
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class RStitchLLMScheduler(BaseTokenRoutingScheduler):

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
        scheduler_kwargs = server_args.scheduler_kwargs
        self.threshold = scheduler_kwargs.pop("threshold", 0.5)

    def route(self, batch, result) -> torch.Tensor:
        logits = result.logits_output.next_token_logits  # batch_size, vocab_size
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(probs.shape[-1])
        return (entropy <= self.threshold).cpu()
