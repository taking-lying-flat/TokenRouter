from typing import Optional

import torch
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class R2RLLMScheduler(BaseTokenRoutingScheduler):

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

    def route(self, batch, result) -> torch.Tensor:
        return torch.ones((batch.batch_size(),))
