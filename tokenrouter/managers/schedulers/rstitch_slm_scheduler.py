import math
from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class RStitchSLMScheduler(BaseTokenRoutingScheduler):

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
        return (entropy > self.threshold).cpu()

    def send(self, req: Req, destination: Optional[str] = None) -> PeerReq:
        destination = destination or self.peer_name
        req.model_tokens[self.server_args.server_name][1] -= 1
        current_token_ids = req.origin_input_ids + req.output_ids[:-1]
        req_to_send = PeerReq.construct_from_req(
            req=req,
            source=self.server_args.server_name,
            destination=destination,
            current_token_ids=current_token_ids,
            entry_model=self.server_args.entry_model_name,
            handoff_seq=req.handoff_seq,
        )
        return req_to_send
