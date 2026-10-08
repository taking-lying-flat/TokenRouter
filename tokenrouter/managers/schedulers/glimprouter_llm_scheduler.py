from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class GlimpRouterLLMScheduler(BaseTokenRoutingScheduler):

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
        self.delimiter_str = scheduler_kwargs.pop("delimiter_str", "\n\n")
        delimiter_token_ids = [
            self.tokenizer.encode(delimiter_str, add_special_tokens=False)
            for delimiter_str in self.delimiter_str
        ]
        self.delimiter_str_max_len = max(
            len(token_ids) for token_ids in delimiter_token_ids
        )
        self.delimiter_token_ids = delimiter_token_ids
        self.finish_thinking_rids = []
        self.delimiter_check_base_lengths = {}

    def receive(self, msgs: list) -> None:
        super().receive(msgs)
        for msg in msgs:
            if msg.status == "finished":
                if msg.rid in self.finish_thinking_rids:
                    self.finish_thinking_rids.remove(msg.rid)
                self.delimiter_check_base_lengths.pop(msg.rid, None)
            else:
                if hasattr(msg, "finish_thinking") and msg.finish_thinking:
                    self.finish_thinking_rids.append(msg.rid)

    def route(self, batch, result) -> torch.Tensor:
        batch_size = batch.batch_size()
        choices = [False] * batch_size

        for i, req in enumerate(batch.reqs):
            if req.rid not in self.delimiter_check_base_lengths:
                self.delimiter_check_base_lengths[req.rid] = len(req.output_ids)
            output_ids = req.output_ids + [result.next_token_ids[i].item()]
            output_ids = output_ids[self.delimiter_check_base_lengths[req.rid] :]
            tail_str = self.tokenizer.decode(
                output_ids[-(self.delimiter_str_max_len + 1) :]
            )
            if self.delimiter_str in tail_str:
                choices[i] = True
                self.delimiter_check_base_lengths.pop(req.rid)

        return torch.tensor(choices)

    def send(self, req: Req, destination: Optional[str] = None) -> PeerReq:
        req_to_send = super().send(req, destination)
        if req.finished() and req.rid in self.finish_thinking_rids:
            self.finish_thinking_rids.remove(req.rid)
            self.delimiter_check_base_lengths.pop(req.rid, None)

        return req_to_send
