import math
from typing import Optional

import torch
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class GlimpRouterSLMScheduler(BaseTokenRoutingScheduler):

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
        self.delimiter_str = scheduler_kwargs.pop("delimiter_str", "\n\n")
        self.finished_str = scheduler_kwargs.pop("finished_str", "</think>")
        self.top_logprobs_nums = scheduler_kwargs.pop("top_logprobs_nums", 20)
        delimiter_token_ids = []
        delimiter_token_ids = [
            self.tokenizer.encode(delimiter_str, add_special_tokens=False)
            for delimiter_str in self.delimiter_str
        ]
        self.delimiter_str_max_len = max(
            len(token_ids) for token_ids in delimiter_token_ids
        )
        self.delimiter_token_ids = delimiter_token_ids
        self.first_token_check_rids = []
        self.finish_thinking_rids = []

    def process_input_requests(self, recv_reqs: list) -> None:
        super().process_input_requests(recv_reqs)
        for req in recv_reqs:
            if isinstance(req, TokenizedGenerateReqInput):
                self.first_token_check_rids.append(req.rid)
                req.top_logprobs_num = self.top_logprobs_nums

    def receive(self, msgs: list) -> None:
        super().receive(msgs)
        for msg in msgs:
            if msg.status == "finished":
                if msg.rid in self.first_token_check_rids:
                    self.first_token_check_rids.remove(msg.rid)
            else:
                self.first_token_check_rids.append(msg.rid)

    def route(self, batch, result) -> torch.Tensor:
        batch_size = batch.batch_size()
        choices = [False] * batch_size

        for i, req in enumerate(batch.reqs):
            if req.rid in self.first_token_check_rids:
                top_logprobs_val = result.logits_output.next_token_top_logprobs_val[i]
                top_probs_val = [math.exp(logprob) for logprob in top_logprobs_val]
                total_probs = sum(top_probs_val)
                top_probs = [prob / total_probs for prob in top_probs_val]
                entropy = -sum([prob * math.log(prob) for prob in top_probs])
                choices[i] = entropy >= self.threshold
                self.first_token_check_rids.remove(req.rid)
            else:
                output_ids = req.output_ids + [result.next_token_ids[i].item()]
                tail_str = self.tokenizer.decode(
                    output_ids[-(self.delimiter_str_max_len + 1) :]
                )
                if tail_str.endswith(self.finished_str):
                    choices[i] = True
                    self.finish_thinking_rids.append(req.rid)
                elif self.delimiter_str in tail_str:
                    self.first_token_check_rids.append(req.rid)

        return torch.tensor(choices)

    def send(self, req: Req, destination: Optional[str] = None) -> PeerReq:
        destination = destination or self.peer_name
        if req.rid in self.finish_thinking_rids:
            current_token_ids = req.origin_input_ids + req.output_ids
        else:
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
        if req.rid in self.finish_thinking_rids:
            self.finish_thinking_rids.remove(req.rid)
            req_to_send.finish_thinking = True
        if req.finished() and req.rid in self.first_token_check_rids:
            self.first_token_check_rids.remove(req.rid)

        return req_to_send
