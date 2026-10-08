from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta
from sglang.srt.utils import broadcast_pyobj

from tokenrouter.algorithms.collm.threshold_schedule import (
    create_deferral_threshold_schedule,
)
from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class CoLLMBaseScheduler(BaseTokenRoutingScheduler):

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
        lm_head_weight = self.tp_worker.model_runner.model.lm_head.weight.data
        local_shard_rows = lm_head_weight.shape[0]
        deferral_global_idx = self.model_config.vocab_size - 1
        owning_rank = deferral_global_idx // local_shard_rows
        deferral_local_idx = deferral_global_idx - owning_rank * local_shard_rows
        if tp_rank == owning_rank:
            self.defer_weight = lm_head_weight[deferral_local_idx].clone()
            if self.tp_size > 1:
                self.defer_weight = self.defer_weight.tolist()
        else:
            self.defer_weight = []
        if self.tp_size > 1:
            self.defer_weight = broadcast_pyobj(
                self.defer_weight,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[owning_rank],
            )
            self.defer_weight = torch.tensor(
                self.defer_weight, device=self.device, dtype=self.model_config.dtype
            )
        self.defer_threshold = scheduler_kwargs.get("defer_threshold", 0.5)
        self.warmup_steps = scheduler_kwargs.get("warmup_steps", 15)
        self.warmup_schedule = scheduler_kwargs.get("warmup_schedule", "constant")
        self.warmup_defer_thresholds = create_deferral_threshold_schedule(
            warmup_schedule=self.warmup_schedule,
            warmup_steps=self.warmup_steps,
            start_value=1.0,
            end_value=self.defer_threshold,
        )

    def route(self, batch, result) -> torch.Tensor:
        device = batch.seq_lens.device
        batch_size = batch.batch_size()
        is_prefill = result.logits_output.hidden_states.shape[0] != batch_size

        if is_prefill:
            # For prefill, use cumsum of extend_lens to get correct indices
            extend_lens = torch.tensor(batch.extend_lens, device=device)
            hidden_indices = torch.cumsum(extend_lens, dim=0) - 1
        else:
            # For decode, use sequential indices
            hidden_indices = torch.arange(batch_size, device=device)

        # Get hidden states for the relevant positions
        hidden_states = result.logits_output.hidden_states[
            hidden_indices, :
        ]  # batch_size, hidden_size

        defer_probs = torch.sigmoid(torch.mv(hidden_states, self.defer_weight)).cpu()
        defer_thresholds = [self.defer_threshold] * batch_size
        output_steps = [len(req.output_ids) for req in batch.reqs]
        for idx, step in enumerate(output_steps):
            if step < self.warmup_steps:
                defer_thresholds[idx] = self.warmup_defer_thresholds[step]

        return defer_probs > torch.tensor(defer_thresholds)

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
