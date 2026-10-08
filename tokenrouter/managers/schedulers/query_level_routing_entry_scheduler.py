from typing import Optional

import torch
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class QueryLevelRoutingEntryScheduler(BaseTokenRoutingScheduler):
    """Admission-time routing baseline with token-level routing disabled."""

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
        route_to_peer = scheduler_kwargs.get("route_to_peer", True)
        self.route_to_peer_p = float(
            scheduler_kwargs.get("route_to_peer_p", 1.0 if route_to_peer else 0.0)
        )

    def process_input_requests(self, recv_reqs: list) -> None:
        target_by_rid = {
            req.rid: getattr(req, "target_model")
            for req in recv_reqs
            if isinstance(req, TokenizedGenerateReqInput)
            and getattr(req, "target_model", None) is not None
        }
        super().process_input_requests(recv_reqs)
        self._propagate_target_model(
            target_by_rid, (self.waiting_queue, self.grammar_queue)
        )

    @staticmethod
    def _propagate_target_model(target_by_rid, queues) -> None:
        remaining = dict(target_by_rid)
        for queue in queues:
            for req in queue:
                target_model = remaining.pop(req.rid, None)
                if target_model is not None:
                    setattr(req, "target_model", target_model)
                if not remaining:
                    return

    def route(self, batch, result) -> torch.Tensor:
        choices = torch.zeros((batch.batch_size(),), dtype=torch.bool)
        for i, req in enumerate(batch.reqs):
            if len(req.output_ids) == 0:  # Only route at the first token
                target_model = getattr(req, "target_model", None)
                if target_model == self.peer_name:
                    choices[i] = True
                elif target_model == self.server_args.server_name:
                    choices[i] = False
                else:
                    choices[i] = torch.rand(()) < self.route_to_peer_p
        return choices

    def send(self, req: Req, destination: Optional[str] = None) -> PeerReq:
        req_to_send = super().send(req, destination)
        return req_to_send
