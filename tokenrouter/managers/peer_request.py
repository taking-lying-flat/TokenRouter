from typing import Dict, List, Optional

from sglang.srt.sampling.sampling_params import SamplingParams


class SimpleSamplingParams:
    """Minimal, pickle-safe subset of ``SamplingParams``.

    Carried inside a ``PeerReq`` so the receiving scheduler can
    create a full ``sglang`` ``Req`` without needing access to the
    original request object.
    """

    __slots__ = (
        "temperature",
        "top_k",
        "top_p",
        "max_new_tokens",
        "ignore_eos",
    )

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        max_new_tokens: int = 128,
        ignore_eos: bool = False,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos

    def derive_sampling_params(self) -> SamplingParams:
        """Expand into a full ``SamplingParams`` for sglang."""
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            ignore_eos=self.ignore_eos,
        )

    @classmethod
    def from_sampling_params(cls, sp: SamplingParams) -> "SimpleSamplingParams":
        """Extract the lightweight subset from a full ``SamplingParams``."""
        return cls(
            temperature=sp.temperature,
            top_k=sp.top_k,
            top_p=sp.top_p,
            max_new_tokens=sp.max_new_tokens,
            ignore_eos=sp.ignore_eos,
        )


class PeerReq:
    """A single message exchanged between cooperative peer schedulers.

    Attributes:
        rid: The request id shared across all cooperating schedulers.
        new_token_ids: Token ids produced by the sending peer.
            Empty list for control messages (``"finished"`` / ``"new"``).
        sampling_params: Needed only for **first-time** requests so the
            receiving scheduler can construct a full ``Req``.  ``None``
            for subsequent token-return messages.
        status: Routing / lifecycle control:
            ``"need"``      — normal token return (default).
            ``"finished"``  — the sending peer has finished this request.
            ``"new"``       — a new cooperative request was created.
            ``"retracted"`` — the sender freed its KV for this rid (typically
                              to break a KV-pool deadlock); the receiver
                              should resume local decoding and treat the next
                              delegation as ``first_send=True``.
    """

    def __init__(
        self,
        rid: str,
        new_token_ids: Optional[List[int]],
        kvcache_loc: Optional[Dict[str, int]],
        origin_input_text: Optional[str] = None,
        origin_input_len: Optional[int] = None,
        sampling_params: Optional[SimpleSamplingParams] = None,
        status: str = "need",
        model_tokens: Optional[Dict[str, List[int]]] = None,
        routing_step: int = 0,
        routing_cooldown: int = 0,
        source_model: Optional[str] = None,
        destination_model: Optional[str] = None,
        entry_model: Optional[str] = None,
        handoff_seq: int = 0,
        final_output_ids: Optional[List[int]] = None,
    ):
        self.rid = rid
        self.new_token_ids = new_token_ids
        self.kvcache_loc = kvcache_loc
        self.origin_input_text = origin_input_text
        self.origin_input_len = origin_input_len
        self.sampling_params = sampling_params
        self.status = status
        self.model_tokens = model_tokens if model_tokens is not None else {}
        self.routing_step = routing_step
        self.routing_cooldown = routing_cooldown
        self.source_model = source_model
        self.destination_model = destination_model
        self.entry_model = entry_model
        self.handoff_seq = handoff_seq
        self.final_output_ids = final_output_ids

    @classmethod
    def construct_from_req(
        cls,
        req,
        source: str,
        destination: str,
        current_token_ids: List[int],
        entry_model: Optional[str] = None,
        handoff_seq: Optional[int] = None,
    ):
        """Construct a PeerReq from an sglang Req and generated tokens."""
        if req.kvcache_loc[destination] is None:
            first_send = True
            new_token_ids = current_token_ids
        else:
            first_send = False
            new_token_ids = current_token_ids[req.kvcache_loc[destination] :]
        return cls(
            rid=req.rid,
            new_token_ids=new_token_ids,
            kvcache_loc=req.kvcache_loc,
            sampling_params=(
                SimpleSamplingParams(
                    temperature=req.sampling_params.temperature,
                    top_k=req.sampling_params.top_k,
                    top_p=req.sampling_params.top_p,
                    max_new_tokens=req.sampling_params.max_new_tokens,
                    ignore_eos=req.sampling_params.ignore_eos,
                )
                if first_send
                else None
            ),
            origin_input_text=req.origin_input_text if first_send else None,
            origin_input_len=len(req.origin_input_ids) if first_send else None,
            status="need" if not req.finished() else "finished",
            model_tokens=req.model_tokens,
            routing_step=getattr(req, "_tokenrouter_random_routing_step", 0),
            routing_cooldown=getattr(
                req, "_tokenrouter_random_routing_cooldown", 0
            ),
            source_model=source,
            destination_model=destination,
            entry_model=entry_model or getattr(req, "entry_model", None),
            handoff_seq=(
                handoff_seq
                if handoff_seq is not None
                else getattr(req, "handoff_seq", 0)
            ),
        )

    @classmethod
    def construct_control(
        cls,
        *,
        rid: str,
        status: str,
        source: str,
        destination: str,
        entry_model: str,
        handoff_seq: int,
        model_tokens: Optional[Dict[str, List[int]]] = None,
        final_output_ids: Optional[List[int]] = None,
    ) -> "PeerReq":
        return cls(
            rid=rid,
            new_token_ids=[],
            kvcache_loc=None,
            status=status,
            model_tokens=model_tokens,
            source_model=source,
            destination_model=destination,
            entry_model=entry_model,
            handoff_seq=handoff_seq,
            final_output_ids=final_output_ids,
        )

    def __repr__(self) -> str:
        n = len(self.new_token_ids or [])
        return (
            f"PeerReq(rid={self.rid!r}, tokens={n}, status={self.status!r}, "
            f"source={self.source_model!r}, destination={self.destination_model!r}, "
            f"seq={self.handoff_seq})"
        )
