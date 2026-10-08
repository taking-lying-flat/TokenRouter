import faulthandler
import logging
import os
import signal
import threading
import time
from abc import abstractmethod
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import psutil
import setproctitle
import torch
import zmq
from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ExpertDistributionReq,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    IdleSleeper,
    Scheduler,
    require_mlp_sync,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_output_processor_mixin import (
    DEFAULT_FORCE_STREAM_INTERVAL,
)
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.session_controller import Session
import sglang.srt.managers.tp_worker_overlap_thread as tp_worker_overlap_thread
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.utils import DPBalanceMeta
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    get_available_gpu_memory,
    get_bool_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

from tokenrouter.managers.io_struct import BatchTokenIDOut
from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedule_batch import ScheduleBatch
from tokenrouter.managers.tp_worker import TpModelWorker
from tokenrouter.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

_SKIP_UNFINISHED_CACHE_ATTR = "_tokenrouting_skip_unfinished_radix_cache"
_STALE_RADIX_PREFIX_ATTR = "_tokenrouting_radix_cached_prefix_len"


class BaseTokenRoutingScheduler(Scheduler):
    """
    Base scheduler for multi-model cooperative inference.

    Each instance manages one model.  Requests whose ``status`` is set to
    ``"notneed"`` (via ``mark_as_not_need``) are parked in
    ``batch_not_need`` while a peer model handles them.  The request's
    KV-cache allocation is preserved.  When the peer returns tokens the
    subclass calls ``accept_peer_tokens`` which routes the request back
    into either the decode path (0-1 accepted tokens) or the
    extend/prefill path (N>1 accepted tokens).
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
        # Parse args
        self.server_args = server_args
        self.peer_names = [
            name
            for name in server_args.all_server_names
            if name != server_args.server_name
        ]
        # Backwards-compatible alias for the existing two-model schedulers.
        # Multi-model schedulers must always pass an explicit destination.
        self.peer_name = self.peer_names[0] if len(self.peer_names) == 1 else None
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.enable_lora = server_args.enable_lora
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = server_args.kv_events_config is not None
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.page_size = server_args.page_size

        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init model config
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init inter-process communication
        context = zmq.Context(2)
        self.idle_sleeper = None

        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if self.dp_size > 1:
                if len(self.peer_names) != 1:
                    raise NotImplementedError(
                        "Multi-model routing with data parallelism is not supported"
                    )
                # Preserve the existing two-model DP-controller bridge.
                peer_bind_endpoint = (
                    port_args.peer_bind_endpoint or port_args.peer_ipc_name
                )
                self.send_to_peer = get_zmq_socket(
                    context, zmq.PUSH, peer_bind_endpoint, True
                )
                peer_connect_endpoint = port_args.peer_connect_endpoint
                if peer_connect_endpoint is None:
                    peer_connect_endpoint = getattr(
                        port_args, "peer_recv_ipc_name", None
                    )
                if peer_connect_endpoint is None:
                    peer_connect_endpoint = server_args.port_args_map[
                        self.peer_name
                    ].peer_ipc_name
                self.recv_from_peer = get_zmq_socket(
                    context, zmq.PULL, peer_connect_endpoint, False
                )
                self.send_to_peers = {self.peer_name: self.send_to_peer}
                logger.info(
                    "Peer transport for %s (DP bridge): PUSH bind=%s, "
                    "PULL connect=%s",
                    server_args.server_name,
                    peer_bind_endpoint,
                    peer_connect_endpoint,
                )
            else:
                # Every model binds one inbox. A dedicated PUSH socket connects
                # to each destination, so messages are addressed instead of
                # being load-balanced across multiple PULL consumers.
                peer_bind_endpoint = (
                    port_args.peer_bind_endpoint or port_args.peer_ipc_name
                )
                self.recv_from_peer = get_zmq_socket(
                    context, zmq.PULL, peer_bind_endpoint, True
                )
                peer_endpoints = {}
                for name in self.peer_names:
                    if len(self.peer_names) == 1 and port_args.peer_connect_endpoint:
                        endpoint = port_args.peer_connect_endpoint
                    else:
                        endpoint = server_args.port_args_map[name].peer_ipc_name
                    peer_endpoints[name] = endpoint
                self.send_to_peers = {
                    name: get_zmq_socket(context, zmq.PUSH, endpoint, False)
                    for name, endpoint in peer_endpoints.items()
                }
                self.send_to_peer = (
                    self.send_to_peers[self.peer_name]
                    if self.peer_name is not None
                    else None
                )
                logger.info(
                    "Peer transport for %s: PULL bind=%s, PUSH connects=%s",
                    server_args.server_name,
                    peer_bind_endpoint,
                    peer_endpoints,
                )
            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.recv_from_peer = None
            self.send_to_peer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_peers = {
                name: SimpleNamespace(send_pyobj=lambda x: None)
                for name in self.peer_names
            }

        if self.current_scheduler_metrics_enabled():
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config
        self.init_moe_config()

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # Launch a tensor parallel worker
        if self.enable_overlap:
            tp_worker_overlap_thread.TpModelWorker = TpModelWorker
            TpWorkerClass = TpModelWorkerClient
        else:
            TpWorkerClass = TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        else:
            self.draft_worker = None

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if global_server_args_dict["max_micro_batch_size"] is None:
            global_server_args_dict["max_micro_batch_size"] = max(
                self.max_running_requests // server_args.pp_size, 1
            )

        self.tp_group = self.tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        # Hybrid memory pool
        self.is_hybrid = self.tp_worker.is_hybrid
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Print debug info
        if tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"available_gpu_mem={avail_mem:.2f} GB"
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.return_health_check_ct = 0
        self.num_paused_reqs: int = 0
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0
        self.sessions: Dict[str, Session] = {}
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
        )
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.offload_tags = set()
        self.init_profier()

        self.recv_skipper = SchedulerRecvSkipper.maybe_create(server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)
        self.init_kv_events(server_args.kv_events_config)

        # Init disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.init_disaggregation()

        if get_bool_env_var("SGLANG_GC_LOG"):
            configure_gc_logger()

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
            ]
        )

        self.balance_meta = dp_balance_meta
        if (
            server_args.enable_dp_attention
            and server_args.load_balance_method == "minimum_tokens"
        ):
            assert dp_balance_meta is not None

        self.recv_dp_balance_id_this_term = []

        self.active_rids: set[str] = set()
        self.latest_handoff_seq: Dict[str, int] = {}
        self.terminal_rids: Dict[str, int] = {}
        self.terminal_tombstone_limit = int(
            self.server_args.scheduler_kwargs.get(
                "terminal_tombstone_limit", 100_000
            )
        )

        self.init_batch_not_need()

    # ------------------------------------------------------------------
    # Cooperative event loop
    # ------------------------------------------------------------------

    def event_loop_normal(self):
        """Main loop: receive from peers and system, schedule, run, let subclass process."""
        while True:
            # 1. Peer messages
            peer_msgs = self.recv_requests_from_peer()
            if peer_msgs:
                self.receive(peer_msgs)

            # 2. System / tokenizer messages
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # 3. Schedule and run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                # else:
                #     self.self_check_during_idle()

                self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap(self):
        """SGLang overlap loop with a fuller result-batch copy for routing.

        SGLang's stock overlap loop stores ``batch.copy()`` in ``result_queue``.
        That copy is intentionally minimal because upstream only needs it for
        output processing. TokenRouting also parks routed requests in
        ``batch_not_need`` and can later merge them back into decode, so it
        needs the batch tensors and ``sampling_info`` that ``merge_batch`` /
        ``filter_batch`` consume.
        """
        self.result_queue = deque()
        while True:
            peer_msgs = self.recv_requests_from_peer()
            if peer_msgs:
                self.receive(peer_msgs)

            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                batch.launch_done = threading.Event()
                result = self.run_batch(batch)
                self.result_queue.append(
                    (self._copy_batch_for_overlap_result(batch), result)
                )

                if self.last_batch is None:
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None, batch.launch_done)

            if self.last_batch is not None:
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(
                    tmp_batch, tmp_result, batch.launch_done if batch else None
                )

            self.last_batch = batch

    @staticmethod
    def _copy_batch_for_overlap_result(batch: ScheduleBatch) -> ScheduleBatch:
        copied = batch.copy()
        for attr in (
            "req_to_token_pool",
            "token_to_kv_pool_allocator",
            "tree_cache",
            "is_hybrid",
            "enable_overlap",
            "batch_is_full",
            "chunked_req",
            "sampling_info",
            "input_ids",
            "input_embeds",
            "token_type_ids",
            "req_pool_indices",
            "seq_lens",
            "out_cache_loc",
            "output_ids",
            "multimodal_inputs",
            "seq_lens_sum",
            "orig_seq_lens",
            "extend_lens",
            "return_logprob",
            "top_logprobs_nums",
            "token_ids_logprobs",
            "has_stream",
            "has_grammar",
            "device",
            "spec_info",
            "return_hidden_states",
            "extend_num_tokens",
            "encoder_cached",
            "encoder_lens",
            "encoder_lens_cpu",
            "encoder_out_cache_loc",
        ):
            setattr(copied, attr, getattr(batch, attr, None))
        return copied

    # ------------------------------------------------------------------
    # Peer communication
    # ------------------------------------------------------------------

    def recv_requests_from_peer(self) -> list:
        if self.recv_from_peer is not None:
            recv_reqs: list = []
            if self.attn_tp_rank == 0 and self.pp_rank == 0:
                while True:
                    try:
                        msg = self.recv_from_peer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.extend(msg)
        else:
            recv_reqs = []

        if self.tp_size > 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)

        return recv_reqs

    def send_requests_to_peer(self, destination: str, reqs: list) -> None:
        if not reqs:
            return
        socket = self.send_to_peers.get(destination)
        if socket is None:
            raise ValueError(
                f"Unknown routing destination {destination!r}; "
                f"available peers: {self.peer_names}"
            )
        socket.send_pyobj(reqs)

    def send_requests_to_peers(self, reqs_by_destination) -> None:
        """Send addressed peer messages.

        A list remains accepted for backwards compatibility in the two-model
        case. Multi-model callers must pass ``destination -> messages``.
        """
        if isinstance(reqs_by_destination, list):
            if self.peer_name is None:
                raise ValueError(
                    "Multi-model routing requires an explicit destination"
                )
            self.send_requests_to_peer(self.peer_name, reqs_by_destination)
            return
        for destination, reqs in reqs_by_destination.items():
            self.send_requests_to_peer(destination, reqs)

    def broadcast_control(
        self,
        *,
        rid: str,
        status: str,
        handoff_seq: int,
        model_tokens=None,
        final_output_ids=None,
    ) -> None:
        source = self.server_args.server_name
        entry_model = self.server_args.entry_model_name
        messages = {
            destination: [
                PeerReq.construct_control(
                    rid=rid,
                    status=status,
                    source=source,
                    destination=destination,
                    entry_model=entry_model,
                    handoff_seq=handoff_seq,
                    model_tokens=model_tokens,
                    final_output_ids=final_output_ids,
                )
            ]
            for destination in self.peer_names
        }
        self.send_requests_to_peers(messages)

    def _mark_terminal(self, rid: str, handoff_seq: int) -> None:
        self.terminal_rids[rid] = max(
            handoff_seq, self.terminal_rids.get(rid, -1)
        )
        # Dicts preserve insertion order. Bound tombstones so a long-running
        # server does not retain every completed rid forever.
        while len(self.terminal_rids) > self.terminal_tombstone_limit:
            oldest_rid = next(iter(self.terminal_rids))
            del self.terminal_rids[oldest_rid]
            self.latest_handoff_seq.pop(oldest_rid, None)
            self.active_rids.discard(oldest_rid)

    def process_input_requests(self, recv_reqs: list) -> None:
        args = self.server_args
        # When any force flag is on, capture the user's original `return_*`
        # values keyed by rid so we can re-attach them to the freshly-built
        # `Req` after `super()`. sglang's `handle_generate_request` builds
        # `Req` from explicit `TokenizedGenerateReqInput` fields only and
        # silently drops attributes added by the routing layer.
        capture_ori = (
            args.force_return_hidden_states
            or args.force_return_logprob
            or args.force_return_logprobs_num
        )
        ori_by_rid: Dict[str, Dict[str, Any]] = {}
        for req in recv_reqs:
            if isinstance(req, TokenizedGenerateReqInput):
                if capture_ori:
                    ori: Dict[str, Any] = {}
                    if args.force_return_hidden_states:
                        ori["ori_return_hidden_states"] = req.return_hidden_states
                        req.return_hidden_states = True
                    if args.force_return_logprob:
                        ori["ori_return_logprob"] = req.return_logprob
                        req.return_logprob = True
                    if args.force_return_logprobs_num:
                        ori["ori_top_logprobs_num"] = req.top_logprobs_num
                        req.top_logprobs_num = args.force_return_logprobs_num
                    if ori:
                        ori_by_rid[req.rid] = ori
                self.active_rids.add(req.rid)
                self.latest_handoff_seq[req.rid] = 0
                if self.server_args.is_entry_model:
                    self.broadcast_control(
                        rid=req.rid,
                        status="new",
                        handoff_seq=0,
                    )
        super().process_input_requests(recv_reqs)
        if ori_by_rid:
            self._propagate_ori_attrs(
                ori_by_rid, (self.waiting_queue, self.grammar_queue)
            )

    @staticmethod
    def _propagate_ori_attrs(
        ori_by_rid: Dict[str, Dict[str, Any]],
        queues: Iterable[Iterable[Req]],
    ) -> None:
        """Copy preserved ``ori_*`` attrs onto the freshly-built ``Req`` objects.

        ``handle_generate_request`` constructs a ``Req`` only from the
        explicit fields of ``TokenizedGenerateReqInput`` — any attribute the
        routing layer tacks on (``ori_return_logprob`` etc.) is dropped. We
        re-attach them here so ``stream_output_generation`` can honour the
        user's original ``return_*`` ask even when the force flags forced
        the upstream values to True for the router's benefit.
        """
        remaining = dict(ori_by_rid)
        for queue in queues:
            for req in queue:
                ori = remaining.pop(req.rid, None)
                if ori is None:
                    continue
                for attr, value in ori.items():
                    setattr(req, attr, value)
                if not remaining:
                    return

    def receive(self, msgs: list) -> None:
        """Process addressed lifecycle and token-handoff messages."""
        flat: list[PeerReq] = []
        for item in msgs:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        current_model = self.server_args.server_name
        flat = [
            msg
            for msg in flat
            if getattr(msg, "destination_model", current_model)
            in (None, current_model)
        ]

        rid_to_req: Dict[str, Req] = {req.rid: req for req in self.batch_not_need.reqs}
        finished_rids: List[str] = []

        latest_need = self._dedup_peer_data_msgs(flat)
        latest_finished = self._dedup_peer_finished_msgs(flat)

        known_reqs: List[Req] = []
        known_token_ids: List[List[int]] = []
        known_msgs: List[PeerReq] = []
        first_time_msgs: list[PeerReq] = []
        decode_rids: List[str] = []

        for msg in flat:
            rid = msg.rid
            status = getattr(msg, "status", "need")
            handoff_seq = int(getattr(msg, "handoff_seq", 0))

            if status == "finished":
                if latest_finished.get(rid) is not msg:
                    continue
                terminal_seq = self.terminal_rids.get(rid)
                if terminal_seq is not None and handoff_seq <= terminal_seq:
                    continue

                self._mark_terminal(rid, handoff_seq)
                self.latest_handoff_seq[rid] = max(
                    handoff_seq, self.latest_handoff_seq.get(rid, -1)
                )
                self.active_rids.discard(rid)

                if rid in rid_to_req:
                    finished_req = rid_to_req[rid]
                    # Release/cache only the KV positions actually owned by
                    # this model before replacing output_ids with the global
                    # final sequence.
                    self.tree_cache.cache_finished_req(finished_req)
                    finished_rids.append(rid)
                    finished_req.output_ids = self._final_output_ids_from_msg(
                        finished_req, msg
                    )
                    finished_req.check_finished()
                    if msg.model_tokens:
                        finished_req.model_tokens = msg.model_tokens
                    finished_req.handoff_seq = handoff_seq
                    finished_req.status = "finished"
                else:
                    waiting_req = self._pop_waiting_req_by_rid(rid)
                    if waiting_req is not None:
                        self._finish_waiting_req_from_peer(waiting_req, msg)
                continue

            if status == "new":
                if rid in self.terminal_rids:
                    continue
                self.active_rids.add(rid)
                self.latest_handoff_seq[rid] = max(
                    handoff_seq, self.latest_handoff_seq.get(rid, -1)
                )
                continue

            if latest_need.get(rid) is not msg:
                continue
            if rid in self.terminal_rids:
                continue
            if handoff_seq <= self.latest_handoff_seq.get(rid, -1):
                continue

            self.active_rids.add(rid)
            self.latest_handoff_seq[rid] = handoff_seq
            new_token_ids: List[int] = msg.new_token_ids or []
            if rid in rid_to_req:
                known_reqs.append(rid_to_req[rid])
                known_msgs.append(msg)
                known_token_ids.append(new_token_ids)
            else:
                first_time_msgs.append(msg)

        if known_reqs:
            n_accepted_list = self.accept_peer_tokens(known_reqs, known_token_ids)
        else:
            n_accepted_list = []

        extend_rids: List[str] = []
        any_decode_acceptance = False

        for req, peer_tids, n_accepted, msg in zip(
            known_reqs, known_token_ids, n_accepted_list, known_msgs
        ):
            accepted = peer_tids[:n_accepted]
            req._tokenrouter_random_routing_step = max(
                getattr(req, "_tokenrouter_random_routing_step", 0),
                getattr(msg, "routing_step", 0),
            )
            req._tokenrouter_random_routing_cooldown = getattr(
                msg, "routing_cooldown", 0
            )
            req.handoff_seq = int(getattr(msg, "handoff_seq", 0))
            req.entry_model = getattr(
                msg, "entry_model", self.server_args.entry_model_name
            )

            if n_accepted <= 0:
                req.status = "need"
                req.kvcache_loc = msg.kvcache_loc
                req.model_tokens = msg.model_tokens
                decode_rids.append(req.rid)

            elif (
                n_accepted == 1
                and self.server_args.forward_mode_for_single_token == "decode"
            ):
                req.output_ids = req.output_ids[:-1] + accepted
                req.status = "need"
                req.kvcache_loc = msg.kvcache_loc
                req.model_tokens = msg.model_tokens
                req.check_finished()
                decode_rids.append(req.rid)
                any_decode_acceptance = True

            else:
                req.status = "need"
                req.kvcache_loc = msg.kvcache_loc
                req.model_tokens = msg.model_tokens
                req.extend_input_len = n_accepted
                req.output_ids = req.output_ids[:-1] + accepted
                self.waiting_queue.append(req)
                extend_rids.append(req.rid)

        if any_decode_acceptance and self.batch_not_need.reqs:
            output_ids_list = [r.output_ids[-1] for r in self.batch_not_need.reqs]
            self.batch_not_need.output_ids = torch.tensor(
                output_ids_list, dtype=torch.int64
            ).to(self.batch_not_need.device, non_blocking=True)

        for msg in first_time_msgs:
            if not msg.new_token_ids:
                continue
            sampling_params = getattr(msg, "sampling_params", None)
            if sampling_params is None or msg.origin_input_len is None:
                logger.warning(
                    "Dropping non-first-send peer msg for unknown rid=%s "
                    "(sampling_params=%s, origin_input_len=%s); peer will "
                    "re-prefill on next first_send.",
                    msg.rid,
                    sampling_params,
                    msg.origin_input_len,
                )
                continue
            if hasattr(sampling_params, "derive_sampling_params"):
                sampling_params = sampling_params.derive_sampling_params()
                sampling_params.normalize(self.tokenizer)
            new_req = Req(
                rid=msg.rid,
                origin_input_text=msg.origin_input_text,
                origin_input_ids=msg.new_token_ids[: msg.origin_input_len],
                sampling_params=sampling_params,
                eos_token_ids=self.model_config.hf_eos_token_id,
                return_hidden_states=False,
                vocab_size=self.model_config.vocab_size,
            )
            new_req.kvcache_loc = msg.kvcache_loc
            new_req.model_tokens = msg.model_tokens
            new_req._tokenrouter_random_routing_step = getattr(
                msg, "routing_step", 0
            )
            new_req._tokenrouter_random_routing_cooldown = getattr(
                msg, "routing_cooldown", 0
            )
            new_req.handoff_seq = int(getattr(msg, "handoff_seq", 0))
            new_req.entry_model = getattr(
                msg, "entry_model", self.server_args.entry_model_name
            )
            new_req.status = "need"
            new_req.output_ids = msg.new_token_ids[msg.origin_input_len :]
            self.waiting_queue.append(new_req)

        if finished_rids:
            if self.server_args.is_entry_model:
                self.stream_output_generation(
                    [rid_to_req[rid] for rid in finished_rids],
                    return_logprob=False,
                )
            for rid in finished_rids:
                del rid_to_req[rid]

        if decode_rids:
            if self.last_batch is not None and getattr(
                self.last_batch, "reqs", None
            ) is not None:
                decode_rids_to_keep = {req.rid for req in self.last_batch.reqs}
                decode_rids_to_keep.update(decode_rids)
                self.last_batch.merge_batch(self.batch_not_need)
                keep = [
                    i
                    for i, req in enumerate(self.last_batch.reqs)
                    if req.rid in decode_rids_to_keep
                ]
                self.last_batch.filter_batch(keep_indices=keep)
            else:
                for req in self.batch_not_need.reqs:
                    if req.rid in decode_rids:
                        req.extend_input_len = 1
                        self.waiting_queue.append(req)

        keep_indices = self._batch_not_need_keep_indices(self.batch_not_need.reqs)
        self.batch_not_need.filter_batch(keep_indices=keep_indices)

    @staticmethod
    def _final_output_ids_from_msg(req: Req, msg: PeerReq) -> List[int]:
        final_output_ids = getattr(msg, "final_output_ids", None)
        if final_output_ids is not None:
            return list(final_output_ids)
        return req.output_ids[:-1] + list(msg.new_token_ids or [])

    @staticmethod
    def _dedup_peer_data_msgs(flat: List["PeerReq"]) -> Dict[str, "PeerReq"]:
        """Collapse a list of peer messages to one entry per rid (latest wins),
        skipping any rid that has a "finished" message in the same batch.

        Each non-finished `PeerReq` from the peer carries cumulative
        `new_token_ids` (see `send`), so when the ZMQ queue
        holds several messages for the same rid only the latest is useful.
        Earlier duplicates would otherwise be applied multiple times in
        phases 2-3 and cause the same `Req` to appear in `waiting_queue`
        more than once.

        Returns an insertion-ordered mapping rid → latest non-control msg.
        Control messages ("finished", "new") are not returned; the caller
        still iterates `flat` separately for those.
        """
        finished_rids_in_batch = {
            m.rid for m in flat if getattr(m, "status", "need") == "finished"
        }
        latest: Dict[str, "PeerReq"] = {}
        for m in flat:
            status = getattr(m, "status", "need")
            if status in ("finished", "new"):
                continue
            if m.rid in finished_rids_in_batch:
                # Will be cleaned up by the finished msg later in this batch.
                continue
            previous = latest.get(m.rid)
            if previous is None or int(getattr(m, "handoff_seq", 0)) >= int(
                getattr(previous, "handoff_seq", 0)
            ):
                latest[m.rid] = m
        return latest

    @staticmethod
    def _dedup_peer_finished_msgs(flat: List["PeerReq"]) -> Dict[str, "PeerReq"]:
        """Collapse finished peer messages to one entry per rid.

        Finished messages carry cumulative token ids just like normal
        peer data messages.  If the receiver drains multiple finished
        messages for the same rid in one poll, only the latest should be
        applied; otherwise the same request is streamed and removed
        repeatedly.
        """
        latest: Dict[str, "PeerReq"] = {}
        for m in flat:
            if getattr(m, "status", "need") == "finished":
                previous = latest.get(m.rid)
                if previous is None or int(getattr(m, "handoff_seq", 0)) >= int(
                    getattr(previous, "handoff_seq", 0)
                ):
                    latest[m.rid] = m
        return latest

    @staticmethod
    def _batch_not_need_keep_indices(reqs: List[Req]) -> List[int]:
        """Indices of ``batch_not_need`` reqs to retain after `receive`.

        A req stays only if it is still being handled by the peer
        (``status == "notneed"``) AND has not finished. Already-finished
        reqs were streamed and cached in phase 5; keeping them here would
        defer eviction until the next ``process_batch_result`` call with a
        non-empty ``reqs_to_send`` — leaking ``Req`` objects across cycles
        and risking stale ``req_pool_idx`` slots being merged back into
        ``last_batch`` after ``cache_finished_req`` released them.

        Mirrors the predicate used in ``process_batch_result``.
        """
        return [
            i
            for i, req in enumerate(reqs)
            if req.status == "notneed" and not req.finished()
        ]

    # ------------------------------------------------------------------
    # Scheduling overrides
    # ------------------------------------------------------------------

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:

        # `running_batch.batch_is_full` is set by `get_new_batch_prefill` when
        # the prefill adder can't fit a new req under the conservative KV
        # budget. Sglang's stock scheduler resets it inside
        # `update_running_batch` (when `batch.is_empty()` or the batch shrunk),
        # which fires every decode iteration. Token-routing schedulers can
        # leave `running_batch` empty for many iterations while waiting on
        # peer responses (all reqs sit in `batch_not_need`), so that reset
        # path is bypassed. Without this guard, a once-set `batch_is_full`
        # latches indefinitely and blocks prefill of reqs that later arrive
        # in the waiting_queue via the peer extend path — producing a hard
        # deadlock at large batch+length where the peer's eventual
        # delegate-back-with-multi-token msg goes to extend.
        if self.running_batch.is_empty() and self.running_batch.batch_is_full:
            self.running_batch.batch_is_full = False

        chunked_req_to_exclude = set()
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req)
            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                elif not self._batches_have_overlapping_rids(
                    self.running_batch, self.last_batch
                ):
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()

        need_dp_attn_preparation = require_mlp_sync(self.server_args)

        if need_dp_attn_preparation and not self.spec_algorithm.is_none():
            new_batch = self.prepare_mlp_sync_batch(new_batch)
            need_dp_attn_preparation = new_batch is None

        if new_batch is not None:
            ret = new_batch
            for req in ret.reqs:
                req.status = "need"
                if not hasattr(req, "kvcache_loc"):
                    req.kvcache_loc = {
                        name: None for name in self.server_args.all_server_names
                    }
                if not hasattr(req, "model_tokens"):
                    req.model_tokens = {
                        name: [0, 0] for name in self.server_args.all_server_names
                    }
                if not hasattr(req, "handoff_seq"):
                    req.handoff_seq = self.latest_handoff_seq.get(req.rid, 0)
                if not hasattr(req, "entry_model"):
                    req.entry_model = self.server_args.entry_model_name
        else:
            # ---- decode path ----
            if not self.running_batch.is_empty():
                n_need_reqs = len(
                    [req for req in self.running_batch.reqs if req.status == "need"]
                )
                min_batch_size = self._get_delayed_batching_threshold()
                if self._has_overlap_peer_routed_req(
                    req
                    for req in self.running_batch.reqs
                    if req.status == "need"
                ):
                    min_batch_size = min(min_batch_size, n_need_reqs)
                if n_need_reqs >= min_batch_size:
                    self.running_batch = self.update_running_batch(self.running_batch)
                    ret = (
                        self.running_batch
                        if not self.running_batch.is_empty()
                        else None
                    )
                else:
                    ret = None
            else:
                ret = None

        if need_dp_attn_preparation:
            if (
                self.server_args.load_balance_method == "minimum_tokens"
                and self.forward_ct % 40 == 0
            ):
                self.handle_dp_balance_data(ret)
            ret = self.prepare_mlp_sync_batch(ret)

        return ret

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked request has just been released.
        # In PP case, a chunked req can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked request to be added, otherwise, there will be a memory leak.
        if self.get_num_allocatable_reqs(running_bs) <= 0 and not self.chunked_req:
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            lora_set = set([req.lora_id for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch.
        # `for/else`: `adder_saturated` stays True only if the loop broke —
        # used below to relax `delayed_batching_threshold` when the budget caps the batch.
        adder_saturated = True
        for req in self.waiting_queue:

            if self.enable_lora and not self.tp_worker.can_run_lora_batch(
                lora_set
                | set([req.lora_id for req in adder.can_run_list])
                | set([req.lora_id])
            ):
                self.running_batch.batch_is_full = True
                break

            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
                break

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue

            owns_prefix_lock = self._owns_radix_cache_lock(req)
            if owns_prefix_lock:
                # This request already owns the radix node lock from its
                # previous running round. PrefillAdder will acquire the lock
                # again if it admits the request, so release the old reference
                # first to avoid accumulating protected cache.
                req._tokenrouting_old_req_pool_idx = req.req_pool_idx
                self._dec_req_radix_lock(req)
                req.fill_ids = req.origin_input_ids + req.output_ids
                req.host_hit_length = 0
                req.last_host_node = None
            else:
                prefix_indices = req.prefix_indices
                req.init_next_round_input(self.tree_cache)
                if prefix_indices is not None and len(prefix_indices) > 0:
                    req.prefix_indices = prefix_indices
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)

            try:
                res = adder.add_one_req(
                    req, has_chunked_req=(self.chunked_req is not None)
                )
            except Exception:
                if owns_prefix_lock:
                    self._inc_req_radix_lock(req)
                raise

            if owns_prefix_lock and req not in adder.can_run_list:
                self._inc_req_radix_lock(req)
                if hasattr(req, "_tokenrouting_old_req_pool_idx"):
                    delattr(req, "_tokenrouting_old_req_pool_idx")

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break
        else:
            adder_saturated = False

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        min_batch_size = self._get_delayed_batching_threshold()
        running_need_reqs = sum(
            1 for req in self.running_batch.reqs if req.status == "need"
        )
        # In overlap mode, a peer-returned request can otherwise wait for this
        # scheduler while this scheduler waits for the delayed-batching
        # threshold. The escape hatch below is intentionally disabled for the
        # normal scheduler so it preserves throughput-oriented batching.
        min_batch_size = self._effective_min_prefill_bs(
            len(can_run_list),
            running_need_reqs,
            min_batch_size,
            adder_saturated,
            self._has_overlap_peer_routed_req(can_run_list),
        )
        if running_need_reqs + len(can_run_list) < min_batch_size:
            self._undo_unlaunched_prefill_admissions(can_run_list)
            return None

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.current_scheduler_metrics_enabled():
            self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()
        self._free_reused_req_pool_slots(can_run_list)

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def _get_delayed_batching_threshold(self) -> int:
        """Return the current min batch size for extend/decode gating.

        ``active_rids`` is maintained by idempotent cross-scheduler control
        messages.
        A stale or over-decremented value should not collapse the batching
        threshold while this scheduler still owns live req objects, so list
        configs are indexed by the larger of the control-plane count and the
        live scheduler-state count.
        """
        delayed_batching_threshold = self.server_args.delayed_batching_threshold
        if not isinstance(delayed_batching_threshold, list):
            return int(delayed_batching_threshold)
        if not delayed_batching_threshold:
            return 1

        active_count = max(len(self.active_rids), self._live_scheduler_req_count())
        active_idx = min(max(active_count, 1), len(delayed_batching_threshold))
        return int(delayed_batching_threshold[active_idx - 1])

    def _live_scheduler_req_count(self) -> int:
        rids = set()
        for attr in (
            "running_batch",
            "last_batch",
            "batch_not_need",
            "waiting_queue",
            "grammar_queue",
        ):
            holder = getattr(self, attr, None)
            reqs = getattr(holder, "reqs", holder)
            if reqs is None:
                continue
            for req in reqs:
                rid = getattr(req, "rid", None)
                if rid is None:
                    continue
                if getattr(req, "status", None) == "finished":
                    continue
                finished = getattr(req, "finished", None)
                if callable(finished) and finished():
                    continue
                rids.add(rid)
        return len(rids)

    @staticmethod
    def _batches_have_overlapping_rids(batch_a, batch_b) -> bool:
        reqs_a = getattr(batch_a, "reqs", None)
        reqs_b = getattr(batch_b, "reqs", None)
        if not reqs_a or not reqs_b:
            return False
        rids_a = {getattr(req, "rid", None) for req in reqs_a}
        return any(getattr(req, "rid", None) in rids_a for req in reqs_b)

    def _effective_min_prefill_bs(
        self,
        can_run_count: int,
        running_need_count: int,
        min_batch_size: int,
        adder_saturated: bool,
        has_peer_routed_req: bool,
    ) -> int:
        total_ready = running_need_count + can_run_count
        # A peer-routed request can participate in a circular wait only in the
        # overlap scheduler.  In the normal scheduler it must continue to obey
        # delayed batching; otherwise every returned request immediately
        # launches an under-filled reference-model batch and destroys the
        # throughput benefit of delayed_batching_threshold at low concurrency.
        if can_run_count and (adder_saturated or has_peer_routed_req):
            return min(min_batch_size, total_ready)
        return min_batch_size

    def _is_peer_routed_req(self, req: Req) -> bool:
        kvcache_loc = getattr(req, "kvcache_loc", None)
        if not isinstance(kvcache_loc, dict):
            return False
        return any(
            name != self.server_args.server_name and loc is not None
            for name, loc in kvcache_loc.items()
        )

    def _has_overlap_peer_routed_req(self, reqs: Iterable[Req]) -> bool:
        """Whether overlap's circular-wait escape hatch may be used."""
        return self.enable_overlap and any(
            self._is_peer_routed_req(req) for req in reqs
        )

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        """Run one model step, then keep locally, hand off, or finish."""
        if getattr(batch, "reqs", None) is None or (
            hasattr(batch.forward_mode, "is_idle") and batch.forward_mode.is_idle()
        ) or (
            hasattr(batch.forward_mode, "is_dummy_first")
            and batch.forward_mode.is_dummy_first()
        ):
            super().process_batch_result(batch, result, launch_done)
            return

        cached_overlap_result = None
        if (
            self.enable_overlap
            and self.is_generation
            and (
                batch.forward_mode.is_decode()
                or batch.forward_mode.is_extend()
            )
        ):
            cached_overlap_result = self.tp_worker.resolve_last_batch_result(
                launch_done
            )
            logits_output, next_token_ids, can_run_cuda_graph = cached_overlap_result
            result = GenerationBatchResult(
                logits_output=logits_output,
                pp_hidden_states_proxy_tensors=(
                    result.pp_hidden_states_proxy_tensors
                    if result is not None
                    else None
                ),
                next_token_ids=next_token_ids,
                extend_input_len_per_req=(
                    result.extend_input_len_per_req if result is not None else None
                ),
                extend_logprob_start_len_per_req=(
                    result.extend_logprob_start_len_per_req
                    if result is not None
                    else None
                ),
                bid=result.bid if result is not None else None,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        destinations = self._normalize_route_decisions(
            self.route(batch, result), batch.batch_size()
        )
        current_model = self.server_args.server_name
        for req in batch.reqs:
            req.model_tokens[current_model][0] += 1
            req.model_tokens[current_model][1] += 1

        for i, req in enumerate(batch.reqs):
            if destinations[i] != current_model:
                setattr(req, _SKIP_UNFINISHED_CACHE_ATTR, True)

        try:
            if cached_overlap_result is None:
                super().process_batch_result(batch, result, launch_done)
            else:
                resolve_last_batch_result = self.tp_worker.resolve_last_batch_result
                self.tp_worker.resolve_last_batch_result = (
                    lambda _launch_done=None: cached_overlap_result
                )
                try:
                    super().process_batch_result(batch, result, None)
                finally:
                    self.tp_worker.resolve_last_batch_result = (
                        resolve_last_batch_result
                    )
        finally:
            for req in batch.reqs:
                if hasattr(req, _SKIP_UNFINISHED_CACHE_ATTR):
                    delattr(req, _SKIP_UNFINISHED_CACHE_ATTR)

        if cached_overlap_result is not None and result.next_token_ids is not None:
            batch.output_ids = torch.tensor(
                result.next_token_ids, dtype=torch.int64, device=batch.device
            )

        reqs_to_send: Dict[str, List[PeerReq]] = {}
        need_rids = []
        for i, req in enumerate(batch.reqs):
            if req.finished():
                finish_seq = getattr(req, "handoff_seq", 0) + 1
                req.handoff_seq = finish_seq
                self.latest_handoff_seq[req.rid] = finish_seq
                self.active_rids.discard(req.rid)
                self._mark_terminal(req.rid, finish_seq)
                self.broadcast_control(
                    rid=req.rid,
                    status="finished",
                    handoff_seq=finish_seq,
                    model_tokens=req.model_tokens,
                    final_output_ids=req.output_ids,
                )
                continue

            req.kvcache_loc[self.server_args.server_name] = (
                len(req.origin_input_ids) + len(req.output_ids) - 1
            )
            destination = destinations[i]
            if destination != current_model:
                req.status = "notneed"
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : req.kvcache_loc[self.server_args.server_name]
                ]
                req.handoff_seq = getattr(req, "handoff_seq", 0) + 1
                self.latest_handoff_seq[req.rid] = req.handoff_seq
                req_to_send = self.send(req, destination)
                reqs_to_send.setdefault(destination, []).append(req_to_send)
            else:
                need_rids.append(req.rid)

        if reqs_to_send:
            self.send_requests_to_peers(reqs_to_send)
            self.batch_not_need.merge_batch(batch)
            need_indices = [
                i for i, req in enumerate(batch.reqs) if req.rid in need_rids
            ]
            not_need_rids = [
                req.rid
                for req in self.batch_not_need.reqs
                if req.status == "notneed" and not req.finished()
            ]
            not_keep_in_batch_not_need = [
                i
                for i, req in enumerate(self.batch_not_need.reqs)
                if req.rid in not_need_rids
            ]
            batch.filter_batch(keep_indices=need_indices)
            self.batch_not_need.filter_batch(keep_indices=not_keep_in_batch_not_need)

    def _normalize_route_decisions(
        self, choices, batch_size: int
    ) -> List[str]:
        if isinstance(choices, torch.Tensor):
            choices = choices.detach().cpu().tolist()
        else:
            choices = list(choices)

        if len(choices) != batch_size:
            raise ValueError(
                f"route() returned {len(choices)} decisions for batch size "
                f"{batch_size}"
            )

        current_model = self.server_args.server_name
        destinations: List[str] = []
        for choice in choices:
            if isinstance(choice, str):
                destination = choice
            elif len(self.peer_names) == 1:
                destination = self.peer_name if bool(choice) else current_model
            else:
                choice_idx = int(choice)
                if choice_idx < 0 or choice_idx >= len(
                    self.server_args.all_server_names
                ):
                    raise ValueError(f"Invalid model index returned by route(): {choice}")
                destination = self.server_args.all_server_names[choice_idx]

            if destination not in self.server_args.all_server_names:
                raise ValueError(
                    f"route() returned unknown model {destination!r}; available: "
                    f"{self.server_args.all_server_names}"
                )
            destinations.append(destination)
        return destinations

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        if not self.server_args.is_entry_model:
            return

        rids = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        output_hidden_states = None
        model_tokens = []
        return_logprob = any(
            [
                hasattr(req, "ori_return_logprob") and req.ori_return_logprob
                for req in reqs
            ]
        )

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_abort:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                should_output = True
            else:
                if req.stream:
                    if self.server_args.rollback_enabled:
                        raise ValueError(
                            "Stream is not supported for rollback-enabled scheduler"
                        )
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if not self.model_config.is_multimodal_gen
                        and stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )
                elif self.server_args.rollback_enabled:
                    should_output = False
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        if not self.model_config.is_multimodal_gen
                        else False
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(req.output_ids[send_token_offset:])
                req.send_token_offset = len(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)
                model_tokens.append(req.model_tokens)
                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if (
                    hasattr(req, "ori_return_hidden_states")
                    and req.ori_return_hidden_states
                ):
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)

            if (
                req.finished()
                and self.tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to detokenizer
        if rids:
            if self.model_config.is_multimodal_gen:
                return

            self.send_to_detokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids,
                    finished_reasons,
                    decoded_texts,
                    decode_ids_list,
                    read_offsets,
                    output_ids,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                    no_stop_trim,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    spec_verify_ct,
                    input_token_logprobs_val,
                    input_token_logprobs_idx,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx,
                    output_hidden_states,
                    model_tokens,
                )
            )

    @abstractmethod
    def route(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ) -> torch.Tensor:
        """
        Router decision for cooperative inference.

        Args:
            batch: The batch of requests to route.
            result: The result of the batch.

        Returns:
            A tensor of model choices.
        """
        raise NotImplementedError

    def send(self, req: Req, destination: Optional[str] = None) -> PeerReq:
        """
        Construct a PeerReq from a Req.

        Args:
            req: The request to construct a PeerReq from.

        Returns:
            A PeerReq.
        """
        destination = destination or self.peer_name
        if destination is None:
            raise ValueError("send() requires an explicit destination")
        current_token_ids = req.origin_input_ids + req.output_ids
        req_to_send = PeerReq.construct_from_req(
            req=req,
            source=self.server_args.server_name,
            destination=destination,
            current_token_ids=current_token_ids,
            entry_model=self.server_args.entry_model_name,
            handoff_seq=req.handoff_seq,
        )
        return req_to_send

    # ------------------------------------------------------------------
    # batch_not_need management
    # ------------------------------------------------------------------

    def init_batch_not_need(self):
        """Create an empty side-batch for requests delegated to a peer."""
        self.batch_not_need = ScheduleBatch.init_new(
            reqs=[],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        self.batch_not_need.prepare_for_extend()
        self.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            self.batch_not_need.device, non_blocking=True
        )

    # ------------------------------------------------------------------
    # Peer token acceptance policy
    # ------------------------------------------------------------------

    def accept_peer_tokens(
        self,
        reqs: List[Req],
        peer_token_ids: List[List[int]],
    ) -> List[int]:
        """Decide how many tokens to accept from the peer for each request.

        Called by ``receive`` for every batch of normal
        (non-control) messages whose ``rid`` is already known in
        ``batch_not_need``.

        Args:
            reqs: requests currently in ``batch_not_need``, one per
                message.
            peer_token_ids: the token-id lists the peer sent, aligned
                1-to-1 with *reqs*.

        Returns:
            A list of integers, one per request.  Each value is the
            number of leading tokens from the corresponding
            ``peer_token_ids`` entry that are accepted:

            * **0** — reject all peer tokens.  The request resumes
              decode with its own last generated token.
            * **1** — accept one token (decode path).
            * **N > 1** — accept N tokens (extend / prefill path).

        The default implementation accepts **all** tokens the peer sent.
        Subclasses override this to implement custom verification or
        speculative-acceptance logic.
        """
        return [len(tids) for tids in peer_token_ids]

    # ------------------------------------------------------------------
    # KV cache rollback
    # ------------------------------------------------------------------

    def rollback_kvcache(self, req: Req, n: int) -> None:
        """Roll back the last ``n`` decoded tokens of ``req`` end-to-end.

        Concretely this:

        1. Frees the KV pool slots for positions
           ``[M-n, M)`` (where ``M = len(origin_input_ids) + len(output_ids)``).
        2. Trims the last ``n`` entries off ``req.output_ids`` AND every
           parallel per-decode array sglang grows in
           ``super().process_batch_result``:

           * ``output_token_logprobs_val/idx``,
             ``output_top_logprobs_val/idx``,
             ``output_token_ids_logprobs_val/idx`` (only populated when
             ``return_logprob=True``)
           * ``hidden_states`` (only populated when
             ``return_hidden_states=True``)

           These arrays are advanced once per decoded token regardless
           of whether ``stream_output`` actually fires, so just trimming
           ``output_ids`` would leave them ``n`` entries longer than
           the post-rollback token list and break index alignment in
           the subsequent stream flush.
        3. Defensively clamps the ``send_*_offset`` cursors used by
           :meth:`stream_output_generation` in case a periodic
           ``DEFAULT_FORCE_STREAM_INTERVAL`` flush already advanced
           them past the new tail.  When the caller correctly
           suppresses streaming for these tokens the cursors stay at
           ``0`` and the clamps are no-ops.
        4. Resets ``req.prefix_indices`` so the next extend recomputes
           ``extend_input_len = len(fill_ids) - len(prefix_indices)``
           correctly against the rolled-back state.

        What this method does NOT undo is anything already shipped
        downstream: the detokenizer's ``decode_ids`` / ``decoded_text``
        and the tokenizer-manager's ``state.text`` are append-only.
        Callers that need a fully consistent user-visible output must
        additionally suppress ``stream_output`` for the drafted tokens
        (e.g. by filtering them out of
        :meth:`stream_output_generation`).
        """
        if req.req_pool_idx is None:
            return
        n = min(n, len(req.output_ids))
        if n <= 0:
            return

        cur_total = len(req.origin_input_ids) + len(req.output_ids)
        new_total = cur_total - n

        # 1. Free the KV slots for the last n decoded positions.
        free_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, new_total:cur_total
        ]
        self.token_to_kv_pool_allocator.free(free_indices)

        # 2. Trim output_ids and every per-decode array that sglang
        #    grew in lock-step with it during the forward passes.
        req.output_ids = req.output_ids[:-n]
        for attr in (
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
        ):
            arr = getattr(req, attr, None)
            if isinstance(arr, list) and arr:
                setattr(req, attr, arr[:-n])
        hidden = getattr(req, "hidden_states", None)
        if isinstance(hidden, list) and hidden:
            req.hidden_states = hidden[:-n]

        # 3. Clamp the streaming cursors so the next flush slices
        #    against the new tail rather than into thin air.
        new_n_output = len(req.output_ids)
        if getattr(req, "send_token_offset", 0) > new_n_output:
            req.send_token_offset = new_n_output
        if getattr(req, "send_decode_id_offset", 0) > new_n_output:
            req.send_decode_id_offset = new_n_output
        logprobs_val = getattr(req, "output_token_logprobs_val", None)
        if isinstance(logprobs_val, list):
            new_n_logprobs = len(logprobs_val)
            if getattr(req, "send_output_token_logprobs_offset", 0) > new_n_logprobs:
                req.send_output_token_logprobs_offset = new_n_logprobs

        # 4. Reset prefix_indices so the next extend recomputes
        #    extend_input_len correctly against the rolled-back state.
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :new_total
        ]

    def _owns_radix_cache_lock(self, req: Req) -> bool:
        prefix_indices = getattr(req, "prefix_indices", None)
        return (
            getattr(req, "req_pool_idx", None) is not None
            and getattr(req, "last_node", None) is not None
            and prefix_indices is not None
            and len(prefix_indices) > 0
        )

    def _dec_req_radix_lock(self, req: Req) -> None:
        if getattr(self, "is_hybrid", False):
            self.tree_cache.dec_lock_ref(
                req.last_node, getattr(req, "swa_uuid_for_lock", None)
            )
            req.swa_uuid_for_lock = None
        else:
            self.tree_cache.dec_lock_ref(req.last_node)

    def _inc_req_radix_lock(self, req: Req) -> None:
        if getattr(self, "is_hybrid", False):
            req.swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
        else:
            self.tree_cache.inc_lock_ref(req.last_node)

    def _free_reused_req_pool_slots(self, reqs: List[Req]) -> None:
        for req in reqs:
            old_req_pool_idx = getattr(req, "_tokenrouting_old_req_pool_idx", None)
            if old_req_pool_idx is None:
                continue
            if old_req_pool_idx != req.req_pool_idx:
                self.req_to_token_pool.free(old_req_pool_idx)
            delattr(req, "_tokenrouting_old_req_pool_idx")

    def _undo_unlaunched_prefill_admissions(self, reqs: List[Req]) -> None:
        for req in reqs:
            old_req_pool_idx = getattr(req, "_tokenrouting_old_req_pool_idx", None)
            if old_req_pool_idx is None:
                # The request did not own a waiting_queue radix lock before
                # PrefillAdder admitted it, so the adder's lock must be undone.
                self._dec_req_radix_lock(req)
            else:
                # For a re-admitted waiting_queue request, we released the old
                # lock before AddReq and PrefillAdder reacquired it. The request
                # is not launched, but it still needs exactly one waiting lock.
                delattr(req, "_tokenrouting_old_req_pool_idx")

    def _pop_waiting_req_by_rid(self, rid: str) -> Optional[Req]:
        for i, req in enumerate(self.waiting_queue):
            if req.rid == rid:
                return self.waiting_queue.pop(i)
        return None

    def _finish_waiting_req_from_peer(self, req: Req, msg: PeerReq) -> None:
        # Release local resources against the local sequence length before
        # replacing it with the global final output.
        self._release_waiting_req_resources(req)
        req.output_ids = self._final_output_ids_from_msg(req, msg)
        self._replace_last_generated_hidden_states(
            req, getattr(msg, "hidden_states", None) or []
        )
        req.check_finished()
        if msg.model_tokens:
            req.model_tokens = msg.model_tokens
        req.handoff_seq = int(getattr(msg, "handoff_seq", 0))
        req.status = "finished"
        if self.server_args.is_entry_model:
            self.stream_output_generation([req], return_logprob=False)

    def _release_waiting_req_resources(self, req: Req) -> None:
        if self._owns_radix_cache_lock(req):
            self._dec_req_radix_lock(req)

        if req.req_pool_idx is None:
            return

        if getattr(self.tree_cache, "disable", False):
            end = len(req.origin_input_ids) + len(req.output_ids)
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :end]
            self.token_to_kv_pool_allocator.free(token_indices)
        else:
            prefix_indices = getattr(req, "prefix_indices", None)
            prefix_len = len(prefix_indices) if prefix_indices is not None else 0
            stale_cached_prefix_len = getattr(req, _STALE_RADIX_PREFIX_ATTR, None)
            if stale_cached_prefix_len is None:
                page_size = getattr(self, "page_size", 1)
                stale_cached_prefix_len = (prefix_len // page_size) * page_size
            if prefix_len > stale_cached_prefix_len:
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, stale_cached_prefix_len:prefix_len
                ]
                self.token_to_kv_pool_allocator.free(token_indices)
            if hasattr(req, _STALE_RADIX_PREFIX_ATTR):
                delattr(req, _STALE_RADIX_PREFIX_ATTR)

        self.req_to_token_pool.free(req.req_pool_idx)
        req.req_pool_idx = None

    def print_kv_msg(self):
        if self.tp_rank == 0:
            _, _, available_size, evictable_size = self._get_token_info()
            protected_size = self.tree_cache.protected_size()
            print(
                f"DEBUG: {self.server_args.server_name} KV cache available {available_size} protected {protected_size} evictable {evictable_size}"
            )


def _run_scheduler_process(
    scheduler_class: Type[Scheduler],
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    balance_meta: Optional[DPBalanceMeta] = None,
):
    """Drop-in replacement for sglang's ``run_scheduler_process`` that
    instantiates *scheduler_class* instead of the hard-coded
    ``sglang.srt.managers.scheduler.Scheduler``."""
    global FIRST_SCHEDULER

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"

    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    try:
        scheduler = scheduler_class(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta=balance_meta,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )
        if not server_args.is_entry_model:
            logging.basicConfig(
                level=getattr(logging, "ERROR"),
                format=f"[%(asctime)s{prefix}] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                force=True,
            )

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                if server_args.pp_size > 1:
                    scheduler.event_loop_pp_disagg_prefill()
                else:
                    scheduler.event_loop_normal_disagg_prefill()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(
            f"Scheduler {server_args.server_name} hit an exception: {traceback}"
        )
        parent_process.send_signal(signal.SIGQUIT)
