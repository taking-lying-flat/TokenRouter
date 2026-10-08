import logging
import multiprocessing as mp
import signal
import tempfile
import threading
from typing import Dict, List, Optional, Type

import psutil
import setproctitle
import zmq

from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.data_parallel_controller import (
    DataParallelController as SglangDataParallelController,
)
from sglang.srt.managers.io_struct import (
    BlockReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import DPBalanceMeta
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import bind_port, configure_logger, get_zmq_socket
from sglang.utils import get_exception_traceback

from tokenrouter.managers.schedulers.base_scheduler import _run_scheduler_process

logger = logging.getLogger(__name__)


class DataParallelController(SglangDataParallelController):
    """SGLang DP controller that launches TokenRouting scheduler processes."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_balance_meta: DPBalanceMeta,
        scheduler_class: Type[Scheduler] = Scheduler,
    ) -> None:
        self.scheduler_class = scheduler_class
        self.dp_peer_inputs = []
        self.dp_peer_outputs = []
        self.peer_req_to_dp_rank: Dict[str, int] = {}
        self.peer_name = None
        self.send_to_peer = None
        self.recv_from_peer = None
        super().__init__(server_args, port_args, dp_balance_meta)
        self._init_peer_bridge(server_args, port_args)

    @staticmethod
    def _new_ipc_name() -> str:
        return f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

    def _init_peer_bridge(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        peer_names = [
            name
            for name in getattr(server_args, "all_server_names", [])
            if name != server_args.server_name
        ]
        if server_args.node_rank != 0 or len(peer_names) != 1:
            return

        peer_port_args = server_args.port_args_map.get(peer_names[0])
        if peer_port_args is None:
            return

        self.peer_name = peer_names[0]
        # Expose this model's original peer endpoint to the other model.
        self.send_to_peer = get_zmq_socket(
            self.context, zmq.PUSH, port_args.peer_ipc_name, True
        )
        # Receive replies from the other model's original peer endpoint.
        self.recv_from_peer = get_zmq_socket(
            self.context, zmq.PULL, peer_port_args.peer_ipc_name, False
        )

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        dp_port_args = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = port_args.__class__.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.peer_recv_ipc_name = self._new_ipc_name()
            dp_port_args.append(tmp_port_args)

            self.dp_peer_inputs.append(
                get_zmq_socket(
                    self.context, zmq.PULL, tmp_port_args.peer_ipc_name, False
                )
            )
            self.dp_peer_outputs.append(
                get_zmq_socket(
                    self.context, zmq.PUSH, tmp_port_args.peer_recv_ipc_name, True
                )
            )

            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += server_args.tp_size * server_args.gpu_id_step

        for sock in sockets:
            sock.close()

        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

        return dp_port_args

    def launch_dp_attention_schedulers(self, server_args, port_args):
        self.launch_tensor_parallel_group(server_args, port_args, 0, None)
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            dp_port_args.append(port_args.__class__.init_new(server_args, dp_rank))
        return dp_port_args

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                    )
                    rank_port_args = port_args.__class__.init_new(server_args, dp_rank)
                    rank_port_args.nccl_port = port_args.nccl_port

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                proc = mp.Process(
                    target=_run_scheduler_process,
                    args=(
                        self.scheduler_class,
                        server_args,
                        rank_port_args,
                        gpu_id,
                        tp_rank,
                        moe_ep_rank,
                        pp_rank,
                        dp_rank,
                        writer,
                        self.balance_meta,
                    ),
                )
                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        scheduler_info = []
        for reader in scheduler_pipe_readers:
            scheduler_info.append(reader.recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def _poll_dp_peer_outputs(self) -> None:
        if self.send_to_peer is None:
            return

        for dp_rank, socket in enumerate(self.dp_peer_inputs):
            while True:
                try:
                    reqs = socket.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                for req in reqs:
                    rid = getattr(req, "rid", None)
                    if rid is None:
                        continue
                    if getattr(req, "status", None) == "finished":
                        self.peer_req_to_dp_rank.pop(rid, None)
                    else:
                        self.peer_req_to_dp_rank[rid] = dp_rank

                self.send_to_peer.send_pyobj(reqs)

    def _poll_peer_replies(self) -> None:
        if self.recv_from_peer is None:
            return

        while True:
            try:
                reqs = self.recv_from_peer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break

            by_dp_rank: Dict[int, List] = {}
            for req in reqs:
                rid = getattr(req, "rid", None)
                dp_rank = self.peer_req_to_dp_rank.get(rid, 0)
                by_dp_rank.setdefault(dp_rank, []).append(req)
                if getattr(req, "status", None) == "finished":
                    self.peer_req_to_dp_rank.pop(rid, None)

            for dp_rank, routed_reqs in by_dp_rank.items():
                if dp_rank >= len(self.dp_peer_outputs):
                    logger.warning(
                        "Dropping peer replies for unknown DP rank %s: %s",
                        dp_rank,
                        [getattr(req, "rid", None) for req in routed_reqs],
                    )
                    continue
                self.dp_peer_outputs[dp_rank].send_pyobj(routed_reqs)

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                    ),
                ):
                    self.dispatching(recv_req)
                elif isinstance(recv_req, BlockReqInput):
                    for worker in self.workers:
                        worker.send_pyobj(recv_req)
                else:
                    for worker in self.workers[:: self.control_message_step]:
                        worker.send_pyobj(recv_req)

            self._poll_dp_peer_outputs()
            self._poll_peer_replies()


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    scheduler_class: Optional[Type[Scheduler]] = None,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()
    balance_meta = DPBalanceMeta(server_args.dp_size)

    try:
        controller = DataParallelController(
            server_args,
            port_args,
            dp_balance_meta=balance_meta,
            scheduler_class=scheduler_class or Scheduler,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
    finally:
        balance_meta.destructor()
