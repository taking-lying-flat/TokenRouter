import random
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.server_args import PortArgs as SglangPortArgs
from sglang.srt.server_args import ServerArgs as SglangServerArgs
from sglang.srt.utils import is_port_available


@dataclass
class ServerArgs(SglangServerArgs):
    server_name: str = None
    all_server_names: list[str] = field(default_factory=list)
    is_entry_model: bool = False
    entry_model_name: str = None
    scheduler_class: str = "BaseTokenRoutingScheduler"
    scheduler_kwargs: dict = field(default_factory=dict)
    port_args_map: dict = field(default_factory=dict)
    force_return_hidden_states: bool = False
    force_return_logprob: bool = False
    force_return_logprobs_num: int = None
    cuda_graph_bs_extend: dict = field(default_factory=dict)
    forward_mode_for_single_token: str = "decode"
    param_num_billion: float = 32.0
    delayed_batching_threshold: list[str] = field(default_factory=list)
    rollback_enabled: bool = False


@dataclass
class PortArgs(SglangPortArgs):
    # Each model owns one inbound routing endpoint. All other models connect
    # a dedicated PUSH socket to this endpoint.
    peer_ipc_name: str = None
    peer_bind_endpoint: str = None
    peer_connect_endpoint: str = None

    @staticmethod
    def init_new(server_args, dp_rank: Optional[int] = None) -> "PortArgs":
        if server_args.nccl_port is None:
            nccl_port = server_args.port + random.randint(100, 1000)
            while True:
                if is_port_available(nccl_port):
                    break
                if nccl_port < 60000:
                    nccl_port += 42
                else:
                    nccl_port -= 43
        else:
            nccl_port = server_args.nccl_port

        if not server_args.enable_dp_attention:
            # Normal case, use IPC within a single node
            return PortArgs(
                tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                nccl_port=nccl_port,
                rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                peer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            )
        else:
            # DP attention. Use TCP + port to handle both single-node and multi-node.
            # if server_args.nnodes == 1 and server_args.dist_init_addr is None:
            #     dist_init_addr = ("127.0.0.1", server_args.port + ZMQ_TCP_PORT_DELTA)
            # elif server_args.dist_init_addr.startswith("["):  # ipv6 address
            #     port_num, host = configure_ipv6(server_args.dist_init_addr)
            #     dist_init_addr = (host, str(port_num))
            # else:
            #     dist_init_addr = server_args.dist_init_addr.split(":")

            # assert (
            #     len(dist_init_addr) == 2
            # ), "please provide --dist-init-addr as host:port of head node"

            # dist_init_host, dist_init_port = dist_init_addr
            # port_base = int(dist_init_port) + 1
            # detokenizer_port = port_base + 1
            # rpc_port = port_base + 2
            # metrics_ipc_name = port_base + 3
            # if dp_rank is None:
            #     # TokenizerManager to DataParallelController
            #     scheduler_input_port = port_base + 4
            # else:
            #     scheduler_input_port = port_base + 4 + 1 + dp_rank

            # return PortArgs(
            #     tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
            #     scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
            #     detokenizer_ipc_name=f"tcp://{dist_init_host}:{detokenizer_port}",
            #     nccl_port=nccl_port,
            #     rpc_ipc_name=f"tcp://{dist_init_host}:{rpc_port}",
            #     metrics_ipc_name=f"tcp://{dist_init_host}:{metrics_ipc_name}",
            # )
            raise NotImplementedError("DP attention is not supported yet")
