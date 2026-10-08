import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import uuid
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import uvloop
import yaml

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    configure_logger,
    kill_process_tree,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
)

from tokenrouter.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from tokenrouter.managers.detokenizer_manager import run_detokenizer_process
from tokenrouter.managers.schedulers import SCHEDULER_REGISTRY
from tokenrouter.managers.schedulers.base_scheduler import _run_scheduler_process
from tokenrouter.managers.tokenizer_manager import TokenizerManager
from tokenrouter.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def _resolve_entry_model(model_server_args: Dict[str, ServerArgs], config: dict) -> str:
    """Resolve `system.entry_model` from config and flag its ServerArgs.

    Must be called before `_launch_subprocesses`: each scheduler subprocess
    reads a pickled copy of `server_args`, so `is_entry_model` has to be set
    on the parent's copy first or the flag is lost across the fork.
    """
    model_names = list(model_server_args.keys())
    system_cfg = config.get("system", {})
    entry_model = system_cfg.get("entry_model", model_names[0])
    if entry_model not in model_server_args:
        raise ValueError(
            f"system.entry_model '{entry_model}' not found in model config. "
            f"Available models: {model_names}"
        )
    for model_name, server_args in model_server_args.items():
        server_args.entry_model_name = entry_model
        server_args.is_entry_model = model_name == entry_model
    return entry_model


class TokenRoutingEngine(EngineBase):
    """
    Multi-LLM cooperation inference engine.

    Each model declared in the config is launched as a full sglang engine
    (scheduler + detokenizer + tokenizer_manager). Requests can be routed
    to any model by name; the default target is the ``entry_model``
    model specified in the config.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path

        def range_constructor(loader, node):
            values = loader.construct_sequence(node)
            return list(range(*values))

        yaml.add_constructor("!range", range_constructor)
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        atexit.register(self.shutdown)

        # Collect the set of valid ServerArgs field names for filtering
        server_args_field_names = {f.name for f in dataclasses.fields(ServerArgs)}

        # ---- Build per-model ServerArgs, separating extra config keys ----
        self.model_server_args: Dict[str, ServerArgs] = {}

        for model_name, model_config in self.config["model"].items():
            server_kwargs = {
                k: v for k, v in model_config.items() if k in server_args_field_names
            }
            self.model_server_args[model_name] = ServerArgs(
                server_name=model_name,
                all_server_names=list(self.config["model"].keys()),
                **server_kwargs,
            )

        model_mem_fraction_static_dict = get_mem_fraction_statics(
            self.model_server_args,
            mem_fraction_overall=self.config.get("system", {}).get(
                "mem_fraction_static", 0.85
            ),
        )
        for model_name, server_args in self.model_server_args.items():
            server_args.mem_fraction_static = model_mem_fraction_static_dict[model_name]

        # ---- Launch sglang subprocesses for every model ----
        self.tokenizer_managers: Dict[str, TokenizerManager] = {}
        self.template_managers: Dict[str, TemplateManager] = {}
        self.scheduler_infos: Dict[str, Dict] = {}
        self.port_args_map: Dict[str, PortArgs] = {}

        for model_name, server_args in self.model_server_args.items():
            self.port_args_map[model_name] = PortArgs.init_new(server_args)

        self.entry_model = _resolve_entry_model(self.model_server_args, self.config)
        self.model_names: List[str] = list(self.model_server_args.keys())

        for model_name, server_args in self.model_server_args.items():
            port_args = self.port_args_map[model_name]
            server_args.port_args_map = self.port_args_map
            logger.info("Launching model '%s': %s", model_name, server_args.model_path)

            tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
                server_args=server_args,
                port_args=port_args,
                scheduler_class=SCHEDULER_REGISTRY[server_args.scheduler_class],
            )
            self.tokenizer_managers[model_name] = tokenizer_manager
            self.template_managers[model_name] = template_manager
            self.scheduler_infos[model_name] = scheduler_info

            tokenizer_manager.name = model_name

        self.tokenizer_manager = self.tokenizer_managers[self.entry_model]
        self.template_manager = self.template_managers[self.entry_model]
        self.scheduler_info = self.scheduler_infos[self.entry_model]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _resolve_tokenizer_manager(
        self, target_model: Optional[str] = None
    ) -> TokenizerManager:
        """Return the tokenizer_manager for *target_model* (default: entry model)."""
        if target_model is not None and target_model != self.entry_model:
            raise ValueError(
                "Token-routing responses are owned by the fixed entry model "
                f"{self.entry_model!r}; target_model={target_model!r} is unsupported"
            )
        return self.tokenizer_manager

    # ------------------------------------------------------------------
    # Generation – synchronous
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data=None,
        audio_data=None,
        video_data=None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        Generate completions using the entry model.

        All arguments mirror ``sglang.srt.managers.io_struct.GenerateReqInput``.
        Use :meth:`async_generate` if you need to route to a non-entry model
        via its ``target_model`` argument.
        """
        target_tokenizer_manager = self._resolve_tokenizer_manager()

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )

        self.generate_rids(obj)

        loop = self._get_event_loop()
        generator = target_tokenizer_manager.generate_request(obj, None)

        if stream:

            def _stream():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return _stream()
        else:
            return loop.run_until_complete(generator.__anext__())

    # ------------------------------------------------------------------
    # Generation – asynchronous
    # ------------------------------------------------------------------

    async def async_generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data=None,
        audio_data=None,
        video_data=None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
        target_model: Optional[str] = None,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """Async counterpart of :meth:`generate`."""
        tokenizer_manager = self._resolve_tokenizer_manager(target_model)

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )

        generator = tokenizer_manager.generate_request(obj, None)
        if stream:
            return generator
        else:
            return await generator.__anext__()

    # ------------------------------------------------------------------
    # EngineBase abstract method implementations
    # ------------------------------------------------------------------

    def flush_cache(self):
        """Flush KV caches on all models."""
        loop = self._get_event_loop()
        for tm in self.tokenizer_managers.values():
            loop.run_until_complete(tm.flush_cache())

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        target_model: Optional[str] = None,
    ):
        """Update weights for a specific model (default: entry model)."""
        model_name = target_model or self.entry_model
        tokenizer_manager = self._resolve_tokenizer_manager(model_name)
        server_args = self.model_server_args[model_name]

        if load_format == "flattened_bucket":
            serialized = named_tensors
        else:
            serialized = [
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(server_args.tp_size)
            ]

        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        loop = self._get_event_loop()
        return loop.run_until_complete(
            tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        """Release GPU memory on all models."""
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        loop = self._get_event_loop()
        for tm in self.tokenizer_managers.values():
            loop.run_until_complete(tm.release_memory_occupation(obj, None))

    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        """Resume GPU memory on all models."""
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        loop = self._get_event_loop()
        for tm in self.tokenizer_managers.values():
            loop.run_until_complete(tm.resume_memory_occupation(obj, None))

    def shutdown(self):
        """Kill all child processes spawned by this engine."""
        kill_process_tree(os.getpid(), include_parent=False)

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    def get_server_info(self) -> Dict[str, Dict]:
        """Return server info for every model, keyed by model name."""
        info: Dict[str, Dict] = {}
        loop = self._get_event_loop()
        for model_name in self.model_names:
            tm = self.tokenizer_managers[model_name]
            internal_states = loop.run_until_complete(tm.get_internal_state())
            info[model_name] = {
                **dataclasses.asdict(self.model_server_args[model_name]),
                **self.scheduler_infos[model_name],
                "internal_states": internal_states,
            }
        return info

    def generate_rids(self, obj: GenerateReqInput):
        self.generate_parallel_sample_num(obj)
        obj._determine_batch_size()
        if obj.is_single:
            obj.rid = uuid.uuid4().hex
        else:
            # Calculate expanded batch size
            if obj.parallel_sample_num == 1:
                num = obj.batch_size
            else:
                # Expand parallel_sample_num
                num = obj.batch_size * obj.parallel_sample_num
            obj._normalize_rid(num)

    def generate_parallel_sample_num(self, obj: GenerateReqInput) -> int:
        if obj.sampling_params is None:
            obj.parallel_sample_num = 1
            return
        elif isinstance(obj.sampling_params, dict):
            obj.parallel_sample_num = obj.sampling_params.get("n", 1)
        else:  # isinstance(obj.sampling_params, list):
            obj.parallel_sample_num = obj.sampling_params[0].get("n", 1)
            for sampling_params in obj.sampling_params:
                if obj.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False


def _launch_subprocesses(
    server_args: ServerArgs,
    port_args: Optional[PortArgs] = None,
    scheduler_class: Type[Scheduler] = Scheduler,
) -> Tuple[TokenizerManager, TemplateManager, Dict]:
    """Launch scheduler, detokenizer, and tokenizer subprocesses.

    Mirrors ``sglang.srt.entrypoints.engine._launch_subprocesses`` but
    accepts *scheduler_class* so callers can inject a custom scheduler.
    """
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
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
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                proc = mp.Process(
                    target=_run_scheduler_process,
                    args=(
                        scheduler_class,
                        server_args,
                        port_args,
                        gpu_id,
                        tp_rank,
                        moe_ep_rank,
                        pp_rank,
                        None,
                        writer,
                        None,
                    ),
                )
                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer, scheduler_class),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            return None, None, None

        launch_dummy_health_check_server(
            server_args.host, server_args.port, server_args.enable_metrics
        )
        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(server_args, port_args),
    )
    detoken_proc.start()

    # Launch tokenizer manager in the main process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data.get("status") != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info


def get_mem_fraction_statics(
    model_server_args: Dict[str, ServerArgs],
    mem_fraction_overall: float = 0.85,
) -> Dict[str, float]:
    """Resolve per-model ``mem_fraction_static`` from a single joint budget.

    ``mem_fraction_overall`` is the per-GPU fraction the cooperative engine
    is allowed to occupy. For each GPU we compute the maximum KV-token count
    that fits within (budget - sum of weights on this GPU); the engine-wide
    token count is the min across GPUs. We then back-solve a per-model
    ``mem_fraction_static`` so each model's static memory holds that token
    count plus its own weight shard.

    The same ``mem_fraction_overall`` works whether models share a GPU
    (combined memory ≤ budget on that GPU) or live on disjoint GPUs (each
    model alone ≤ budget on its own GPU): the per-GPU constraint is computed
    independently for every GPU and the tightest one wins.
    """
    gpu_to_models: Dict[int, List[str]] = {}
    model_to_gpus: Dict[str, List[int]] = {}
    for name, server_args in model_server_args.items():
        gpu_ids = [
            server_args.base_gpu_id + i * server_args.gpu_id_step
            for i in range(server_args.tp_size)
        ]
        model_to_gpus[name] = gpu_ids
        for gid in gpu_ids:
            gpu_to_models.setdefault(gid, []).append(name)

    os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"

    model_config_dict = {
        name: ModelConfig.from_server_args(server_args)
        for name, server_args in model_server_args.items()
    }

    num_layers_dict = {
        name: cfg.num_hidden_layers for name, cfg in model_config_dict.items()
    }

    tp_size_dict = {name: args.tp_size for name, args in model_server_args.items()}

    cell_size_dict = {}
    for name, cfg in model_config_dict.items():
        tp_size = tp_size_dict[name]
        cell_size_dict[name] = (
            cfg.get_num_kv_heads(tp_size)
            * cfg.head_dim
            * num_layers_dict[name]
            * 2
            * torch._utils._element_size(cfg.dtype)
        )

    model_weight_mem_dict = {}
    for name, cfg in model_config_dict.items():
        bytes_per_elem = torch._utils._element_size(cfg.dtype)
        model_weight_mem_dict[name] = bytes_per_elem * float(
            model_server_args[name].param_num_billion * 1e9
        )

    total_gpu_memory_dict = {
        i: torch.cuda.mem_get_info(i)[0] for i in gpu_to_models.keys()
    }

    target_mem_dict = {
        i: total_gpu_memory_dict[i] * mem_fraction_overall
        for i in total_gpu_memory_dict.keys()
    }

    token_num_candidates = []

    for gid, models in gpu_to_models.items():
        weight_sum = 0.0
        cell_sum = 0.0

        for name in models:
            tp_size = tp_size_dict[name]
            weight_sum += model_weight_mem_dict[name] / tp_size
            cell_sum += cell_size_dict[name]

        if cell_sum == 0:
            continue

        token_num_max = (target_mem_dict[gid] - weight_sum) / cell_sum
        token_num_candidates.append(token_num_max)

    assert len(token_num_candidates) > 0, "No valid GPU found"

    token_num_max = int(min(token_num_candidates))
    assert token_num_max > 0, "Not enough memory for KV cache"

    total_mem_dict = {}
    model_mem_fraction_static_dict = {}

    for name in model_config_dict:
        tp_size = tp_size_dict[name]

        weight = model_weight_mem_dict[name] / tp_size
        kv = token_num_max * cell_size_dict[name]

        total_mem_dict[name] = weight + kv

    for name in model_config_dict:
        model_mem_fraction_static_dict[name] = total_mem_dict[name] / min(
            [total_gpu_memory_dict[i] for i in model_to_gpus[name]]
        )
        for i in model_to_gpus[name]:
            total_gpu_memory_dict[i] -= total_mem_dict[name]

    return model_mem_fraction_static_dict
