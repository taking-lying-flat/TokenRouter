import datetime
import gc
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization import (
    deep_gemm_wrapper,
    monkey_patch_isinstance_for_vllm_base_layer,
)
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    global_server_args_dict,
)
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator_ascend import AscendPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    AscendMLAPagedTokenToKVPool,
    AscendTokenToKVPool,
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.model_runner import ModelRunner as SglangModelRunner
from sglang.srt.model_executor.model_runner import RankZeroFilter
from sglang.srt.model_executor.npu_graph_runner import NPUGraphRunner
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    cpu_has_amx_support,
    dynamic_import,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_fa3_default_architecture,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_no_spec_infer_or_topk_one,
    is_npu,
    monkey_patch_p2p_access_check,
    monkey_patch_vllm_gguf_config,
    set_cuda_arch,
)
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

from tokenrouter.model_executor.cuda_graph_runner import CudaGraphRunner

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()

# Use a small KV cache pool size for tests in CI
SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)

# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300

logger = logging.getLogger(__name__)


class ModelRunner(SglangModelRunner):
    """ModelRunner runs the forward passes of the models."""

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.server_args.scheduler_class not in ("rstitch_slm", "rstitch_llm"):
            return super().sample(logits_output, forward_batch)

        # SGLang sampling can overwrite logits with probabilities in-place.
        # R-Stitch needs the original model logits for its entropy calculation.
        routing_logits = logits_output.next_token_logits.clone()
        try:
            return super().sample(logits_output, forward_batch)
        finally:
            logits_output.next_token_logits = routing_logits

    def init_device_graphs(self):
        """Capture cuda graphs."""
        self.graph_runner = None
        self.cuda_graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.graph_runner = (
            CudaGraphRunner(self) if not _is_npu else NPUGraphRunner(self)
        )
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.cuda_graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.cuda_graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        def is_cuda_graph(forward_mode):
            return (
                forward_mode == ForwardMode.DECODE
                or forward_mode == ForwardMode.TARGET_VERIFY
                or forward_mode == ForwardMode.IDLE
                or forward_mode == ForwardMode.EXTEND
            )

        can_run_graph = bool(
            is_cuda_graph(forward_batch.forward_mode)
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ret, can_run_graph

        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)

        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_extend():
            ret = self.forward_extend(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif forward_batch.forward_mode.is_idle():
            ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        if (
            forward_batch.global_num_tokens_cpu is not None
            and self.pp_group.is_last_rank
        ):
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ret, can_run_graph

    def _get_attention_backend_from_str(self, backend_str: str):
        if backend_str == "flashinfer":
            if not self.use_mla_backend:
                from tokenrouter.layers.attention.flashinfer_backend import (
                    FlashInferAttnBackend,
                )

                # Init streams
                if self.server_args.speculative_algorithm == "EAGLE":
                    if (
                        not hasattr(self, "plan_stream_for_flashinfer")
                        or not self.plan_stream_for_flashinfer
                    ):
                        self.plan_stream_for_flashinfer = torch.cuda.Stream()
                return FlashInferAttnBackend(self)
            else:
                from sglang.srt.layers.attention.flashinfer_mla_backend import (
                    FlashInferMLAAttnBackend,
                )

                return FlashInferMLAAttnBackend(self)
        elif backend_str == "aiter":
            from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

            return AiterAttnBackend(self)
        elif self.server_args.attention_backend == "wave":
            from sglang.srt.layers.attention.wave_backend import WaveAttnBackend

            return WaveAttnBackend(self)
        elif backend_str == "ascend":
            from sglang.srt.layers.attention.ascend_backend import AscendAttnBackend

            return AscendAttnBackend(self)
        elif backend_str == "triton":
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                from sglang.srt.layers.attention.double_sparsity_backend import (
                    DoubleSparseAttnBackend,
                )

                return DoubleSparseAttnBackend(self)
            else:
                from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

                return TritonAttnBackend(self)
        elif backend_str == "torch_native":
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )

            return TorchNativeAttnBackend(self)
        elif backend_str == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            return FlashMLABackend(self)
        elif backend_str == "fa3":
            assert (
                torch.cuda.get_device_capability()[0] == 8 and not self.use_mla_backend
            ) or torch.cuda.get_device_capability()[0] == 9, (
                "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
                "Please use `--attention-backend flashinfer`."
            )
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            return FlashAttentionBackend(self)
        elif backend_str == "cutlass_mla":
            from sglang.srt.layers.attention.cutlass_mla_backend import (
                CutlassMLABackend,
            )

            return CutlassMLABackend(self)
        elif backend_str == "trtllm_mla":
            if not self.use_mla_backend:
                raise ValueError("trtllm_mla backend can only be used with MLA models.")
            from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

            return TRTLLMMLABackend(self)
        elif backend_str == "trtllm_mha":
            if self.use_mla_backend:
                raise ValueError(
                    "trtllm_mha backend can only be used with non-MLA models."
                )
            from sglang.srt.layers.attention.trtllm_mha_backend import (
                TRTLLMHAAttnBackend,
            )

            return TRTLLMHAAttnBackend(self)
        elif backend_str == "intel_amx":
            from sglang.srt.layers.attention.intel_amx_backend import (
                IntelAMXAttnBackend,
            )

            return IntelAMXAttnBackend(self)
        elif backend_str == "dual_chunk_flash_attn":
            from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
                DualChunkFlashAttentionBackend,
            )

            return DualChunkFlashAttentionBackend(self)
        else:
            raise ValueError(f"Invalid attention backend: {backend_str}")
