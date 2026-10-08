import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchMultimodalOut,
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import ReqState as SglangReqState
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager as SglangTokenizerManager,
)

from tokenrouter.managers.io_struct import BatchStrOut, BatchTokenIDOut

logger = logging.getLogger(__name__)


@dataclass
class ReqState(SglangReqState):
    model_tokens: Dict[str, List[int]] = field(default_factory=dict)


class TokenizerManager(SglangTokenizerManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rid_to_state: Dict[str, ReqState] = {}

    def _send_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        target_model = getattr(obj, "target_model", None)
        if target_model is not None:
            setattr(tokenized_obj, "target_model", target_model)
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = ReqState(
            [],
            False,
            asyncio.Event(),
            obj,
            created_time=created_time,
            model_tokens={name: [0, 0] for name in self.server_args.all_server_names},
        )
        self.rid_to_state[obj.rid] = state
        return state

    def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOut, BatchEmbeddingOut, BatchMultimodalOut, BatchTokenIDOut
        ],
    ):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.server_args.weight_version,
            }

            if getattr(state.obj, "return_logprob", False):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs
                    and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            if not isinstance(recv_obj, BatchEmbeddingOut):
                model_tokens = recv_obj.model_tokens[i]
                for model_name in self.server_args.all_server_names:
                    model_tokens[model_name] = {
                        "generated_tokens": model_tokens[model_name][0],
                        "accepted_tokens": model_tokens[model_name][1],
                    }
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                        "model_tokens": model_tokens,
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOut):
                state.text += recv_obj.output_strs[i]
                if state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "text": state.text,
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchTokenIDOut):
                if self.server_args.stream_output and state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchMultimodalOut):
                raise NotImplementedError("BatchMultimodalOut not implemented")
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.server_args.speculative_algorithm:
                    meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time
                del self.rid_to_state[rid]

                # Mark ongoing LoRA request as finished.
                if self.server_args.enable_lora and state.obj.lora_path:
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            state.out_list.append(out_dict)
            state.event.set()

            # Log metrics and dump
            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)
