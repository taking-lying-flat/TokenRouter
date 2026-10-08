"""Monkey patches for SGLang prefill output handling.

SGLang 0.5.1 slices prefill hidden states with ``len(req.origin_input_ids)``.
With radix cache enabled, the model only computes the uncached extend suffix, so
batched requests can receive hidden states from neighbouring requests.  This
module also lets token-routing schedulers skip an unnecessary unfinished-request
radix insert when a request is immediately handed to the peer and its KV
allocation remains live.  Both patches are installed at runtime without
modifying the installed SGLang package.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Union

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_tokenrouter_hidden_states_prefill_patch"
_SKIP_UNFINISHED_CACHE_ATTR = "_tokenrouting_skip_unfinished_radix_cache"
_STALE_RADIX_PREFIX_ATTR = "_tokenrouting_radix_cached_prefix_len"


def _add_next_token_logprob_return_values(i, req, next_token_ids, logits_output) -> None:
    req.output_token_logprobs_val.append(logits_output.next_token_logprobs[i])
    req.output_token_logprobs_idx.append(next_token_ids[i])

    if req.top_logprobs_num > 0:
        req.output_top_logprobs_val.append(logits_output.next_token_top_logprobs_val[i])
        req.output_top_logprobs_idx.append(logits_output.next_token_top_logprobs_idx[i])

    if req.token_ids_logprob is not None:
        req.output_token_ids_logprobs_val.append(
            logits_output.next_token_token_ids_logprobs_val[i]
        )
        req.output_token_ids_logprobs_idx.append(
            logits_output.next_token_token_ids_logprobs_idx[i]
        )


def _get_extend_input_len(batch, req, index: int, extend_input_len_per_req) -> int:
    if extend_input_len_per_req is not None:
        return int(extend_input_len_per_req[index])

    extend_lens = getattr(batch, "extend_lens", None)
    if extend_lens is not None:
        return int(extend_lens[index])

    extend_input_len = getattr(req, "extend_input_len", None)
    if extend_input_len is not None:
        return int(extend_input_len)

    return len(req.origin_input_ids)


def _patched_process_batch_result_prefill(
    self,
    batch,
    result: Union[object, object],
    launch_done: Optional[threading.Event] = None,
):
    skip_stream_req = None

    if self.is_generation:
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        if self.enable_overlap:
            logits_output, next_token_ids, _ = self.tp_worker.resolve_last_batch_result(
                launch_done
            )
        else:
            if hasattr(next_token_ids, "tolist"):
                next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if (
                    logits_output.next_token_logprobs is not None
                    and hasattr(logits_output.next_token_logprobs, "tolist")
                ):
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if (
                    logits_output.input_token_logprobs is not None
                    and hasattr(logits_output.input_token_logprobs, "tolist")
                ):
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

        hidden_state_offset = 0
        has_hidden_states = logits_output.hidden_states is not None

        logprob_pt = 0
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            hidden_states = None
            if has_hidden_states:
                extend_input_len = _get_extend_input_len(
                    batch, req, i, extend_input_len_per_req
                )
                next_hidden_state_offset = hidden_state_offset + extend_input_len
                hidden_states = logits_output.hidden_states[
                    hidden_state_offset:next_hidden_state_offset
                ]
                hidden_state_offset = next_hidden_state_offset

            if self.is_mixed_chunk and self.enable_overlap and req.finished():
                j = len(batch.out_cache_loc) - len(batch.reqs) + i
                self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                continue

            if req.is_chunked <= 0:
                req.output_ids.append(next_token_id)
                req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                    req.time_stats.completion_time = time.time()
                elif getattr(req, _SKIP_UNFINISHED_CACHE_ATTR, False):
                    if not hasattr(req, _STALE_RADIX_PREFIX_ATTR):
                        page_size = getattr(self, "page_size", 1)
                        prefix_len = len(req.prefix_indices)
                        cached_prefix_len = (
                            (prefix_len // page_size) * page_size
                            if page_size != 1
                            else prefix_len
                        )
                        setattr(req, _STALE_RADIX_PREFIX_ATTR, cached_prefix_len)
                elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                    self.tree_cache.cache_unfinished_req(req)
                    if hasattr(req, _STALE_RADIX_PREFIX_ATTR):
                        delattr(req, _STALE_RADIX_PREFIX_ATTR)

                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    if num_input_logprobs == 0:
                        _add_next_token_logprob_return_values(
                            i, req, next_token_ids, logits_output
                        )
                    else:
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                    logprob_pt += num_input_logprobs

                if req.return_hidden_states and hidden_states is not None:
                    req.hidden_states.append(hidden_states.cpu().clone().tolist())

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        logger.error(
                            f"Grammar accept_token failed for req {req.rid} "
                            f"with token {next_token_id}: {e}"
                        )
                        self.abort_request(AbortReq(req.rid))
                    req.grammar.finished = req.finished()
            else:
                req.is_chunked -= 1
                skip_stream_req = req

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

        self.set_next_batch_sampling_info_done(batch)

    else:
        embeddings, bid = result.embeddings, result.bid
        embeddings = embeddings.tolist()

        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.embedding = embeddings[i]
            if req.is_chunked <= 0:
                req.output_ids.append(0)
                req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                else:
                    self.tree_cache.cache_unfinished_req(req)
            else:
                req.is_chunked -= 1

    self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)


def apply_sglang_hidden_states_patch() -> bool:
    """Install the SGLang prefill output patch once."""

    current = SchedulerOutputProcessorMixin.process_batch_result_prefill
    if getattr(current, _PATCH_ATTR, False):
        return False

    setattr(_patched_process_batch_result_prefill, _PATCH_ATTR, True)
    SchedulerOutputProcessorMixin.process_batch_result_prefill = (
        _patched_process_batch_result_prefill
    )
    return True
