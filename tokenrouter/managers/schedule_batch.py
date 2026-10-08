from itertools import chain
from typing import List, Optional, Union

import torch
from sglang.srt.managers.schedule_batch import GLOBAL_SERVER_ARGS_KEYS, Req
from sglang.srt.managers.schedule_batch import ScheduleBatch as SglangScheduleBatch
from sglang.srt.managers.schedule_batch import (
    get_last_loc,
    write_req_to_token_pool_triton,
)
from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import support_triton

global_server_args_dict = {k: getattr(ServerArgs, k) for k in GLOBAL_SERVER_ARGS_KEYS}


class ScheduleBatch(SglangScheduleBatch):
    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        # Allocate req slots
        bs = len(self.reqs)
        req_pool_indices = [req.req_pool_idx for req in self.reqs]
        new_req_indices = [i for i in range(bs) if req_pool_indices[i] is None]
        if new_req_indices:
            new_alloc_req_pool_indices = self.alloc_req_slots(len(new_req_indices))
            for i, idx in enumerate(new_req_indices):
                req_pool_indices[idx] = new_alloc_req_pool_indices[i]

        # Init tensors
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        orig_seq_lens = [max(len(r.fill_ids), len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]

        token_type_ids = [
            r.token_type_ids for r in reqs if r.token_type_ids is not None
        ]

        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        input_ids_tensor = torch.tensor(
            list(chain.from_iterable(input_ids)), dtype=torch.int64
        ).to(self.device, non_blocking=True)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        orig_seq_lens_tensor = torch.tensor(orig_seq_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        prefix_lens_tensor = torch.tensor(
            prefix_lens, dtype=torch.int64, device=self.device
        )

        token_type_ids_tensor = None
        if len(token_type_ids) > 0:
            token_type_ids_tensor = torch.tensor(
                sum(token_type_ids, []), dtype=torch.int64
            ).to(self.device, non_blocking=True)

        extend_lens_tensor = seq_lens_tensor - prefix_lens_tensor

        # Copy prefix and do some basic check
        input_embeds = []
        extend_input_logprob_token_ids = []
        multimodal_inputs = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )
                if isinstance(self.tree_cache, SWAChunkCache):
                    self.tree_cache.evict_swa(
                        req, pre_len, self.model_config.attention_chunk_size
                    )

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # If req.input_embeds is already a list, append its content directly
                input_embeds.extend(req.input_embeds)  # Use extend to avoid nesting

            multimodal_inputs.append(req.multimodal_inputs)

            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False

            # Compute the relative logprob_start_len in an extend batch
            if req.logprob_start_len >= pre_len:
                req.extend_logprob_start_len = min(
                    req.logprob_start_len - pre_len,
                    req.extend_input_len,
                    req.seqlen - 1,
                )
            else:
                req.extend_logprob_start_len = 0

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    len(req.fill_ids),
                )
                # Apply logprob_start_len
                if global_start_idx < req.logprob_start_len:
                    global_start_idx = req.logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_input_len - req.extend_logprob_start_len number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
        else:
            extend_input_logprob_token_ids = None

        # Allocate memory
        if self.token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = self.alloc_token_slots(extend_num_tokens)
        else:
            last_loc = get_last_loc(
                self.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
            )
            out_cache_loc = self.alloc_paged_token_slots_extend(
                prefix_lens_tensor, seq_lens_tensor, last_loc, extend_num_tokens
            )

        # Set fields
        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.seq_lens = seq_lens_tensor
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds).to(self.device, non_blocking=True)
            if input_embeds
            else None
        )
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            for mm_item in mm_input.mm_items:
                pixel_values = getattr(mm_item, "feature", None)
                if isinstance(pixel_values, torch.Tensor):
                    mm_item.feature = pixel_values.to(self.device, non_blocking=True)
        self.multimodal_inputs = multimodal_inputs
        self.token_type_ids = token_type_ids_tensor
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # Write to req_to_token_pool
        if support_triton(global_server_args_dict.get("attention_backend")):
            # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)

            write_req_to_token_pool_triton[(bs,)](
                self.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
                seq_lens_tensor,
                extend_lens_tensor,
                out_cache_loc,
                self.req_to_token_pool.req_to_token.shape[1],
            )
        else:
            pt = 0
            for i in range(bs):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )
                pt += extend_lens[i]

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        # if keep_indices is None or len(keep_indices) == 0:
        #     # Filter out all requests
        #     self.reqs = []
        #     return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = (
            self.output_ids[keep_indices_device]
            if self.output_ids is not None
            else None
        )
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(keep_indices_device)
