from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tokenrouter.managers.schedulers.rstitch_llm_scheduler import RStitchLLMScheduler
from tokenrouter.managers.schedulers.rstitch_slm_scheduler import RStitchSLMScheduler
from tokenrouter.model_executor.model_runner import ModelRunner, SglangModelRunner


@pytest.mark.parametrize(
    "scheduler_name,scheduler,expected",
    [
        ("rstitch_slm", RStitchSLMScheduler, [False, True]),
        ("rstitch_llm", RStitchLLMScheduler, [True, False]),
    ],
)
@pytest.mark.parametrize("all_greedy", [True, False])
def test_routing_uses_pre_sampling_logits(
    scheduler_name, scheduler, expected, all_greedy
):
    # Compare a greedy request alone with the same request in a mixed batch.
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if all_greedy:
        logits = logits[:1]
    output = SimpleNamespace(next_token_logits=logits.clone())
    token_ids = torch.arange(len(logits))
    logprobs = torch.zeros(len(logits))

    def sample(self, logits_output, batch):
        # SGLang 0.5.1 writes probabilities into this tensor for non-greedy batches.
        if not all_greedy:
            values = logits_output.next_token_logits
            values.div_(torch.tensor([[1.0], [0.7]]))
            values[:] = torch.softmax(values, dim=-1)
        logits_output.next_token_logprobs = logprobs
        return token_ids

    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(scheduler_class=scheduler_name)
    with patch.object(SglangModelRunner, "sample", sample):
        assert runner.sample(output, None) is token_ids
    assert output.next_token_logprobs is logprobs
    decisions = scheduler.route(
        SimpleNamespace(threshold=0.03), None, SimpleNamespace(logits_output=output)
    )
    assert decisions.tolist() == expected[: len(logits)]
    torch.testing.assert_close(output.next_token_logits, logits)
