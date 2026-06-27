import json

import torch

import cs336_alignment.SFT_utils as sft_utils

from .adapters import (
    run_compute_entropy as compute_entropy,
    run_get_response_log_probs as get_response_log_probs,
    run_masked_normalize as masked_normalize,
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_sft_microbatch_train_step as sft_microbatch_train_step,
)

def test_tokenize_prompt_and_output(numpy_snapshot, prompt_strs, output_strs, tokenizer):
    output = tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )
    numpy_snapshot.assert_match(output)

def test_compute_entropy(numpy_snapshot, logits):
    output = compute_entropy(logits)
    numpy_snapshot.assert_match(output)


def test_get_response_log_probs(
    numpy_snapshot,
    model,
    input_ids,
    labels,
):
    output = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=True,
    )
    numpy_snapshot.assert_match(output)

def test_masked_normalize_dim0(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=0,
    )
    numpy_snapshot.assert_match(output)


def test_masked_normalize_dim1(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_normalize_dimlast(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_normalize_dimNone(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
    )
    numpy_snapshot.assert_match(output)

def test_sft_microbatch_train_step(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
):
    policy_log_probs.requires_grad = True
    loss, _ = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=1.0,
    )
    output = {"loss": loss, "policy_log_probs_grad": policy_log_probs.grad}
    numpy_snapshot.assert_match(output)

def test_sft_microbatch_train_step_normalize(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    normalize_constant,
):
    policy_log_probs.requires_grad = True
    loss, _ = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )
    output = {"loss": loss, "policy_log_probs_grad": policy_log_probs.grad}
    numpy_snapshot.assert_match(output)

def test_sft_microbatch_train_step_10_steps(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
):
    policy_log_probs.requires_grad = True

    loss_list = []
    grad_list = []
    for _ in range(10):
        loss, _ = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=1.0,
        )
        loss_list.append(loss)
        grad_list.append(policy_log_probs.grad)

    output = {
        "loss": torch.stack(loss_list),
        "policy_log_probs_grad": torch.stack(grad_list),
    }
    numpy_snapshot.assert_match(output)


def test_log_generations_uses_tqdm(monkeypatch, tmp_path):
    tqdm_calls = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    class FakeGeneration:
        def __init__(self, text):
            self.outputs = [
                type(
                    "FakeOutput",
                    (),
                    {
                        "text": text,
                        "token_ids": [1, 2],
                        "cumulative_logprob": -4.0,
                    },
                )()
            ]

    class FakeVllmModel:
        def generate(self, prompts, sampling_params):
            del sampling_params
            return [FakeGeneration("ok") for _ in prompts]

    def fake_reward_fn(response, ground_truth):
        del response, ground_truth
        return {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}

    monkeypatch.setattr(sft_utils, "tqdm", fake_tqdm)

    output_path = tmp_path / "generations.json"

    result = sft_utils.log_generations(
        vllm_model=FakeVllmModel(),
        examples=[{"prompt": "p", "ground_truth": "ok"}],
        reward_fn=fake_reward_fn,
        sampling_params=None,
        output_path=output_path,
    )

    saved_result = json.loads(output_path.read_text())
    assert result is None
    assert saved_result["summary"]["mean_reward"] == 1.0
    assert saved_result["summary"]["mean_response_logprob"] == -2.0
    assert tqdm_calls == [{"desc": "Logging generations", "total": 1}]
