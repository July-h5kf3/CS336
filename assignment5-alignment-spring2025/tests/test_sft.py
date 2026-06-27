import json
from pathlib import Path

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


def test_format_gsm8k_target_matches_r1_zero_prompt():
    from cs336_alignment import SFT as sft

    target = sft.format_gsm8k_target("2 + 2 = <<2+2=4>>4\n#### 4")

    assert target == "2 + 2 = <<2+2=4>>4\n</think> <answer>4</answer>"


def test_load_gsm8k_sft_examples(tmp_path):
    from cs336_alignment import SFT as sft

    data_path = tmp_path / "gsm8k.jsonl"
    prompt_path = tmp_path / "prompt.txt"
    data_path.write_text(
        json.dumps({"question": "What is 2+2?", "answer": "2+2=4\n#### 4"})
        + "\n"
    )
    prompt_path.write_text("Question: {question}\nAssistant: <think>")

    examples = sft.load_gsm8k_sft_examples(data_path, prompt_path, limit=1)

    assert examples == [
        {
            "prompt": "Question: What is 2+2?\nAssistant: <think>",
            "output": "2+2=4\n</think> <answer>4</answer>",
            "ground_truth": "4",
        }
    ]


def test_clip_gradients_caps_global_norm():
    from cs336_alignment import SFT as sft

    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)

    sft.clip_gradients(model, max_grad_norm=1.0)

    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(parameter.grad.detach(), 2)
                for parameter in model.parameters()
            ]
        ),
        2,
    )
    assert grad_norm <= 1.0 + 1e-6


def test_load_config_uses_yaml_without_defaults(tmp_path):
    from cs336_alignment import SFT as sft

    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: tests/fixtures/tiny-gpt2",
                "epochs: 2",
                "learning_rate: 0.0002",
                "max_eval_examples: 3",
            ]
        )
    )

    config = sft.load_config(config_path)

    assert config.model == "tests/fixtures/tiny-gpt2"
    assert config.epochs == 2
    assert config.learning_rate == 0.0002
    assert config.max_eval_examples == 3


def test_step_paths_include_global_step(tmp_path):
    from types import SimpleNamespace

    from cs336_alignment import SFT as sft

    config = SimpleNamespace(
        output_dir=str(tmp_path / "policy"),
        eval_output_path=str(tmp_path / "eval.json"),
    )

    assert sft.checkpoint_path_for_step(config, 12) == tmp_path / "policy" / "checkpoint-step-12"
    assert sft.eval_output_path_for_step(config, 12) == tmp_path / "eval_step_12.json"


def test_save_checkpoint_uses_global_step(tmp_path):
    from types import SimpleNamespace

    from cs336_alignment import SFT as sft

    class FakeModel:
        def save_pretrained(self, path):
            Path(path, "model.txt").write_text("model")

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path, "tokenizer.txt").write_text("tokenizer")

    config = SimpleNamespace(output_dir=str(tmp_path / "policy"))

    checkpoint_path = sft.save_checkpoint(
        config,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        global_step=7,
    )

    assert checkpoint_path == tmp_path / "policy" / "checkpoint-step-7"
    assert (checkpoint_path / "model.txt").exists()
    assert (checkpoint_path / "tokenizer.txt").exists()


def test_average_loss_for_logging_undoes_accumulation_scaling():
    from cs336_alignment import SFT as sft

    assert sft.average_loss_for_logging([0.5, 1.0], gradient_accumulation_steps=4) == 3.0


def test_log_train_metrics_to_wandb():
    from cs336_alignment import SFT as sft

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

    run = FakeRun()

    sft.log_train_metrics(run, global_step=5, train_loss=2.5, learning_rate=1e-5)

    assert run.logged == [
        (
            {
                "train/loss": 2.5,
                "train/learning_rate": 1e-5,
            },
            5,
        )
    ]


def test_init_wandb_uses_entity_project_and_name(monkeypatch):
    from types import SimpleNamespace

    from cs336_alignment import SFT as sft

    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(sft.wandb, "init", fake_init)

    config = SimpleNamespace(
        wandb_enabled=True,
        wandb_entity="acd66-nankai-university",
        wandb_project="CS336 assignment5",
        wandb_run_name="gsm8k-sft",
    )

    sft.init_wandb(config)

    assert calls == [
        {
            "entity": "acd66-nankai-university",
            "project": "CS336 assignment5",
            "name": "gsm8k-sft",
            "config": vars(config),
        }
    ]


def test_run_eval_and_log_uses_global_step(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from cs336_alignment import SFT as sft

    calls = []
    config = SimpleNamespace(
        config_path="config.yaml",
        eval_output_path=str(tmp_path / "eval.json"),
        wandb_run=object(),
    )

    def fake_run_eval_subprocess(config_path, checkpoint_path, global_step):
        calls.append(("eval", config_path, checkpoint_path, global_step))

    def fake_log_eval_metrics(wandb_run, output_path, global_step):
        calls.append(("log", wandb_run, output_path, global_step))

    monkeypatch.setattr(sft, "run_eval_subprocess", fake_run_eval_subprocess)
    monkeypatch.setattr(sft, "log_eval_metrics", fake_log_eval_metrics)

    sft.run_eval_and_log(config, tmp_path / "checkpoint-step-0", global_step=0)

    assert calls == [
        ("eval", "config.yaml", tmp_path / "checkpoint-step-0", 0),
        ("log", config.wandb_run, tmp_path / "eval_step_0.json", 0),
    ]


def test_log_eval_metrics_to_wandb(tmp_path):
    from cs336_alignment import SFT as sft

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "summary": {
                    "mean_reward": 0.25,
                    "mean_format_reward": 0.5,
                    "mean_answer_reward": 0.25,
                    "mean_response_logprob": -0.75,
                    "mean_response_length": 42,
                }
            }
        )
    )
    run = FakeRun()

    sft.log_eval_metrics(run, eval_path, global_step=10)

    assert run.logged == [
        (
            {
                "eval/mean_reward": 0.25,
                "eval/mean_format_reward": 0.5,
                "eval/mean_answer_reward": 0.25,
                "eval/mean_response_logprob": -0.75,
            },
            10,
        )
    ]
