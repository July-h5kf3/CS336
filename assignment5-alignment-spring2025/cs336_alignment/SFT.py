import gc
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.SFT_utils import (
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenizer_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_config(config_path: str | Path) -> SimpleNamespace:
    with Path(config_path).open() as f:
        config = yaml.safe_load(f) or {}
    return SimpleNamespace(**config)


def format_gsm8k_target(answer: str) -> str:
    reasoning, final_answer = answer.rsplit("####", 1)
    return f"{reasoning.strip()}\n</think> <answer>{final_answer.strip()}</answer>"


def load_gsm8k_sft_examples(
    data_path: str | Path,
    prompt_template_path: str | Path,
    limit: int | None = None,
) -> list[dict[str, str]]:
    prompt_template = Path(prompt_template_path).read_text()
    examples = []
    with Path(data_path).open() as f:
        for line in f:
            if limit is not None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            ground_truth = row["answer"].rsplit("####", 1)[1].strip()
            examples.append(
                {
                    "prompt": prompt_template.format(question=row["question"]),
                    "output": format_gsm8k_target(row["answer"]),
                    "ground_truth": ground_truth,
                }
            )
    return examples


def iter_microbatches(
    examples: list[dict[str, str]],
    micro_batch_size: int,
    shuffle: bool,
    seed: int,
):
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), micro_batch_size):
        yield [examples[i] for i in indices[start : start + micro_batch_size]]


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def clip_gradients(model: torch.nn.Module, max_grad_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def average_loss_for_logging(
    microbatch_losses: list[float],
    gradient_accumulation_steps: int,
) -> float:
    return sum(microbatch_losses) * gradient_accumulation_steps / len(microbatch_losses)


def init_wandb(args: SimpleNamespace):
    if not args.wandb_enabled:
        return None
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def log_train_metrics(
    wandb_run,
    global_step: int,
    train_loss: float,
    learning_rate: float,
) -> None:
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "train/loss": train_loss,
            "train/learning_rate": learning_rate,
        },
        step=global_step,
    )


def log_eval_metrics(wandb_run, output_path: str | Path, global_step: int) -> None:
    if wandb_run is None:
        return
    with Path(output_path).open() as f:
        result = json.load(f)
    eval_keys = [
        "mean_reward",
        "mean_format_reward",
        "mean_answer_reward",
        "mean_response_logprob",
    ]
    metrics = {
        f"eval/{key}": result["summary"][key]
        for key in eval_keys
    }
    wandb_run.log(metrics, step=global_step)


def checkpoint_path_for_step(args: SimpleNamespace, global_step: int) -> Path:
    return Path(args.output_dir) / f"checkpoint-step-{global_step}"


def eval_output_path_for_step(args: SimpleNamespace, global_step: int) -> Path:
    output_path = Path(args.eval_output_path)
    return output_path.with_name(f"{output_path.stem}_step_{global_step}{output_path.suffix}")


def release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_model_tokenizer_optimizer(args: SimpleNamespace, model_path: str | Path):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    return model, tokenizer, optimizer


def save_checkpoint(
    args: SimpleNamespace,
    model,
    tokenizer,
    global_step: int,
    optimizer=None,
) -> Path:
    checkpoint_path = checkpoint_path_for_step(args, global_step)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    if optimizer is not None:
        torch.save(
            {"optimizer": optimizer.state_dict(), "global_step": global_step},
            checkpoint_path / "training_state.pt",
        )
    return checkpoint_path


def load_training_state(args: SimpleNamespace, checkpoint_path: str | Path):
    model, tokenizer, optimizer = load_model_tokenizer_optimizer(args, checkpoint_path)
    state = torch.load(
        Path(checkpoint_path) / "training_state.pt",
        map_location=torch.device(args.device),
    )
    optimizer.load_state_dict(state["optimizer"])
    return model, tokenizer, optimizer


def run_eval_subprocess(
    config_path: str | Path,
    checkpoint_path: str | Path,
    global_step: int | None = None,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "eval",
        str(config_path),
        str(checkpoint_path),
    ]
    if global_step is not None:
        command.append(str(global_step))
    subprocess.run(command, check=True)


def run_eval_and_log(args: SimpleNamespace, checkpoint_path: str | Path, global_step: int) -> None:
    run_eval_subprocess(args.config_path, checkpoint_path, global_step)
    log_eval_metrics(args.wandb_run, eval_output_path_for_step(args, global_step), global_step)


def train_sft(args: SimpleNamespace) -> tuple[Path, int]:
    device = torch.device(args.device)
    model, tokenizer, optimizer = load_model_tokenizer_optimizer(args, args.model)

    examples = load_gsm8k_sft_examples(
        args.train_path,
        args.prompt_template,
        args.max_train_examples,
    )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_steps = 0
    microbatch_losses = []
    checkpoint_path = save_checkpoint(args, model, tokenizer, global_step, optimizer)
    del model, tokenizer, optimizer
    release_cuda_memory()
    run_eval_and_log(args, checkpoint_path, global_step)
    release_cuda_memory()
    model, tokenizer, optimizer = load_training_state(args, checkpoint_path)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        microbatches = iter_microbatches(
            examples,
            micro_batch_size=args.micro_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        )
        for microbatch in tqdm(microbatches, desc=f"SFT epoch {epoch + 1}"):
            tokenized = tokenizer_prompt_and_output(
                prompt_strs=[example["prompt"] for example in microbatch],
                output_strs=[example["output"] for example in microbatch],
                tokenizer=tokenizer,
            )
            tokenized = move_batch_to_device(tokenized, device)
            response_log_probs = get_response_log_probs(
                model=model,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
            )["log_probs"]
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=response_log_probs,
                response_mask=tokenized["response_mask"],
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=args.normalize_constant,
            )
            microbatch_losses.append(loss.detach().item())

            micro_steps += 1
            if micro_steps % args.gradient_accumulation_steps == 0:
                clip_gradients(model, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_loss = average_loss_for_logging(
                    microbatch_losses,
                    args.gradient_accumulation_steps,
                )
                tqdm.write(f"[train] step={global_step} loss={train_loss:.6f}")
                log_train_metrics(
                    args.wandb_run,
                    global_step,
                    train_loss,
                    args.learning_rate,
                )
                microbatch_losses = []
                if global_step % args.eval_every_steps == 0:
                    checkpoint_path = save_checkpoint(
                        args,
                        model,
                        tokenizer,
                        global_step,
                        optimizer,
                    )
                    del model, tokenizer, optimizer
                    release_cuda_memory()
                    run_eval_and_log(args, checkpoint_path, global_step)
                    release_cuda_memory()
                    model, tokenizer, optimizer = load_training_state(args, checkpoint_path)
                    optimizer.zero_grad(set_to_none=True)
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if micro_steps % args.gradient_accumulation_steps != 0:
        clip_gradients(model, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        train_loss = average_loss_for_logging(
            microbatch_losses,
            args.gradient_accumulation_steps,
        )
        tqdm.write(f"[train] step={global_step} loss={train_loss:.6f}")
        log_train_metrics(
            args.wandb_run,
            global_step,
            train_loss,
            args.learning_rate,
        )

    return save_checkpoint(args, model, tokenizer, global_step), global_step


def evaluate_checkpoint_with_vllm(
    args: SimpleNamespace,
    checkpoint_path: Path,
    global_step: int | None = None,
) -> None:
    from vllm import LLM, SamplingParams

    output_path = (
        eval_output_path_for_step(args, global_step)
        if global_step is not None
        else Path(args.eval_output_path)
    )
    sampling_params = SamplingParams(
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        max_tokens=args.eval_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    examples = load_gsm8k_sft_examples(
        args.eval_path,
        args.prompt_template,
        args.max_eval_examples,
    )
    log_generations(
        vllm_model=LLM(model=str(checkpoint_path)),
        examples=examples,
        reward_fn=r1_zero_reward_fn,
        sampling_params=sampling_params,
        output_path=output_path,
    )
    if global_step is not None:
        with output_path.open() as f:
            result = json.load(f)
        result["global_step"] = global_step
        with output_path.open("w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main(config_path: str | Path = "configs/sft_gsm8k.yaml") -> None:
    args = load_config(config_path)
    args.config_path = str(config_path)
    args.wandb_run = init_wandb(args)
    try:
        checkpoint_path, global_step = train_sft(args)
        release_cuda_memory()
        run_eval_subprocess(args.config_path, checkpoint_path)
        log_eval_metrics(
            args.wandb_run,
            args.eval_output_path,
            global_step,
        )
        release_cuda_memory()
    finally:
        if args.wandb_run is not None:
            args.wandb_run.finish()


def eval_main(
    config_path: str | Path,
    checkpoint_path: str | Path,
    global_step: int | None = None,
) -> None:
    args = load_config(config_path)
    evaluate_checkpoint_with_vllm(args, Path(checkpoint_path), global_step)
    release_cuda_memory()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_main(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]) if len(sys.argv) > 4 else None,
        )
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "configs/sft_gsm8k.yaml")
