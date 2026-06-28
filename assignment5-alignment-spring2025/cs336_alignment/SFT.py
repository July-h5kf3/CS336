import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from cs336_alignment.SFT_utils import (
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenizer_prompt_and_output,
)
from cs336_alignment.dataset import build_sft_dataloader, build_sft_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.train_utils import (
    average_loss_for_logging,
    clip_gradients,
    eval_output_path_for_step,
    init_wandb,
    load_config,
    load_model_tokenizer_optimizer,
    load_training_state,
    log_eval_metrics,
    log_train_metrics,
    move_batch_to_device,
    release_cuda_memory,
    save_checkpoint,
)


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

    train_dataset = build_sft_dataset(
        args.dataset,
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
        microbatches = build_sft_dataloader(
            train_dataset,
            micro_batch_size=args.micro_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            num_workers=args.dataloader_num_workers,
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
    eval_dataset = build_sft_dataset(
        args.dataset,
        args.eval_path,
        args.prompt_template,
        args.max_eval_examples,
    )
    log_generations(
        vllm_model=LLM(model=str(checkpoint_path)),
        examples=list(eval_dataset),
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
