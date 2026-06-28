import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
import yaml
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str | Path) -> SimpleNamespace:
    with Path(config_path).open() as f:
        config = yaml.safe_load(f) or {}
    return SimpleNamespace(**config)


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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
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
