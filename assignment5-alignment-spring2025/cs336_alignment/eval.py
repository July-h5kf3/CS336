import sys
from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.SFT_utils import log_generations
from cs336_alignment.dataset import build_sft_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.train_utils import load_config


def main(
    config_path: str | Path = "configs/sft_gsm8k.yaml",
    model_path: str | Path | None = None,
) -> None:
    args = load_config(config_path)
    eval_dataset = build_sft_dataset(
        args.dataset,
        args.eval_path,
        args.prompt_template,
        args.max_eval_examples,
    )
    sampling_params = SamplingParams(
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        max_tokens=args.eval_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    log_generations(
        vllm_model=LLM(model=str(model_path or args.model)),
        examples=list(eval_dataset),
        reward_fn=r1_zero_reward_fn,
        sampling_params=sampling_params,
        output_path=args.eval_output_path,
    )


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "configs/sft_gsm8k.yaml",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
