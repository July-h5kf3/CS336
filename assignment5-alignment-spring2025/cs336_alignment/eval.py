from vllm import LLM,SamplingParams
from collections import Counter
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
from typing import Callable
from pathlib import Path

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    gt: list[str],
    eval_sampling_params: SamplingParams,
):
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    output_path = Path("results/gsm8k_val.jsonl")
    type_counts = Counter()
    records = []
    for prompt, ground_truth, output in zip(prompts,gt,outputs):
        response = output.outputs[0].text
        reward_info = reward_fn(response, ground_truth)
        if reward_info["reward"] == 1.0:
            result_type = "type1"
        elif reward_info["format_reward"] == 1.0:
            result_type = "type2"
        else:
            result_type = "type3"
        type_counts[result_type] += 1
        record = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "response": response,
            "reward": reward_info["reward"],
            "format_reward": reward_info["format_reward"],
            "answer_reward": reward_info["answer_reward"],
        }
        records.append(record)
    result = {
        "summary": dict(type_counts),
        "records": records,
    }
    with output_path.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
def main():
    examples = []
    with open("data/gsm8k/test.jsonl") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    prompts = []
    gt = []
    prompt_template = Path("cs336_alignment/prompts/r1_zero.prompt").read_text()
    for ex in examples:
        problem = ex["problem"]
        answer = ex["answer"].split("####")[-1].strip()
        prompt = prompt_template.format(question=problem)
        prompts.append(prompt)
        gt.append(answer)
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p = 1.0,
        max_tokens = 1024,
        stop = ["</answer>"],
        include_stop_str_in_output = True
    )
    evaluate_vllm(
        vllm_model=LLM(model=""),
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gt=gt,
        eval_sampling_params=sampling_params
    )

if __name__ == "__main__":
    main()