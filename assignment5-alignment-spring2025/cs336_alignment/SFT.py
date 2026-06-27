from vllm import LLM,SamplingParams
from pathlib import Path
import json

from cs336_alignment.SFT_utils import log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def main():
    vllm_model = LLM(model = "")
    sampling_params = SamplingParams(
        temperature= 1.0,
        top_p = 1.0,
        max_tokens= 1024,
        stop= ["</answer>"],
        include_stop_str_in_output= True,
        logprobs=1,
    )
    examples = []
    prompt_template = Path("cs336_alignment/prompts/r1_zero.prompt").read_text()
    with open("data/gsm8k/test.jsonl") as f:
        for i,line in enumerate(f):
            if i >= 100:
                break
            if line.strip():
                line = json.loads(line)
                problem = line["question"]
                answer = line["answer"].split("####")[-1].strip()
                prompt = prompt_template.format(question=problem)
                examples.append({
                    "prompt":prompt,
                    "ground_truth": answer
                })
    log_generations(
        vllm_model= vllm_model,
        examples = examples,
        reward_fn = r1_zero_reward_fn,
        sampling_params= sampling_params,
        output_path="results/sft_gsm8k_test_100.json",
    )
        
if __name__ == "__main__":
    main()
