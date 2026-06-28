import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from pathlib import Path
from typing import Any, Callable
import json
from tqdm import tqdm

def tokenizer_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
):
    prompt_token_ids = tokenizer(prompt_strs, add_special_tokens = False)["input_ids"]
    output_token_ids = tokenizer(output_strs, add_special_tokens = False)["input_ids"]

    pad_token_id = tokenizer.pad_token_id

    full_seqs = []
    full_seq_masks = []
    for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids):
        sequence = prompt_ids + output_ids
        mask = (
            [False] * len(prompt_ids) + [True] * len(output_ids)
        )
        full_seqs.append(sequence)
        full_seq_masks.append(mask)
    max_length = max(len(sequence) for sequence in full_seqs)
    padded_seqs = []
    padded_mask = []
    for seq,mask in zip(full_seqs, full_seq_masks):
        padding_length = max_length - len(seq)
        padded_seqs.append(
            seq + [pad_token_id] * padding_length
        )
        padded_mask.append(
            mask + [False] * padding_length
        )
    token_ids = torch.tensor(padded_seqs, dtype = torch.long)
    full_seq_masks = torch.tensor(padded_mask, dtype = torch.bool)

    input_ids = token_ids[:, :-1]
    labels = token_ids[:,1:]

    response_mask = full_seq_masks[:,1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
def compute_entropy(
    logits: torch.Tensor,
):
    #logits: [bs,s,dim]
    log_probs = F.log_softmax(logits,dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
):
    logits = model(input_ids).logits

    token_logits = torch.gather(
        logits,
        dim = -1,
        index = labels.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = token_logits - torch.logsumexp(logits, dim=-1)

    result = {"log_probs": token_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
):
    masked_sum = tensor.masked_fill(~mask,0).sum(dim = dim)
    return masked_sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps:int,
    normalize_constant: float = 1.0,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(
        tensor= policy_log_probs,
        mask = response_mask,
        normalize_constant= normalize_constant,
        dim = -1,
    ).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss,{}

def log_generations(
    vllm_model,
    examples: list[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params,
    output_path: str | Path | None = None,
) -> None:
    prompts = [example["prompt"] for example in examples]
    ground_truths = [example["ground_truth"] for example in examples]
    outputs = vllm_model.generate(prompts, sampling_params)
    generations = []
    
    for output in outputs:
        completion = output.outputs[0]
        response = completion.text
        token_ids = getattr(completion, "token_ids", None)
        response_length = len(token_ids) if token_ids is not None else len(response)
        cumulative_logprob = getattr(completion, "cumulative_logprob", None)
        mean_response_logprob = (
            cumulative_logprob / response_length
            if cumulative_logprob is not None and response_length > 0
            else None
        )
        generations.append(
            {
                "response": response,
                "response_length": response_length,
                "cumulative_logprob": cumulative_logprob,
                "mean_response_logprob": mean_response_logprob,
            }
        )

    records = []
    correct_lengths = []
    incorrect_lengths = []
    rewards = []
    format_rewards = []
    answer_rewards = []
    mean_response_logprobs = []

    for prompt, generation, ground_truth in tqdm(
        zip(prompts, generations, ground_truths),
        desc="Logging generations",
        total=len(prompts),
    ):
        response = generation["response"]
        response_length = generation["response_length"]
        mean_response_logprob = generation["mean_response_logprob"]
        if mean_response_logprob is not None:
            mean_response_logprobs.append(mean_response_logprob)
        reward_info = reward_fn(response, ground_truth)
        is_correct = reward_info["reward"] == 1.0
        if is_correct:
            correct_lengths.append(response_length)
        else:
            incorrect_lengths.append(response_length)
        rewards.append(reward_info["reward"])
        format_rewards.append(reward_info["format_reward"])
        answer_rewards.append(reward_info["answer_reward"])

        records.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": reward_info["reward"],
                "format_reward": reward_info["format_reward"],
                "answer_reward": reward_info["answer_reward"],
                "response_length": response_length,
                "cumulative_logprob": generation["cumulative_logprob"],
                "mean_response_logprob": mean_response_logprob,
            }
        )

    result = {
        "summary": {
            "mean_reward": sum(rewards) / len(rewards),
            "mean_format_reward": sum(format_rewards) / len(format_rewards),
            "mean_answer_reward": sum(answer_rewards) / len(answer_rewards),
            "mean_response_logprob": (
                sum(mean_response_logprobs) / len(mean_response_logprobs)
                if mean_response_logprobs
                else None
            ),
            "mean_response_length": sum(
                record["response_length"] for record in records
            ) / len(records),
            "mean_correct_response_length": (
                sum(correct_lengths) / len(correct_lengths)
                if correct_lengths
                else 0.0
            ),
            "mean_incorrect_response_length": (
                sum(incorrect_lengths) / len(incorrect_lengths)
                if incorrect_lengths
                else 0.0
            ),
        },
        "records": records,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


    
