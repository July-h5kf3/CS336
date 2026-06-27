import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,PreTrainedModel

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

    all_log_probs = F.log_softmax(logits,dim = -1)
    token_log_probs = torch.gather(
        all_log_probs,
        dim = -1,
        index = labels.unsqueeze(-1),
    ).squeeze(-1)

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
        normalize_constant= normalize_constant
        dim = -1,
    ).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss,{}


    
