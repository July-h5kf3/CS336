import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class SFTExampleDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.examples[index]


def stringify_answer(answer) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def load_sft_reason_examples(
    data_path: str | Path,
    prompt_template_path: str | Path,
    limit: int | None = None,
) -> list[dict[str, str]]:
    prompt_template = Path(prompt_template_path).read_text()
    examples = []
    rows = json.loads(Path(data_path).read_text())
    for row in rows[:limit]:
        example = {
            "prompt": prompt_template.format(question=row["problem"]),
            "ground_truth": stringify_answer(row["expected_answer"]),
        }
        if "reasoning_trace" in row:
            example["output"] = row["reasoning_trace"]
        examples.append(example)
    return examples


def format_gsm8k_target(answer: str) -> str:
    reasoning, final_answer = answer.rsplit("####", 1)
    return f"{reasoning.strip()}\n</think> <answer>{final_answer.strip()}</answer>"


def load_gsm8k_examples(
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
            row = json.loads(line)
            examples.append(
                {
                    "prompt": prompt_template.format(question=row["question"]),
                    "output": format_gsm8k_target(row["answer"]),
                    "ground_truth": row["answer"].rsplit("####", 1)[1].strip(),
                }
            )
    return examples


DATASET_LOADERS = {
    "sft_reason": load_sft_reason_examples,
    "gsm8k": load_gsm8k_examples,
}


def build_sft_dataset(
    dataset_name: str,
    data_path: str | Path,
    prompt_template_path: str | Path,
    limit: int | None = None,
) -> SFTExampleDataset:
    examples = DATASET_LOADERS[dataset_name](data_path, prompt_template_path, limit)
    return SFTExampleDataset(examples)


def collate_sft_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    return examples


def build_sft_dataloader(
    dataset: SFTExampleDataset,
    micro_batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_sft_examples,
    )
