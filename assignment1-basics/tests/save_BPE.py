import multiprocessing as mp
import regex as re
import os
import json
import pickle
from pathlib import Path
from typing import BinaryIO
from collections import Counter
from adapters import run_train_bpe
from common import gpt2_bytes_to_unicode

def main():
    filepath = "/root/project/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # filepath = "/root/project/CS336/assignment1-basics/data/owt_train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(filepath, vocab_size, special_tokens)
    print(f"The longest tokens in the vocabulary are:{sorted(vocab.values(), key=len, reverse=True)[:10]}")
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(filepath).stem
    vocab_path = output_dir / f"{dataset_name}_vocab.pkl"
    merges_path = output_dir / f"{dataset_name}_merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()
