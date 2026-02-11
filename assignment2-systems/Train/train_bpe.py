import multiprocessing as mp
import regex as re
import os
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from cs336_basics.BPE_utils import find_chunk_boundaries, _pretokenize_range
from tqdm import tqdm

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 词表的初始化，初始时词表应该只有从字节串 token 到整数 ID 的一一映射,以及规定的special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for token in special_tokens:
        b = token.encode("utf-8")
        if b not in vocab.values():
            vocab[len(vocab)] = b
    
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    total_token_counts = Counter()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(
            _pretokenize_range,
            [(input_path,s,e) for s,e in ranges],
        )
    for token_counts in results:
        total_token_counts.update(token_counts)
    word_freqs = total_token_counts
    word_symbols = {
        token: tuple(vocab[b] for b in token)
        for token in word_freqs.keys()
    }
    pair2token = defaultdict(set)
    pair_counts = Counter()
    for token,freq in word_freqs.items():
        symbols = word_symbols[token]
        if len(symbols) < 2:
            continue
        for a,b in zip(symbols, symbols[1:]):
            pair_counts[(a,b)] += freq
            pair2token[(a,b)].add(token)
            
    with tqdm(total=vocab_size, initial=len(vocab), desc="Training BPE") as pbar:
        while len(vocab) < vocab_size:
            if not pair_counts:
                break
            
            # 合并出现频次最高的pair,如果有多个pair出现频次相同，则选择字典序最大的那个
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            # 更新vocab和word_symbols
            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = merged_token
            pbar.update(1)

            tokens_to_update = pair2token[best_pair]

            for token in list(tokens_to_update):
                freq = word_freqs[token]
                symbols = word_symbols[token]
                if len(symbols) < 2:
                    continue
                new_symbols = []
                # 这里和原来不同了，我们需要减去旧的pair的计数
                i = 0
                while i < len(symbols) - 1:
                    p = (symbols[i], symbols[i+1])
                    pair_counts[p] -= freq
                    pair2token[p].discard(token)
                    i += 1
                # 然后进行合并
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                word_symbols[token] = tuple(new_symbols)
                for i in range(len(new_symbols) - 1):
                    p = (new_symbols[i], new_symbols[i+1])
                    pair_counts[p] += freq
                    pair2token[p].add(token)
    return vocab, merges


def main():
    # Updated path to the existing sample file (tinystories_sample_5M.txt)
    # The original path was .../data/TinyStoriesV2-GPT4-train.txt which did not exist
    filepath = "/home/acd66/project/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
    
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found at {filepath}")
        return

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Starting BPE training on {filepath}...")
    vocab, merges = run_train_bpe(filepath, vocab_size, special_tokens)
    
    print(f"The longest tokens in the vocabulary are: {sorted(vocab.values(), key=len, reverse=True)[:10]}")
    
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(filepath).stem
    # Save with specific names as expected by other scripts or general names
    vocab_path = output_dir / f"{dataset_name}_vocab.pkl"
    merges_path = output_dir / f"{dataset_name}_merges.pkl"

    print(f"Saving vocab to {vocab_path}")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving merges to {merges_path}")
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")

if __name__ == "__main__":
    main()
