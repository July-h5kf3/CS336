import os
import regex as re
from typing import IO, Any, BinaryIO
from collections import Counter,defaultdict

def split_keep_special(text: str, special_tokens: list[str]) -> list[str]:
    #这里需要补充的一点在于，这里之所以对special token进行保留，是因为后续在tokenizer中我们可以直接复用这个函数
    if not special_tokens:
        return [text]
    #由于会出现special token串联的情况，因此需要先将special token按照长度降序排列
    #比如:Special token = ["<|endoftext|>", "<|endoftext|><|endoftext|>"].我们需要先匹配后者，才能保证正确split
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))
    #需要避免special token中有|等正则符号
    pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p != ""]

def _pretokenize_range(args: tuple[str | os.PathLike, int, int]) -> dict[bytes, int]:
    #首先将chunk按照special token进行split,避免出现跨doc的合并问题
    input_path, start, end = args
    token_counts = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        to_read = end - start
        data = f.read(to_read)
        parts = split_keep_special(data.decode("utf-8", errors="ignore"), ["<|endoftext|>"])
    for part in parts:
        if part == "<|endoftext|>":
            # token_counts[part.encode("utf-8")] += 1
            continue
        for m in re.finditer(PAT, part):
            token_counts[m.group(0).encode("utf-8")] += 1
    return token_counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    #调整chunk的分割点，保证不会出现doc1和doc2在边界合并的情况
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))