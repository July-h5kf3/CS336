"""
数据预处理脚本：将原始文本文件转换为 tokenized 的 numpy 文件
"""
import argparse
import os
import yaml
import numpy as np
from typing import Iterator
from cs336_basics.BPE_tokenizer import BPETokenizer


def read_file_in_chunks(filepath: str, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """流式读取大型文本文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def preprocess(input_path: str, output_path: str, vocab_path: str, merges_path: str, 
               special_tokens: list[str] = None):
    """
    将原始文本转换为 tokenized 的 numpy 文件
    
    Args:
        input_path: 输入文本文件路径
        output_path: 输出 numpy 文件路径
        vocab_path: vocab 序列化文件路径（.pkl）
        merges_path: merges 序列化文件路径（.pkl）
        special_tokens: 特殊 token 列表
    """
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    print(f"Tokenizing {input_path}...")
    # 使用 encode_iterable 进行流式编码
    token_ids = list(tokenizer.encode_iterable(read_file_in_chunks(input_path)))
    print(f"Total tokens: {len(token_ids)}")
    
    # 选择合适的 dtype
    vocab_size = len(tokenizer.vocab)
    if vocab_size <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32
    print(f"Using dtype: {dtype}")
    
    # 转换并保存
    data = np.array(token_ids, dtype=dtype)
    np.save(output_path, data)
    print(f"Saved to {output_path}")
    
    # 验证
    print("\n=== Verification ===")
    print(f"Shape: {data.shape}")
    print(f"Max token id: {data.max()}")
    print(f"Min token id: {data.min()}")
    
    if data.max() >= vocab_size:
        print(f"WARNING: Max token id ({data.max()}) >= vocab size ({vocab_size})!")
    else:
        print("Token ids look valid.")
    
    # 解码一小部分验证
    sample = data[:100].tolist()
    decoded = tokenizer.decode(sample)
    print(f"\nFirst 100 tokens decoded:\n{decoded[:200]}...")


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data for language model training")
    parser.add_argument("--config", default="cs336_basics/config.yaml", help="Path to config.yaml file")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    preprocess_config = config['preprocess']
    
    # 处理 special_tokens：可能是字符串或列表
    special_tokens = preprocess_config.get('special_tokens')
    if isinstance(special_tokens, str):
        special_tokens = [special_tokens]
    
    preprocess(
        input_path=preprocess_config['input'],
        output_path=preprocess_config['output'],
        vocab_path=preprocess_config['vocab'],
        merges_path=preprocess_config['merges'],
        special_tokens=special_tokens
    )

