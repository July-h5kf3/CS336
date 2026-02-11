"""
Transformer 语言模型推理脚本
所有参数从 config.yaml 读取，命令行仅指定 --config 和 --prompt
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.Transformer import Transformer
from cs336_basics.BPE_tokenizer import BPETokenizer
from cs336_basics.data_utils import load_checkpoint


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def apply_top_p(logits, top_p):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_mask] = -float('inf')
    return torch.gather(sorted_logits, -1, torch.argsort(sorted_indices, dim=-1))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, device, context_length,
             temperature=0.8, top_p=1.0, eos_token=None):
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 获取 eos token id
    eos_id = None
    if eos_token is not None:
        try:
            eos_ids = tokenizer.encode(eos_token)
            if len(eos_ids) == 1:
                eos_id = eos_ids[0]
        except Exception:
            pass

    generated_ids = []
    for _ in range(max_new_tokens):
        if tokens.size(1) > context_length:
            tokens = tokens[:, -context_length:]

        logits = model(tokens)[:, -1, :]

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        # 仅使用 top-p 采样
        if top_p < 1.0:
            logits = apply_top_p(logits, top_p)

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

        tokens = torch.cat((tokens, next_token), dim=-1)
        next_id = next_token.item()
        generated_ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(prompt_ids + generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Transformer 语言模型推理")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--prompt", type=str, required=True, help="输入提示文本")
    args = parser.parse_args()

    # ─── 加载配置 ───
    config = load_config(args.config)
    model_config = config['model']
    preprocess_config = config.get('preprocess', {})
    train_config = config.get('training', {})
    infer_config = config.get('inference', {})

    device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📌 Device: {device}")

    # ─── Tokenizer ───
    special_tokens = preprocess_config.get('special_tokens')
    if isinstance(special_tokens, str):
        special_tokens = [special_tokens]
    tokenizer = BPETokenizer.from_files(
        preprocess_config['vocab'],
        preprocess_config['merges'],
        special_tokens
    )
    print(f"✓ Tokenizer (vocab={len(tokenizer.vocab)})")

    # ─── 模型 ───
    context_length = int(model_config['context_length'])
    model = Transformer(
        vocab_size=int(model_config['vocab_size']),
        context_length=context_length,
        num_layers=int(model_config['num_layers']),
        d_model=int(model_config['d_model']),
        num_heads=int(model_config['num_heads']),
        device=device,
    )
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ─── Checkpoint ───
    checkpoint_path = infer_config.get('checkpoint')
    if checkpoint_path is None:
        checkpoint_dir = train_config.get('checkpoint_dir', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_final.pt')

    if os.path.exists(checkpoint_path):
        # 使用 data_utils 中的 load_checkpoint
        try:
            step = load_checkpoint(checkpoint_path, model, optimizer=None)
            print(f"✓ 加载 checkpoint (step={step})")
        except Exception as e:
            print(f"⚠ 加载 checkpoint 失败: {e}")
            print("  将使用随机权重")
    else:
        print(f"⚠ Checkpoint 不存在: {checkpoint_path}（使用随机权重）")

    # ─── 推理参数（仅 top-p） ───
    max_new_tokens = int(infer_config.get('max_new_tokens', 200))
    temperature = float(infer_config.get('temperature', 0.8))
    top_p = float(infer_config.get('top_p', 0.95))
    eos_token = infer_config.get('eos_token', None)

    # ─── 生成 ───
    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature={temperature}  Top-p={top_p}  Max tokens={max_new_tokens}")
    print("─" * 60)

    generated_text = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        context_length=context_length,
        temperature=temperature,
        top_p=top_p,
        eos_token=eos_token,
    )

    print(generated_text)
    print("─" * 60)
    prompt_tokens = len(tokenizer.encode(args.prompt))
    total_tokens = len(tokenizer.encode(generated_text))
    print(f"Prompt: {prompt_tokens} tokens → 生成: {total_tokens - prompt_tokens} tokens")


if __name__ == "__main__":
    main()