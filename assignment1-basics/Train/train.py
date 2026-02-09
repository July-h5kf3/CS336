"""
完整的训练循环脚本（支持 WandB 日志记录）
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# 添加 cs336_basics 到路径
sys.path.insert(0, '/home/acd66/project/CS336/assignment1-basics')

from cs336_basics.Transformer import Transformer
from cs336_basics.Cross_entropy import cross_entropy
from cs336_basics.Adamw import Adamw
from cs336_basics.lr_scheduling import lr_cosine_schedule
from cs336_basics.gradient_clip import gradient_clip
from cs336_basics.data_utils import get_batch, save_checkpoint, load_checkpoint

# 尝试导入 wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train(config_path: str, resume_path: str = None):
    """
    主训练函数
    
    Args:
        config_path: 配置文件路径
        resume_path: 恢复训练的 checkpoint 路径（可选）
    """
    # 加载配置
    config = load_config(config_path)
    model_config = config['model']
    train_config = config['training']
    preprocess_config = config['preprocess']
    wandb_config = config.get('wandb', {})
    
    # WandB 初始化
    use_wandb = WANDB_AVAILABLE and wandb_config.get('enabled', False)
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'cs336-transformer'),
            name=wandb_config.get('run_name', None),
            config={
                'model': model_config,
                'training': train_config,
            }
        )
        print("WandB logging enabled.")
    else:
        print("WandB logging disabled.")
    
    # 设备设置（从配置文件读取，默认自动检测）
    device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据（memory-mapped）
    data_path = preprocess_config['output']
    print(f"Loading data from {data_path}...")
    train_data = np.load(data_path, mmap_mode='r')
    print(f"Data shape: {train_data.shape}, dtype: {train_data.dtype}")
    
    # 创建模型
    print("Initializing model...")
    model = Transformer(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        num_layers=model_config['num_layers'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        device=device
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # 创建优化器
    optimizer = Adamw(
        model.parameters(),
        lr=float(train_config['max_lr']),
        weight_decay=float(train_config['weight_decay'])
    )
    
    # 恢复训练（可选）
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        start_step = load_checkpoint(resume_path, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    # 创建 checkpoint 目录
    checkpoint_dir = train_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练参数（添加类型转换确保正确）
    batch_size = int(train_config['batch_size'])
    context_length = int(model_config['context_length'])
    total_steps = int(train_config['total_steps'])
    max_lr = float(train_config['max_lr'])
    min_lr = float(train_config['min_lr'])
    warmup_steps = int(train_config['warmup_steps'])
    max_grad_norm = float(train_config['max_grad_norm'])
    log_interval = int(train_config['log_interval'])
    save_interval = int(train_config['save_interval'])
    wandb_log_interval = int(wandb_config.get('log_interval', log_interval))
    
    print(f"\n=== Training Config ===")
    print(f"Batch size: {batch_size}")
    print(f"Context length: {context_length}")
    print(f"Total steps: {total_steps}")
    print(f"LR: {min_lr} -> {max_lr} (warmup: {warmup_steps})")
    print(f"Starting from step: {start_step}")
    print()
    
    # 训练循环
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(range(start_step, total_steps), desc="Training", initial=start_step, total=total_steps)
    for step in pbar:
        # 1. 采样 batch
        xb, yb = get_batch(train_data, batch_size, context_length, device)
        
        # 2. 前向传播
        logits = model.forward(xb)  # (batch, seq_len, vocab_size)
        
        # 3. 计算损失（展平）
        logits_flat = logits.view(-1, model_config['vocab_size'])
        yb_flat = yb.view(-1)
        loss = cross_entropy(logits_flat, yb_flat)
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 5. 梯度裁剪
        gradient_clip(model.parameters(), max_grad_norm)
        
        # 6. 更新学习率
        lr = lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 7. 更新参数
        optimizer.step()
        
        # 记录 loss
        running_loss += loss.item()
        
        # 8. 更新进度条和日志
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
            running_loss = 0.0
        
        # 9. WandB 日志
        if use_wandb and (step + 1) % wandb_log_interval == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': lr,
                'train/step': step + 1,
            }, step=step + 1)
        
        # 10. 保存 checkpoint
        if (step + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'ckpt_step_{step + 1}.pt')
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            tqdm.write(f"Saved checkpoint to {ckpt_path}")
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, 'ckpt_final.pt')
    save_checkpoint(model, optimizer, total_steps, final_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    
    # 关闭 wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args.config, args.resume)
