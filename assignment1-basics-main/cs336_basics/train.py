import argparse
import math
import time

import numpy as np
import torch
import os
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from tests.adapters import (
    get_adamw_cls,
    run_get_lr_cosine_schedule,
    run_get_batch,
    run_gradient_clipping,
    run_save_checkpoint,
    run_load_checkpoint,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TinyStories Transformer LM")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    # 训练超参(后面要 sweep 的)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # 数据 / 设备 / 日志
    parser.add_argument("--train-data", type=str, default="data/tinystories_train_tokens.npy")
    parser.add_argument("--valid-data", type=str, default="data/tinystories_valid_tokens.npy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = np.load(args.train_data, mmap_mode="r")
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    model.to(args.device)
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model = torch.compile(model, mode="reduce-overhead")
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, betas=(args.beta1, args.beta2), eps = args.eps)
    valid_data = np.load(args.valid_data, mmap_mode="r")
    assert train_data.dtype == np.uint16 and train_data.max() < args.vocab_size
    start_step = 0
    if args.resume:
        start_step = run_load_checkpoint(args.resume, model, optimizer)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for step in range(start_step, args.num_steps):
        lr = run_get_lr_cosine_schedule(step+1, args.lr, args.min_lr, args.warmup_iters, args.num_steps)
        optimizer.param_groups[0]["lr"] = lr
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        loss.backward()
        run_gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if(step + 1) % args.log_interval == 0:
            print(f"step {step+1}/{args.num_steps} | lr {lr:.2e} | loss{loss.item():.4f}")
        if (step + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(8):
                    x_val, y_val = run_get_batch(valid_data, args.batch_size, args.context_length, args.device)
                    losses.append(F.cross_entropy(model(x_val).view(-1, args.vocab_size),y_val.view(-1)))
                val_loss = torch.stack(losses).mean().item()
                model.train()
                print(f"val loss {val_loss:.4f} perplexity {math.exp(val_loss):.2f}")
        if (step + 1) % args.checkpoint_interval == 0:
            run_save_checkpoint(model, optimizer, step+1, os.path.join(args.checkpoint_dir, f"checkpoint-{step+1:07d}.pt"))

if __name__ == "__main__":
    main()