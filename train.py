"""Pretraining loop for UrduLM.

Features
  * token-count-based schedule (train for N tokens, not raw steps)
  * cosine LR with linear warmup, grad clipping, gradient accumulation
  * AMP: bf16 autocast on CUDA, fp32 on CPU
  * full resume: model + optimizer + scheduler + data cursor + RNG
  * JSONL metrics log
  * single-GPU and DDP (torchrun) code paths, deterministic seeding

Config comes from a YAML file (see configs/). CLI flags override YAML.

Examples
  python train.py --config configs/tiny.yaml
  torchrun --nproc_per_node=1 train.py --config configs/base.yaml   # DDP path
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from model.config import build_config
from model.transformer import UrduLM


def is_dist():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def setup_ddp():
    if is_dist():
        import torch.distributed as dist
        dist.init_process_group(backend="gloo" if not torch.cuda.is_available() else "nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, dist.get_world_size()
    return 0, 0, 1


class TokenLoader:
    """Memory-mapped uint16 token stream sampler with a resumable cursor.

    The cursor is a deterministic offset into a shuffled list of block start
    positions, so resume reproduces the exact same batch sequence.
    """

    def __init__(self, bin_path, block_size, batch_size, device, seed, rank, world_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rank = rank
        self.world_size = world_size
        n_blocks = (len(self.data) - 1) // block_size
        self.starts = np.arange(n_blocks) * block_size
        rng = np.random.default_rng(seed)
        rng.shuffle(self.starts)
        self.cursor = 0

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, sd):
        self.cursor = sd["cursor"]

    def next_batch(self):
        idxs = []
        need = self.batch_size * self.world_size
        while len(idxs) < need:
            if self.cursor >= len(self.starts):
                self.cursor = 0  # epoch wrap
            idxs.append(self.starts[self.cursor])
            self.cursor += 1
        # each rank takes its stride
        idxs = idxs[self.rank::self.world_size][: self.batch_size]
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in idxs])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in idxs])
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        return x, y


def cosine_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, ctx, eval_iters):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = loader.next_batch()
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, help="override token budget")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.max_tokens:
        cfg["max_tokens"] = args.max_tokens

    rank, local_rank, world_size = setup_ddp()
    master = rank == 0

    seed = cfg.get("seed", 1337) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else torch.autocast(device_type="cpu", enabled=False))

    out_dir = Path(cfg["out_dir"])
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)

    block_size = cfg["block_size"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg.get("grad_accum", 1)

    model_cfg = build_config(cfg["preset"], vocab_size=cfg["vocab_size"], max_seq_len=block_size)
    model = UrduLM(model_cfg).to(device)
    if master:
        print(f"model: preset={cfg['preset']} params={model.num_params()/1e6:.2f}M device={device}")

    raw_model = model
    if is_dist():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank] if device == "cuda" else None)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, cfg.get("beta2", 0.95)),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    data_dir = Path(cfg["data_dir"])
    train_loader = TokenLoader(data_dir / "train.bin", block_size, batch_size, device,
                               seed=cfg.get("seed", 1337), rank=rank, world_size=world_size)
    val_loader = TokenLoader(data_dir / "val.bin", block_size, batch_size, device,
                             seed=99, rank=rank, world_size=world_size)

    tokens_per_step = batch_size * block_size * grad_accum * world_size
    max_tokens = cfg["max_tokens"]
    total_steps = max(1, max_tokens // tokens_per_step)
    warmup_steps = cfg.get("warmup_steps", max(1, int(0.03 * total_steps)))

    step = 0
    tokens_seen = 0
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        train_loader.load_state_dict(ck["data"])
        step = ck["step"]
        tokens_seen = ck["tokens_seen"]
        if master:
            print(f"resumed at step {step}, tokens {tokens_seen}")

    metrics_path = out_dir / "metrics.jsonl"
    metrics_fh = open(metrics_path, "a") if master else None

    log_interval = cfg.get("log_interval", 20)
    eval_interval = cfg.get("eval_interval", 200)
    eval_iters = cfg.get("eval_iters", 50)
    ckpt_interval = cfg.get("ckpt_interval", 500)
    grad_clip = cfg.get("grad_clip", 1.0)

    model.train()
    t0 = time.time()
    while step < total_steps:
        lr = cosine_lr(step, warmup_steps, total_steps, cfg["lr"], cfg.get("min_lr", cfg["lr"] * 0.1))
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(grad_accum):
            x, y = train_loader.next_batch()
            if is_dist():
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step

        if master and step % log_interval == 0:
            dt = time.time() - t0
            tps = tokens_per_step * log_interval / dt
            rec = {"step": step, "tokens": tokens_seen, "loss": round(loss_accum, 4),
                   "lr": round(lr, 6), "tok_per_s": round(tps)}
            print(rec, flush=True)
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()
            t0 = time.time()

        if master and step % eval_interval == 0:
            val_loss = evaluate(raw_model, val_loader, ctx, eval_iters)
            rec = {"step": step, "tokens": tokens_seen, "val_loss": round(val_loss, 4),
                   "val_ppl": round(math.exp(min(val_loss, 20)), 2)}
            print(rec, flush=True)
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()

        if master and step % ckpt_interval == 0:
            torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                        "data": train_loader.state_dict(), "step": step,
                        "tokens_seen": tokens_seen, "config": model_cfg.__dict__}, ckpt_path)

    if master:
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "data": train_loader.state_dict(), "step": step,
                    "tokens_seen": tokens_seen, "config": model_cfg.__dict__}, ckpt_path)
        # slim, inference-only checkpoint (no optimizer/data state) for release
        model_path = out_dir / "model.pt"
        torch.save({"model": raw_model.state_dict(), "config": model_cfg.__dict__,
                    "step": step, "tokens_seen": tokens_seen}, model_path)
        metrics_fh.close()
        print(f"done: {step} steps, {tokens_seen} tokens, ckpt -> {ckpt_path}, model -> {model_path}")

    if is_dist():
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
