"""Held-out perplexity on a tokenized .bin split."""

import argparse
import math

import numpy as np
import torch

from model.config import ModelConfig
from model.transformer import UrduLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny/model.pt")
    ap.add_argument("--bin", default="data/val.bin")
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ck["config"])
    model = UrduLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    block = args.block_size or cfg.max_seq_len
    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    n_blocks = (len(data) - 1) // block
    starts = np.arange(min(n_blocks, args.max_batches * args.batch_size)) * block

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), args.batch_size):
            batch = starts[i : i + args.batch_size]
            x = torch.from_numpy(np.stack([data[s : s + block].astype(np.int64) for s in batch])).to(device)
            y = torch.from_numpy(np.stack([data[s + 1 : s + 1 + block].astype(np.int64) for s in batch])).to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg = total_loss / total_tokens
    print(f"tokens={total_tokens} loss={avg:.4f} perplexity={math.exp(avg):.2f}")


if __name__ == "__main__":
    main()
