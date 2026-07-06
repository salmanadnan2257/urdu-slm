"""Plot the training/val loss curve from a metrics.jsonl log."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/tiny/metrics.jsonl")
    ap.add_argument("--out", default="docs/tiny_run_loss.png")
    ap.add_argument("--title", default="UrduLM tiny (13M) CPU proof run")
    args = ap.parse_args()

    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    for line in Path(args.metrics).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "loss" in r:
            train_steps.append(r["step"])
            train_loss.append(r["loss"])
        if "val_loss" in r:
            val_steps.append(r["step"])
            val_loss.append(r["val_loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_steps, train_loss, label="train loss", color="#2b6cb0", linewidth=1.2)
    if val_loss:
        ax.plot(val_steps, val_loss, label="val loss", color="#c05621",
                marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}  (train points={len(train_loss)}, val points={len(val_loss)})")
    if train_loss:
        print(f"first train loss={train_loss[0]:.3f} last={train_loss[-1]:.3f}")
    if val_loss:
        print(f"first val loss={val_loss[0]:.3f} last={val_loss[-1]:.3f}")


if __name__ == "__main__":
    main()
