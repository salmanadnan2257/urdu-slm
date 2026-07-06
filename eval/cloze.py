"""Few-shot cloze probe harness.

Each item has a `stem` (context up to a blank) and a list of `options`; the
model fills each option in and we score the total log-probability of the
completion tokens. The lowest-loss option is the model's answer. Accuracy is
reported over the hand-written eval set in evaldata/cloze.jsonl.

This harness is built for the phase-2 base model. On the tiny proof-run
checkpoint accuracy will be near chance; that is expected and reported honestly.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from eval._shared import load_model
from tokenizers import Tokenizer


def option_loss(model, tok, device, context, option, max_len):
    ctx_ids = tok.encode(context).ids
    opt_ids = tok.encode(option).ids
    ids = (ctx_ids + opt_ids)[-max_len:]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    # only score the option tokens
    mask_start = len(ids) - 1 - len(opt_ids)
    y = y.clone()
    y[0, :max(mask_start, 0)] = -100
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny/model.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--data", default="evaldata/cloze.jsonl")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples prepended")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)

    items = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]

    # build a few-shot prefix from the first `shots` items (kept out of scoring set)
    shot_items = items[: args.shots]
    test_items = items[args.shots:] if args.shots else items
    prefix = ""
    for s in shot_items:
        prefix += s["stem"] + s["options"][s["answer_idx"]] + "\n"

    correct = 0
    for it in test_items:
        losses = [option_loss(model, tok, device, prefix + it["stem"], opt, cfg.max_seq_len)
                  for opt in it["options"]]
        pred = int(min(range(len(losses)), key=lambda i: losses[i]))
        correct += int(pred == it["answer_idx"])

    n = len(test_items)
    acc = correct / n if n else 0.0
    print(f"cloze accuracy: {correct}/{n} = {acc:.3f} (shots={args.shots}, items scored={n})")


if __name__ == "__main__":
    main()
