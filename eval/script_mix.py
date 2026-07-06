"""Script-mix diagnostic: share of generated characters in Arabic vs Latin script.

A healthy Urdu model should generate almost entirely Arabic-script characters.
A high Latin share signals the model is leaking English or degenerating.
"""

import argparse

import torch

from eval._shared import load_model
from pipeline.normalize import is_arabic_char
from tokenizers import Tokenizer


def script_shares(text):
    arabic = latin = other = 0
    for c in text:
        if not c.isalpha():
            continue
        if is_arabic_char(c):
            arabic += 1
        elif c.isascii():
            latin += 1
        else:
            other += 1
    total = arabic + latin + other
    if total == 0:
        return {"arabic": 0.0, "latin": 0.0, "other": 0.0}
    return {"arabic": arabic / total, "latin": latin / total, "other": other / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny/model.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)

    prompts = ["پاکستان", "اردو", "دنیا", "علم", "کتاب"]
    agg = {"arabic": 0.0, "latin": 0.0, "other": 0.0}
    n = 0
    for i in range(args.n_samples):
        p = prompts[i % len(prompts)]
        ids = tok.encode(p).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, args.max_new_tokens, temperature=0.9, top_k=40)
        text = tok.decode(out[0].tolist())
        s = script_shares(text)
        for k in agg:
            agg[k] += s[k]
        n += 1
    for k in agg:
        agg[k] = round(agg[k] / n, 4)
    print("mean script shares over", n, "samples:", agg)


if __name__ == "__main__":
    main()
