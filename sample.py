"""Generate sample continuations from a trained checkpoint."""

import argparse

import torch
from tokenizers import Tokenizer

from model.config import ModelConfig
from model.transformer import UrduLM


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ck["config"])
    model = UrduLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny/model.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)

    prompts = args.prompt or ["پاکستان ایک", "اردو زبان", "آج موسم"]
    for p in prompts:
        ids = tok.encode(p).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        text = tok.decode(out[0].tolist())
        print("PROMPT:", p)
        print("OUTPUT:", text)
        print("-" * 60)


if __name__ == "__main__":
    main()
