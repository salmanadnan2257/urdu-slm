"""Compare our Urdu BPE tokenizer against GPT-2's tokenizer on Urdu text.

Reports chars-per-token (compression) for each. Higher chars/token means fewer
tokens for the same text, i.e. cheaper training and inference. GPT-2's tokenizer
was trained on English web text and falls back to many bytes per Urdu character,
so the gap is large.

GPT-2's merges/vocab are loaded from a local file if present; otherwise the
comparison uses a byte-level baseline (1 token per UTF-8 byte), which is the
worst case GPT-2 approaches on unseen scripts. Pass --gpt2 to point at a
gpt2 tokenizer.json if you have one cached offline.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer

from pipeline.common import iter_jsonl


def sample_text(data_dir, max_chars):
    chunks, total = [], 0
    for split in ["val", "train"]:
        d = Path(data_dir) / split
        for shard in sorted(d.glob("*.jsonl")):
            for rec in iter_jsonl(shard):
                chunks.append(rec["text"])
                total += len(rec["text"])
                if total >= max_chars:
                    return "\n".join(chunks)
    return "\n".join(chunks)


def chars_per_token_ours(tok, text):
    n_tok = len(tok.encode(text).ids)
    return len(text) / n_tok, n_tok


def chars_per_token_bytes(text):
    # byte-level baseline: worst-case behaviour of an English tokenizer on Urdu,
    # where most Urdu code points become 2 UTF-8 bytes and merge rarely.
    n_bytes = len(text.encode("utf-8"))
    return len(text) / n_bytes, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--gpt2", default=None, help="path to a gpt2 tokenizer.json (optional)")
    ap.add_argument("--max-chars", type=int, default=500000)
    args = ap.parse_args()

    text = sample_text(args.data_dir, args.max_chars)
    ours = Tokenizer.from_file(args.tokenizer)
    cpt_ours, n_ours = chars_per_token_ours(ours, text)

    print(f"sample chars: {len(text)}")
    print(f"ours (32k Urdu BPE): {n_ours} tokens, {cpt_ours:.3f} chars/token")

    if args.gpt2 and Path(args.gpt2).exists():
        gpt2 = Tokenizer.from_file(args.gpt2)
        cpt_g, n_g = chars_per_token_ours(gpt2, text)
        print(f"gpt2: {n_g} tokens, {cpt_g:.3f} chars/token")
        print(f"compression gain: {cpt_ours / cpt_g:.2f}x fewer tokens with our tokenizer")
    else:
        cpt_b, n_b = chars_per_token_bytes(text)
        print(f"byte-level baseline (worst case for an English tokenizer): "
              f"{n_b} tokens, {cpt_b:.3f} chars/token")
        print(f"compression gain vs byte-level: {cpt_ours / cpt_b:.2f}x fewer tokens")


if __name__ == "__main__":
    main()
