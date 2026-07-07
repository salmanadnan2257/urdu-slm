"""Tokenize cleaned JSONL shards into flat uint16 .bin token streams.

Each document is encoded and terminated with the <|endoftext|> id, then all ids
are concatenated into data/train.bin and data/val.bin. uint16 is enough for a
32k vocabulary and halves the file size versus int32. Training memory-maps
these files, so the corpus never has to fit in RAM.
"""

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .common import iter_jsonl, human


def encode_split(tok, shard_dir, out_path, eot_id, upsample=None, only=None):
    shards = sorted(Path(shard_dir).glob("*.jsonl"))
    total = 0
    with open(out_path, "wb") as fh:
        for shard in shards:
            buf = []
            for rec in iter_jsonl(shard):
                src = rec.get("source")
                if only and src not in only:
                    continue
                ids = tok.encode(rec["text"]).ids
                ids.append(eot_id)
                reps = upsample.get(src, 1) if upsample else 1
                buf.extend(ids * reps)
            arr = np.array(buf, dtype=np.uint16)
            arr.tofile(fh)
            total += len(arr)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=None, help="where to write .bin files (default: data-dir)")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--upsample", action="append", default=[],
                    help="repeat a source's docs in train.bin, spec 'source:k' (repeatable); "
                         "val.bin is never upsampled")
    ap.add_argument("--filter-source", default=None,
                    help="comma list of sources to keep in train.bin (default: all); "
                         "val.bin is never filtered")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id("<|endoftext|>")
    data = Path(args.data_dir)
    out = Path(args.out_dir) if args.out_dir else data
    out.mkdir(parents=True, exist_ok=True)

    upsample = {}
    for spec in args.upsample:
        src, k = spec.rsplit(":", 1)
        upsample[src] = int(k)
    only = set(args.filter_source.split(",")) if args.filter_source else None

    for split in ["train", "val"]:
        train = split == "train"
        n = encode_split(tok, data / split, out / f"{split}.bin", eot_id,
                         upsample=upsample if train else None,
                         only=only if train else None)
        print(f"{split}: {human(n)} tokens -> {out/f'{split}.bin'}")


if __name__ == "__main__":
    main()
