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


def encode_split(tok, shard_dir, out_path, eot_id):
    shards = sorted(Path(shard_dir).glob("*.jsonl"))
    total = 0
    with open(out_path, "wb") as fh:
        for shard in shards:
            buf = []
            for rec in iter_jsonl(shard):
                ids = tok.encode(rec["text"]).ids
                ids.append(eot_id)
                buf.extend(ids)
            arr = np.array(buf, dtype=np.uint16)
            arr.tofile(fh)
            total += len(arr)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id("<|endoftext|>")
    data = Path(args.data_dir)

    for split in ["train", "val"]:
        n = encode_split(tok, data / split, data / f"{split}.bin", eot_id)
        print(f"{split}: {human(n)} tokens -> {data/f'{split}.bin'}")


if __name__ == "__main__":
    main()
