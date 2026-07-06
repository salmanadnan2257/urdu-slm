"""Train a byte-level BPE tokenizer on the cleaned corpus (HF tokenizers).

32k vocabulary by default. Byte-level BPE guarantees no out-of-vocabulary
symbols on any UTF-8 input, which matters for a script the base tokenizers of
English models handle poorly.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

from .common import iter_jsonl

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


def corpus_lines(shard_dirs):
    for d in shard_dirs:
        for shard in sorted(Path(d).glob("*.jsonl")):
            for rec in iter_jsonl(shard):
                yield rec["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="dir with train/ and val/ subdirs")
    ap.add_argument("--out-dir", default="tokenizer")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    data = Path(args.data_dir)
    shard_dirs = [data / "train"]

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(corpus_lines(shard_dirs), trainer=trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    print(f"saved tokenizer ({tok.get_vocab_size()} tokens) to {out/'tokenizer.json'}")


if __name__ == "__main__":
    main()
