"""End-to-end corpus builder. One command, resumable stages, streaming to disk.

Stages
  1. extract   : read each raw source, normalize, filter, write per-source shards
  2. dedup     : exact + MinHash near-dedup across all sources -> clean shards
  3. split     : deterministic train/val split -> data/train, data/val

Run:
  python -m pipeline.build_corpus --raw-dir <scratch/raw> --out-dir data \
      --wiki urwiki-latest-pages-articles.xml.bz2 \
      --leipzig urd_newscrawl_2016_1M.tar.gz:leipzig_newscrawl_2016 \
      --leipzig urd_newscrawl_2011_300K.tar.gz:leipzig_newscrawl_2011

Each stage writes a `.stage.done` marker with its stats; re-running skips
finished stages unless --force is passed.
"""

import argparse
import json
import random
from pathlib import Path

from .common import ShardWriter, Stage, iter_jsonl, human
from .dedup import Deduper
from .normalize import normalize, keep_line
from .sources import read_wiki_dump, read_leipzig_tar, read_plain_text


def _paragraphs(text):
    for para in text.split("\n"):
        para = para.strip()
        if para:
            yield para


def stage_extract(args, work):
    stage = Stage("extract", work)
    if stage.is_done() and not args.force:
        print(f"[extract] skip (done): {stage.load_stats()}")
        return stage.load_stats()

    per_source = {}
    extract_dir = work / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    def run_source(name, records):
        kept = 0
        raw_docs = 0
        chars = 0
        with ShardWriter(extract_dir, name) as sw:
            for rec in records:
                raw_docs += 1
                for para in _paragraphs(normalize(rec["text"])):
                    if keep_line(para):
                        out = {"text": para, "source": rec["source"]}
                        sw.write(out)
                        kept += 1
                        chars += len(para)
                if args.limit_docs and raw_docs >= args.limit_docs:
                    break
        per_source[name] = {"raw_docs": raw_docs, "kept_paragraphs": kept, "chars": chars}
        print(f"[extract] {name}: raw={human(raw_docs)} kept={human(kept)} chars={human(chars)}")

    if args.wiki:
        run_source("wikipedia", read_wiki_dump(Path(args.raw_dir) / args.wiki,
                                               max_docs=args.limit_docs or None))
    for spec in args.leipzig:
        fname, name = spec.split(":", 1)
        run_source(name, read_leipzig_tar(Path(args.raw_dir) / fname, name,
                                          max_docs=args.limit_docs or None))
    for spec in args.plain:
        fname, name = spec.split(":", 1)
        run_source(name, read_plain_text(Path(args.raw_dir) / fname, name,
                                         max_docs=args.limit_docs or None))

    stats = {"per_source": per_source}
    stage.mark_done(stats)
    return stats


def stage_dedup(args, work):
    stage = Stage("dedup", work)
    if stage.is_done() and not args.force:
        print(f"[dedup] skip (done): {stage.load_stats()}")
        return stage.load_stats()

    extract_dir = work / "extract"
    clean_dir = work / "clean"
    shards = sorted(extract_dir.glob("*.jsonl"))
    deduper = Deduper(threshold=args.near_threshold)
    kept_chars = 0
    with ShardWriter(clean_dir, "clean", docs_per_shard=50000) as sw:
        for shard in shards:
            for rec in iter_jsonl(shard):
                if deduper.keep(rec["text"]):
                    sw.write(rec)
                    kept_chars += len(rec["text"])
    stats = deduper.stats()
    stats["kept_chars"] = kept_chars
    print(f"[dedup] {stats}")
    stage.mark_done(stats)
    return stats


def stage_split(args, work, out_dir):
    stage = Stage("split", work)
    if stage.is_done() and not args.force:
        print(f"[split] skip (done): {stage.load_stats()}")
        return stage.load_stats()

    clean_dir = work / "clean"
    rng = random.Random(args.seed)
    train_w = ShardWriter(out_dir / "train", "train", docs_per_shard=100000)
    val_w = ShardWriter(out_dir / "val", "val", docs_per_shard=100000)
    n_train = n_val = 0
    src_counts = {}
    for shard in sorted(clean_dir.glob("*.jsonl")):
        for rec in iter_jsonl(shard):
            src_counts[rec["source"]] = src_counts.get(rec["source"], 0) + 1
            if rng.random() < args.val_frac:
                val_w.write(rec)
                n_val += 1
            else:
                train_w.write(rec)
                n_train += 1
    train_w.close()
    val_w.close()
    stats = {"train_docs": n_train, "val_docs": n_val, "per_source": src_counts}
    print(f"[split] train={human(n_train)} val={human(n_val)} sources={src_counts}")
    stage.mark_done(stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build the cleaned Urdu corpus.")
    ap.add_argument("--raw-dir", required=True, help="directory holding downloaded raw files")
    ap.add_argument("--out-dir", default="data", help="project data/ output dir")
    ap.add_argument("--work-dir", default=None, help="intermediate dir (default: <raw-dir>/work)")
    ap.add_argument("--wiki", default=None, help="wiki dump filename in raw-dir")
    ap.add_argument("--leipzig", action="append", default=[],
                    help="leipzig spec 'file.tar.gz:source_name' (repeatable)")
    ap.add_argument("--plain", action="append", default=[],
                    help="plain-text spec 'file[.xz|.gz]:source_name' (repeatable)")
    ap.add_argument("--near-threshold", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--limit-docs", type=int, default=0, help="cap raw docs per source (0=all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--force", action="store_true", help="rerun all stages")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    work = Path(args.work_dir) if args.work_dir else Path(args.raw_dir) / "work"
    work.mkdir(parents=True, exist_ok=True)

    stats = {
        "extract": stage_extract(args, work),
        "dedup": stage_dedup(args, work),
        "split": stage_split(args, work, out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n[done] wrote", out_dir / "corpus_stats.json")


if __name__ == "__main__":
    main()
