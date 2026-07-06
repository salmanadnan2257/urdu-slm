"""Shared helpers: streaming JSONL I/O, sharded writing, resume markers, stats."""

import json
import os
from pathlib import Path


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class ShardWriter:
    """Write JSONL records to fixed-size shards: prefix-00000.jsonl, ..."""

    def __init__(self, out_dir, prefix, docs_per_shard=20000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.docs_per_shard = docs_per_shard
        self.shard_idx = 0
        self.count_in_shard = 0
        self.total = 0
        self._fh = None

    def _open(self):
        p = self.out_dir / f"{self.prefix}-{self.shard_idx:05d}.jsonl"
        self._fh = open(p, "w", encoding="utf-8")

    def write(self, record):
        if self._fh is None:
            self._open()
        if self.count_in_shard >= self.docs_per_shard:
            self._fh.close()
            self.shard_idx += 1
            self.count_in_shard = 0
            self._open()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count_in_shard += 1
        self.total += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Stage:
    """A resumable pipeline stage. Writes a `.done` marker with its stats so a
    re-run skips completed stages unless --force is given."""

    def __init__(self, name, out_dir):
        self.name = name
        self.out_dir = Path(out_dir)
        self.marker = self.out_dir / f".{name}.done"

    def is_done(self):
        return self.marker.exists()

    def mark_done(self, stats):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.marker, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def load_stats(self):
        if self.marker.exists():
            with open(self.marker, encoding="utf-8") as f:
                return json.load(f)
        return {}


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000
    return f"{n:.1f}T"
