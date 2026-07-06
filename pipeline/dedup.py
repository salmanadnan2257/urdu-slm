"""Exact and MinHash near-duplicate removal, streaming.

Exact dedup: 64-bit blake2b of the normalized text, held in a Python set.
Near dedup: MinHashLSH over character-5-gram shingles (datasketch). Documents
whose Jaccard estimate exceeds the threshold against an already-kept document
are dropped. Both passes stream shard by shard so peak memory is the hash/LSH
index, not the corpus.
"""

import hashlib

import numpy as np
import xxhash
from datasketch import MinHash, MinHashLSH

# Vectorized MinHash. datasketch's per-shingle update() loops in Python and runs
# a small numpy op per shingle, which measured at ~250 docs/s on this corpus
# (about two hours for the full run). We compute the whole signature with two
# numpy operations per document instead: hash every shingle to a 32-bit value,
# then take the min over shingles of the affine permutations. The resulting
# hashvalues are set directly on a datasketch MinHash so MinHashLSH banding is
# unchanged. Element hashing uses xxhash (deterministic across processes, unlike
# Python's salted hash()).
_MERSENNE = (1 << 61) - 1
_MAX_HASH = np.uint64((1 << 32) - 1)
_NUM_PERM = 64
_seed = MinHash(num_perm=_NUM_PERM, seed=1, scheme="affine32")
_A, _B = _seed.permutations  # affine coefficients, shape (num_perm,)


def exact_key(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _shingles(text: str, k: int = 5):
    text = "".join(text.split())
    if len(text) < k:
        return [text] if text else []
    return list({text[i : i + k] for i in range(len(text) - k + 1)})


def build_minhash(text: str, num_perm: int = _NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1, permutations=(_A, _B), scheme="affine32")
    shs = _shingles(text)
    if not shs:
        return m
    hv = np.fromiter((xxhash.xxh32(s).intdigest() for s in shs),
                     dtype=np.uint64, count=len(shs))
    # (E, num_perm): a_i * hv_e + b_i, uint64 wraparound matches datasketch
    phv = (np.outer(hv, _A) + _B) % _MERSENNE
    phv = np.bitwise_and(phv, _MAX_HASH)
    m.hashvalues = phv.min(axis=0).astype(np.uint64)
    return m


class Deduper:
    """Combined exact + near-duplicate filter with running counters."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 64):
        self.seen_exact = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._next_id = 0
        self.n_exact = 0
        self.n_near = 0
        self.n_kept = 0

    def keep(self, text: str) -> bool:
        key = exact_key(text)
        if key in self.seen_exact:
            self.n_exact += 1
            return False
        self.seen_exact.add(key)

        m = build_minhash(text, self.num_perm)
        if self.lsh.query(m):
            self.n_near += 1
            return False
        self.lsh.insert(str(self._next_id), m)
        self._next_id += 1
        self.n_kept += 1
        return True

    def stats(self):
        total = self.n_kept + self.n_exact + self.n_near
        rate = (self.n_exact + self.n_near) / total if total else 0.0
        return {
            "input": total,
            "kept": self.n_kept,
            "dropped_exact": self.n_exact,
            "dropped_near": self.n_near,
            "dedup_rate": round(rate, 4),
        }
