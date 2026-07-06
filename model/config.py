"""Model configuration and size presets.

The three presets track the phased plan: `tiny` is small enough to train to a
visibly decreasing loss on a CPU, `small` is a mid-point sanity check, and
`base` is the ~125M target reserved for the GPU run in phase 2.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_kv_head: int = 0          # 0 means use n_head (no grouped-query attention)
    mlp_ratio: float = 8 / 3    # SwiGLU keeps params comparable to a 4x GELU MLP
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must divide by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must divide by n_kv_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# Parameter counts below are the non-embedding + embedding totals measured by
# tests/test_model.py at vocab_size=32000. Comments give the measured figure.
PRESETS = {
    # ~13M params (of which ~8.2M are the 32k embedding table), CPU-trainable
    "tiny": dict(n_layer=6, n_head=4, n_embd=256, max_seq_len=512),
    # ~36M params
    "small": dict(n_layer=6, n_head=8, n_embd=512, max_seq_len=1024),
    # ~124M params, phase-2 GPU target
    "base": dict(n_layer=14, n_head=12, n_embd=768, max_seq_len=1024),
}


def build_config(preset: str, vocab_size: int = 32000, **overrides) -> ModelConfig:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    params = dict(PRESETS[preset])
    params["vocab_size"] = vocab_size
    params.update(overrides)
    return ModelConfig(**params)
