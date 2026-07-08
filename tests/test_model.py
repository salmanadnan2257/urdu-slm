"""Correctness tests for the model: shapes, causal masking, parameter counts."""

import torch

from model.config import build_config, PRESETS
from model.transformer import UrduLM


def _tiny_model(vocab=256, block=64):
    cfg = build_config("tiny", vocab_size=vocab, max_seq_len=block, n_layer=2, n_embd=64, n_head=4)
    return UrduLM(cfg), cfg


def test_forward_shapes():
    model, cfg = _tiny_model()
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(x)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss is None
    _, loss = model(x, x)
    assert loss.ndim == 0 and loss.item() > 0


def test_causal_masking():
    """Changing a future token must not change logits at earlier positions."""
    model, cfg = _tiny_model()
    model.eval()
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # perturb last token
        pert, _ = model(x2)
    # all positions except the last must be identical
    assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-5)
    assert not torch.allclose(base[:, -1], pert[:, -1], atol=1e-5)


def test_tied_embeddings():
    model, _ = _tiny_model()
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_generate_runs():
    model, cfg = _tiny_model()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=8, top_k=10)
    assert out.shape == (1, 12)


def test_preset_param_counts():
    """Each preset lands in its intended size band at the real 32k vocab."""
    bands = {"tiny": (9e6, 16e6), "small": (30e6, 45e6), "base": (115e6, 135e6),
             "medium": (320e6, 350e6)}
    for name in PRESETS:
        cfg = build_config(name, vocab_size=32000)
        model = UrduLM(cfg)
        n = model.num_params()
        lo, hi = bands[name]
        assert lo <= n <= hi, f"{name}: {n/1e6:.1f}M outside [{lo/1e6},{hi/1e6}]M"


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        c = build_config(name, vocab_size=32000)
        m = UrduLM(c)
        print(f"{name}: {m.num_params()/1e6:.2f}M params, {c.n_layer}L {c.n_embd}d {c.n_head}h")
