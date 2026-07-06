"""Shared eval helpers."""

import torch

from model.config import ModelConfig
from model.transformer import UrduLM


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ck["config"])
    model = UrduLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg
