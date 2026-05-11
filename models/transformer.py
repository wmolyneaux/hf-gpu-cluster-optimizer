"""modallabs.models.transformer -- built-in encoder-only transformer sequence trainer."""
from __future__ import annotations

from typing import Any, Dict

from modallabs.registry import register
from modallabs.models._seq_base import SequenceTrainerBase


@register("transformer")
class EncoderTransformerTrainer(SequenceTrainerBase):
    """Encoder-only transformer for sequence-to-class / regression."""

    def build_model(self, in_dim: int, n_out: int):
        import torch.nn as nn
        d_model = int(self.config.get("d_model", max(self.hidden_dim, 32)))
        nhead = int(self.config.get("nhead", 4))
        dim_ff = int(self.config.get("dim_feedforward", d_model * 2))
        # Ensure d_model is divisible by nhead.
        if d_model % nhead != 0:
            d_model = ((d_model // nhead) + 1) * nhead

        in_proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=float(self.config.get("dropout", 0.1)),
        )
        encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        out_proj = nn.Linear(d_model, n_out)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = in_proj
                self.encoder = encoder
                self.out = out_proj

            def forward(self, x):
                h = self.in_proj(x)
                z = self.encoder(h)
                return self.out(z.mean(dim=1))

        return _Net()
