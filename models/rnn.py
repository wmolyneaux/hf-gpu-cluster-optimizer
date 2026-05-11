"""modallabs.models.rnn -- vanilla RNN and GRU sequence trainers."""
from __future__ import annotations

from typing import Any, Dict

from modallabs.registry import register
from modallabs.models._seq_base import SequenceTrainerBase


def _make_rnn_module(cell: str, in_dim: int, hidden: int, layers: int, n_out: int):
    import torch.nn as nn
    rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU}[cell]
    rnn = rnn_cls(in_dim, hidden, num_layers=layers, batch_first=True)

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn
            self.fc = nn.Linear(hidden, n_out)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out.mean(dim=1))

    return _Head()


@register("rnn")
class VanillaRNNTrainer(SequenceTrainerBase):
    """Vanilla RNN sequence classifier."""

    def build_model(self, in_dim: int, n_out: int):
        return _make_rnn_module("rnn", in_dim, self.hidden_dim, self.num_layers, n_out)


@register("gru")
class GRUTrainer(SequenceTrainerBase):
    """GRU sequence classifier."""

    def build_model(self, in_dim: int, n_out: int):
        return _make_rnn_module("gru", in_dim, self.hidden_dim, self.num_layers, n_out)
