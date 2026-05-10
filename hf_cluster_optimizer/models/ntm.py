"""hf_cluster_optimizer.models.ntm -- Neural Turing Machine (read+write external memory).

A small NTM with one read head + one write head over an N x M memory.
The controller is an LSTM cell. Used as a sequence classifier here so it
plugs into the same train/eval surface as the other built-in models.
"""
from __future__ import annotations

from typing import Any, Dict

from hf_cluster_optimizer.registry import register
from hf_cluster_optimizer.models._seq_base import SequenceTrainerBase


@register("ntm")
class NTMTrainer(SequenceTrainerBase):
    """NTM with single read+write head over an N x M differentiable memory."""

    def build_model(self, in_dim: int, n_out: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        mem_n = int(self.config.get("memory_n", 16))
        mem_m = int(self.config.get("memory_m", 16))
        ctrl_hidden = int(self.config.get("hidden_dim", 32))

        class _NTMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.mem_n = mem_n
                self.mem_m = mem_m
                self.ctrl = nn.LSTMCell(in_dim + mem_m, ctrl_hidden)
                # Heads emit (key, beta, gate, shift, gamma) for content+location addressing
                # plus (erase, add) for the write head. Simplified to key + erase + add.
                self.read_key = nn.Linear(ctrl_hidden, mem_m)
                self.write_key = nn.Linear(ctrl_hidden, mem_m)
                self.write_erase = nn.Linear(ctrl_hidden, mem_m)
                self.write_add = nn.Linear(ctrl_hidden, mem_m)
                self.out = nn.Linear(ctrl_hidden + mem_m, n_out)

            def _addr(self, key, memory):
                # Content-based softmax addressing
                k = key.unsqueeze(1)  # (B,1,M)
                sim = F.cosine_similarity(memory, k, dim=-1)  # (B,N)
                return F.softmax(sim, dim=-1)

            def forward(self, x):
                B, T, _ = x.shape
                device = x.device
                memory = torch.zeros(B, self.mem_n, self.mem_m, device=device)
                h = torch.zeros(B, self.ctrl.hidden_size, device=device)
                c = torch.zeros(B, self.ctrl.hidden_size, device=device)
                read = torch.zeros(B, self.mem_m, device=device)
                for t in range(T):
                    inp = torch.cat([x[:, t], read], dim=-1)
                    h, c = self.ctrl(inp, (h, c))
                    rk = self.read_key(h)
                    wk = self.write_key(h)
                    we = torch.sigmoid(self.write_erase(h))
                    wa = torch.tanh(self.write_add(h))
                    rw = self._addr(rk, memory)  # (B,N)
                    ww = self._addr(wk, memory)
                    # Write
                    erase = ww.unsqueeze(-1) * we.unsqueeze(1)  # (B,N,M)
                    add = ww.unsqueeze(-1) * wa.unsqueeze(1)
                    memory = memory * (1.0 - erase) + add
                    # Read
                    read = (rw.unsqueeze(-1) * memory).sum(dim=1)  # (B,M)
                final = torch.cat([h, read], dim=-1)
                return self.out(final)

        return _NTMHead()
