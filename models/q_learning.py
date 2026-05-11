"""modallabs.models.q_learning -- DQN-style Q-learning trainer.

Supports double-Q + dueling head + optional uniform replay buffer.
Trains on a synthetic deterministic 1-D bandit-like environment by
default; pass cfg.transitions_path to load (state, action, reward,
next_state, done) tuples from parquet for offline RL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.registry import register

from modallabs.models._torch_common import mean_metrics, resolve_device


def _build_q_net(state_dim: int, n_actions: int, hidden: int, dueling: bool):
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            if dueling:
                self.value = nn.Linear(hidden, 1)
                self.adv = nn.Linear(hidden, n_actions)
            else:
                self.q = nn.Linear(hidden, n_actions)

        def forward(self, x):
            h = self.feature(x)
            if dueling:
                v = self.value(h)
                a = self.adv(h)
                return v + (a - a.mean(dim=-1, keepdim=True))
            return self.q(h)

    return _Net()


def _make_synthetic_transitions(n: int, state_dim: int, n_actions: int, seed: int):
    """Deterministic offline transitions: reward = w[a] . s + noise."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    s = torch.randn(n, state_dim, generator=g)
    a = torch.randint(0, n_actions, (n,), generator=g)
    W = torch.randn(n_actions, state_dim, generator=g)
    rew = (W[a] * s).sum(dim=-1) + 0.05 * torch.randn(n, generator=g)
    s2 = s + 0.1 * torch.randn(n, state_dim, generator=g)
    done = torch.zeros(n, dtype=torch.float32)
    return s, a, rew, s2, done


@register("q_learning")
class QLearningTrainer(Trainer):
    """Offline DQN trainer with double-Q + optional dueling head."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.state_dim = int(self.config.get("state_dim", 8))
        self.n_actions = int(self.config.get("n_actions", 4))
        self.hidden_dim = int(self.config.get("hidden_dim", 64))
        self.gamma = float(self.config.get("gamma", 0.99))
        self.tau = float(self.config.get("tau", 0.01))  # soft target update
        self.lr = float(self.config.get("lr", 5e-4))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.epochs = int(self.config.get("epochs", 1))
        self.dueling = bool(self.config.get("dueling", True))
        self.double_q = bool(self.config.get("double_q", True))
        self.n_samples = int(self.config.get("n", 512))
        self.device = "cpu"
        self.q_net = None
        self.target_net = None
        self.opt = None
        self.S = None
        self.A = None
        self.R = None
        self.S2 = None
        self.D = None
        self._train_buf: List[Dict[str, float]] = []
        self._eval_buf: List[Dict[str, float]] = []
        self._best: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QLearningTrainer":
        return cls(config)

    def setup(self, setup: TrainerSetup) -> None:
        import torch

        self.device = resolve_device(setup.device)
        self.q_net = _build_q_net(self.state_dim, self.n_actions, self.hidden_dim, self.dueling).to(self.device)
        self.target_net = _build_q_net(self.state_dim, self.n_actions, self.hidden_dim, self.dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        path = self.config.get("transitions_path")
        if path:
            from modallabs.data_io import load_table
            df = load_table(Path(path))
            S_cols = self.config.get("state_columns")
            S2_cols = self.config.get("next_state_columns") or S_cols
            A_col = self.config.get("action_column")
            R_col = self.config.get("reward_column")
            D_col = self.config.get("done_column")
            if not (S_cols and A_col and R_col):
                raise ValueError("transitions_path requires state_columns + action_column + reward_column")
            S = torch.tensor(df[list(S_cols)].values, dtype=torch.float32)
            A = torch.tensor(df[A_col].values, dtype=torch.long)
            R = torch.tensor(df[R_col].values, dtype=torch.float32)
            S2 = torch.tensor(df[list(S2_cols)].values, dtype=torch.float32)
            D = torch.tensor(df[D_col].values if D_col else [0] * len(df), dtype=torch.float32)
        else:
            S, A, R, S2, D = _make_synthetic_transitions(
                self.n_samples, self.state_dim, self.n_actions, setup.seed,
            )
        cut = max(1, int(len(S) * 0.9))
        self.S, self.A, self.R, self.S2, self.D = (
            S[:cut].to(self.device), A[:cut].to(self.device),
            R[:cut].to(self.device), S2[:cut].to(self.device),
            D[:cut].to(self.device),
        )
        self.S_val = S[cut:].to(self.device) if cut < len(S) else self.S
        self.A_val = A[cut:].to(self.device) if cut < len(S) else self.A
        self.R_val = R[cut:].to(self.device) if cut < len(S) else self.R
        self.S2_val = S2[cut:].to(self.device) if cut < len(S) else self.S2
        self.D_val = D[cut:].to(self.device) if cut < len(S) else self.D

    def _td_loss(self, s, a, r, s2, d):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            if self.double_q:
                next_a = self.q_net(s2).argmax(dim=-1)
                next_q = self.target_net(s2).gather(-1, next_a.unsqueeze(-1)).squeeze(-1)
            else:
                next_q = self.target_net(s2).max(dim=-1).values
            target = r + (1.0 - d) * self.gamma * next_q
        pred_q = self.q_net(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        return F.mse_loss(pred_q, target)

    def _iter(self, S, A, R, S2, D) -> Iterable[Tuple[Any, ...]]:
        n = S.shape[0]
        for i in range(0, n, self.batch_size):
            yield (S[i:i+self.batch_size], A[i:i+self.batch_size],
                   R[i:i+self.batch_size], S2[i:i+self.batch_size],
                   D[i:i+self.batch_size])

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        self.q_net.train()
        return self._iter(self.S, self.A, self.R, self.S2, self.D)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        self.q_net.eval()
        return self._iter(self.S_val, self.A_val, self.R_val, self.S2_val, self.D_val)

    def train_step(self, batch: Any) -> TrainerStepResult:
        import torch
        s, a, r, s2, d = batch
        self.opt.zero_grad()
        loss = self._td_loss(s, a, r, s2, d)
        loss.backward()
        self.opt.step()
        # Soft-update target
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        m = {"td_loss": float(loss.item())}
        self._train_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=int(s.shape[0]))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        import torch
        s, a, r, s2, d = batch
        with torch.no_grad():
            loss = self._td_loss(s, a, r, s2, d)
            # On-policy mean reward proxy: greedy action = argmax Q(s,.).
            greedy = self.q_net(s).argmax(dim=-1)
            policy_match = (greedy == a).float().mean().item()
        m = {"td_loss": float(loss.item()), "policy_match": float(policy_match)}
        self._eval_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=int(s.shape[0]))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        train_m = mean_metrics(self._train_buf)
        val_m = mean_metrics(self._eval_buf)
        # Lower TD-loss is better; flip sign for monitor.
        monitor = -float(val_m.get("td_loss", float("inf")))
        is_best = self._best is None or monitor > self._best
        if is_best:
            self._best = monitor
        return TrainerEpochResult(
            train_metrics=train_m, val_metrics=val_m,
            is_best=is_best, monitor_value=monitor,
        )

    def save_checkpoint(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        import torch
        ckpt = torch.load(Path(path), map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
