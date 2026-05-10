"""hf_cluster_optimizer.models._seq_base -- shared sequence-classifier training loop.

Subclasses provide build_model(in_dim, n_out) returning a torch.nn.Module
that consumes (B, T, F) and outputs (B, n_out). The base handles data,
optimizer, loss, batching, metrics, monitor, checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hf_cluster_optimizer.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)

from hf_cluster_optimizer.models._torch_common import (
    make_synthetic_sequences,
    mean_metrics,
    resolve_device,
)


class SequenceTrainerBase(Trainer):
    """Base for sequence -> class (or scalar) trainers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.in_dim = int(self.config.get("in_dim", 8))
        self.hidden_dim = int(self.config.get("hidden_dim", 32))
        self.num_layers = int(self.config.get("num_layers", 1))
        self.n_classes = int(self.config.get("n_classes", 3))
        self.seq_len = int(self.config.get("seq_len", 16))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.lr = float(self.config.get("lr", 1e-3))
        self.epochs = int(self.config.get("epochs", 1))
        self.n_samples = int(self.config.get("n", 256))
        self.task = str(self.config.get("task", "classification")).lower()
        self.device = "cpu"
        self.model = None
        self.opt = None
        self.loss_fn = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self._train_buf: List[Dict[str, float]] = []
        self._eval_buf: List[Dict[str, float]] = []
        self._best: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SequenceTrainerBase":
        return cls(config)

    # --- subclass hook ---
    def build_model(self, in_dim: int, n_out: int):
        raise NotImplementedError

    def setup(self, setup: TrainerSetup) -> None:
        import torch
        import torch.nn as nn

        self.device = resolve_device(setup.device)
        n_out = 1 if self.task == "regression" else self.n_classes
        self.model = self.build_model(self.in_dim, n_out).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        X, y = make_synthetic_sequences(
            self.n_samples, self.seq_len, self.in_dim, self.n_classes, setup.seed,
        )
        cut = max(1, int(len(X) * 0.9))
        self.X_train, self.y_train = X[:cut].to(self.device), y[:cut].to(self.device)
        self.X_val, self.y_val = X[cut:].to(self.device), y[cut:].to(self.device)
        if self.X_val.shape[0] == 0:
            self.X_val, self.y_val = self.X_train, self.y_train

    def _iter(self, X, y) -> Iterable[Tuple[Any, Any]]:
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i: i + self.batch_size], y[i: i + self.batch_size]

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        self.model.train()
        return self._iter(self.X_train, self.y_train)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        self.model.eval()
        return self._iter(self.X_val, self.y_val)

    def train_step(self, batch: Any) -> TrainerStepResult:
        import torch
        Xb, yb = batch
        self.opt.zero_grad()
        out = self.model(Xb)
        if self.task == "regression":
            out = out.squeeze(-1)
            loss = self.loss_fn(out, yb.float())
            metrics = {"loss": float(loss.item())}
        else:
            loss = self.loss_fn(out, yb.long())
            with torch.no_grad():
                acc = (out.argmax(dim=-1) == yb).float().mean().item()
            metrics = {"loss": float(loss.item()), "acc": float(acc)}
        loss.backward()
        self.opt.step()
        self._train_buf.append(metrics)
        return TrainerStepResult(metrics=metrics, n_examples=int(Xb.shape[0]))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        import torch
        Xb, yb = batch
        with torch.no_grad():
            out = self.model(Xb)
            if self.task == "regression":
                out = out.squeeze(-1)
                loss = self.loss_fn(out, yb.float())
                metrics = {"loss": float(loss.item())}
            else:
                loss = self.loss_fn(out, yb.long())
                acc = (out.argmax(dim=-1) == yb).float().mean().item()
                metrics = {"loss": float(loss.item()), "acc": float(acc)}
        self._eval_buf.append(metrics)
        return TrainerStepResult(metrics=metrics, n_examples=int(Xb.shape[0]))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        train_m = mean_metrics(self._train_buf)
        val_m = mean_metrics(self._eval_buf)
        if self.task == "regression":
            monitor = -float(val_m.get("loss", float("inf")))
        else:
            monitor = float(val_m.get("acc", 0.0))
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
        torch.save({"state_dict": self.model.state_dict(), "config": self.config}, path)

    def load_checkpoint(self, path: Path) -> None:
        import torch
        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
