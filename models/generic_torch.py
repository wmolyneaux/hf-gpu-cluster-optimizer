"""modallabs.models.generic_torch -- wrap any torch.nn.Module from a dotted path.

Cfg fields:
  module_path: "package.module:ClassName" or "package.module.ClassName"
  module_kwargs: dict passed to ClassName(**kwargs)
  loss: "cross_entropy" | "mse" | "l1" | "bce_with_logits"
  optimizer: "adam" | "adamw" | "sgd"
  lr: float
  weight_decay: float
  epochs: int
  batch_size: int
  task: "classification" | "regression"
  # Data: data_path (parquet/csv) + feature_columns + label_column;
  # else falls back to synthetic deterministic data.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.registry import register

from modallabs.models._torch_common import (
    load_xy_table,
    mean_metrics,
    resolve_device,
)


def _import_dotted(spec: str):
    """Import 'pkg.mod:Name' or 'pkg.mod.Name' and return the attribute."""
    if ":" in spec:
        mod, attr = spec.split(":", 1)
    else:
        mod, _, attr = spec.rpartition(".")
    if not mod or not attr:
        raise ValueError(f"module_path must be 'pkg.mod:Name' or 'pkg.mod.Name', got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


@register("torch_module")
class GenericTorchTrainer(Trainer):
    """Generic supervised trainer for any torch.nn.Module."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.task = str(self.config.get("task", "classification")).lower()
        self.lr = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.epochs = int(self.config.get("epochs", 1))
        self.module_path = self.config.get("module_path")
        self.module_kwargs = dict(self.config.get("module_kwargs") or {})
        self.loss_name = str(self.config.get("loss", "cross_entropy")).lower()
        self.opt_name = str(self.config.get("optimizer", "adam")).lower()
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.device = "cpu"
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self._train_buf: List[Dict[str, float]] = []
        self._eval_buf: List[Dict[str, float]] = []
        self._best = None
        self._monitor_mode = "min" if self.task == "regression" else "max"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GenericTorchTrainer":
        return cls(config)

    def _build_loss(self):
        import torch.nn as nn
        m = self.loss_name
        if m == "cross_entropy":
            return nn.CrossEntropyLoss()
        if m == "mse":
            return nn.MSELoss()
        if m == "l1":
            return nn.L1Loss()
        if m == "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"unknown loss {self.loss_name!r}")

    def _build_optimizer(self):
        import torch
        params = self.model.parameters()
        if self.opt_name == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.opt_name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.opt_name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"unknown optimizer {self.opt_name!r}")

    def setup(self, setup: TrainerSetup) -> None:
        self.device = resolve_device(setup.device)
        if not self.module_path:
            raise ValueError("generic_torch requires cfg.module_path (dotted-path)")
        cls = _import_dotted(self.module_path)
        self.model = cls(**self.module_kwargs).to(self.device)
        self.loss_fn = self._build_loss()
        self.optimizer = self._build_optimizer()

        # Data
        X, y = load_xy_table(
            self.config, seed=setup.seed,
            task=("regression" if self.task == "regression" else "classification"),
        )
        n = X.shape[0]
        cut = max(1, int(n * 0.9))
        self.X_train, self.y_train = X[:cut].to(self.device), y[:cut].to(self.device)
        self.X_val, self.y_val = X[cut:].to(self.device), y[cut:].to(self.device)
        if self.X_val.shape[0] == 0:  # tiny dataset edge case
            self.X_val, self.y_val = self.X_train, self.y_train

    def _iter_batches(self, X, y) -> Iterable[Tuple[Any, Any]]:
        n = X.shape[0]
        for i in range(0, n, self.batch_size):
            yield X[i: i + self.batch_size], y[i: i + self.batch_size]

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        self.model.train()
        return self._iter_batches(self.X_train, self.y_train)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        self.model.eval()
        return self._iter_batches(self.X_val, self.y_val)

    def train_step(self, batch: Any) -> TrainerStepResult:
        import torch
        Xb, yb = batch
        self.optimizer.zero_grad()
        out = self.model(Xb)
        if self.task == "regression":
            out = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
            loss = self.loss_fn(out, yb.float())
            metrics = {"loss": float(loss.item())}
        else:
            loss = self.loss_fn(out, yb.long())
            with torch.no_grad():
                acc = (out.argmax(dim=-1) == yb).float().mean().item()
            metrics = {"loss": float(loss.item()), "acc": float(acc)}
        loss.backward()
        self.optimizer.step()
        self._train_buf.append(metrics)
        return TrainerStepResult(metrics=metrics, n_examples=int(Xb.shape[0]))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        import torch
        Xb, yb = batch
        with torch.no_grad():
            out = self.model(Xb)
            if self.task == "regression":
                out = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
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
            monitor = -float(val_m.get("loss", float("inf")))  # higher = better
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
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        import torch
        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
