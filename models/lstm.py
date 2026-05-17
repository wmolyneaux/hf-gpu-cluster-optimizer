"""modallabs.models.lstm -- built-in LSTM sequence trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.registry import register

from modallabs.models._torch_common import (
    make_synthetic_sequences,
    mean_metrics,
    resolve_device,
)


@register("lstm")
class LSTMTrainer(Trainer):
    """Sequence classifier on top of an LSTM encoder."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.in_dim = int(self.config.get("in_dim", 8))
        self.hidden_dim = int(self.config.get("hidden_dim", 32))
        self.num_layers = int(self.config.get("num_layers", 1))
        self.bidirectional = bool(self.config.get("bidirectional", False))
        self.dropout = float(self.config.get("dropout", 0.0))
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
    def from_config(cls, config: Dict[str, Any]) -> "LSTMTrainer":
        return cls(config)

    def setup(self, setup: TrainerSetup) -> None:
        import torch
        import torch.nn as nn

        self.device = resolve_device(setup.device)

        class _LSTMHead(nn.Module):
            def __init__(self, in_dim, hidden, layers, n_out, bi, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    in_dim, hidden, num_layers=layers,
                    batch_first=True, bidirectional=bi,
                    dropout=dropout if layers > 1 else 0.0,
                )
                d = hidden * (2 if bi else 1)
                self.head = nn.Linear(d, n_out)

            def forward(self, x):
                out, _ = self.lstm(x)
                pooled = out.mean(dim=1)
                return self.head(pooled)

        n_out = 1 if self.task == "regression" else self.n_classes
        self.model = _LSTMHead(
            self.in_dim, self.hidden_dim, self.num_layers,
            n_out, self.bidirectional, self.dropout,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        # Data: try parquet if data_path; else synthetic deterministic.
        path = self.config.get("data_path")
        if path:
            from modallabs.data_io import load_table
            df = load_table(Path(path))
            feat_cols = list(self.config.get("feature_columns") or [])
            label_col = self.config.get("label_column")
            if not feat_cols or not label_col:
                raise ValueError("cfg.data_path requires feature_columns + label_column")
            seq_id_col = self.config.get("sequence_id_column")
            if seq_id_col:
                # Group rows by seq_id, pad to seq_len.
                groups = []
                labels = []
                for sid, sub in df.groupby(seq_id_col):
                    arr = sub[feat_cols].values.astype("float32")
                    if len(arr) > self.seq_len:
                        arr = arr[: self.seq_len]
                    elif len(arr) < self.seq_len:
                        pad = self.seq_len - len(arr)
                        import numpy as _np
                        arr = _np.vstack([arr, _np.zeros((pad, len(feat_cols)), dtype="float32")])
                    groups.append(arr)
                    labels.append(sub[label_col].iloc[-1])
                import numpy as _np
                X = torch.tensor(_np.stack(groups), dtype=torch.float32)
                y = torch.tensor(_np.array(labels), dtype=torch.long if self.task != "regression" else torch.float32)
            else:
                # Treat each row as a 1-step sequence (degenerate but valid).
                X = torch.tensor(df[feat_cols].values, dtype=torch.float32).unsqueeze(1)
                y_arr = df[label_col].values
                y = torch.tensor(y_arr, dtype=torch.long if self.task != "regression" else torch.float32)
        else:
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
