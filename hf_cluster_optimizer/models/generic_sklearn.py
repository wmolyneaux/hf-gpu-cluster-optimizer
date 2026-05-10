"""hf_cluster_optimizer.models.generic_sklearn -- wrap any sklearn estimator.

Cfg fields:
  estimator_path: "package.module:Class"
  estimator_kwargs: dict
  data_path, feature_columns, label_column (optional; falls back to synthetic)
  task: "classification" | "regression"
  metric: "accuracy" | "f1" | "r2" | "neg_mse" (default by task)
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hf_cluster_optimizer.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from hf_cluster_optimizer.registry import register


def _import_dotted(spec: str):
    """Import 'pkg.mod:Name' or 'pkg.mod.Name' and return the attribute."""
    if ":" in spec:
        mod, attr = spec.split(":", 1)
    else:
        mod, _, attr = spec.rpartition(".")
    if not mod or not attr:
        raise ValueError(f"estimator_path must be 'pkg.mod:Name' or 'pkg.mod.Name', got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


@register("sklearn")
class SklearnTrainer(Trainer):
    """One-shot sklearn-style estimator: fit() once, eval = .score()."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.task = str(self.config.get("task", "classification")).lower()
        self.estimator_path = self.config.get("estimator_path")
        self.estimator_kwargs = dict(self.config.get("estimator_kwargs") or {})
        self.metric_name = str(
            self.config.get(
                "metric",
                "r2" if self.task == "regression" else "accuracy",
            )
        )
        self.estimator = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self._fitted = False
        self._train_score: Optional[float] = None
        self._val_score: Optional[float] = None
        self._best: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SklearnTrainer":
        return cls(config)

    def _load_data(self, seed: int):
        import numpy as np
        path = self.config.get("data_path")
        n = int(self.config.get("n", 256))
        in_dim = int(self.config.get("in_dim", 8))
        n_classes = int(self.config.get("n_classes", 3))
        if path:
            from hf_cluster_optimizer.data_io import load_table
            df = load_table(Path(path))
            feat_cols = self.config.get("feature_columns")
            label_col = self.config.get("label_column")
            if not feat_cols or not label_col:
                raise ValueError(
                    "cfg.data_path requires cfg.feature_columns + cfg.label_column"
                )
            X = df[list(feat_cols)].values.astype("float32")
            y_arr = df[label_col].values
            y = y_arr.astype("float32") if self.task == "regression" else y_arr
        else:
            rng = np.random.default_rng(int(seed))
            X = rng.standard_normal((n, in_dim)).astype("float32")
            W = rng.standard_normal((in_dim, n_classes if self.task != "regression" else 1)).astype("float32")
            scores = X @ W
            if self.task == "regression":
                y = scores.squeeze(-1) + 0.01 * rng.standard_normal(n).astype("float32")
            else:
                y = scores.argmax(axis=-1)
        cut = max(1, int(len(X) * 0.9))
        return X[:cut], y[:cut], X[cut:] if cut < len(X) else X[:cut], y[cut:] if cut < len(y) else y[:cut]

    def setup(self, setup: TrainerSetup) -> None:
        if not self.estimator_path:
            raise ValueError("sklearn trainer requires cfg.estimator_path")
        cls = _import_dotted(self.estimator_path)
        # Some sklearn estimators accept random_state; pass when accepted.
        kwargs = dict(self.estimator_kwargs)
        try:
            self.estimator = cls(random_state=int(setup.seed), **kwargs)
        except TypeError:
            self.estimator = cls(**kwargs)
        self.X_train, self.y_train, self.X_val, self.y_val = self._load_data(setup.seed)

    def _score(self, X, y) -> float:
        from sklearn.metrics import (
            accuracy_score, f1_score, r2_score, mean_squared_error,
        )
        pred = self.estimator.predict(X)
        m = self.metric_name.lower()
        if m == "accuracy":
            return float(accuracy_score(y, pred))
        if m == "f1":
            return float(f1_score(y, pred, average="macro", zero_division=0))
        if m == "r2":
            return float(r2_score(y, pred))
        if m == "neg_mse":
            return float(-mean_squared_error(y, pred))
        return float(self.estimator.score(X, y))

    def train_iter(self) -> Iterable[Any]:
        # Single "batch" — sklearn fits in one shot.
        if self._fitted:
            return iter([])
        return iter([(self.X_train, self.y_train)])

    def eval_iter(self) -> Iterable[Any]:
        return iter([(self.X_val, self.y_val)])

    def train_step(self, batch: Any) -> TrainerStepResult:
        Xb, yb = batch
        self.estimator.fit(Xb, yb)
        self._fitted = True
        s = self._score(Xb, yb)
        self._train_score = s
        return TrainerStepResult(metrics={"score": float(s)}, n_examples=int(len(Xb)))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        Xb, yb = batch
        s = self._score(Xb, yb)
        self._val_score = s
        return TrainerStepResult(metrics={"score": float(s)}, n_examples=int(len(Xb)))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        monitor = float(self._val_score if self._val_score is not None else 0.0)
        is_best = self._best is None or monitor > self._best
        if is_best:
            self._best = monitor
        return TrainerEpochResult(
            train_metrics={"score": float(self._train_score or 0.0)},
            val_metrics={"score": monitor},
            is_best=is_best, monitor_value=monitor,
        )

    def num_epochs(self) -> int:
        return 1  # sklearn estimators fit in one pass

    def save_checkpoint(self, path: Path) -> None:
        import joblib
        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)

    def load_checkpoint(self, path: Path) -> None:
        import joblib
        p = Path(path)
        if p.suffix != ".joblib":
            p = p.with_suffix(".joblib")
        self.estimator = joblib.load(p)
        self._fitted = True
