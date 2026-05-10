"""hf_cluster_optimizer.models.xgboost -- XGBoost gradient-boosting trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hf_cluster_optimizer.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from hf_cluster_optimizer.registry import register


@register("xgboost")
class XGBoostTrainer(Trainer):
    """XGBoost classification or regression."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.task = str(self.config.get("task", "classification")).lower()
        self.params = dict(self.config.get("params") or {})
        self.num_boost_round = int(self.config.get("num_boost_round", 50))
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self._train_score: Optional[float] = None
        self._val_score: Optional[float] = None
        self._best: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "XGBoostTrainer":
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
                raise ValueError("cfg.data_path requires feature_columns + label_column")
            X = df[list(feat_cols)].values.astype("float32")
            y = df[label_col].values
        else:
            rng = np.random.default_rng(int(seed))
            X = rng.standard_normal((n, in_dim)).astype("float32")
            W = rng.standard_normal((in_dim, n_classes if self.task == "classification" else 1))
            scores = X @ W
            if self.task == "regression":
                y = scores.squeeze(-1).astype("float32")
            else:
                y = scores.argmax(axis=-1).astype("int32")
        cut = max(1, int(len(X) * 0.9))
        return X[:cut], y[:cut], X[cut:] if cut < len(X) else X[:cut], y[cut:] if cut < len(y) else y[:cut]

    def setup(self, setup: TrainerSetup) -> None:
        self.X_train, self.y_train, self.X_val, self.y_val = self._load_data(setup.seed)
        defaults = {
            "seed": int(setup.seed),
            "verbosity": 0,
            "nthread": int(self.config.get("num_threads", 1)),
        }
        if self.task == "regression":
            defaults["objective"] = "reg:squarederror"
            defaults["eval_metric"] = "rmse"
        else:
            n_classes = int(self.config.get("n_classes", 3))
            if n_classes == 2:
                defaults["objective"] = "binary:logistic"
                defaults["eval_metric"] = "logloss"
            else:
                defaults["objective"] = "multi:softprob"
                defaults["num_class"] = n_classes
                defaults["eval_metric"] = "mlogloss"
        for k, v in defaults.items():
            self.params.setdefault(k, v)

    def _score(self, X, y) -> float:
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, r2_score
        d = xgb.DMatrix(X)
        pred = self.model.predict(d)
        if self.task == "regression":
            return float(r2_score(y, pred))
        if pred.ndim == 2:
            pred = pred.argmax(axis=-1)
        else:
            pred = (pred > 0.5).astype(int)
        return float(accuracy_score(y, pred))

    def train_iter(self) -> Iterable[Any]:
        if self.model is not None:
            return iter([])
        return iter([(self.X_train, self.y_train)])

    def eval_iter(self) -> Iterable[Any]:
        return iter([(self.X_val, self.y_val)])

    def train_step(self, batch: Any) -> TrainerStepResult:
        import xgboost as xgb
        Xb, yb = batch
        d = xgb.DMatrix(Xb, label=yb)
        self.model = xgb.train(self.params, d, num_boost_round=self.num_boost_round)
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
        return 1

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load_checkpoint(self, path: Path) -> None:
        import xgboost as xgb
        p = Path(path)
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        self.model = xgb.Booster()
        self.model.load_model(str(p))
