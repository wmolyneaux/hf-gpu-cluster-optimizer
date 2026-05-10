"""hf_cluster_optimizer.models._torch_common -- shared helpers for torch trainers.

Light helpers used by multiple built-in trainers: synthetic-fallback
data loaders, mean-aggregation of per-step metrics, monitor extraction.
Kept private so external porting only depends on the public Trainer
contract in hf_cluster_optimizer.base.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def make_synthetic_classification(
    n: int, in_dim: int, n_classes: int, seed: int,
) -> Tuple[Any, Any]:
    """Deterministic torch tensors (X, y) for smoke tests."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    X = torch.randn(n, in_dim, generator=g)
    # Linear separator + noise -> 2+ class problem
    W = torch.randn(in_dim, n_classes, generator=g)
    logits = X @ W
    y = logits.argmax(dim=-1)
    return X, y


def make_synthetic_regression(
    n: int, in_dim: int, seed: int,
) -> Tuple[Any, Any]:
    """Deterministic torch tensors (X, y) for regression smoke tests."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    X = torch.randn(n, in_dim, generator=g)
    W = torch.randn(in_dim, 1, generator=g)
    y = (X @ W).squeeze(-1) + 0.01 * torch.randn(n, generator=g)
    return X, y


def make_synthetic_sequences(
    n: int, seq_len: int, in_dim: int, n_classes: int, seed: int,
) -> Tuple[Any, Any]:
    """Deterministic (B, T, F) sequences with per-sequence label."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    X = torch.randn(n, seq_len, in_dim, generator=g)
    W = torch.randn(in_dim, n_classes, generator=g)
    # Label from mean-pooled sequence -> linear classifier
    pooled = X.mean(dim=1)
    y = (pooled @ W).argmax(dim=-1)
    return X, y


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean-aggregate a list of per-step metric dicts."""
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None
                and isinstance(r[k], (int, float)) and math.isfinite(float(r[k]))]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def resolve_device(spec: Optional[str]) -> str:
    """Resolve a device spec to a concrete torch device string."""
    s = (spec or "auto").lower()
    try:
        import torch
        if s == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if s.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return s
    except Exception:
        return "cpu"


def load_xy_table(
    cfg: Dict[str, Any],
    *,
    default_n: int = 256,
    default_in_dim: int = 8,
    default_n_classes: int = 3,
    seed: int = 0,
    task: str = "classification",
) -> Tuple[Any, Any]:
    """Load (X, y) tensors from cfg.data_path or fall back to synthetic.

    cfg keys honored:
      data_path, feature_columns, label_column, n, in_dim, n_classes
    Falls back to deterministic synthetic data when data_path is absent.
    """
    import torch
    from hf_cluster_optimizer.data_io import load_table

    n = int(cfg.get("n", default_n))
    in_dim = int(cfg.get("in_dim", default_in_dim))
    n_classes = int(cfg.get("n_classes", default_n_classes))

    path = cfg.get("data_path")
    if path:
        df = load_table(Path(path))
        feat_cols = cfg.get("feature_columns")
        label_col = cfg.get("label_column")
        if not feat_cols or not label_col:
            raise ValueError(
                "cfg.data_path requires cfg.feature_columns + cfg.label_column"
            )
        X = torch.tensor(df[list(feat_cols)].values, dtype=torch.float32)
        y_arr = df[label_col].values
        if task == "regression":
            y = torch.tensor(y_arr, dtype=torch.float32)
        else:
            y = torch.tensor(y_arr, dtype=torch.long)
        return X, y

    if task == "regression":
        return make_synthetic_regression(n, in_dim, seed)
    return make_synthetic_classification(n, in_dim, n_classes, seed)
