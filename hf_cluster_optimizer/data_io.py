"""hf_cluster_optimizer — minimal data-loading abstraction.

Pandas-backed parquet/CSV reader plus a simple column-extraction API.
Each Trainer typically uses pandas directly; this module only exists
so paths can be resolved relative to the cfg's `data_root` and so
HuggingFace `datasets` integrations have a unified entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def resolve_path(p: str, data_root: Optional[Path] = None) -> Path:
    """Resolve a possibly-relative path to absolute.

    Absolute paths pass through. Relative paths are joined with
    `data_root` if provided; otherwise the current working directory.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if data_root is not None:
        return Path(data_root) / pp
    return pp.resolve()


def load_table(
    path: Path,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Any:
    """Load a parquet or CSV into a pandas DataFrame.

    Lazy import so hf_cluster_optimizer without pandas still imports.
    """
    import pandas as pd
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hf_cluster_optimizer.data_io.load_table: {p} does not exist")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p, columns=list(columns) if columns else None)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, usecols=list(columns) if columns else None)
    if p.suffix.lower() in (".json", ".jsonl"):
        return pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
    raise ValueError(
        f"hf_cluster_optimizer.data_io.load_table: unsupported extension {p.suffix!r}"
    )


def split_train_val(
    df: Any,
    *,
    val_frac: float = 0.1,
    by_column: Optional[str] = None,
    seed: int = 0,
) -> Tuple[Any, Any]:
    """Split a DataFrame into (train, val).

    If `by_column` is given AND it's a sortable column (timestamp etc.),
    the split is time-based: last `val_frac` rows become val.
    Otherwise it's a wallet-disjoint / random split with seed.
    """
    import numpy as np
    import pandas as pd
    n = len(df)
    if n == 0:
        return df, df
    if by_column is not None and by_column in df.columns:
        sorted_df = df.sort_values(by_column)
        cutoff = max(1, int(n * (1.0 - float(val_frac))))
        return sorted_df.iloc[:cutoff], sorted_df.iloc[cutoff:]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    cutoff = max(1, int(n * (1.0 - float(val_frac))))
    train_idx = perm[:cutoff]
    val_idx = perm[cutoff:]
    return df.iloc[train_idx], df.iloc[val_idx]


__all__ = ["resolve_path", "load_table", "split_train_val"]
