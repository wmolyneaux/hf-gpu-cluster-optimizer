"""modallabs — JSON-line metrics writer + reader.

One file per run at runs/<run_id>/<run_name>/metrics.jsonl.
Each line is a JSON object with keys:
    ts:      ISO8601 timestamp
    epoch:   int
    step:    int (global step within run)
    kind:    "train" | "eval" | "epoch" | "info" | "error"
    payload: dict of scalars (or strings for info / error)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class MetricsWriter:
    """Append-only JSON-line writer.

    Thread-safe within a single process; not multi-process safe (each
    Trainer runs in its own process). Flushes on every write so a
    crash leaves the file valid up to the last newline.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._step = 0

    def write(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        epoch: int = -1,
        step: Optional[int] = None,
    ) -> None:
        if step is None:
            step = self._step
            self._step += 1
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "epoch": int(epoch),
            "step": int(step),
            "kind": str(kind),
            "payload": dict(payload),
        }
        self._fh.write(json.dumps(record, ensure_ascii=True))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_metrics(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream-read a metrics.jsonl file."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


__all__ = ["MetricsWriter", "read_metrics"]
