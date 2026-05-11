"""modallabs -- summarize a finished run directory.

`modallabs-report runs/<run_id>` prints a per-run table (phase, best
metric, epochs, wall-clock, checkpoint). Point it at a single
`runs/<run_id>/<run_name>` directory instead and it additionally prints
that run's per-epoch metric history.

No dependencies beyond the standard library + `modallabs` itself --
it reads the artifacts the runner already writes (`summary.json`,
each run's `metrics.jsonl`, `status.json`, and the `.modallabs_done`
sentinel).

    modallabs-report runs/experiment_01
    modallabs-report runs/experiment_01 --json
    modallabs-report runs/experiment_01/bert_classifier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from modallabs.checkpoint import SENTINEL, _resolve_existing_checkpoint, final_checkpoint_path
from modallabs.metrics import read_metrics


# ---------- helpers ----------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_secs(s: Optional[float]) -> str:
    if s is None:
        return "-"
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def _fmt_metric(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _is_run_dir(path: Path) -> bool:
    """A leaf run dir has a metrics.jsonl, a status.json, or the sentinel."""
    p = Path(path)
    return any((p / f).exists() for f in (SENTINEL, "metrics.jsonl", "status.json"))


# ---------- data extraction ----------

def _run_summary_from_dir(run_dir: Path) -> Dict[str, Any]:
    """Build a one-run summary dict from a leaf run directory's artifacts.

    Prefers the done-sentinel (written only on clean completion), falls
    back to status.json (which is also written for failed/running runs).
    """
    run_dir = Path(run_dir)
    name = run_dir.name
    done = _load_json(run_dir / SENTINEL)
    status = _load_json(run_dir / "status.json")
    src = done or status or {}

    phase = src.get("phase") or ("succeeded" if done else "unknown")
    # If only status.json exists and it says "running" but the sentinel is
    # absent, the run either crashed or is still in flight; surface that.
    if done is None and status is not None and status.get("phase") == "running":
        phase = "running-or-crashed"

    n_epochs = src.get("n_epochs")
    if n_epochs is None:
        # Count distinct "epoch"-kind records in metrics.jsonl.
        epochs_seen = {
            r.get("epoch") for r in read_metrics(run_dir / "metrics.jsonl")
            if r.get("kind") == "epoch"
        }
        n_epochs = len(epochs_seen) if epochs_seen else None

    ckpt = _resolve_existing_checkpoint(final_checkpoint_path(run_dir))
    return {
        "name": name,
        "phase": phase,
        "best_metric": src.get("best_metric"),
        "n_epochs": n_epochs,
        "elapsed_sec": src.get("elapsed_sec"),
        "type": src.get("type"),
        "error": src.get("error"),
        "checkpoint_path": str(ckpt) if ckpt.exists() else None,
        "_run_dir": str(run_dir),
    }


def collect(path: Path) -> Dict[str, Any]:
    """Return a structured report for `path` (a run-group dir or a run dir)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such path: {path}")

    if _is_run_dir(path):
        run = _run_summary_from_dir(path)
        epochs = [
            {"epoch": r.get("epoch"), **(r.get("payload") or {})}
            for r in read_metrics(path / "metrics.jsonl")
            if r.get("kind") == "epoch"
        ]
        return {"kind": "run", "run_id": path.parent.name, "run": run, "epochs": epochs}

    # Run-group directory: prefer summary.json; otherwise scan subdirs.
    summary = _load_json(path / "summary.json")
    if summary and isinstance(summary.get("runs"), list) and summary["runs"]:
        runs = [dict(r) for r in summary["runs"]]
        # The orchestrator's in-memory result dicts omit a few fields the
        # per-run done-sentinel carries (n_epochs, type). Backfill from
        # each run's sentinel so the table is complete.
        for r in runs:
            sub = path / str(r.get("name", ""))
            if not sub.is_dir():
                continue
            sent = _load_json(sub / SENTINEL) or {}
            for k in ("n_epochs", "type", "best_metric"):
                if r.get(k) is None and sent.get(k) is not None:
                    r[k] = sent[k]
            if not r.get("checkpoint_path"):
                ckpt = _resolve_existing_checkpoint(final_checkpoint_path(sub))
                if ckpt.exists():
                    r["checkpoint_path"] = str(ckpt)
    else:
        runs = [
            _run_summary_from_dir(d)
            for d in sorted(p for p in path.iterdir() if p.is_dir())
            if _is_run_dir(d)
        ]
        if not runs:
            raise ValueError(
                f"{path} has no summary.json and no run subdirectories -- "
                f"is it a runs/<run_id> directory?"
            )
    n_ok = sum(1 for r in runs if r.get("phase") == "succeeded")
    n_fail = sum(1 for r in runs if r.get("phase") in ("failed", "interrupted",
                                                       "running-or-crashed"))
    return {
        "kind": "group",
        "run_id": (summary or {}).get("run_id", path.name),
        "elapsed_sec": (summary or {}).get("elapsed_sec"),
        "n_runs": len(runs),
        "n_succeeded": n_ok,
        "n_failed": n_fail,
        "runs": runs,
    }


# ---------- rendering ----------

def _render_group(rep: Dict[str, Any]) -> str:
    runs: List[Dict[str, Any]] = rep["runs"]
    # The CHECKPOINT column shows just the filename -- the run name (first
    # column) already identifies the runs/<run_id>/<name>/ directory.
    rows = [("RUN", "PHASE", "TYPE", "EPOCHS", "BEST", "ELAPSED", "CHECKPOINT")]
    for r in runs:
        ckpt = r.get("checkpoint_path")
        rows.append((
            str(r.get("name", "?")),
            str(r.get("phase", "?")),
            str(r.get("type") or "-"),
            str(r.get("n_epochs") if r.get("n_epochs") is not None else "-"),
            _fmt_metric(r.get("best_metric")),
            _fmt_secs(r.get("elapsed_sec")),
            (Path(ckpt).name if ckpt else "-"),
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    lines.append(f"run_id: {rep.get('run_id')}   "
                 f"runs: {rep.get('n_runs')}   "
                 f"succeeded: {rep.get('n_succeeded')}   "
                 f"failed: {rep.get('n_failed')}   "
                 f"wall-clock: {_fmt_secs(rep.get('elapsed_sec'))}")
    lines.append("")
    for ri, row in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(row))))
    # Surface errors verbatim below the table.
    errs = [(r.get("name"), r.get("error")) for r in runs if r.get("error")]
    if errs:
        lines.append("")
        lines.append("errors:")
        for name, err in errs:
            lines.append(f"  {name}: {err}")
    return "\n".join(lines)


def _render_run(rep: Dict[str, Any]) -> str:
    run = rep["run"]
    lines = [
        f"run_id: {rep.get('run_id')}   run: {run.get('name')}",
        f"  phase:      {run.get('phase')}",
        f"  type:       {run.get('type') or '-'}",
        f"  epochs:     {run.get('n_epochs') if run.get('n_epochs') is not None else '-'}",
        f"  best:       {_fmt_metric(run.get('best_metric'))}",
        f"  elapsed:    {_fmt_secs(run.get('elapsed_sec'))}",
        f"  checkpoint: {run.get('checkpoint_path') or '-'}",
    ]
    if run.get("error"):
        lines.append(f"  error:      {run['error']}")
    epochs = rep.get("epochs") or []
    if epochs:
        lines.append("")
        lines.append("per-epoch metrics:")
        # Column union over a stable, useful subset.
        keys: List[str] = []
        for e in epochs:
            for k in e:
                if k != "epoch" and k not in keys:
                    keys.append(k)
        # Keep it readable: cap to a sane number of columns.
        prefer = [k for k in keys if k.startswith(("train/", "val/"))
                  or k in ("monitor", "is_best", "train_sec", "eval_sec")]
        keys = (prefer or keys)[:8]
        header = ["epoch", *keys]
        rows = [header]
        for e in epochs:
            rows.append([str(e.get("epoch"))] + [_fmt_metric(e.get(k)) for k in keys])
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        for ri, row in enumerate(rows):
            lines.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
            if ri == 0:
                lines.append("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    return "\n".join(lines)


def render(rep: Dict[str, Any]) -> str:
    return _render_run(rep) if rep.get("kind") == "run" else _render_group(rep)


# ---------- CLI ----------

def main(argv: Optional[List[str]] = None) -> int:
    av = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="modallabs-report",
        description="Summarize a modallabs run directory "
                    "(runs/<run_id> or runs/<run_id>/<run_name>).",
    )
    parser.add_argument("path", help="path to a runs/<run_id> dir or a single run dir")
    parser.add_argument("--json", action="store_true", help="emit the structured report as JSON")
    args = parser.parse_args(av)

    try:
        rep = collect(Path(args.path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"modallabs-report: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render(rep))

    # Exit non-zero if any run did not succeed -- handy for CI / scripts.
    if rep.get("kind") == "group":
        return 0 if rep.get("n_failed", 0) == 0 else 1
    return 0 if rep["run"].get("phase") == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
