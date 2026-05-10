"""hf_cluster_optimizer — local concurrent orchestrator.

Parses the orchestrator cfg, dispatches each run as a separate process,
collects results, writes a consolidated summary.

Usage:
    python -m hf_cluster_optimizer.concurrent_train --config configs/all_models.yaml
    python -m hf_cluster_optimizer.concurrent_train --config X.yaml --resume
    python -m hf_cluster_optimizer.concurrent_train --config X.yaml --max-workers 2

Process-pool gives clean GPU/CUDA isolation per run -- a model that
OOMs only kills its own process.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import hf_cluster_optimizer.models  # noqa: F401  -- registers all built-in trainers
from hf_cluster_optimizer.checkpoint import (
    _resolve_existing_checkpoint,
    final_checkpoint_path,
    is_done as _ckpt_is_done,
)
from hf_cluster_optimizer.runner import train_one


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_count() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def _default_max_workers(n_runs: int) -> int:
    """Default = number of GPUs, falling back to min(n_runs, cpu_count())."""
    n_gpus = _gpu_count()
    if n_gpus > 0:
        return min(n_gpus, n_runs)
    cpu = mp.cpu_count() or 1
    return min(n_runs, max(1, cpu // 2))  # leave half the cores free


def _proc_target(args: tuple) -> Dict[str, Any]:
    """Worker entry point -- runs in subprocess. Must be top-level for pickling."""
    run_cfg, run_id, output_root, resume, force_cpu, gpu_idx = args
    if gpu_idx is not None:
        # Pin this worker to one GPU. Set BEFORE torch is imported in
        # the worker process.
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    return train_one(
        run_cfg,
        run_id=run_id,
        output_root=Path(output_root),
        resume=resume,
        force_cpu=force_cpu,
    )


def run(
    cfg_path: Path,
    *,
    output_root: Path = Path("runs"),
    resume: bool = False,
    max_workers: Optional[int] = None,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """Run an orchestrator config end-to-end. Returns the summary dict."""
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"orchestrator cfg must be a dict, got {type(cfg).__name__}")
    run_id = str(cfg.get("run_id") or _utcnow_iso().replace(":", "-"))
    runs: List[Dict[str, Any]] = list(cfg.get("runs") or [])
    if not runs:
        raise ValueError("orchestrator cfg has no `runs:` list")

    # Detect duplicate / blank run names BEFORE any subprocess spawns. Two
    # runs with the same `name` would share runs/<run_id>/<name>/ -- they
    # would clobber each other's metrics.jsonl, status.json, and checkpoints
    # while the orchestrator silently reported both as "succeeded". Fail
    # loud at config-load time. A blank name would land at runs/<run_id>/
    # and clobber the orchestrator summary.json.
    names = [str(r.get("name", "")) for r in runs]
    seen: Dict[str, int] = {}
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
    dups = sorted(nm for nm, c in seen.items() if nm and c > 1)
    if dups:
        raise ValueError(
            f"orchestrator cfg has duplicate run names {dups!r}; "
            f"each run must have a unique `name` (used as the runs/ subdir)."
        )
    blanks = sum(1 for nm in names if not nm)
    if blanks:
        raise ValueError(
            f"orchestrator cfg has {blanks} run(s) with missing/empty `name`; "
            f"every run requires a non-empty `name`."
        )

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_dir = out_root / run_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    # GPU pinning: round-robin assignment across available GPUs.
    n_gpus = _gpu_count()
    workers = int(max_workers if max_workers is not None else _default_max_workers(len(runs)))
    workers = max(1, min(workers, len(runs)))

    started = time.time()
    results: List[Dict[str, Any]] = []

    # --resume short-circuit -- skip done runs BEFORE submitting them to
    # the process pool. On a Modal-equivalent pay-per-second backend this
    # is the difference between zero burn (correct) and one full GPU
    # spin-up per already-done run (the bug we are preventing).
    pending_runs: List[Dict[str, Any]] = []
    if resume:
        for rc in runs:
            run_dir = out_root / run_id / str(rc["name"])
            if _ckpt_is_done(run_dir):
                # Verify the actual checkpoint exists on disk -- the
                # sentinel can be present without the artifact (manual
                # cleanup, partial-disk-fail). Resolve across known
                # Trainer suffixes (.pt / .joblib / .txt / .json / .cbm /
                # save_pretrained dir). If still missing, fall through
                # so the run executes again and re-emits the artifact.
                resolved = _resolve_existing_checkpoint(final_checkpoint_path(run_dir))
                if resolved.exists():
                    logger.info("hfco: pre-skipping %s (already done)", rc["name"])
                    results.append({
                        "name": str(rc["name"]),
                        "phase": "succeeded",
                        "skipped": True,
                        "elapsed_sec": 0.0,
                        "checkpoint_path": str(resolved),
                    })
                else:
                    logger.warning(
                        "hfco: %s has done-sentinel but no checkpoint "
                        "(possibly deleted); re-running to restore artifact",
                        rc["name"],
                    )
                    pending_runs.append(rc)
            else:
                pending_runs.append(rc)
    else:
        pending_runs = list(runs)

    args_list = []
    for i, rc in enumerate(pending_runs):
        gpu_idx = (i % n_gpus) if (n_gpus > 0 and not force_cpu) else None
        args_list.append((rc, run_id, str(out_root), resume, force_cpu, gpu_idx))

    if not args_list:
        logger.info("hfco: nothing to dispatch (all runs already done).")
    elif workers == 1:
        # Serial fallback -- easier to debug, no pool overhead.
        for a in args_list:
            try:
                results.append(_proc_target(a))
            except KeyboardInterrupt:
                # Surface Ctrl-C as a partial result so the caller can
                # see what was completed; re-raise to abort the orchestrator.
                results.append({
                    "name": a[0]["name"],
                    "phase": "interrupted",
                    "error": "KeyboardInterrupt",
                    "elapsed_sec": 0.0,
                })
                raise
    else:
        ctx = mp.get_context("spawn")  # spawn on Windows + clean GPU init on Linux
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_proc_target, a): a[0]["name"] for a in args_list}
            try:
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        results.append({
                            "name": name,
                            "phase": "failed",
                            "error": f"executor: {type(exc).__name__}: {exc}",
                            "elapsed_sec": 0.0,
                        })
            except KeyboardInterrupt:
                # User Ctrl-C -- cancel pending futures so we stop billing
                # for any worker the pool is about to start. In-flight
                # workers will receive SIGTERM via the pool's __exit__.
                logger.warning("hfco: KeyboardInterrupt -- cancelling pending futures.")
                for fut in futures:
                    fut.cancel()
                raise

    elapsed = time.time() - started
    n_succeeded = sum(1 for r in results if r.get("phase") == "succeeded")
    n_failed = sum(1 for r in results if r.get("phase") == "failed")
    n_interrupted = sum(1 for r in results if r.get("phase") == "interrupted")
    summary = {
        "run_id": run_id,
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(timespec="seconds"),
        "finished_at": _utcnow_iso(),
        "elapsed_sec": elapsed,
        "n_runs": len(runs),
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "n_interrupted": n_interrupted,
        "max_workers": workers,
        "n_gpus_visible": n_gpus,
        "force_cpu": bool(force_cpu),
        "resumed": bool(resume),
        "runs": results,
    }
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "hfco: run_id=%s done. succeeded=%d failed=%d interrupted=%d elapsed=%.1fs",
        run_id, n_succeeded, n_failed, n_interrupted, elapsed,
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    av = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="HF Cluster Optimizer concurrent training orchestrator (local).",
    )
    parser.add_argument("--config", required=True, help="path to orchestrator yaml")
    parser.add_argument("--output-root", default="runs", help="output directory root")
    parser.add_argument("--resume", action="store_true", help="skip already-done runs")
    parser.add_argument("--max-workers", type=int, default=None, help="parallel run cap")
    parser.add_argument("--force-cpu", action="store_true", help="force device=cpu")
    parser.add_argument("--log-level", default="INFO", help="python logging level")
    args = parser.parse_args(av)

    # Resolve --log-level. `getattr` with `logging.INFO` as fallback would
    # silently swallow typos ("--log-level NOPE" -> INFO). Surface a stderr
    # warning so the user sees their typo without making the orchestrator
    # fail-loud on a cosmetic setting.
    _level_name = (args.log_level or "INFO").upper()
    _level = getattr(logging, _level_name, None)
    if not isinstance(_level, int):
        print(
            f"hfco/concurrent_train: unknown --log-level {args.log_level!r}; "
            f"falling back to INFO. Valid: DEBUG INFO WARNING ERROR CRITICAL.",
            file=sys.stderr,
        )
        _level = logging.INFO
    logging.basicConfig(
        level=_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = run(
        Path(args.config),
        output_root=Path(args.output_root),
        resume=args.resume,
        max_workers=args.max_workers,
        force_cpu=args.force_cpu,
    )
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
