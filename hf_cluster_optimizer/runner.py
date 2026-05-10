"""hf_cluster_optimizer — single-run executor.

`train_one(run_cfg, run_id, output_root)` runs ONE Trainer end-to-end:
construct -> setup -> train epochs -> save final/best checkpoints ->
write done sentinel.

Returns a structured result dict the orchestrator aggregates.

This module is the cross-platform pivot point: local processes call
train_one directly; Modal Lab's `@app.function(...)` wraps the same
train_one. No code path divergence.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from hf_cluster_optimizer.base import Trainer, TrainerSetup
from hf_cluster_optimizer.checkpoint import (
    SENTINEL,
    _resolve_existing_checkpoint,
    best_checkpoint_path,
    final_checkpoint_path,
    is_done,
    write_done,
)
from hf_cluster_optimizer.metrics import MetricsWriter
from hf_cluster_optimizer.registry import get as registry_get
from hf_cluster_optimizer.seed import set_global_seed

# Importing this package fires every Trainer's @register decorator. Without
# this, a caller using `train_one` directly (no orchestrator) would see an
# empty registry and KeyError on the first cfg type lookup. The orchestrator
# (concurrent_train.py) and the Modal worker (_remote in modal_app.py) also
# import hf_cluster_optimizer.models, but doing it here makes runner.py the single
# source of registration truth for any caller -- subprocess, modal function,
# or direct Python invocation. Cost is one one-time import.
import hf_cluster_optimizer.models  # noqa: F401  -- side-effect: register all built-in trainers


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_device(requested: Optional[str]) -> str:
    """Resolve device string. Accepts: 'cuda', 'cuda:N', 'cpu', 'mps', 'auto'.

    'auto' picks cuda if available, else mps, else cpu.
    """
    req = (requested or "auto").lower()
    try:
        import torch
        if req == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if req.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return req
    except Exception:
        return "cpu"


def _write_status(run_dir: Path, payload: Dict[str, Any]) -> None:
    status_path = run_dir / "status.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("updated_at", _utcnow_iso())
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _yaml_safe(obj: Any) -> Any:
    """Recursively coerce a cfg-like object into yaml.safe_dump-safe types.

    yaml.safe_dump rejects pathlib.Path, custom classes, sets, etc. by
    raising RepresenterError. Without this coercion, a cfg that includes
    a pathlib.Path (e.g. someone passing data_path through Path()) would
    crash _write_resolved_cfg AFTER the run started -- writing a partial
    status.json but no resolved cfg, then unwinding via the exception
    handler. Coerce defensively into JSON-ish primitives.
    """
    from pathlib import Path as _PPath
    import datetime as _dt
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, _PPath):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_yaml_safe(v) for v in sorted(obj, key=lambda x: str(x))]
    # Last resort -- repr instead of crashing. Keeps the reproducibility
    # signal even when the user passed a custom class through cfg.
    return repr(obj)


def _write_resolved_cfg(run_dir: Path, cfg: Dict[str, Any]) -> None:
    safe = _yaml_safe(cfg)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(safe, sort_keys=False), encoding="utf-8",
    )


class _Tee:
    """File-like that writes to multiple sinks. Used to mirror stdout to log.txt."""

    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, data: str) -> int:
        for s in self._sinks:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._sinks:
            try:
                s.flush()
            except Exception:
                pass


def train_one(
    run_cfg: Dict[str, Any],
    *,
    run_id: str,
    output_root: Path,
    resume: bool = False,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """Train one model end-to-end.

    Args:
        run_cfg: dict with required keys {name, type, config}; optional
            {seed, device, epochs}. `config` is the Trainer-specific cfg.
        run_id: parent grouping id; output lands at
            output_root / run_id / run_cfg["name"] /.
        output_root: root output directory.
        resume: if True and the run's done-sentinel exists, skip and
            report success. If False, remove any existing sentinel and
            re-run from scratch.
        force_cpu: override device to cpu (used for smoke tests).

    Returns:
        dict {name, phase, elapsed_sec, error?, best_metric?, ...}.
    """
    name = str(run_cfg["name"])
    rtype = str(run_cfg["type"])
    seed = int(run_cfg.get("seed", 0))
    epochs_override = run_cfg.get("epochs")
    cfg = dict(run_cfg.get("config", {}))
    if epochs_override is not None:
        cfg.setdefault("epochs", int(epochs_override))

    run_dir = Path(output_root) / run_id / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume and is_done(run_dir):
        # Defensive: a sentinel without its checkpoint means the user (or
        # a cleanup tool) deleted the checkpoint after the run finished.
        # Reporting `succeeded` here would let downstream `load_checkpoint`
        # silently fail. Re-run instead -- the cost is one fresh training
        # pass; the alternative is a misleading "succeeded" with a missing
        # artifact, which is the silent-cloaking pattern we want to avoid.
        ckpt = final_checkpoint_path(run_dir)
        # Trainer types that save to a directory (HF save_pretrained) drop
        # the .pt suffix; check both the suffixed file AND the bare-name
        # directory before declaring success.
        ckpt_dir = ckpt.with_suffix("")
        if ckpt.exists() or ckpt_dir.exists():
            logger.info("hfco: skipping %s (already done)", name)
            return {
                "name": name,
                "phase": "succeeded",
                "skipped": True,
                "elapsed_sec": 0.0,
                "checkpoint_path": str(ckpt if ckpt.exists() else ckpt_dir),
            }
        logger.warning(
            "hfco: %s has done-sentinel but no checkpoint at %s "
            "(possibly deleted); re-running to restore the artifact",
            name, ckpt,
        )
        # Fall through to fresh re-run; sentinel is removed below.

    # Fresh run -- clear any stale sentinel from prior failed attempts
    sentinel = run_dir / SENTINEL
    if sentinel.exists():
        sentinel.unlink()

    log_path = run_dir / "log.txt"
    log_fh = log_path.open("a", encoding="utf-8")
    log_fh.write(f"\n=== hfco run start {_utcnow_iso()} ===\n")

    metrics_path = run_dir / "metrics.jsonl"
    started_at = time.time()
    _write_status(run_dir, {
        "phase": "running",
        "name": name,
        "type": rtype,
        "seed": seed,
        "started_at": _utcnow_iso(),
    })
    _write_resolved_cfg(run_dir, run_cfg)

    result: Dict[str, Any] = {
        "name": name,
        "type": rtype,
        "phase": "failed",
        "error": None,
        "elapsed_sec": 0.0,
    }
    best_metric: Optional[float] = None
    last_epoch_metrics: Dict[str, float] = {}

    try:
        with MetricsWriter(metrics_path) as metric_fn:
            def log_fn(msg: str) -> None:
                line = f"{_utcnow_iso()} {msg}\n"
                log_fh.write(line)
                log_fh.flush()
                logger.info("[%s] %s", name, msg)

            log_fn(f"resolving device, requested={run_cfg.get('device', 'auto')}")
            device = "cpu" if force_cpu else _resolve_device(run_cfg.get("device"))
            log_fn(f"device={device}")

            log_fn(f"set_global_seed({seed})")
            set_global_seed(seed)

            log_fn(f"registry.get({rtype!r})")
            cls = registry_get(rtype)
            log_fn(f"constructing {cls.__name__}")
            trainer: Trainer = cls.from_config(cfg)

            setup = TrainerSetup(
                config=cfg,
                seed=seed,
                device=device,
                output_dir=run_dir,
                log_fn=log_fn,
                metric_fn=metric_fn.write,
            )
            log_fn("trainer.setup()")
            trainer.setup(setup)

            n_epochs = int(cfg.get("epochs", trainer.num_epochs()))
            for epoch in range(n_epochs):
                log_fn(f"epoch {epoch}/{n_epochs} train")
                t_train_start = time.time()
                n_train = 0
                for batch in trainer.train_iter():
                    step_res = trainer.train_step(batch)
                    n_train += int(step_res.n_examples or 0)
                    metric_fn.write("train", step_res.metrics, epoch=epoch)
                t_train = time.time() - t_train_start

                log_fn(f"epoch {epoch}/{n_epochs} eval")
                t_eval_start = time.time()
                n_eval = 0
                for batch in trainer.eval_iter():
                    step_res = trainer.eval_step(batch)
                    n_eval += int(step_res.n_examples or 0)
                    metric_fn.write("eval", step_res.metrics, epoch=epoch)
                t_eval = time.time() - t_eval_start

                ep_res = trainer.epoch_summary(epoch)
                metric_fn.write("epoch", {
                    **{f"train/{k}": v for k, v in ep_res.train_metrics.items()},
                    **{f"val/{k}": v for k, v in ep_res.val_metrics.items()},
                    "is_best": int(bool(ep_res.is_best)),
                    "monitor": (
                        float(ep_res.monitor_value)
                        if ep_res.monitor_value is not None else float("nan")
                    ),
                    "n_train_examples": n_train,
                    "n_eval_examples": n_eval,
                    "train_sec": t_train,
                    "eval_sec": t_eval,
                }, epoch=epoch)
                last_epoch_metrics = {
                    **{f"val_{k}": v for k, v in ep_res.val_metrics.items()},
                }
                if ep_res.is_best:
                    best_path = best_checkpoint_path(run_dir)
                    log_fn(f"epoch {epoch}: new best -> {best_path}")
                    trainer.save_checkpoint(best_path)
                    if ep_res.monitor_value is not None:
                        best_metric = float(ep_res.monitor_value)

            final_path = final_checkpoint_path(run_dir)
            log_fn(f"saving final checkpoint -> {final_path}")
            trainer.save_checkpoint(final_path)

            log_fn("trainer.teardown()")
            try:
                trainer.teardown()
            except Exception as exc:
                log_fn(f"teardown raised (non-fatal): {exc}")
            # Defensive GPU memory release. The subprocess boundary
            # (concurrent_train) and Modal container teardown also free
            # GPU memory, but if a Trainer is invoked in-process (smoke
            # test, custom script, sequential workers=1) without this,
            # the next run can OOM on memory the previous one leaked.
            try:
                import torch  # local import: torch may not be installed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

            elapsed = time.time() - started_at
            # Resolve actual on-disk checkpoint paths -- Trainers that save
            # to .joblib / .txt / .json / .cbm or to a save_pretrained dir
            # write a different filename than final_checkpoint_path returned
            # (which assumes .pt). Without this, callers get a path that
            # silently does not exist on disk for sklearn / LightGBM / etc.
            actual_final = _resolve_existing_checkpoint(final_path)
            actual_best = _resolve_existing_checkpoint(best_checkpoint_path(run_dir))
            result.update({
                "phase": "succeeded",
                "elapsed_sec": elapsed,
                "best_metric": best_metric,
                "checkpoint_path": str(actual_final),
                "best_checkpoint_path": str(actual_best),
                "last_epoch_metrics": last_epoch_metrics,
            })
            write_done(run_dir, {
                "name": name,
                "type": rtype,
                "phase": "succeeded",
                "elapsed_sec": elapsed,
                "best_metric": best_metric,
                "n_epochs": n_epochs,
                "finished_at": _utcnow_iso(),
            })
            _write_status(run_dir, {
                "phase": "succeeded",
                "name": name,
                "type": rtype,
                "started_at": _utcnow_iso(),
                "finished_at": _utcnow_iso(),
                "elapsed_sec": elapsed,
                "best_metric": best_metric,
            })

    except KeyboardInterrupt:
        result.update({
            "phase": "interrupted",
            "error": "KeyboardInterrupt",
            "elapsed_sec": time.time() - started_at,
        })
        _write_status(run_dir, {**result, "interrupted_at": _utcnow_iso()})
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        log_fh.write(tb + "\n")
        result.update({
            "phase": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "elapsed_sec": time.time() - started_at,
        })
        _write_status(run_dir, result)
        # Do NOT write done sentinel -- let --resume re-attempt this run.
    finally:
        try:
            log_fh.close()
        except Exception:
            pass

    return result


__all__ = ["train_one"]
