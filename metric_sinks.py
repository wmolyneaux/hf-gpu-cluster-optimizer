"""modallabs -- optional external metric sinks (TensorBoard / Weights & Biases).

The canonical metrics record is always `runs/<run_id>/<run_name>/metrics.jsonl`
(see `modallabs.metrics`). A run may *additionally* mirror every metric
line into TensorBoard and/or Weights & Biases by setting a `logger:`
field on the run (or the `MODALLABS_LOGGER` env var):

    runs:
      - name: my_run
        type: hf_causal_lm
        logger: tensorboard            # or: wandb  | "tensorboard,wandb"
        config: {...}

    # or with backend options:
        logger:
          tensorboard: true
          wandb:
            project: my-project
            entity: my-team

These are best-effort: a requested backend that isn't installed is
skipped with a log line, the run continues, and `metrics.jsonl` is
always written regardless. `pip install -e .[tensorboard]` /
`pip install -e .[wandb]` to enable them.

The public entry point is `make_metrics_writer(...)`, which returns
either a plain `MetricsWriter` (no extra sinks -- identical behavior to
before this module existed) or a `CompositeMetricsWriter` that fans
`.write(...)` out to the JSONL file plus each extra sink.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modallabs.metrics import MetricsWriter

logger = logging.getLogger(__name__)

# A logger spec is a string ("tensorboard", "wandb", "tensorboard,wandb"),
# a list of such names, or a dict {backend: True | {opts}}.
LoggerSpec = Union[str, List[str], Dict[str, Any], None]


def _scalars(payload: Dict[str, Any]) -> Dict[str, float]:
    """Keep only the finite numeric entries of a metric payload."""
    import math
    out: Dict[str, float] = {}
    for k, v in (payload or {}).items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def _parse_spec(spec: LoggerSpec) -> Dict[str, Dict[str, Any]]:
    """Normalize a logger spec to {backend_name: options_dict}."""
    if not spec:
        return {}
    if isinstance(spec, str):
        return {name.strip().lower(): {} for name in spec.split(",") if name.strip()}
    if isinstance(spec, (list, tuple)):
        return {str(name).strip().lower(): {} for name in spec if str(name).strip()}
    if isinstance(spec, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in spec.items():
            key = str(k).strip().lower()
            if v in (False, None):
                continue
            out[key] = dict(v) if isinstance(v, dict) else {}
        return out
    logger.warning("modallabs: ignoring unrecognized logger spec %r", spec)
    return {}


class _Sink:
    """Interface for an extra metric sink. All methods are best-effort."""

    def write(self, kind: str, payload: Dict[str, Any], *, epoch: int, step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _TensorBoardSink(_Sink):
    def __init__(self, run_dir: Path, run_name: str, opts: Dict[str, Any]) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:  # pragma: no cover - depends on optional install
            try:
                from tensorboardX import SummaryWriter  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "logger=tensorboard requested but neither "
                    "torch.utils.tensorboard nor tensorboardX is available "
                    "(pip install -e .[tensorboard])"
                ) from exc
        log_dir = opts.get("log_dir") or str(Path(run_dir) / "tensorboard")
        self._w = SummaryWriter(log_dir=log_dir)
        logger.info("modallabs: tensorboard logging -> %s", log_dir)

    def write(self, kind: str, payload: Dict[str, Any], *, epoch: int, step: int) -> None:
        for k, v in _scalars(payload).items():
            self._w.add_scalar(f"{kind}/{k}", v, global_step=step)

    def close(self) -> None:
        try:
            self._w.flush()
            self._w.close()
        except Exception:
            pass


class _WandbSink(_Sink):
    def __init__(self, run_dir: Path, run_name: str, run_id: str, opts: Dict[str, Any]) -> None:
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "logger=wandb requested but the 'wandb' package is not "
                "installed (pip install -e .[wandb])"
            ) from exc
        self._wandb = wandb
        init_kwargs = dict(opts)
        init_kwargs.setdefault("project", "modallabs")
        init_kwargs.setdefault("name", run_name)
        init_kwargs.setdefault("group", run_id)
        init_kwargs.setdefault("dir", str(run_dir))
        init_kwargs.setdefault("reinit", True)
        # Don't let wandb hijack stdout/stderr inside a worker subprocess.
        init_kwargs.setdefault("settings", wandb.Settings(console="off"))
        self._run = wandb.init(**init_kwargs)
        logger.info("modallabs: wandb logging -> %s", getattr(self._run, "url", "(offline)"))

    def write(self, kind: str, payload: Dict[str, Any], *, epoch: int, step: int) -> None:
        data = {f"{kind}/{k}": v for k, v in _scalars(payload).items()}
        if data:
            data["epoch"] = epoch
            self._run.log(data, step=step)

    def close(self) -> None:
        try:
            self._run.finish()
        except Exception:
            pass


class CompositeMetricsWriter:
    """A MetricsWriter plus zero or more extra sinks.

    Owns the global-step counter so the JSONL file and the extra sinks
    all see the same `step`. Exposes the same `.write(kind, payload, *,
    epoch=-1, step=None)` surface and context-manager protocol as
    `MetricsWriter`, so it is a drop-in for it.
    """

    def __init__(self, primary: MetricsWriter, extras: List[_Sink]) -> None:
        self._primary = primary
        self._extras = extras
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
        # The JSONL file is authoritative -- write it first and let any
        # exception propagate (same contract as MetricsWriter).
        self._primary.write(kind, payload, epoch=epoch, step=step)
        for s in self._extras:
            try:
                s.write(kind, payload, epoch=epoch, step=step)
            except Exception as exc:  # noqa: BLE001 -- a flaky sink must not kill the run
                logger.debug("modallabs: metric sink %s.write failed: %s",
                             type(s).__name__, exc)

    def close(self) -> None:
        for s in self._extras:
            s.close()
        self._primary.close()

    def __enter__(self) -> "CompositeMetricsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_metrics_writer(
    metrics_path: Path,
    *,
    run_id: str,
    run_name: str,
    run_dir: Path,
    logger_spec: LoggerSpec = None,
) -> Union[MetricsWriter, CompositeMetricsWriter]:
    """Build the metrics writer for a run.

    With no `logger_spec` (and no `MODALLABS_LOGGER` env var) this returns
    a plain `MetricsWriter` -- behavior is exactly as before. Otherwise it
    returns a `CompositeMetricsWriter` that also mirrors into TensorBoard
    and/or wandb. Unavailable backends are skipped with a log line.
    """
    primary = MetricsWriter(metrics_path)

    spec = logger_spec
    if not spec:
        env = os.environ.get("MODALLABS_LOGGER", "").strip()
        spec = env or None
    backends = _parse_spec(spec)
    if not backends:
        return primary

    run_dir = Path(run_dir)
    extras: List[_Sink] = []
    for name, opts in backends.items():
        try:
            if name in ("tensorboard", "tb"):
                extras.append(_TensorBoardSink(run_dir, run_name, opts))
            elif name in ("wandb", "wb"):
                extras.append(_WandbSink(run_dir, run_name, run_id, opts))
            else:
                logger.warning("modallabs: unknown logger backend %r (ignored)", name)
        except Exception as exc:
            logger.warning("modallabs: logger backend %r unavailable, skipping: %s",
                           name, exc)

    if not extras:
        return primary
    return CompositeMetricsWriter(primary, extras)


__all__ = ["CompositeMetricsWriter", "make_metrics_writer", "LoggerSpec"]
