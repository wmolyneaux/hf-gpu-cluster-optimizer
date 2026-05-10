"""hf_cluster_optimizer — abstract Trainer interface.

Every model type registered in hf_cluster_optimizer implements the Trainer protocol
defined here. The interface is intentionally tiny: 6 methods + 1
classmethod. Anything more model-specific lives inside the Trainer
implementation.

Design: pure-Python abstract base; no torch/numpy imports at module
load so the registry can enumerate available trainers without forcing
heavy deps. Heavy deps are imported inside trainer methods.

ASCII only. No emojis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TrainerSetup:
    """Inputs to Trainer.setup -- everything a Trainer needs to begin."""
    config: Dict[str, Any]
    seed: int
    device: str            # "cuda", "cuda:0", "cpu", "mps"
    output_dir: Path       # runs/<run_id>/<run_name>/
    log_fn: Any            # callable(msg: str) -> None; structured log sink
    metric_fn: Any         # callable(step: int, kind: str, payload: dict) -> None


@dataclass
class TrainerStepResult:
    """One training-step result. Returned from train_step / eval_step."""
    metrics: Dict[str, float] = field(default_factory=dict)
    n_examples: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerEpochResult:
    """One full-epoch result. Returned from epoch_summary at end of epoch."""
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False
    monitor_value: Optional[float] = None  # the scalar used for "best" tracking


class Trainer(ABC):
    """Abstract Trainer for one model in a concurrent training run.

    Lifecycle (called by runner.train_one):
        1. cls.from_config(cfg)         -> Trainer
        2. trainer.setup(setup)         -> None     (single call)
        3. for epoch in range(N):
              for batch in trainer.train_iter():
                  trainer.train_step(batch)
              for batch in trainer.eval_iter():
                  trainer.eval_step(batch)
              result = trainer.epoch_summary(epoch)
              if result.is_best:
                  trainer.save_checkpoint(out / "best_checkpoint")
        4. trainer.save_checkpoint(out / "checkpoint")
        5. trainer.teardown()

    The framework handles seeding, logging, metric writing, checkpoint
    paths, and the per-run process boundary. The Trainer concentrates
    on the model + data.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "Trainer":
        """Construct the trainer from a config dict. Validate cfg here."""

    @abstractmethod
    def setup(self, setup: TrainerSetup) -> None:
        """Prepare model, optimizer, loaders. Called once after construction.

        Implementations MUST honor `setup.seed` -- set torch / numpy /
        random seeds before instantiating any module that uses random
        initialization. Use hf_cluster_optimizer.seed.set_global_seed() for the
        canonical recipe.
        """

    @abstractmethod
    def train_iter(self) -> Iterable[Any]:
        """Yield training batches. Each batch is opaque to the framework."""

    @abstractmethod
    def eval_iter(self) -> Iterable[Any]:
        """Yield validation batches."""

    @abstractmethod
    def train_step(self, batch: Any) -> TrainerStepResult:
        """Execute one training step on `batch`. Returns metrics."""

    @abstractmethod
    def eval_step(self, batch: Any) -> TrainerStepResult:
        """Execute one eval step on `batch`. Returns metrics."""

    @abstractmethod
    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        """Produce the per-epoch summary. Aggregates step metrics.

        Implementations decide how is_best is computed (min/max of
        val loss, accuracy, custom). The framework logs the result and
        triggers best_checkpoint save when is_best=True.
        """

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Write checkpoint to disk.

        For torch models: torch.save(model.state_dict(), path) or
        torch.save({"model": ..., "optimizer": ..., "epoch": ...}, path).
        For sklearn/lightgbm: joblib.dump(model, path).
        For HF: model.save_pretrained(path); tokenizer.save_pretrained(path).

        Implementations choose the file extension. The framework will
        not append one.
        """

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Restore from checkpoint. Used by --resume."""

    def num_epochs(self) -> int:
        """Total epoch count. Default reads `config["epochs"]`."""
        cfg = getattr(self, "config", {}) or {}
        return int(cfg.get("epochs", 1))

    def teardown(self) -> None:
        """Optional cleanup. Default no-op. Called after final checkpoint."""
        return None


class FailFastTrainer(Trainer):
    """Mixin that raises on any non-finite metric.

    Use as base class when you want any NaN/Inf in train_loss to
    stop the run rather than silently produce a degenerate model.
    """

    def _assert_finite(self, metrics: Dict[str, float]) -> None:
        import math
        bad = [k for k, v in metrics.items()
               if isinstance(v, (int, float)) and not math.isfinite(v)]
        if bad:
            raise RuntimeError(
                f"FailFastTrainer: non-finite metrics {bad} in {metrics}"
            )


__all__ = [
    "Trainer",
    "FailFastTrainer",
    "TrainerSetup",
    "TrainerStepResult",
    "TrainerEpochResult",
]
