# Porting a new model into HF Cluster Optimizer

HF Cluster Optimizer is a thin contract around `Trainer`. Adding a new model is
three steps. We'll work through them with a tiny example.

## The contract

Open `hf_cluster_optimizer/base.py` and you'll see the full `Trainer` interface:
nine abstract methods + an optional `teardown`. The framework calls
them in this order:

```
cls.from_config(cfg) -> trainer
trainer.setup(setup) -> None
for epoch in range(N):
    for batch in trainer.train_iter():
        trainer.train_step(batch)
    for batch in trainer.eval_iter():
        trainer.eval_step(batch)
    result = trainer.epoch_summary(epoch)
    if result.is_best:
        trainer.save_checkpoint(out / "best_checkpoint")
trainer.save_checkpoint(out / "checkpoint")
trainer.teardown()
```

Everything outside that loop -- seed setup, log file, metrics file,
checkpoint paths, the per-run subprocess boundary -- is handled by
the framework.

## Step 1: implement the Trainer

Let's add a fictional `MyModelTrainer` that wraps a custom torch module.
Drop this in `hf_cluster_optimizer/models/my_model.py`:

```python
from pathlib import Path
from typing import Any, Dict, Iterable
import torch

from hf_cluster_optimizer.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from hf_cluster_optimizer.registry import register


@register("my_model")
class MyModelTrainer(Trainer):
    """One-line docstring -- always required."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.model = None
        self.opt = None
        self._best = None
        self._train_buf = []
        self._eval_buf = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MyModelTrainer":
        return cls(config)

    def setup(self, setup: TrainerSetup) -> None:
        self.device = setup.device
        # Build your model; respect setup.seed for any random init.
        self.model = build_my_model(**self.config).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        # Load data -- use hf_cluster_optimizer.data_io.load_table for parquet/CSV.

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        return iter(self.train_batches)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        return iter(self.val_batches)

    def train_step(self, batch: Any) -> TrainerStepResult:
        loss = ...  # your forward + backward
        m = {"loss": float(loss.item())}
        self._train_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=len(batch))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        m = ...  # eval metrics
        self._eval_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=len(batch))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        train_m = mean_of(self._train_buf)
        val_m = mean_of(self._eval_buf)
        monitor = float(val_m.get("acc", 0.0))
        is_best = self._best is None or monitor > self._best
        if is_best:
            self._best = monitor
        return TrainerEpochResult(
            train_metrics=train_m, val_metrics=val_m,
            is_best=is_best, monitor_value=monitor,
        )

    def save_checkpoint(self, path: Path) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "config": self.config}, path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
```

That's the whole port. The `@register("my_model")` decorator wires it
up so any orchestrator config with `type: my_model` will train your
model.

## Step 2: import the module so the registry fires

Add a single line to `hf_cluster_optimizer/models/__init__.py`:

```python
_safe_import("hf_cluster_optimizer.models.my_model")
```

The `_safe_import` wrapper means a missing optional dep (e.g., a
custom CUDA kernel that won't compile on macOS) downgrades to a clean
"skipped" rather than blocking the whole package import.

## Step 3: add a smoke test entry

Drop a tuple into `_LOCAL_CASES` in `hf_cluster_optimizer/tests/smoke.py`:

```python
("my_model", {
    # Minimal config that trains in a few seconds on CPU.
    "lr": 1e-3, "epochs": 1, "batch_size": 8, "n": 32,
}, ["torch"]),  # required modules
```

Run `python -m hf_cluster_optimizer.tests.smoke` to verify the new model trains
end-to-end + checkpoints + writes the done sentinel.

## Things to keep in mind

- **Honor the seed.** `setup.seed` is set globally before your trainer
  is constructed, but if you do additional random init inside `setup`,
  use a `torch.Generator().manual_seed(setup.seed)` to keep reruns
  bit-identical.
- **One-line docstrings.** Multi-paragraph essays drift from the code.
  Keep the docstring tight and let the type signatures speak.
- **No silent neutral fallback.** If your data path is missing or your
  model can't load, raise. The orchestrator catches and reports;
  swallowing the error to "return zeros" hides bugs.
- **`save_checkpoint` writes a path the framework chose.** Don't
  rename or move it; the orchestrator looks for `best_checkpoint*`
  and `checkpoint*` exactly where it told you.
- **`teardown` is a free hook** if you need to release a Triton
  context, close a database, etc. Default no-op is fine for most
  trainers.

## Burn-conscious Trainer rules (read before you ship a Trainer that runs on Modal)

When a Trainer runs on Modal, every second of `setup` / `train_iter` /
`train_step` is billed against a real GPU. Honor these rules so your
Trainer doesn't blow up the user's bill:

- **No unbounded `while True:` loops.** Every wait must have a budget.
  If you poll for a checkpoint, dataset, or HTTP resource, give up
  after a fixed number of attempts and raise.
- **No silent retry on dataset / network failures.** If
  `datasets.load_dataset(...)` fails, raise. Do NOT wrap it in
  `for _ in range(10): try ... except: time.sleep(60)`. That pattern
  silently turns a 5-minute failure into a 50-minute GPU bill.
- **No `time.sleep(N)` longer than 1 second in any hot path.**
  Each sleep on an A100 is roughly N cents. If you must sleep, sleep
  on CPU before `setup` (i.e., before the GPU is hot), not inside
  `train_step`.
- **No background heartbeat threads.** A `Thread(target=ping_loop)`
  that does `while True: time.sleep(60)` will hold the container
  alive past `train_one`'s return.
- **Eager checkpoint saves.** `save_checkpoint` should return promptly.
  If you compress / upload to a remote bucket, do it AFTER the
  framework returns from `train_one`, not during.
- **Free GPU memory in `teardown`.** The framework calls
  `torch.cuda.empty_cache()` defensively after `teardown`, but if your
  Trainer holds tensors in long-lived attributes, set them to `None`
  in `teardown` so empty_cache can actually reclaim them.
- **Fail loud on missing required config.** If your Trainer needs
  `cfg.data_path` or `cfg.tokenizer_name`, raise on missing values in
  `from_config` BEFORE `setup` starts allocating GPU memory. Failing
  at minute 0 is free; failing after 30 minutes of training has cost.

## Cost knobs your port inherits for free

Your new Trainer port automatically gets every HF Cluster Optimizer cost control;
you don't write any code to opt in. Document the right defaults for
YOUR model in your trainer's module docstring so users can copy-paste
them into their cfg without reading the framework source:

| Knob | Where it goes in the cfg | Default |
|---|---|---|
| GPU type | `runs[*].modal.gpu: T4` (or `L4`/`A10G`/`L40S`/`A100-40G`/`A100-80G`/`H100`/`H200`) | `T4` |
| Auto GPU selector | `runs[*].modal.gpu: auto` (string heuristic on hf_model_name) | off |
| Per-run hard timeout | `runs[*].modal.max_runtime_sec: 1800` (Modal-enforced; gates the cost ceiling) | `14400` (4 h) |
| Cost-preview hint | `runs[*].modal.est_sec_per_epoch: 60` (informational; does NOT gate ceiling) | `60` |
| Total cost ceiling | `export HFCO_MAX_USD=<dollars>` | `25.0` |

The dry-run preview is the single source of truth -- run it before any
launch:

```bash
python hf_cluster_optimizer/modal_app.py --config <your.yaml> --dry-run
```

You'll see per-run GPU type, per-run timeout (with `*` if overridden
from the default), worst-case cost (`max_runtime_sec * GPU rate`),
optimistic estimate, and the ceiling status. If the ceiling is
breached, the CLI exits with code `2` and prints `BLOCKED`.

## When in doubt

Look at `hf_cluster_optimizer/models/generic_torch.py` for the simplest possible
torch trainer, or `hf_cluster_optimizer/models/hf_transformer.py` for the
HuggingFace-AutoModel-driven one. Both follow the same contract; the
HF one happens to dispatch on `cfg.hf_task` to pick the right
AutoModel head.
