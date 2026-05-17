# HF GPU Cluster Optimizer

[![CI](https://github.com/wmolyneaux/hf-gpu-cluster-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/wmolyneaux/hf-gpu-cluster-optimizer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/wmolyneaux/hf-gpu-cluster-optimizer)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Train a fleet of ML models in one run. Locally on your dev box, or
> on Modal Labs cloud. Same code path, same checkpoints, same metrics.
>
> The Python package is still imported as `modallabs` (renaming an
> import name breaks every downstream `import` statement); the repo
> name is the discoverable product name.

`modallabs` is a thin, opinionated training harness: write a tiny
`Trainer` once, register it under a name, and now your model trains
concurrently alongside every other registered model from a single
YAML config. Heavy on HuggingFace support out of the box, with
escape hatches for any `torch.nn.Module` or sklearn estimator you
already have.

## Why you might want this

You probably already have one bespoke training script per model.
That's fine for one model. With five it's friction; with ten it's
your full-time job.

`modallabs` flattens the fleet:

- **One config, many models.** A single `concurrent_train --config X.yaml`
  trains every entry in parallel. Each runs in its own subprocess,
  so a CUDA OOM in one doesn't take the others down.
- **HuggingFace native.** Every popular HF architecture trains through
  one wrapper. BERT, GPT, T5, LLaMA, ViT, Whisper, Wav2Vec2,
  Stable Diffusion encoder backbones -- pick the model, point at it
  by name, and go.
- **Modal Labs deployable.** The same config runs on Modal cloud
  with a one-line change. GPU type per run, hard cost ceiling,
  pre-flight cost preview, no warm-pool keep-alive.
- **Deterministic.** `seed=42` + same cfg produces the same metrics,
  every time. The smoke test enforces this within `1e-6`.
- **Crash-isolated.** A model that segfaults or OOMs only kills its
  own process. The other models keep training.
- **Resumable.** `--resume` skips runs that already completed; failed
  runs restart from scratch.
- **Friendly checkpoints.** HuggingFace models save via
  `save_pretrained` so you can `push_to_hub` directly.

## Supported architectures

Out of the box:

| Architecture family | Example HF checkpoint | modallabs `type` |
|---|---|---|
| BERT-style encoders | `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base` | `hf_sequence_classification` |
| GPT / Llama / Mistral | `gpt2`, `meta-llama/Llama-3.2-1B`, `mistralai/Mistral-7B-v0.3` | `hf_causal_lm` |
| T5 / BART / Pegasus | `t5-small`, `facebook/bart-base`, `google/pegasus-xsum` | `hf_seq2seq` |
| ViT / Swin / ConvNeXt | `google/vit-base-patch16-224`, `microsoft/swin-tiny-patch4-window7-224` | `hf_image_classification` |
| Whisper / Wav2Vec2 | `openai/whisper-tiny`, `facebook/wav2vec2-base` | `hf_whisper` / `hf_audio_classification` |
| Token classification | `dslim/bert-base-NER` | `hf_token_classification` |
| Question answering | `distilbert-base-cased-distilled-squad` | `hf_qa` |
| Embedding (any AutoModel) | `sentence-transformers/all-MiniLM-L6-v2` | `hf_embedding` |
| Stable Diffusion family | (toy DDPM bundled; swap your own UNet via `torch_module`) | `diffusion` |
| Generic `torch.nn.Module` | (your own model class) | `torch_module` |
| Generic sklearn estimator | `RandomForest`, `GradientBoosting`, etc. | `sklearn` |
| LightGBM / XGBoost / CatBoost | -- | `lightgbm` / `xgboost` / `catboost` |
| Offline RL (Q-learning) | DQN with double-Q + dueling head | `q_learning` |

If your architecture isn't on the list, see `PORTING.md` -- adding
a new one is three steps and ~80 lines.

### Parameter-efficient fine-tuning (LoRA)

Any `hf_*` run can be wrapped in a PEFT adapter (LoRA by default, IA3
also supported) by adding a `peft:` block to its `config:`. Full
fine-tunes of 1B+ models don't fit on a consumer GPU; LoRA does. Only
the adapter is written to the checkpoint directory -- a few MB, not
the full model -- so it stays `push_to_hub`-friendly. The block is
inert when absent, so existing configs are unchanged. Needs the `peft`
extra (`pip install -e .[peft]`).

```yaml
runs:
  - name: llama_lora
    type: hf_causal_lm
    config:
      hf_model_name: meta-llama/Llama-3.2-1B
      epochs: 1
      batch_size: 4
      lr: 2e-4
      peft:
        method: lora            # or: ia3
        r: 16
        lora_alpha: 32
        lora_dropout: 0.05
        target_modules: [q_proj, v_proj]   # omit to let PEFT auto-infer
```

Everything except `method` is forwarded verbatim to the PEFT config
constructor, so any `LoraConfig` / `IA3Config` keyword works.

## Quick start (local)

### 1. Install

```bash
pip install -e .[hf]            # HuggingFace + torch
pip install -e .[full]          # everything (boosting + Modal + peft + tensorboard)
pip install -e .[peft]          # LoRA / IA3 parameter-efficient fine-tuning
pip install -e .[tensorboard]   # mirror metrics into TensorBoard
pip install -e .[wandb]         # mirror metrics into Weights & Biases
pip install -e .[dev]           # ruff + mypy + pre-commit + build
```

### 2. Pick a config

```yaml
# my_run.yaml
run_id: experiment_01
runs:
  - name: bert_classifier
    type: hf_sequence_classification
    seed: 42
    config:
      hf_model_name: bert-base-uncased
      n_classes: 2
      max_length: 128
      epochs: 3
      batch_size: 16
      lr: 2e-5

  - name: gpt2_finetune
    type: hf_causal_lm
    seed: 42
    config:
      hf_model_name: gpt2
      max_length: 256
      epochs: 1
      batch_size: 4
      lr: 5e-5

  - name: vit_classifier
    type: hf_image_classification
    seed: 42
    config:
      hf_model_name: google/vit-base-patch16-224
      n_classes: 10
      epochs: 5
      batch_size: 32
      lr: 1e-4
```

### 3. Train them all in parallel

```bash
python -m modallabs.concurrent_train --config my_run.yaml
# or, if you installed via pyproject.toml:
modallabs-train --config my_run.yaml
```

Each model trains in its own process; per-run status, metrics, and
checkpoints land in `runs/<run_id>/<run_name>/`.

### 4. Resume after a crash

```bash
modallabs-train --config my_run.yaml --resume
```

Successful runs are skipped; failed or interrupted runs restart from
scratch.

### 5. Summarize the results

```bash
modallabs-report runs/experiment_01
# point it at one run for the per-epoch metric history:
modallabs-report runs/experiment_01/bert_classifier
# machine-readable:
modallabs-report runs/experiment_01 --json
```

```
run_id: experiment_01   runs: 3   succeeded: 3   failed: 0   wall-clock: 42.2m

RUN              PHASE      TYPE                          EPOCHS  BEST    ELAPSED  CHECKPOINT
---------------  ---------  ----------------------------  ------  ------  -------  ----------------
bert_classifier  succeeded  hf_sequence_classification    3       0.873   6.9m     checkpoint
gpt2_finetune    succeeded  hf_causal_lm                  1       0.41    7.1m     checkpoint
vit_classifier   succeeded  hf_image_classification       5       0.612   28.2m    checkpoint
```

Exit code is non-zero if any run did not succeed, so it doubles as a
CI gate.

## Quick start (Modal Labs cloud)

### 1. Install + authenticate

```bash
pip install modal
modal token new
```

### 2. Pre-flight cost preview

```bash
python modallabs/modal_app.py --config my_run.yaml --dry-run
```

You'll see something like:

```
[DRY RUN] 3 runs queued (run_id=experiment_01)
   run #1: bert_classifier      gpu=T4        timeout=0.50h*  worst=0.50h ~= $0.18  (est=0.03h ~= $0.01)
   run #2: gpt2_finetune        gpu=L4        timeout=1.00h*  worst=1.00h ~= $0.50  (est=0.13h ~= $0.07)
   run #3: vit_classifier       gpu=T4        timeout=2.00h*  worst=2.00h ~= $0.72  (est=0.50h ~= $0.18)
   --
   Total WORST-CASE cost (every run hits its max_runtime_sec timeout): $1.40
   Cost ceiling (gates on worst-case): $25.00 (override via env MODALLABS_MAX_USD)
   Hard per-run timeout: 4.0h default (override per-run via cfg.modal.max_runtime_sec)
   * = per-run override (3 of 3 runs)

Proceed: re-run without --dry-run to actually launch.
```

If the worst-case total exceeds your `MODALLABS_MAX_USD` ceiling, you
get a `BLOCKED` line instead and the CLI exits with code `2` -- safe
for `set -e` shell pipelines and CI gates.

### 3. Launch

```bash
modal run modallabs/modal_app.py --config my_run.yaml
```

Each run gets its own `@app.function(gpu=...)` instance with a hard
4-hour timeout (configurable per run). Output is mirrored to a
Modal volume; sync back with:

```bash
modal volume get modallabs-runs runs/
```

## Cost controls (Modal)

Cloud GPU bills get out of hand fast. `modallabs` ships with five
guardrails on by default:

| Guardrail | Default | How to override |
|---|---|---|
| Per-run timeout (Modal-enforced) | 4 hours | `cfg.modal.max_runtime_sec` per run |
| Default GPU | `T4` (cheapest) | `cfg.modal.gpu` per run |
| Total-cost ceiling (worst-case) | `$25` across all runs | `export MODALLABS_MAX_USD=<dollars>` |
| Auto GPU selector | `T4` for <7B, `A10G` for 7-30B, `A100-80G` for >30B | Set `cfg.modal.gpu: auto` to opt in; explicit values still win |
| Warm-pool keep-alive | OFF (function tears down on completion) | not configurable -- containers always shut down |

The dry-run preview prints **worst-case cost** -- what you actually
pay if a model hangs and burns through its `max_runtime_sec` timeout.
The cost ceiling gates on that worst-case number, not the optimistic
`epochs * est_sec_per_epoch` estimate. If total worst-case exceeds
`MODALLABS_MAX_USD`, the launcher refuses to start.

## GPU burn minimization checklist

Every minute of an idle GPU is dollars burned. Read this before
launching a Modal run:

1. **Always start with `--dry-run`.** It prints worst-case cost per
   run and ceiling status. Review the breakdown before you spend a
   dollar.
2. **Set `MODALLABS_MAX_USD` as a hard kill-switch.**
   `export MODALLABS_MAX_USD=10` blocks any launch whose worst-case
   total exceeds $10. Make this your default in `~/.bashrc`.
3. **Use `--resume` to skip done runs.** Already-done runs are
   filtered BEFORE GPU allocation -- a resume on a fully-completed
   run_id costs zero GPU dollars. (A tiny CPU probe runs to consult
   the volume; that's pennies at most.)
4. **Iterate on CPU first.** `python -m modallabs.concurrent_train
   --config X.yaml --force-cpu --max-workers 1` runs the orchestrator
   end-to-end with no GPU allocations anywhere. Reserve GPU for runs
   you've already smoke-tested locally.
5. **Tighten `max_runtime_sec` per run.** Worst-case bill =
   `max_runtime_sec * GPU rate`. If your model trains in 20 minutes,
   set `max_runtime_sec: 1800`, not the 4-hour default. Modal enforces
   the timeout in C, not Python -- a hung CUDA op can't outrun it.
6. **Pin GPUs explicitly in production.** `cfg.modal.gpu: auto` is a
   string heuristic. For production, pick the smallest tier that fits
   and pin it (e.g., `gpu: T4`). T4 is ~$0.36/hr; A100-80G is
   ~$4.00/hr -- 11x burn for the wrong default.
7. **Failed runs do NOT auto-retry.** A run that raises is reported as
   `phase: failed`; the next launch is your decision. There is no
   silent retry that doubles your bill.
8. **`Ctrl-C` cancels in-flight futures.** The orchestrator catches
   `KeyboardInterrupt`, cancels pending Modal futures, and signals
   in-flight subprocess workers. Containers tear down and GPUs release.
9. **Monitor `runs/<run_id>/<name>/summary.json`.** After every run,
   `elapsed_sec` and `best_metric` land in the summary. Compare
   `elapsed_sec` against your `max_runtime_sec` to spot runs that are
   eating their full timeout (= probable hang, not progress).
10. **No warm-pool keep-alive.** `modallabs` deliberately does not
    pass `keep_warm=N` or `min_containers=N`. Containers exit on
    function return; your bill stops the second the run finishes.

## What lands on disk

For run `<run_name>` under run_id `<run_id>`:

```
runs/<run_id>/<run_name>/
  status.json              -- {phase, started_at, ...}
  manifest.json            -- provenance: git commit + dirty flag, resolved
                              library versions, config SHA-256, python/platform
  metrics.jsonl            -- one JSON line per train + eval step
  checkpoint.pt            -- final model checkpoint (or .joblib / dir; LoRA = adapter dir)
  best_checkpoint.pt       -- best-on-val checkpoint
  config_resolved.yaml     -- the exact cfg used
  log.txt                  -- captured stdout + stderr
  tensorboard/             -- present only if logger: tensorboard
  .modallabs_done          -- presence = run completed cleanly
```

Run `modallabs-report runs/<run_id>` for a per-run table, or
`modallabs-report runs/<run_id>/<run_name>` for that run's per-epoch
metric history.

The orchestrator writes a consolidated `runs/<run_id>/summary.json`
after every run finishes:

```json
{
  "run_id": "experiment_01",
  "n_runs": 3,
  "n_succeeded": 3,
  "n_failed": 0,
  "elapsed_sec": 2531.4,
  "runs": [
    {
      "name": "bert_classifier",
      "phase": "succeeded",
      "best_metric": 0.873,
      "checkpoint_path": "runs/experiment_01/bert_classifier/checkpoint.pt",
      "elapsed_sec": 412.3
    }
  ]
}
```

## External metric trackers (optional)

`metrics.jsonl` is always the source of truth. A run can *additionally*
mirror every metric line into TensorBoard and/or Weights & Biases by
setting `logger:` on the run (string `tensorboard` / `wandb` /
`tensorboard,wandb`, or a dict with backend options) -- or set the
`MODALLABS_LOGGER` env var as a default for all runs. Best-effort: a
backend that isn't installed is skipped with a log line and the run
continues. `pip install -e .[tensorboard]` / `.[wandb]` to enable.

```yaml
runs:
  - name: my_run
    type: hf_causal_lm
    logger: tensorboard          # writes runs/<run_id>/my_run/tensorboard/
    config: {...}
```

## Concurrency model

- Each cfg `runs:` entry becomes one process (local) or one Modal
  function (cloud).
- Local: `multiprocessing.Pool(processes=N)` where `N` defaults to
  the number of GPUs available (or `cpu_count()/2` if no CUDA).
- Modal: each function scales independently. No worker cap.
- Each subprocess is a clean Python instance -- no module state
  shared between runs. CUDA contexts are fresh; one run's leaked
  memory cannot bleed into the next.

## Determinism

Set a seed and you get the same metrics every run. `set_global_seed`
covers `random`, `numpy`, `torch` (CPU + CUDA), and `transformers`,
plus `torch.use_deterministic_algorithms(True)`. The smoke test
asserts identical metrics across two seeded runs within `1e-6` on
CPU. GPU determinism depends on cuDNN flags + CUDA version; we set
the deterministic-algorithm flag, but expect minor float drift across
GPU types.

## Crash isolation

Subprocess boundaries mean OOM, segfault, or unhandled exceptions in
one Trainer cannot block the others. The orchestrator captures the
exit and writes `phase: failed` + traceback to `status.json`. The
done-sentinel is only written on clean completion, so `--resume`
re-runs failed entries.

## Adding your own model

See `PORTING.md`. Three steps: implement the `Trainer` interface,
register it, add a smoke entry. About 80 lines for a typical port.

## Module layout

```
modallabs/
  base.py              -- abstract Trainer + result dataclasses
  registry.py          -- name -> Trainer class
  runner.py            -- single-run executor (cfg -> result)
  concurrent_train.py  -- local multiprocessing orchestrator
  modal_app.py         -- Modal Labs orchestrator + cost controls
  seed.py              -- deterministic seed setup
  metrics.py           -- JSON-line metrics writer + reader
  metric_sinks.py      -- optional TensorBoard / wandb mirrors
  report.py            -- `modallabs-report`: summarize a run directory
  checkpoint.py        -- done-sentinel + path helpers
  data_io.py           -- parquet/CSV loader, train/val split
  models/
    __init__.py        -- imports every concrete trainer
    hf_transformer.py  -- HuggingFace AutoModel wrapper (12 task heads)
    generic_torch.py   -- any torch.nn.Module via dotted-path
    generic_sklearn.py -- any sklearn estimator via dotted-path
    lightgbm.py        -- LightGBM gradient boosting
    xgboost.py         -- XGBoost gradient boosting
    catboost.py        -- CatBoost gradient boosting
    q_learning.py      -- DQN with double-Q + dueling head
    diffusion.py       -- minimal DDPM (smoke-friendly)
    ...
  configs/
    _template.yaml     -- documented schema
    all_models.yaml    -- demo: 6+ model families in one run
    hf_examples.yaml   -- one example per HF architecture
    cost_controlled_modal.yaml  -- Modal cost-knob demo
  tests/
    smoke.py           -- end-to-end smoke + determinism check
```

## Design constraints

- **No network at training time.** Models read from local parquet /
  CSV / image dirs. The Modal version downloads to its volume first.
- **No pickle-of-arbitrary-class.** Checkpoint format is `state_dict`
  (torch), `joblib` (sklearn), `.txt` / `.json` / `.cbm` (LightGBM /
  XGBoost / CatBoost), or `save_pretrained` (HF).
- **No silent neutral fallback.** A Trainer that hits an unrecoverable
  error raises. The orchestrator catches and reports.
- **Resource boundaries are explicit.** GPU memory and timeout per
  run are configurable; the orchestrator kills the process if
  exceeded.

## License

MIT. See [LICENSE](LICENSE).
