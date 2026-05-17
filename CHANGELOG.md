# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-17

Repo renamed to **hf-gpu-cluster-optimizer**; the import name `modallabs`
is unchanged. This release ships the fixes that make a Modal cluster
launch survive end-to-end on Modal SDK v1.4.x without surprise burns.

### Added

- **Preflight validation framework** (`scripts/preflight_validate.py`
  and Windows wrapper `scripts/preflight.ps1`). Free local probe that
  gates a paid `modal run` on INFRA / VOL / CODE / COST checks before
  the GPU clock starts. Exit codes: `0` proceed, `1` infra/code fail,
  `2` cost ceiling breach. Read-only by default; mutations require
  explicit flags (`--auto-create-volume`, `--remote-smoke`). See
  [PREFLIGHT.md](PREFLIGHT.md).
- **NCCL env defaults** (`nccl_env_defaults.py`). Single-source-of-truth
  for the NCCL env vars `torch.distributed` reads at init time, tuned
  for 8 x H100 NVLink with NVSwitch SHARP. Idempotent; never
  overwrites operator overrides. Importable from
  `modallabs.nccl_env_defaults`.
- **Fault-tolerance handlers** (`fault_tolerance.py`). Recovery probes
  + handler scaffolding for the post-allocation failure modes the
  preflight gate can't catch (NCCL all-reduce timeout surfacing as
  Python exception, etc.).
- **PEFT / LoRA support for HuggingFace heads.** A `peft:` block in a
  run's `config:` wraps the model in a PEFT adapter (LoRA by default)
  before training, so 1B+ models fine-tune on a single consumer GPU.
  Inert when the block is absent; requires the new `peft` extra
  (`pip install -e .[peft]`).
- **`modallabs-report` CLI.** Summarize a finished run directory:
  per-run table, best epoch, final metrics, wall-clock, failures.
  `modallabs-report runs/<run_id>`. Reads `summary.json` + each run's
  `metrics.jsonl` / `.modallabs_done`; no extra dependencies.
- **Per-run `manifest.json`.** Every run directory now gets a
  `manifest.json` capturing the git commit, dirty-tree flag, resolved
  versions of the key libraries (torch / transformers / numpy / ...),
  a SHA-256 of the resolved config, Python / platform, device, and
  start/finish timestamps. Makes the "deterministic" claim auditable.
  Writing it never fails the run (best-effort, errors are logged).
- **Optional external metric sinks.** A `logger:` field on a run
  (or the `MODALLABS_LOGGER` env var) mirrors every metric line into
  TensorBoard (`logger: tensorboard`) and/or Weights & Biases
  (`logger: wandb`) in addition to the canonical `metrics.jsonl`.
  No hard dependency — an unavailable backend is skipped with a log
  line, the run continues, and `metrics.jsonl` is always written.
- `CONTRIBUTING.md`, GitHub issue / PR templates, Dependabot config,
  and a `.pre-commit-config.yaml`.
- Ruff + mypy configuration in `pyproject.toml`; a `lint` job in CI.
- PyPI publish workflow (`.github/workflows/publish.yml`) — builds an
  sdist + wheel on a `v*` tag and publishes via PyPI Trusted Publishing
  (no stored token).
- README status / license / Python-version badges.

### Changed

- **Repository renamed** to `hf-gpu-cluster-optimizer` (the import name
  `modallabs` is unchanged; renaming a Python package would break every
  downstream `import` statement). GitHub redirects the old URL.
- `modal_app.py` works on Modal SDK v1.4.2:
  - `@app.function` decorators are hoisted to module scope
    (`.with_options` was removed in v1.4.x; resource overrides now
    require multiple module-level decorators, one per
    `(gpu, timeout)` tuple).
  - `serialized=True` on per-run functions for stable dispatch.
  - `gpu` and `timeout` are fixed at decoration time; `main()` fails
    loud if a config requests a different `(gpu, timeout)` than the
    declared remote variants.
- CI: bumped `actions/checkout` to v5 and `actions/setup-python` to v6
  (clears the Node 20 deprecation warning); the smoke job now installs
  the package itself (`pip install -e .`) instead of re-listing the
  core dependencies.

## [0.1.0] — 2026-05-10

Initial public release.

### Added

- `Trainer` abstract interface (6 methods + 1 classmethod) and the
  decorator-based type registry.
- Local concurrent orchestrator (`modallabs.concurrent_train`):
  one subprocess per run, per-GPU pinning, `--resume`, crash isolation.
- Modal Labs orchestrator (`modal_app.py`) with built-in cost controls:
  per-run hard timeout, cheap default GPU, worst-case cost ceiling
  (`MODALLABS_MAX_USD`), `--dry-run` cost preview, optional GPU
  auto-selector, no warm-pool keep-alive.
- HuggingFace `AutoModel` wrapper covering twelve task heads
  (sequence / token classification, causal & masked LM, seq2seq, QA,
  multiple choice, image classification, object detection, audio
  classification, speech-seq2seq, embedding) through one Trainer.
- Generic escape hatches: any `torch.nn.Module` via dotted path
  (`torch_module`), any sklearn estimator (`sklearn`), LightGBM /
  XGBoost / CatBoost, a minimal DDPM (`diffusion`), an LSTM / RNN / GRU
  / Transformer / manifold-autoencoder / NTM sequence head, and an
  offline DQN (`q_learning`) with double-Q + dueling head.
- Deterministic seeding (`set_global_seed`) across `random`, `numpy`,
  `torch` CPU + CUDA, and `transformers`.
- JSON-line metrics writer/reader, done-sentinel checkpoint helpers,
  parquet/CSV data loader with train/val split.
- Cross-platform smoke test (`modallabs.tests.smoke`): trains every
  registered local model on a tiny synthetic run, checks `seed=42`
  determinism to `1e-6`, and verifies crash isolation.
- GitHub Actions CI matrix: ubuntu / macOS / windows × Python
  3.10 / 3.11 / 3.12.

### Notes

- The `0.1.0` tag's first CI runs failed because the repository content
  sits at the workspace root (flat layout) while `import modallabs`
  needs the package importable under that name; fixed immediately after
  by checking the repo out into a `modallabs/` subdirectory in CI and
  wiring the flat layout through `pyproject.toml`
  (`[tool.setuptools.package-dir]`). No code change to the package
  itself.

[0.2.0]: https://github.com/wmolyneaux/hf-gpu-cluster-optimizer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/wmolyneaux/hf-gpu-cluster-optimizer/releases/tag/v0.1.0
