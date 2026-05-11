# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **PEFT / LoRA support for HuggingFace heads.** A `peft:` block in a
  run's `config:` wraps the model in a PEFT adapter (LoRA by default)
  before training, so 1B+ models fine-tune on a single consumer GPU.
  Inert when the block is absent; requires the new `peft` extra
  (`pip install -e .[peft]`).
- **`modallabs-report` CLI.** Summarize a finished run directory:
  per-run table, best epoch, final metrics, wall-clock, failures.
  `modallabs-report runs/<run_id>` (or point it at a single
  `runs/<run_id>/<run_name>` dir). Reads `summary.json` + each run's
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
- `CHANGELOG.md`, `CONTRIBUTING.md`, GitHub issue / PR templates,
  Dependabot config, and a `.pre-commit-config.yaml`.
- Ruff + mypy configuration in `pyproject.toml`; a `lint` job in CI.
- PyPI publish workflow (`.github/workflows/publish.yml`) — builds an
  sdist + wheel on a `v*` tag and publishes via PyPI Trusted Publishing
  (no stored token).
- README status / license / Python-version badges.

### Changed

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

[Unreleased]: https://github.com/wmolyneaux/hf-cluster-optimizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wmolyneaux/hf-cluster-optimizer/releases/tag/v0.1.0
