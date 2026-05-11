# Contributing to modallabs

Thanks for your interest. This is a small, deliberately-scoped project;
the bar for changes is "does it keep the harness thin and the smoke
test green on three OSes".

## Ground rules

- **The smoke test is the contract.** `python -m modallabs.tests.smoke`
  must pass on Linux, macOS, and Windows. CI runs it on all three ×
  Python 3.10 / 3.11 / 3.12. If your change adds a model type, it adds
  a smoke case for it (see below).
- **No network at training time.** Trainers read from local parquet /
  CSV / image dirs. The synthetic-fallback batch must work offline so
  the smoke test does not hit the Hub.
- **No silent neutral fallback.** A Trainer that hits an unrecoverable
  error raises. The orchestrator catches and reports `phase: failed`.
  Do not swallow exceptions and return a degenerate result.
- **Heavy deps stay lazy.** `import modallabs` must not pull in torch /
  transformers / lightgbm / modal. Import those inside the trainer
  methods that need them, and wrap optional-dep modules so a missing
  library skips that trainer rather than breaking the registry.
- **ASCII only in source and docstrings.** No emojis in code.
- **Deterministic.** Honor `setup.seed`; set every RNG before
  instantiating anything random. Use `modallabs.seed.set_global_seed`.

## Dev setup

```bash
git clone https://github.com/wmolyneaux/hf-cluster-optimizer.git
cd hf-cluster-optimizer
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[full]"                          # or just torch+sklearn for the smoke
pip install ruff mypy pre-commit
pre-commit install
```

## Before you open a PR

```bash
ruff check .                       # must be clean
python -m modallabs.tests.smoke    # must exit 0
mypy --ignore-missing-imports .    # informational; try not to add new errors
```

Update `CHANGELOG.md` under `## [Unreleased]` describing your change.

## Adding a new model type

See [`PORTING.md`](PORTING.md). The short version:

1. Implement the `Trainer` interface (`modallabs/base.py`) in a new
   `modallabs/models/<name>.py`. Heavy imports go inside methods.
2. Decorate the class with `@register("your_type")`.
3. Add the module to `modallabs/models/__init__.py` (wrapped in
   `_safe_import` if it has an optional dependency).
4. Add a tiny CPU smoke case to `_LOCAL_CASES` in
   `modallabs/tests/smoke.py` (1-2 epochs, n<=128, deps listed).
5. If it takes user-facing config, document the fields in
   `configs/_template.yaml`.

About 80 lines for a typical port.

## Reporting bugs / requesting features

Open an issue using the templates. For bugs, include: OS, Python
version, the exact config that failed, and the contents of
`runs/<run_id>/<run_name>/status.json` (it has the traceback).

## Scope: things that are out of scope

- Per-architecture bespoke trainers (the whole point is one HF wrapper).
- Distributed / multi-node training (this is a single-box + Modal
  fan-out harness, not a DeepSpeed replacement).
- Warm-pool keep-alive on Modal (deliberately off — it bills idle GPU).
- Pickle-of-arbitrary-class checkpoints (state_dict / joblib /
  save_pretrained only).

## License

By contributing you agree your contributions are licensed under the
MIT License (see [LICENSE](LICENSE)).
