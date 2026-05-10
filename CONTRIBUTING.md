# Contributing to HF Cluster Optimizer

Thanks for your interest in contributing! `hf_cluster_optimizer` aims to stay small,
opinionated, and reliable. PRs that fit those goals are welcome.

## Getting set up

```bash
git clone https://github.com/<owner>/hf-cluster-optimizer.git
cd hf-cluster-optimizer
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[full]"
python -m hf_cluster_optimizer.tests.smoke
```

The smoke should report `ran=13 skipped=0 failed=0` in roughly 90 seconds
on a clean Python 3.11 box. If anything fails on your machine, that is
itself a useful bug report.

## What good PRs look like

- **Bug fixes with a reproducer.** Drop a minimal script that triggers
  the bug into the PR description, then the fix, then the verification
  output (post-fix smoke + orchestrator).
- **New Trainers.** Follow `PORTING.md`. The smoke entry is required;
  the new Trainer must train a tiny synthetic case end-to-end on CPU
  in <5 seconds so CI stays fast.
- **Cost-control improvements.** Anything that makes the Modal dry-run
  more honest (better worst-case math, more visible per-run knobs,
  tighter ceiling enforcement). Include a before/after of the dry-run
  output in the PR.
- **Docs / typo fixes.** Always welcome.

## What does NOT fit HF Cluster Optimizer

- **Trainer-specific scheduling logic** (LR schedulers beyond what the
  Trainer wants, gradient accumulation policies, mixed-precision
  toggles) -- those live inside the Trainer, not in the framework. The
  framework is intentionally tiny.
- **Background heartbeat threads, retry-loops, sleep-poll patterns.**
  See `PORTING.md` -- the framework is GPU-burn-conscious by design.
- **Silent neutral fallback.** A Trainer that hits an unrecoverable
  error must raise. The orchestrator catches and reports.

## Tests

Every PR must:

1. Pass `python -m hf_cluster_optimizer.tests.smoke` (13/13 OK).
2. Pass `python -m hf_cluster_optimizer.concurrent_train --config configs/all_models.yaml --force-cpu --max-workers 1` (6/6 succeeded).
3. Pass `python hf_cluster_optimizer/modal_app.py --config configs/cost_controlled_modal.yaml --dry-run` (exit 0).
4. Pass the cost-ceiling kill-switch: `HFCO_MAX_USD=0.01 python hf_cluster_optimizer/modal_app.py --config configs/cost_controlled_modal.yaml --dry-run` (exit 2, `BLOCKED` line printed).

CI runs all four on Ubuntu / macOS / Windows x Python 3.10 / 3.11 / 3.12.

## Style

- ASCII only. No emojis in code or docs.
- One-line docstrings on top of multi-paragraph essays. Let type
  signatures speak.
- File-line citations on bug reports, never "this might be wrong
  somewhere."
- `from __future__ import annotations` at the top of every new module.
- No new top-level dependencies without justification. Optional deps go
  in `pyproject.toml` `[project.optional-dependencies]` extras.

## License

By contributing, you agree your contributions will be licensed under
the project's MIT License.
