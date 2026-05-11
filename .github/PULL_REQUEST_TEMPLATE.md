## What this changes

Short description.

## Why

The motivation / linked issue (`Closes #...`).

## Checklist

- [ ] `ruff check .` is clean
- [ ] `python -m modallabs.tests.smoke` exits 0 locally
- [ ] If a new model type was added: smoke case in `_LOCAL_CASES`,
      registered, wired in `models/__init__.py`, config documented in
      `configs/_template.yaml`
- [ ] Heavy deps (torch / transformers / ...) are imported lazily, not
      at package import time
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] No emojis / non-ASCII in source

## Notes for reviewers

Anything non-obvious — design trade-offs, things you're unsure about.
