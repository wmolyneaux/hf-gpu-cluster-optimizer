---
name: Bug report
about: Something the harness did wrong
title: "[bug] "
labels: bug
---

## What happened

A clear description of the bug.

## Expected

What you expected instead.

## Repro

The exact orchestrator config (or the relevant `runs:` entry) and the
command you ran:

```yaml
# config.yaml
```

```bash
python -m modallabs.concurrent_train --config config.yaml ...
```

## Environment

- OS:
- Python version:
- `modallabs` version / commit:
- Relevant libs (torch / transformers / lightgbm / modal) + versions:
- Local or Modal:

## Run artifacts

Contents of `runs/<run_id>/<run_name>/status.json` (it includes the
traceback for failed runs), and the tail of `log.txt` if relevant.

```json
```
