# Preflight validation framework

A validation framework for the infra/deploy side of
hf-gpu-cluster-optimizer. It runs the checks that catch the failure
modes a paid `modal run` would hit *after* the GPU clock has started.

## Why this exists

Modal failures that fire **before** GPU allocation cost nothing. Failures
that fire **after** allocation cost real money. The preflight gate moves
the most common post-allocation failures into a free local probe so you
either fix the problem or know exactly what you're paying for.

## Checks

### INFRA — Modal CLI itself

- **INFRA-1**  modal CLI installed (`modal --version`)
- **INFRA-2**  modal CLI on PATH (`shutil.which("modal")`) — WARN if installed but not on PATH (prints fix)
- **INFRA-3**  modal auth working (`modal volume list` returns 0)
- **INFRA-4**  UTF-8 stdio configured (`PYTHONUTF8 / PYTHONIOENCODING`)
- **INFRA-5**  modal CLI version ≥ 1.4.0 (older clients lack `--force` on `modal volume put`)

### VOL — Modal Volume state

- **VOL-1**  target volume exists (auto-create with `--auto-create-volume`)

### CODE — Code resolves

- **CODE-1**  `modal_app.py` imports cleanly (catches syntax / pip-install drift)
- **CODE-2**  config YAML parses + has `runs` key
- **CODE-3**  every `cfg.runs[*].type` is registered after `import modallabs.models`
- **CODE-3-REMOTE** *(opt-in via `--remote-smoke`)* dispatches a 1-second CPU function
  on Modal that imports the trainer **inside an actual container**. Catches
  pip-install drift the local check misses. Costs ~$0.0001.
- **CODE-5**  `modal_app.py` is free of known Modal SDK v1.4 antipatterns
  (non-global `@app.function` decoration, removed `.with_options`, etc.).

### COST — Money

- **COST-1**  dry-run worst-case ≤ ceiling (delegates to `modal_app.py --dry-run`)
- **COST-2**  per-run `max_runtime_sec` ≤ 4h (WARN unless `--allow-long-timeout`)

## Exit codes

| Code | Meaning                                              |
|------|------------------------------------------------------|
| 0    | All PASS or WARN-only — proceed to paid launch       |
| 1    | At least one INFRA/VOL/CODE check FAILed             |
| 2    | COST ceiling breached (same code as `--dry-run`)     |

Exit 0 with WARNs means "you can proceed but you should know about X."
Exit 1 means "do not launch; fix first." Exit 2 specifically signals a
cost issue so CI gates can react differently than they would to
infrastructure failures.

## Usage

### Direct Python (cross-platform)

```bash
python scripts/preflight_validate.py --config configs/all_models.yaml
python scripts/preflight_validate.py --config <path> --auto-create-volume
python scripts/preflight_validate.py --config <path> --remote-smoke
python scripts/preflight_validate.py --config <path> --json
python scripts/preflight_validate.py --config <path> --skip COST-1 CODE-3
```

### PowerShell wrapper (Windows; sets UTF-8 + PATH automatically)

```powershell
.\scripts\preflight.ps1 -Config configs\all_models.yaml
.\scripts\preflight.ps1 -Config configs\cost_controlled_modal.yaml -AutoCreateVolume
.\scripts\preflight.ps1 -Config configs\cost_controlled_modal.yaml -RemoteSmoke
.\scripts\preflight.ps1 -Config configs\cost_controlled_modal.yaml -Skip COST-1,CODE-3
```

Use the wrapper on Windows — it sets `$env:PYTHONIOENCODING="utf-8"`,
`$env:PYTHONUTF8="1"`, and prepends the user-site Python Scripts dir
to `PATH` before the validator runs. Without those env vars, the
validator itself will FAIL INFRA-4.

## Read-only by default

The validator is read-only by default. The only Modal mutations it
will ever perform are:

| Mutation                                     | Trigger                | Cost     |
|----------------------------------------------|------------------------|----------|
| `modal volume create <name>`                 | `--auto-create-volume` | free     |
| `modal run scripts/_remote_import_probe.py`  | `--remote-smoke`       | ~$0.0001 |

The remote-smoke probe is a 1-second CPU function (no GPU). The script
it dispatches is written to `scripts/_remote_import_probe.py` so it
can be audited / re-run manually.

## Adding a new check

1. Write a `check_<name>(report, ...)` that calls `report.add(CheckResult(...))`.
2. Wire it into `run_preflight(...)` in the matching category block.
3. Add the ID to the `--skip` allowlist — any unknown ID is silently
   ignored, which is the right behavior for forward-compat.
4. Document it in this file's table.

Categories: `INFRA` / `VOL` / `CODE` / `COST`. Statuses:
`PASS` / `FAIL` / `WARN` / `SKIP`. A `WARN` is a soft fail — the verdict
is still PROCEED, but the message is rendered in the report.
