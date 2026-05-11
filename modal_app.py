"""modallabs — Modal Labs cloud orchestrator with cost controls.

Run the same orchestrator config on Modal Labs. One Modal function per
run, isolated GPU per run, automatic shutdown on completion, hard
timeout per run.

Cost controls (default-on):
  * `MAX_RUNTIME_SEC` -- hard timeout per run (default 4 hours).
  * `MIN_GPU_FOR_TYPE` -- GPU type defaults are conservative (T4 / A10G);
    upgrade only via explicit cfg `modal.gpu` field.
  * `IDLE_SHUTDOWN_SEC` -- function returns immediately after the run
    completes; no warm-pool keep-alive.
  * `volume_path` -- runs/ output is mirrored to a Modal Volume so you
    pay storage only for outputs, not the full container image.
  * `--dry-run` -- prints the GPU + estimated-cost preview WITHOUT
    starting any function.

Usage:
    modal token new                              # one-time
    modal run modallabs/modal_app.py --config configs/all_models.yaml
    modal run modallabs/modal_app.py --config X --dry-run
    modal volume get modallabs-runs runs/        # download outputs

Pre-flight cost preview (printed before any GPU spin-up):
    [DRY RUN] 5 runs queued
       run #1: bert_finetune      gpu=T4   est=  4h ~= $1.44
       run #2: gpt2_finetune      gpu=T4   est=  2h ~= $0.72
       ...
       Total estimated cost: $6.40
    Proceed? (--no-dry-run to actually run)

The script intentionally fails LOUD on any cost control breach so you
never accidentally leave a $40/hr A100 instance spinning.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -- Modal SDK (optional import; the file is also runnable as a CLI
# preview without modal installed).
try:
    import modal
    _HAS_MODAL = True
except Exception:
    modal = None  # type: ignore
    _HAS_MODAL = False


# Approximate Modal pricing (USD / hour, 2026 list price; ALWAYS verify
# against modal.com/pricing before relying on these numbers). Used only
# for the dry-run preview; actual billing is whatever Modal charges.
_GPU_HOURLY_USD: Dict[str, float] = {
    "cpu":     0.20,
    "T4":      0.36,
    "L4":      0.50,
    "A10G":    1.10,
    "L40S":    2.00,
    "A100-40G": 3.10,
    "A100-80G": 4.00,
    "H100":    5.50,
    "H200":    8.00,
}


_MAX_RUNTIME_SEC_DEFAULT = 4 * 60 * 60  # 4 hours
_DEFAULT_GPU = "T4"
_VOLUME_NAME = "modallabs-runs"
# Hard upper bound on the dry-run cost-preview total. Override via env.
_DEFAULT_MAX_TOTAL_USD = 25.0


def _max_total_usd() -> float:
    """Read the cost ceiling from env. Default $25."""
    try:
        return float(os.environ.get("MODALLABS_MAX_USD", _DEFAULT_MAX_TOTAL_USD))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOTAL_USD


def auto_select_gpu(rc: Dict[str, Any]) -> str:
    """Pick the smallest Modal GPU that should fit a model.

    Heuristic on cfg.config.hf_model_name OR cfg.config.params (param
    count proxy). Override per run by setting cfg.modal.gpu explicitly;
    this helper is only consulted when cfg.modal.gpu == 'auto' or unset.

    Tiers:
      * <7B params      -> T4   (16GB)
      * 7B-30B          -> A10G or L40S
      * >30B            -> A100-80G

    The user can always override per run via cfg.modal.gpu.
    """
    modal_section = rc.get("modal") or {}
    explicit = modal_section.get("gpu")
    if explicit and str(explicit).lower() != "auto":
        return str(explicit)
    # Param-count proxy from model name. We do NOT download the model
    # for sizing -- this is a string heuristic.
    name = str((rc.get("config") or {}).get("hf_model_name", "")).lower()
    big = ("70b" in name or "65b" in name or "180b" in name
           or "40b" in name or "32b" in name)
    mid = ("7b" in name or "8b" in name or "13b" in name or "11b" in name)
    if big:
        return "A100-80G"
    if mid:
        return "A10G"
    return _DEFAULT_GPU


def _gpu_for_run(rc: Dict[str, Any]) -> str:
    modal_section = rc.get("modal") or {}
    g = modal_section.get("gpu")
    if not g or str(g).lower() == "auto":
        return auto_select_gpu(rc)
    return str(g)


def _max_runtime_sec(rc: Dict[str, Any]) -> int:
    modal_section = rc.get("modal") or {}
    return int(modal_section.get("max_runtime_sec", _MAX_RUNTIME_SEC_DEFAULT))


def _expected_runtime_sec(rc: Dict[str, Any]) -> int:
    """Best-effort wall-time estimate from cfg. Informational only."""
    cfg = rc.get("config") or {}
    epochs = int(cfg.get("epochs", 1))
    sec_per_epoch = int((rc.get("modal") or {}).get("est_sec_per_epoch", 60))
    return epochs * sec_per_epoch


def _worst_case_runtime_sec(rc: Dict[str, Any]) -> int:
    """Worst-case billable seconds = the hard timeout that Modal will actually
    enforce. If a model hangs, the user pays for `max_runtime_sec`, not the
    optimistic `epochs * est_sec_per_epoch`. The cost ceiling and the dry-run
    preview must both gate on THIS number."""
    return _max_runtime_sec(rc)


_UNKNOWN_GPU_WARNED: set = set()


def _estimate_cost_usd(gpu: str, sec: int) -> float:
    if gpu not in _GPU_HOURLY_USD:
        # Unknown GPU type -- silently using the T4 rate would let a user
        # request an expensive new tier (e.g. "B200") and see a surprisingly
        # low cost preview. Warn ONCE per unknown type per process so the
        # dry-run output is not flooded but the user is not deceived.
        if gpu not in _UNKNOWN_GPU_WARNED:
            _UNKNOWN_GPU_WARNED.add(gpu)
            import sys as _sys
            print(
                f"   !! WARNING: GPU type {gpu!r} not in modallabs price table. "
                f"Cost preview using {_DEFAULT_GPU} rate (${_GPU_HOURLY_USD[_DEFAULT_GPU]:.2f}/h); "
                f"actual Modal billing may differ. Verify against modal.com/pricing.",
                file=_sys.stderr,
            )
    rate = _GPU_HOURLY_USD.get(gpu, _GPU_HOURLY_USD.get(_DEFAULT_GPU, 0.36))
    return rate * (sec / 3600.0)


def _load_orchestrator_cfg(path: str) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def estimate_total_cost_usd(cfg: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Sum WORST-CASE billable cost across every run. The worst case is
    `max_runtime_sec * GPU rate` — what the user actually pays if a model
    hangs to its hard timeout. We deliberately do NOT use the optimistic
    `epochs * est_sec_per_epoch` for ceiling decisions."""
    runs = cfg.get("runs") or []
    breakdown = []
    total = 0.0
    for rc in runs:
        gpu = _gpu_for_run(rc)
        worst_sec = _worst_case_runtime_sec(rc)
        est_sec = _expected_runtime_sec(rc)
        worst_cost = _estimate_cost_usd(gpu, worst_sec)
        est_cost = _estimate_cost_usd(gpu, est_sec)
        total += worst_cost
        breakdown.append({
            "name": rc.get("name", "?"),
            "gpu": gpu,
            "worst_sec": worst_sec,
            "worst_usd": worst_cost,
            "est_sec": est_sec,
            "est_usd": est_cost,
        })
    return total, breakdown


def _print_dry_run(cfg: Dict[str, Any]) -> bool:
    """Print the dry-run preview. Returns True iff the ceiling was breached.

    The caller is expected to translate that bool into an exit code so a
    CI gate (or `set -e` shell pipeline) can react -- `_print_dry_run`
    itself does not exit.
    """
    runs = cfg.get("runs") or []
    print(f"[DRY RUN] {len(runs)} runs queued (run_id={cfg.get('run_id', 'auto')})")
    total, breakdown = estimate_total_cost_usd(cfg)
    # Show per-run hard timeout next to each run so override-vs-default is visible.
    timeout_overrides = []
    for i, (b, rc) in enumerate(zip(breakdown, runs), 1):
        timeout_h = _max_runtime_sec(rc) / 3600.0
        is_override = _max_runtime_sec(rc) != _MAX_RUNTIME_SEC_DEFAULT
        if is_override:
            timeout_overrides.append((b["name"], timeout_h))
        timeout_str = f"timeout={timeout_h:.2f}h{'*' if is_override else ' '}"
        print(f"   run #{i}: {b['name']:<24}  "
              f"gpu={b['gpu']:<8}  "
              f"{timeout_str}  "
              f"worst={b['worst_sec']/3600:.2f}h ~= ${b['worst_usd']:.2f}  "
              f"(est={b['est_sec']/3600:.2f}h ~= ${b['est_usd']:.2f})")
    print("   --")
    print(f"   Total WORST-CASE cost (every run hits its max_runtime_sec timeout): ${total:.2f}")
    print(f"   Cost ceiling (gates on worst-case): ${_max_total_usd():.2f} "
          f"(override via env MODALLABS_MAX_USD)")
    print(f"   Hard per-run timeout: {_MAX_RUNTIME_SEC_DEFAULT/3600:.1f}h default "
          f"(override per-run via cfg.modal.max_runtime_sec)")
    if timeout_overrides:
        print(f"   * = per-run override ({len(timeout_overrides)} of {len(runs)} runs)")
    blocked = total > _max_total_usd()
    if blocked:
        print()
        print(f"   !! BLOCKED: worst-case ${total:.2f} > ${_max_total_usd():.2f} ceiling.")
        print("   !! Either lower per-run max_runtime_sec / GPU tier,")
        print("   !! or raise the ceiling: export MODALLABS_MAX_USD=<dollars>")
        print()
        print("Cannot proceed: lower the cost or raise MODALLABS_MAX_USD.")
    else:
        print()
        print("Proceed: re-run without --dry-run to actually launch.")
    return blocked


# ---------------------------------------------------------------------------
# Modal-only definitions. We define these inside a function so the file
# imports cleanly without modal installed (for dry-run on a local box).
# ---------------------------------------------------------------------------

_HF_CACHE_VOLUME_NAME = "modallabs-hf-cache"
_HF_CACHE_MOUNT = "/hf_cache"


if _HAS_MODAL:
    # Direct GPU-burn risk: without a persistent HF cache, every cold start
    # re-downloads every transformers / datasets artifact. A first-run llama
    # checkpoint can be tens of GB; on an A10G ($1.10/h) a 5-minute download
    # is ~9 cents, multiplied across N runs. Mount a persistent volume at
    # /hf_cache and point the standard HF env vars at it so the second cold
    # start sees the cache already populated. Volume is shared across all
    # Modal containers in this app; per Modal docs, reads are concurrent-safe.
    # (Per Q-5c H6 deferral; Q-5b's audit content was overwritten and the
    # GPU-burn-minimization owner role is vacant, so Q-5d applies it.)
    modal_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1",
            "numpy",
            "pandas",
            "pyarrow",
            "pyyaml",
            "scikit-learn",
            "lightgbm",
            "transformers",
            "datasets",
            "accelerate",
            "safetensors",
            "tokenizers",
        )
        .env({
            "HF_HOME": _HF_CACHE_MOUNT,
            "TRANSFORMERS_CACHE": f"{_HF_CACHE_MOUNT}/transformers",
            "HUGGINGFACE_HUB_CACHE": f"{_HF_CACHE_MOUNT}/hub",
            "HF_DATASETS_CACHE": f"{_HF_CACHE_MOUNT}/datasets",
        })
        .add_local_python_source("modallabs")  # bundle the framework
    )
    app = modal.App("modallabs", image=modal_image)
    runs_volume = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)
    hf_cache_volume = modal.Volume.from_name(_HF_CACHE_VOLUME_NAME, create_if_missing=True)

    # Tiny CPU-only function that inspects the volume and tells the local
    # entrypoint which runs already have a `.modallabs_done` sentinel.
    # CPU container is ~$0.20/hr and runs in <1 sec; the alternative
    # (allocating a GPU per run just so train_one can call is_done()) is
    # the catastrophic burn we are preventing here.
    @app.function(
        cpu=1.0,
        timeout=120,
        volumes={"/runs": runs_volume},
    )
    def _done_runs(run_id: str, names: list) -> list:
        """Return the subset of `names` whose run dir has a done-sentinel."""
        from modallabs.checkpoint import is_done as _is_done
        done = []
        for n in names:
            if _is_done(Path("/runs") / run_id / n):
                done.append(n)
        return done

    def _make_remote_runner(gpu: str, max_runtime_sec: int):
        """Create a per-(gpu, timeout) remote function.

        `timeout` is forwarded to Modal so the container is hard-killed
        at the deadline regardless of what the Trainer does -- no
        Python-side `signal.alarm`, no soft timeout.

        No `keep_warm` and no `min_containers` -- container exits on
        function return and the GPU is released.

        Mounts the persistent `modallabs-hf-cache` volume at
        `/hf_cache` so HuggingFace downloads survive container teardown
        across runs. The image's HF_HOME / TRANSFORMERS_CACHE /
        HUGGINGFACE_HUB_CACHE / HF_DATASETS_CACHE env vars all point at
        that mount, so the first cold start populates the cache and
        every subsequent cold start reuses it.
        """
        @app.function(
            gpu=gpu,
            timeout=max_runtime_sec,
            volumes={
                "/runs": runs_volume,
                _HF_CACHE_MOUNT: hf_cache_volume,
            },
            # The container is torn down at function exit -- no idle keep-alive.
            # We deliberately do NOT pass keep_warm or min_containers.
        )
        def _remote(run_cfg: dict, run_id: str, resume: bool) -> dict:
            from modallabs.runner import train_one
            import modallabs.models  # noqa: F401  -- register trainers
            return train_one(
                run_cfg,
                run_id=run_id,
                output_root=Path("/runs"),
                resume=resume,
                force_cpu=False,
            )
        return _remote

    @app.local_entrypoint()
    def main(
        config: str,
        resume: bool = False,
        dry_run: bool = False,
    ) -> None:
        cfg = _load_orchestrator_cfg(config)
        if dry_run:
            blocked = _print_dry_run(cfg)
            if blocked:
                # Surface ceiling breach via non-zero exit so CI gates catch it.
                raise SystemExit(2)
            return
        runs: List[Dict[str, Any]] = list(cfg.get("runs") or [])
        run_id = str(cfg.get("run_id") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))

        # Resume filter -- runs through a CPU-only function so we never
        # allocate a GPU for an already-done run.
        if resume and runs:
            try:
                already_done = set(_done_runs.remote(run_id, [str(r.get("name")) for r in runs]))
            except Exception as exc:
                print(f"modallabs/modal: resume probe failed ({exc!r}); "
                      f"falling back to letting each worker check is_done. "
                      f"Note: this may allocate GPUs for already-done runs.")
                already_done = set()
            if already_done:
                pre_skipped = [r for r in runs if str(r.get("name")) in already_done]
                runs = [r for r in runs if str(r.get("name")) not in already_done]
                print(f"modallabs/modal: --resume skipping {len(pre_skipped)} "
                      f"already-done runs (no GPU allocated): "
                      f"{[r.get('name') for r in pre_skipped]}")

        # Hard kill-switch on WORST-CASE cost (gates pre-spawn, before any
        # GPU container materializes).
        cfg_after_resume = dict(cfg, runs=runs)
        total, _ = estimate_total_cost_usd(cfg_after_resume)
        ceiling = _max_total_usd()
        if total > ceiling:
            _print_dry_run(cfg_after_resume)  # purely informational here
            raise RuntimeError(
                f"modallabs/modal: refusing to launch -- worst-case total "
                f"${total:.2f} exceeds ceiling ${ceiling:.2f}. "
                f"Override with `export MODALLABS_MAX_USD=<dollars>` if intentional."
            )
        if not runs:
            print("modallabs/modal: nothing to launch (all runs already done or empty config).")
            print(json.dumps({"run_id": run_id, "n_runs": 0, "runs": []}, indent=2))
            return

        print(f"modallabs/modal: launching {len(runs)} runs (run_id={run_id}); "
              f"worst-case total ${total:.2f} (ceiling ${ceiling:.2f}); "
              f"per-run hard timeout enforced by Modal; tear-down on completion.")
        # Modal fans these out concurrently (each .spawn() call is async).
        futures = []
        for rc in runs:
            gpu = _gpu_for_run(rc)
            timeout_sec = _max_runtime_sec(rc)
            remote = _make_remote_runner(gpu, timeout_sec)
            futures.append((rc.get("name"), remote.spawn(rc, run_id, resume)))

        results = []
        try:
            for name, fut in futures:
                try:
                    results.append(fut.get())
                except Exception as exc:
                    results.append({
                        "name": name,
                        "phase": "failed",
                        "error": f"modal.remote: {type(exc).__name__}: {exc}",
                    })
        except KeyboardInterrupt:
            # User Ctrl-C -- cancel any in-flight futures so containers
            # terminate (and GPUs release) instead of running to timeout.
            print("modallabs/modal: KeyboardInterrupt -- cancelling outstanding futures.")
            for name, fut in futures:
                try:
                    fut.cancel()
                except Exception:
                    pass
            raise

        # Persist a top-level summary inside the volume.
        summary = {
            "run_id": run_id,
            "n_runs": len(runs),
            "n_succeeded": sum(1 for r in results if r.get("phase") == "succeeded"),
            "n_failed": sum(1 for r in results if r.get("phase") == "failed"),
            "runs": results,
        }
        print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI fallback when modal isn't installed: print the dry-run preview only.
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="modallabs Modal orchestrator (preview when modal not installed)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = _load_orchestrator_cfg(args.config)
    if args.dry_run or not _HAS_MODAL:
        blocked = _print_dry_run(cfg)
        if not _HAS_MODAL:
            print("\n(modal SDK not installed; install with `pip install modal` to actually run)")
        # Exit code 2 when the cost ceiling is breached so CI / `set -e`
        # shells can detect it. This mirrors what the modal-installed
        # launch path does (it raises RuntimeError before spawning any
        # function).
        return 2 if blocked else 0
    print("Use `modal run modallabs/modal_app.py --config <path>` to launch on Modal.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
