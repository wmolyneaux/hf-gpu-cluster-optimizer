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
    starting any function AND without hydrating any image (the check runs
    at import time; see _dry_run_short_circuit).

Usage:
    modal token new                              # one-time
    modal run modallabs/modal_app.py --config configs/all_models.yaml
    modal run modallabs/modal_app.py --config X --dry-run
    python modallabs/modal_app.py --config X --dry-run   # same preview, no modal needed
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

from modallabs import credentials as _creds


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

# ---------------------------------------------------------------------------
# BILL SAFETY. Four independent layers, each of which alone bounds the spend.
# Ordered outermost (cannot be bypassed) to innermost (cheapest to trip).
#
#   L1  _ABSOLUTE_MAX_USD   -- a ceiling on the ceiling. MODALLABS_MAX_USD may
#                              LOWER the gate but can never raise it past this.
#                              Not overridable by env, by config, or by flag.
#   L2  Modal `timeout=`    -- server-side container kill. Enforced by Modal
#                              even if the container is wedged and ignoring us.
#                              worst_case_cost == timeout * rate, which is
#                              exactly what estimate_total_cost_usd() charges.
#   L3  deadline watchdog   -- in-container, fires _DEADLINE_MARGIN_SEC before
#                              L2 so checkpoints flush instead of being lost to
#                              a hard kill.
#   L4  dead-man switch     -- terminates after _NO_PROGRESS_KILL_SEC with no
#                              forward progress. THIS IS THE ONE THAT SAVES
#                              REAL MONEY: without it a hung job silently bills
#                              the FULL lane (a wedged 4h H100 run is ~$15.80
#                              instead of the ~$2 it should have cost).
# ---------------------------------------------------------------------------
_ABSOLUTE_MAX_USD = 100.0
_DEADLINE_MARGIN_SEC = 120
_NO_PROGRESS_KILL_SEC = 600
# Cold start + image pull + HuggingFace download all happen before the first
# write, so startup gets its own, larger budget. Too tight and a legitimate
# cold start is killed; too loose and a job that dies at import burns it.
_STARTUP_GRACE_SEC = 900

# Timeout lanes. Modal v1.4.2 fixes (gpu, timeout) at decoration time, so one
# lane == one module-level @app.function. Keep this table SHORT: every lane is
# additional worst-case billing exposure, and main() always routes a run to the
# SMALLEST lane that fits its requested max_runtime_sec.
_REMOTE_GPU = "H100"
# Run types that have their OWN image and volumes. Declared unconditionally, so that a
# lane which failed to declare becomes a loud refusal in main() instead of a silent
# dispatch to the generic training image.
_TYPED_LANES_REQUIRED = frozenset({"wan_vace_shot", "longcat_avatar", "tram_motion",
                                   "heroshot_take", "trellis2_recon", "kimodo_motion",
                                   "comfy_sheet"})
_LANES: Dict[str, int] = {
    # 10 min - short measured jobs. Added 2026-08-11 off MEASURED runtimes, not a guess:
    # an orpheus_voice 3-epoch LoRA train is 217s and an orpheus_tts 4-clip generation is
    # 187s, both on H100. Routing those to `short` gated each at $2.75 for ~3 minutes of
    # work, which stops being rounding error the moment you fan out: 7 concurrent runs
    # gate at $19.25 instead of $6.42. The lane timeout IS the worst-case bill.
    "brief":   600,
    "short":  1800,    # 30 min - Wan 2.2 shots (~1250s measured). Default.
    "medium": 5400,    # 90 min
    "long":  14400,    # 4 h    - full FPO training runs (1.5-4h estimated)
}
_DEFAULT_LANE = "short"


def _max_total_usd() -> float:
    """Cost ceiling, clamped to the absolute cap.

    MODALLABS_MAX_USD can only ever LOWER the gate. Attempting to raise it
    past _ABSOLUTE_MAX_USD is clamped and warned about rather than honoured --
    a typo in an env var must not be able to authorise an unbounded bill.
    """
    try:
        requested = float(os.environ.get("MODALLABS_MAX_USD", _DEFAULT_MAX_TOTAL_USD))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOTAL_USD
    if requested <= 0:
        return _DEFAULT_MAX_TOTAL_USD
    if requested > _ABSOLUTE_MAX_USD:
        print(
            f"[modallabs] MODALLABS_MAX_USD={requested:.2f} exceeds the hard cap "
            f"${_ABSOLUTE_MAX_USD:.2f}; CLAMPED to ${_ABSOLUTE_MAX_USD:.2f}. "
            "Raise _ABSOLUTE_MAX_USD in modal_app.py only with a deliberate "
            "code change and review.",
            file=sys.stderr,
        )
        return _ABSOLUTE_MAX_USD
    return requested


def _lane_for(timeout_sec: int) -> str:
    """Smallest lane that fits `timeout_sec`. Raises if none does.

    Routing to the smallest sufficient lane is a cost control: the lane's
    timeout IS the worst-case bill, so over-provisioning the lane directly
    over-provisions the maximum spend.
    """
    for name, cap in sorted(_LANES.items(), key=lambda kv: kv[1]):
        if timeout_sec <= cap:
            return name
    raise RuntimeError(
        f"modallabs/modal: max_runtime_sec={timeout_sec} exceeds the largest "
        f"lane ({max(_LANES.values())}s). Split the job, checkpoint and resume, "
        "or add a larger lane in modal_app.py -- deliberately, with review, "
        "because a larger lane raises the worst-case bill for every run in it."
    )


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
    explicit = modal_section.get("max_runtime_sec")
    if explicit is None:
        # Defaulting to 4h was safe when a non-matching timeout was REFUSED before
        # .spawn(). Now it ROUTES, so the default silently buys the most expensive
        # lane ($22.00 worst case on H100) for a config that just forgot a line.
        # Default to the SMALLEST lane; a longer run must ask for it in writing.
        return _LANES[_DEFAULT_LANE]
    return int(explicit)


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


# --------------------------------------------------------------------------
# STALE-VOLUME ASSERTION for heroshot_take.
#
# THE FAILURE THIS EXISTS TO CONVERT INTO A REFUSAL. `berkeley-usd-take` was
# staged on 2026-08-11 with a place_rig.py whose sky Mapping rotation was
# (0, 0, 90). The corrected value is (-90, 0, 90); with the old one the camera
# ray lands in the map's below-horizon half and the sky renders EXACTLY 0.0 --
# min = max = 0 (MEASURED, place_rig.py's own comment block). A fleet run off
# that volume produces black skies on every worker, and because every worker
# reads the SAME stale script, the cross-worker equality band passes: eight
# workers agreeing is evidence of determinism, not of correctness. The invoice
# arrives either way.
#
# So the launcher stamps the sha256 of the LOCAL scripts into each run's config
# and the trainer refuses if the volume disagrees. The comparison is done
# against files the launcher can actually read; what it does NOT cover is
# stated rather than implied:
#   COVERED     the render/repair scripts below, byte-for-byte.
#   NOT COVERED usd/, textures/, rig/rigged.glb and the retarget solve. Those
#               are hashed by place_rig into the manifest (world_sha256,
#               textures_sha256, glb_sha256, retarget_sha256) but there is no
#               local expectation to compare them to without duplicating
#               place_rig's _sha_tree, and a re-implementation that drifts
#               would refuse good runs. Re-stage after ANY input change.
_BUSD_LOCAL_ENV = "BUSD_LOCAL"
_BUSD_LOCAL_DEFAULT = Path.home() / "Desktop" / "berkeley-usd"
# place_rig.py is REQUIRED: it is the file that had the bug, and it is a
# digest input to its own manifest (script_sha256), so the trainer can
# cross-check the pin against what the render itself certified.
_HEROSHOT_PIN_REQUIRED = "tools/heroshot/place_rig.py"
_HEROSHOT_PINS = (_HEROSHOT_PIN_REQUIRED,
                  "tools/render/lut_repair.py",
                  "tools/render/material_lut.json")


def _sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _heroshot_pin_run(rc: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp local script hashes into one heroshot run's config, in place.

    Refuses AT THE LAUNCHER (free) as well as in the trainer (cheap): the
    launcher check costs nothing, and the trainer check still fires for
    anything that reaches a container by another route.
    """
    cfg = rc.setdefault("config", {})
    if cfg.get("allow_unpinned"):
        print(f"[modallabs] WARNING: {rc.get('name')!r} sets allow_unpinned -- the "
              "stale-volume assertion is DISABLED for this run. A pre-fix place_rig "
              "on berkeley-usd-take renders black skies that pass every cross-worker "
              "equality check.")
        return rc
    if cfg.get("expect_sha256"):
        return rc                      # pinned explicitly in the YAML; respect it
    local = Path(os.environ.get(_BUSD_LOCAL_ENV) or _BUSD_LOCAL_DEFAULT)
    found = {rel: local / rel for rel in _HEROSHOT_PINS if (local / rel).is_file()}
    if _HEROSHOT_PIN_REQUIRED not in found:
        raise RuntimeError(
            f"modallabs/modal: refusing to launch {rc.get('name')!r} -- cannot pin "
            f"{_HEROSHOT_PIN_REQUIRED}: no such file under {local}. The launcher "
            "compares the volume's copy against the local one, and without the local "
            "copy there is nothing to compare. Set "
            f"{_BUSD_LOCAL_ENV}=<path to berkeley-usd>, or set config.allow_unpinned "
            "to launch unverified (which is how black-sky frames got rendered).")
    cfg["expect_sha256"] = {rel: _sha256_file(p) for rel, p in sorted(found.items())}
    skipped = [rel for rel in _HEROSHOT_PINS if rel not in found]
    print(f"[modallabs] heroshot pin ({rc.get('name')}): "
          + ", ".join(f"{rel}={h[:12]}" for rel, h in cfg["expect_sha256"].items())
          + (f"; NOT pinned (absent locally): {skipped}" if skipped else ""))
    return rc


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
# --dry-run MUST NOT BUILD IMAGES.
#
# The check used to live inside main(), which is an @app.local_entrypoint. By the time
# `modal run modal_app.py --config X --dry-run` reaches main(), Modal has already created
# the App and hydrated every image every lane declares -- so the "free" preview could
# trigger a from-source DROID-SLAM/detectron2 compile. That is not free, and for the lane
# whose image had never built it was the expensive path.
#
# This fires at IMPORT time, which is strictly earlier: `modal run` imports the target file
# to discover the app, and only then builds. VERIFIED 2026-08-04 with an argv probe --
# under `modal run`, sys.argv is the modal CLI's own argv
#   ['.../bin/modal', 'run', '<file>', '--config', 'X', '--dry-run']
# and a SystemExit raised at import stops the process before "Initialized"/"Created
# objects" -- no app, no image, no container, nothing billed.
#
# It only fires for `modal run <this file> ... --dry-run`. `python modal_app.py --dry-run`
# already never hydrates anything (see _cli), and importing this module from another modal
# file (scripts/build_image.py does) must not be hijacked.
# ---------------------------------------------------------------------------

def _dry_run_short_circuit() -> None:
    argv = list(sys.argv)
    if "--dry-run" not in argv or len(argv) < 3 or argv[1] != "run":
        return
    ref = argv[2].split("::", 1)[0]
    try:
        same_file = Path(ref).resolve() == Path(__file__).resolve()
    except OSError:
        same_file = False
    if not (same_file or ref.replace(".py", "").replace(".", "/").endswith("modal_app")):
        return
    cfg_path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
        elif a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
    if cfg_path is None:
        return  # let modal's own argument parsing produce the error
    print("[modallabs] --dry-run short-circuit: previewing cost BEFORE any image is "
          "hydrated. No app is created, nothing is built, nothing is billed.")
    blocked = _print_dry_run(_load_orchestrator_cfg(cfg_path))
    raise SystemExit(2 if blocked else 0)


_dry_run_short_circuit()


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
            # README documents a `peft:` block on any hf_* run and the
            # orpheus_voice lane is LoRA-only, but peft was missing here -- so
            # every documented PEFT run failed on import after the GPU was hot.
            "peft",
            # SNAC audio codec, for the orpheus_tts lane. Pure-python + torch, so
            # it needs no apt packages and does not justify a separate typed lane
            # and image. Note orpheus_tts writes WAV with the stdlib `wave` module
            # rather than soundfile, precisely to keep libsndfile out of here.
            "snac",
        )
        .env({
            "HF_HOME": _HF_CACHE_MOUNT,
            "TRANSFORMERS_CACHE": f"{_HF_CACHE_MOUNT}/transformers",
            "HUGGINGFACE_HUB_CACHE": f"{_HF_CACHE_MOUNT}/hub",
            "HF_DATASETS_CACHE": f"{_HF_CACHE_MOUNT}/datasets",
        })
        .add_local_python_source("modallabs")
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

    # Module-level remote function. Modal SDK v1.4.2 fixes (gpu, timeout)
    # at @app.function decoration time -- there is no per-call override
    # (Function.with_options() doesn't exist in this version, .spawn()
    # only takes the function args). Resource overrides require
    # multiple module-level @app.function decorators, one per (gpu,
    # timeout) tuple.
    #
    # This file declares ONE remote function pinned to (_REMOTE_GPU,
    # _REMOTE_TIMEOUT_SEC). main() enforces this at launch time: if any
    # run requests a different (gpu, timeout), it fails loudly rather
    # than silently mis-routing. To support multiple (gpu, timeout)
    # pairs in a single sweep, declare additional module-level @app.function
    # variants and route on cfg.modal.gpu in main().
    #
    # The container is torn down at function exit -- no idle keep-alive.
    # We deliberately do NOT pass keep_warm or min_containers.
    _REMOTE_TIMEOUT_SEC = _LANES[_DEFAULT_LANE]  # back-compat alias

    def _remote_body(run_cfg: dict, run_id: str, resume: bool,
                     lane_timeout_sec: int) -> dict:
        """Shared body for every lane. Arms L3 (deadline) and L4 (dead-man)
        before handing off to the trainer.

        Mounts the persistent `modallabs-hf-cache` volume at `/hf_cache` so
        HuggingFace downloads survive container teardown across runs.
        """
        import threading
        from modallabs.runner import train_one
        import modallabs.models  # noqa: F401  -- register trainers

        started = time.monotonic()
        deadline = started + max(1, lane_timeout_sec - _DEADLINE_MARGIN_SEC)
        run_dir = Path("/runs") / run_id
        stop = threading.Event()

        def _newest_mtime() -> float:
            """Most recent write anywhere under the run dir, or -1 if none.

            Deliberately filesystem-based rather than callback-based:
            `train_one` takes no progress hook, and requiring one would make
            this guard depend on trainer cooperation. Every trainer writes
            checkpoints and logs, so mtime is the one progress signal that is
            true for all of them and cannot be forgotten by a new trainer.
            """
            newest = -1.0
            try:
                for p in run_dir.rglob("*"):
                    try:
                        if p.is_file():
                            newest = max(newest, p.stat().st_mtime)
                    except OSError:
                        continue
            except OSError:
                return newest
            return newest

        def _watch() -> None:
            # L3 + L4. os._exit is deliberate: a wedged trainer may be stuck in
            # a C extension where an exception can never be delivered, and the
            # entire point of this thread is that it works ANYWAY. The container
            # dies, Modal stops billing, and the run resumes from its last
            # checkpoint.
            last_seen = -1.0
            last_change = time.monotonic()
            while not stop.wait(15.0):
                now = time.monotonic()
                if now >= deadline:
                    print(f"[modallabs] L3 deadline watchdog: {lane_timeout_sec}s lane "
                          f"nearly exhausted; terminating to stop billing.", file=sys.stderr)
                    sys.stderr.flush()
                    os._exit(75)
                m = _newest_mtime()
                if m > last_seen:
                    last_seen, last_change = m, now
                    continue
                idle = now - last_change
                # Cold start, image pull and HF download all precede the first
                # write, so the startup budget is separate and larger.
                limit = (_STARTUP_GRACE_SEC if last_seen < 0 else _NO_PROGRESS_KILL_SEC)
                if idle >= limit:
                    what = "no output written since container start" if last_seen < 0 \
                        else f"no write under {run_dir} for {idle:.0f}s"
                    print(f"[modallabs] L4 dead-man switch: {what} (limit {limit}s); "
                          f"terminating to stop billing.", file=sys.stderr)
                    sys.stderr.flush()
                    os._exit(76)

        watcher = threading.Thread(target=_watch, name="modallabs-billguard", daemon=True)
        watcher.start()
        print(f"[modallabs] bill guard armed: lane={lane_timeout_sec}s, "
              f"L3 deadline at {lane_timeout_sec - _DEADLINE_MARGIN_SEC}s, "
              f"L4 idle limit {_NO_PROGRESS_KILL_SEC}s "
              f"(startup grace {_STARTUP_GRACE_SEC}s)", file=sys.stderr)
        try:
            return train_one(
                run_cfg,
                run_id=run_id,
                output_root=Path("/runs"),
                resume=resume,
                force_cpu=False,
            )
        finally:
            stop.set()

    # HF credentials for gated pulls and `push_to_hub` from a Trainer. Attached by name
    # from modallabs.credentials, which is also what the consuming lane checks against --
    # same convention as tram-motion's smpl-credentials, so the launcher and the consumer
    # cannot drift. Attached unconditionally rather than best-effort: a silently absent
    # secret turns into a 401 after the GPU is hot, which is the expensive way to find out.
    _COMMON = dict(
        gpu=_REMOTE_GPU,
        volumes={"/runs": runs_volume, _HF_CACHE_MOUNT: hf_cache_volume},
        secrets=[modal.Secret.from_name(_creds.HF_SECRET_NAME)],
    )

    @app.function(timeout=_LANES["brief"], **_COMMON)
    def _remote_brief(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """10 min lane. Worst case ~$0.92/run on H100."""
        return _remote_body(run_cfg, run_id, resume, _LANES["brief"])

    @app.function(timeout=_LANES["short"], **_COMMON)
    def _remote(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """Default lane, 30 min. Name kept as `_remote` for back-compat."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    @app.function(timeout=_LANES["medium"], **_COMMON)
    def _remote_medium(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """90 min lane."""
        return _remote_body(run_cfg, run_id, resume, _LANES["medium"])

    @app.function(timeout=_LANES["long"], **_COMMON)
    def _remote_long(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """4 h lane. Worst case ~$15.80/run on H100 -- route here only when
        the job genuinely needs it."""
        return _remote_body(run_cfg, run_id, resume, _LANES["long"])

    _LANE_FNS = {"brief": _remote_brief, "short": _remote,
                 "medium": _remote_medium, "long": _remote_long}

    # ------------------------------------------------------------- Wan lane
    # Separate image from `modal_image`: the training image has no video stack,
    # and adding one would slow the cold start of all existing trainers.
    # ComfyUI route (RENDER_CONTRACT M-5 option 2, the one M-6 is written
    # against): core nodes only (WanVaceToVideo, LoraLoaderModelOnly, LoadVideo,
    # CreateVideo/SaveVideo are all core as of v0.28.x). Linux torch wheels on
    # PyPI bundle CUDA, so ComfyUI's own requirements pull a CUDA torch.
    wan_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg")
        .run_commands(
            # Pinned to the EXACT commit of the locally verified install — every
            # node schema the wan_vace_shot trainer emits was read from this
            # commit's comfy_extras sources (WanVaceToVideo, LoadVideo,
            # GetVideoComponents, CreateVideo, SaveVideo, TrimVideoLatent).
            "git init /comfyui && cd /comfyui && "
            "git remote add origin https://github.com/comfyanonymous/ComfyUI.git && "
            "git fetch --depth 1 origin b08debceca73bd3732b731c571bd0b4710281310 && "
            "git checkout FETCH_HEAD",
            "pip install -r /comfyui/requirements.txt",
        )
        .env({
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            "HF_HOME": _HF_CACHE_MOUNT,
        })
        .add_local_python_source("modallabs")
    )
    # create_if_missing=False ON PURPOSE: a typo in the volume name must fail at
    # launch, not mount an empty volume discovered 34.6 GiB short on a hot GPU.
    wan_weights_volume = modal.Volume.from_name("swarmcrp-wan-weights", create_if_missing=False)

    _WAN_COMMON = dict(
        image=wan_image,
        gpu=_REMOTE_GPU,
        volumes={"/runs": runs_volume, _HF_CACHE_MOUNT: hf_cache_volume,
                 "/wan_models": wan_weights_volume},
    )

    @app.function(timeout=_LANES["medium"], **_WAN_COMMON)
    def _remote_wan(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """Wan 2.2 VACE shot-batch lane. One epoch is one shot; weights load once."""
        return _remote_body(run_cfg, run_id, resume, _LANES["medium"])

    # --------------------------------------------------------- LongCat lane
    # LongCat-Video-Avatar-1.5 (Meituan, MIT). Separate image again: upstream pins
    # torch 2.6.0+cu124 and flash_attn 2.7.4.post1, neither of which the training
    # image nor the Wan image carries.
    #
    # UNVERIFIED, flagged rather than hidden:
    #   - the flash_attn wheel URL below is the standard Dao-AILab release naming for
    #     (2.7.4.post1, cu12, torch2.6, cp310, abiFALSE) but has not been fetched. A 404
    #     at build time means the wheel name is wrong; fall back to a source build with
    #     `pip install flash_attn==2.7.4.post1 --no-build-isolation` (slow, ~20 min).
    #   - LongCat-Video is cloned at a moving ref because upstream publishes no tags.
    #     The trainer records the resolved SHA in every manifest (C-004). Pin it here
    #     once a run is accepted, exactly as the ComfyUI clone above is pinned.
    longcat_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
        )
        .apt_install("git", "ffmpeg", "libsndfile1", "build-essential", "ninja-build")
        .pip_install(
            "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install("ninja", "psutil", "packaging", "wheel")
        .pip_install(
            "https://github.com/Dao-AILab/flash-attention/releases/download/"
            "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-"
            "cp310-cp310-linux_x86_64.whl"
        )
        .run_commands(
            # Pinned to the EXACT commit whose source the trainer was written against:
            # the argparse choices (lowercase 480p/720p), the save_fps=25 branch, the
            # '../LongCat-Video' sibling load and the deterministic output filenames were
            # all read from this commit. An unpinned clone can change any of them.
            "git init /opt/LongCat-Video && cd /opt/LongCat-Video && "
            "git remote add origin https://github.com/meituan-longcat/LongCat-Video.git && "
            "git fetch --depth 1 origin 6b3f4b8582a8bc3f20f795735f5383716c4ba794 && "
            "git checkout FETCH_HEAD",
            "pip install -r /opt/LongCat-Video/requirements.txt",
            # Upstream's requirements_avatar.txt lists two packages that do not exist on
            # PyPI (both verified 404 against pypi.org/pypi/<name>/json on 2026-07-31):
            #   libsndfile1==0.0.1      an APT package, not a Python one. `soundfile` and
            #                           `librosa` bind to the SYSTEM libsndfile, which is
            #                           installed via apt_install above. Never imported
            #                           as a Python module anywhere in the repo.
            #   tritonserverclient==0.0.6  does not exist (the real package is
            #                           `tritonclient`). Imported nowhere in the repo --
            #                           verified by grep over the pinned tarball.
            # Filtering them is required: pip fails the whole file on either one, so the
            # image cannot build at all otherwise. Re-check this filter if the pin moves.
            "grep -vE '^(libsndfile1|tritonserverclient)==' "
            "/opt/LongCat-Video/requirements_avatar.txt > /tmp/req_avatar.txt",
            "pip install -r /tmp/req_avatar.txt",
        )
        .env({
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            "HF_HOME": _HF_CACHE_MOUNT,
        })
        .add_local_python_source("modallabs", "longcatavatar")
    )
    # create_if_missing=False ON PURPOSE, same rule as the Wan volume: a typo must fail
    # at launch, not mount an empty volume discovered 21.5 GB short on a hot GPU.
    # Stage it first: `modal run scripts/stage_weights.py` in the LongCatAvatar repo.
    longcat_weights_volume = modal.Volume.from_name(
        "longcat-avatar-weights", create_if_missing=False
    )

    _LONGCAT_COMMON = dict(
        image=longcat_image,
        # NOTE: _REMOTE_GPU is ONE H100. Upstream documents only --nproc_per_node=2
        # --context_parallel_size=2; the trainer runs context_parallel_size=1 and says
        # so in its log and manifest. Moving to gpu="H100:2" doubles per-second spend
        # while _estimate_cost_usd still prices the run at the 1x H100 rate, so it must
        # land together with an "H100x2" entry in _GPU_HOURLY_USD or the gate lies.
        gpu=_REMOTE_GPU,
        volumes={"/runs": runs_volume, _HF_CACHE_MOUNT: hf_cache_volume,
                 "/longcat_weights": longcat_weights_volume},
    )

    @app.function(timeout=_LANES["medium"], **_LONGCAT_COMMON)
    def _remote_longcat(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """LongCat avatar batch lane. One epoch is one job; weights load once."""
        return _remote_body(run_cfg, run_id, resume, _LANES["medium"])

    # --------------------------------------------------------- heroshot lane
    # Cycles take chunks for the berkeley-usd 270-walk heroshot (type
    # "heroshot_take"). Separate image on purpose: it is a Blender tarball plus
    # numpy/pillow and NOTHING else -- no torch, no HF stack -- so its cold
    # start stays light and no existing lane's start slows down.
    #
    # Blender install pattern is the PROVEN one from actionmesh_lane.py:104-184
    # (wget tarball -> /opt/blender -> `blender --version` probe fails the
    # BUILD, not the H100 run, if a headless lib is missing). 4.5.12 pinned:
    # bpy.app.version_string is a place_rig digest input, so the fleet must
    # run the same Blender the local takes run (4.5.12 LTS on the M5).
    # download.blender.org 403s some clients (observed 2026-08-12 from the
    # launcher); the OCF Berkeley mirror carries the identical file, hence the
    # fallback. The probe still gates whichever one won.
    _BLENDER_VER = "4.5.12"
    _BLENDER_TAR = f"blender-{_BLENDER_VER}-linux-x64.tar.xz"
    heroshot_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "wget", "xz-utils", "ca-certificates",
            # Blender headless (-b) runtime deps: the list proven for the
            # 3.5.1 image in actionmesh_lane.py. If 4.5 ever needs one more,
            # the --version probe below fails the build loudly.
            "libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxi6",
            "libxxf86vm1", "libxfixes3", "libxkbcommon0", "libx11-6", "libxext6",
        )
        .pip_install("numpy", "pillow")   # trainer-side band comparison only
        .run_commands(
            "mkdir -p /opt/blender && "
            f"(wget -q https://download.blender.org/release/Blender4.5/{_BLENDER_TAR} "
            "-O /tmp/blender.tar.xz || "
            f"wget -q https://mirrors.ocf.berkeley.edu/blender/release/Blender4.5/{_BLENDER_TAR} "
            "-O /tmp/blender.tar.xz) && "
            "tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && "
            "rm /tmp/blender.tar.xz && "
            "/opt/blender/blender --version"
        )
        # pyyaml: modallabs.runner imports yaml at module scope, so EVERY lane
        # image needs it -- found the paid way on pilot r1 (container import
        # died in seconds; torch is NOT needed: _resolve_device and
        # set_global_seed both guard their torch imports). A separate layer
        # AFTER the blender step on purpose: the cached 300 MB tarball layer
        # survives this fix.
        .pip_install("pyyaml")
        .env({"PYTHONUNBUFFERED": "1"})
        .add_local_python_source("modallabs")
    )
    # create_if_missing=False ON PURPOSE, same rule as the Wan/LongCat/TRAM
    # volumes: a typo must fail at launch, not mount an empty volume that
    # place_rig then hashes as a 0-byte world on a hot GPU. Stage it first:
    #   hf-gpu-cluster-optimizer/scripts/stage_berkeley_take.sh
    busd_take_volume = modal.Volume.from_name(
        "berkeley-usd-take", create_if_missing=False
    )
    # CYCLES KERNEL CACHE.
    #
    # WHY THE FIRST RENDER IN A FRESH CONTAINER STALLS. MEASURED 2026-08-12 by
    # listing /opt/blender: the 4.5.12 tarball ships CUDA cubins for sm_30, 35,
    # 37, 50, 52, 60, 61, 70, 75, 86, 89 and 120 -- and NOT sm_90. H100 and
    # H200 are sm_90, so Cycles finds no cubin, falls back to the one shipped
    # PTX (lib/kernel_compute_75.ptx.zst), and the CUDA DRIVER JIT-compiles it
    # to sm_90 SASS at module-load time. That is the "Loading render kernels
    # (may take a few minutes the first time)" stall. On a cold H100 with an
    # empty cache it MEASURED 548.8 s (probe prime run, 2026-08-12) -- longer
    # than the brief lane's whole 480 s L3 window, which is how fleet r1 lost
    # all eight workers to the watchdog and then paid for the crash-retries.
    # The same render on a T4 (sm_75, cubin SHIPPED) finishes in 3.37 s with no
    # stall at all: the stall is a property of the ARCH CHOICE, not of Cycles.
    #
    # WHERE THE CACHE ACTUALLY GOES -- and where it does NOT.
    # This mount used to be "/root/.cache", on the reasoning that Cycles caches
    # under ~/.cache. It does not, and the volume was decorative. A
    # filesystem-wide before/after diff around a real compile found the bytes
    # in two places, NEITHER of them ~/.cache:
    #    51,159,125 B  CUDA driver JIT cache   default $HOME/.nv/ComputeCache
    #     1,114,112 B  OptiX disk cache        default /var/tmp/OptixCache_root
    # Both honour an env-var redirect (CUDA_CACHE_PATH / OPTIX_CACHE_PATH),
    # MEASURED: with the vars set, 52,273,269 B landed on this volume and the
    # default locations stayed empty. So the volume is mounted at a path of our
    # own and the two compilers are pointed into it explicitly.
    #
    # The subdirectory names are the ones the priming run actually wrote and
    # the cold test actually read. They are pinned, not tidied: renaming them
    # orphans a cache that cost 548.8 s of H100 to build.
    #
    # IS THE CACHE FEATURE-DEPENDENT? Cycles does pick OptiX module variants by
    # kernel feature set, so a cache primed against one look is not
    # automatically a cache for another -- a fair objection, and it was TESTED
    # rather than argued. A fresh container running the post-NPR place_rig with
    # the maximal feature set (--npr cel --outline both --exr-aov: twelve light
    # AOVs on top of Z, cryptomatte and object-index) against a cache primed by
    # the PLAIN look loaded kernels in 0.81 s and left the volume byte-identical
    # at 52,273,269 B -- zero additional compilation. That is what the mechanism
    # predicts: the tarball ships exactly ONE CUDA artifact and the 548.8 s is
    # the driver JIT-ing that single fixed module, which render passes cannot
    # change. Feature-adaptive compilation is a compile-from-source path needing
    # nvcc and kernel.cu, neither of which is in the tarball.
    #
    # WHAT WOULD INVALIDATE IT: the cache key includes the DRIVER VERSION. Both
    # proving containers ran 580.95.05. A worker landing on a host with a
    # different driver misses and pays the JIT again -- a slow render, never a
    # wrong one, but it must fit the lane or the watchdog turns it into a
    # crash-retry loop. That is the residual risk, and it is why the attempt
    # bound above matters.
    #
    # create_if_missing=True is correct HERE (unlike the weights volumes): an
    # empty cache is not an error state, it is just the first run. CONFIRMED
    # lazily created on first app hydration -- the volume did not exist before
    # the first probe run and no explicit `modal volume create` was needed.
    heroshot_kernel_cache = modal.Volume.from_name(
        "heroshot-kernel-cache", create_if_missing=True
    )
    _KCACHE_MOUNT = "/kcache"
    _KCACHE_OPTIX = f"{_KCACHE_MOUNT}/optix_env"
    _KCACHE_CUDA = f"{_KCACHE_MOUNT}/nv_env"
    # .env() is a BUILD step and heroshot_image already ends with
    # .add_local_python_source (see the TRAM note below for the InvalidError
    # this avoids), so these go in as a runtime Secret. Nothing secret in it.
    _heroshot_env = modal.Secret.from_dict({
        "OPTIX_CACHE_PATH": _KCACHE_OPTIX,
        "CUDA_CACHE_PATH": _KCACHE_CUDA,
        "CUDA_CACHE_DISABLE": "0",
        # The driver's default compute-cache cap can evict a 51 MB megakernel.
        # Nothing may be evicted between the priming run and the fleet.
        "CUDA_CACHE_MAXSIZE": str(4 * 1024 ** 3),
    })

    _HEROSHOT_COMMON = dict(
        image=heroshot_image,
        gpu=_REMOTE_GPU,
        secrets=[_heroshot_env],
        volumes={"/runs": runs_volume, "/busd": busd_take_volume,
                 _KCACHE_MOUNT: heroshot_kernel_cache},
    )

    @app.function(timeout=_LANES["brief"], **_HEROSHOT_COMMON)
    def _remote_heroshot_brief(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """10-min heroshot chunk lane. The fleet default: a time-balanced
        ~95-frame chunk is ~421 s of M5-Metal work, and the whole point of
        the pilot is to prove it fits under 600 s on H100-OptiX. Worst case
        $0.92/run at the gate's table rate."""
        return _remote_body(run_cfg, run_id, resume, _LANES["brief"])

    @app.function(timeout=_LANES["short"], **_HEROSHOT_COMMON)
    def _remote_heroshot(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min heroshot lane, for chunks that measure too slow for brief
        (or a full 753-frame single-worker take). BILL_SAFETY: the timeout is
        the worst-case bill, so route here only when the pilot's measured s/f
        says brief does not fit."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    # ---------------- heroshot NON-sm_90 variants (Will, 2026-08-12) --------
    # Blender 4.5.12 ships CUDA cubins for sm_86 (A10G) and sm_89 (L4, L40S)
    # and NOT for sm_90 (H100) -- modal_app's own kernel-cache comment block
    # documents the measured consequence: 548.8 s of driver JIT on a cold H100
    # vs 3.37 s on a T4 whose cubin ships. On these three cards the JIT never
    # happens, so they carry NO dependency on the primed /kcache volume
    # surviving, no priming step, and no JIT-driven crash-retry mode. H100
    # stays declared above as the fallback. The _heroshot_env Secret is
    # harmless here (cache dirs are arch+driver keyed; sm_8x writes pennies of
    # bytes). Each (gpu, lane) pair is worst-case exposure only when a config
    # routes to it; dispatch is the explicit _HEROSHOT_FNS table, never a
    # heuristic.
    @app.function(timeout=_LANES["brief"], **{**_HEROSHOT_COMMON, "gpu": "L40S"})
    def _remote_heroshot_brief_l40s(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """10-min heroshot lane, L40S (sm_89: shipped cubin, zero JIT)."""
        return _remote_body(run_cfg, run_id, resume, _LANES["brief"])

    @app.function(timeout=_LANES["short"], **{**_HEROSHOT_COMMON, "gpu": "L40S"})
    def _remote_heroshot_short_l40s(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min heroshot lane, L40S."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    @app.function(timeout=_LANES["brief"], **{**_HEROSHOT_COMMON, "gpu": "A10G"})
    def _remote_heroshot_brief_a10g(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """10-min heroshot lane, A10G (sm_86: shipped cubin, zero JIT)."""
        return _remote_body(run_cfg, run_id, resume, _LANES["brief"])

    @app.function(timeout=_LANES["short"], **{**_HEROSHOT_COMMON, "gpu": "A10G"})
    def _remote_heroshot_short_a10g(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min heroshot lane, A10G."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    @app.function(timeout=_LANES["brief"], **{**_HEROSHOT_COMMON, "gpu": "L4"})
    def _remote_heroshot_brief_l4(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """10-min heroshot lane, L4 (sm_89: shipped cubin, zero JIT)."""
        return _remote_body(run_cfg, run_id, resume, _LANES["brief"])

    @app.function(timeout=_LANES["short"], **{**_HEROSHOT_COMMON, "gpu": "L4"})
    def _remote_heroshot_short_l4(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min heroshot lane, L4."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    # (gpu, lane) -> fn for heroshot_take, consulted by main() BEFORE the
    # generic homogeneous-GPU check. H100 rows point at the original two
    # functions, so an H100 config routes byte-identically to before.
    _HEROSHOT_FNS = {
        ("H100", "brief"): _remote_heroshot_brief,
        ("H100", "short"): _remote_heroshot,
        ("L40S", "brief"): _remote_heroshot_brief_l40s,
        ("L40S", "short"): _remote_heroshot_short_l40s,
        ("A10G", "brief"): _remote_heroshot_brief_a10g,
        ("A10G", "short"): _remote_heroshot_short_a10g,
        ("L4", "brief"): _remote_heroshot_brief_l4,
        ("L4", "short"): _remote_heroshot_short_l4,
    }

    @app.function(timeout=_LANES["short"], **_LONGCAT_COMMON)
    def _remote_longcat_short(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min LongCat lane, for fanning single-job runs out concurrently.

        Exists so the cost gate stays truthful. estimate_total_cost_usd prices
        cfg.modal.max_runtime_sec, but a type-routed run used to land in the medium
        container regardless -- so a config asking for 1800s would be GATED at $2.75
        while actually being allowed to burn 5400s. Six such runs would gate at $16.50
        and bill up to $49.50. With this lane the container timeout equals what the
        gate charged for.
        """
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    # ------------------------------------------------------------ TRAM lane
    # TRAM (yufu-wang/tram, ECCV 2024, MIT): DROID-SLAM camera solve robustified against
    # dynamic humans, then VIMO for SMPL body motion.
    #
    # THE IMAGE DEFINITION IS NOT HERE, AND THAT IS THE FIX.
    # ------------------------------------------------------------------------------
    # This file used to carry its own inline cu118 tram_image, ~120 lines. It was DELETED
    # on 2026-08-04. Two definitions of one image existed, they had diverged, and the one
    # in this file was the one that had never successfully built:
    #
    #   deleted (inline, cu118)      never built. Its build-time probe was
    #                                `cd /opt/tram && python -c "import lib.pipeline"`,
    #                                which CANNOT succeed on a CPU builder:
    #                                lib/pipeline/__init__.py -> tools.py:12 ->
    #                                deva_track.py runs DEVA(cfg).cuda().eval() AND
    #                                torch.load('data/pretrain/DEVA-propagation.pth') at
    #                                MODULE SCOPE. That needs a GPU the builder does not
    #                                have and a weight file that lives on the Volume, not
    #                                in the image. Proven by running it. Every build of
    #                                this image was therefore going to fail at that step,
    #                                after paying for detectron2 + pytorch3d + DROID-SLAM
    #                                from source.
    #
    #   kept (tram-motion/modal/     BUILT AND PROVEN: im-Uc0Lnpt85CAv3Jz0TDpbJ9. DROID-SLAM
    #   tram_image.py, cu121)        compiled, cuobjdump-verified sm_80/86/90 cubins,
    #                                build_check.py rc 0, $0.27. It probes the image with
    #                                modal/build_check.py, which imports the LEAF modules
    #                                that matter (torch, droid_backends, lietorch_backends,
    #                                detectron2, pytorch3d, smplx, chumpy, cv2 ...) and
    #                                verifies the compiled cubins with cuobjdump -- none of
    #                                which needs a GPU or a Volume. That is the difference
    #                                between a probe that validates the image and a probe
    #                                that validates the image PLUS a GPU PLUS the weights.
    #
    # The kept definition deviates from upstream's cu118 pin deliberately and says so in
    # its own docstring: pytorch3d publishes a prebuilt py310_cu121_pyt240 wheel and 403s
    # on cu118, so cu121 removes a 30-60 minute from-source nvcc build. torch stays at
    # upstream's exact 2.4.0.
    #
    # It is IMPORTED, not copied, so this file cannot drift from the artifact that was
    # actually proven.
    #
    # RESOLUTION ORDER. Each step exists for a process that actually occurs, and the third
    # one is the one that was learned the hard way:
    #
    #   1. `import tram_image` -- succeeds INSIDE a TRAM container, where the proven
    #      definition is mounted at /root/tram_image.py by its own
    #      .add_local_python_source("tram_image"). Verified that it imports with its
    #      sibling files absent (add_local_file does not stat at definition time), which
    #      is what makes this work in a container that carries only the one module.
    #   2. the definition FILE on this machine: $TRAM_IMAGE_DEF, else repo-relative.
    #      This is the local launcher path.
    #   3. neither -> DO NOT declare the lane, and say so on stderr. modal_app is imported
    #      inside every OTHER lane's container too (_done_runs on modal_image, _remote_wan,
    #      _remote_longcat), and none of them has this file or this module. An earlier
    #      version raised here, which crash-looped a container that has nothing to do with
    #      TRAM: OBSERVED 2026-08-04, app ap-tXmJLExtGZmF3bIzXCBPoe --
    #      "RuntimeError: the proven TRAM image definition is not at
    #      /root/tram-motion/modal/tram_image.py". Found by hydrating the image for real.
    _tram_image_mod = None
    try:
        import tram_image as _tram_image_mod  # noqa: E402 -- mounted in a TRAM container
    except ImportError:
        _tram_def = Path(os.environ.get(
            "TRAM_IMAGE_DEF",
            Path(__file__).resolve().parent.parent / "tram-motion" / "modal" / "tram_image.py",
        ))
        if _tram_def.exists():
            if str(_tram_def.parent) not in sys.path:
                sys.path.insert(0, str(_tram_def.parent))
            import tram_image as _tram_image_mod  # noqa: E402 -- path-dependent by design
        else:
            print(
                "[modallabs] TRAM lane NOT declared: no `tram_image` module on sys.path "
                f"and no definition file at {_tram_def}. Expected inside a non-TRAM "
                "container. On a launcher it means the tram-motion repo moved -- set "
                "TRAM_IMAGE_DEF=<...>/tram-motion/modal/tram_image.py. The inline copy "
                "that used to live in this file was deleted deliberately: it had never "
                "built.",
                file=sys.stderr,
            )

    # trammotion is the lane package (tram-motion/lane/trammotion); modallabs is this
    # harness. The proven definition already carries `.add_local_python_source("tram_image")`
    # for its own build entrypoint; these two are what the RUNTIME needs. Adding a mount
    # layer does not invalidate any build layer, so this still resolves to the proven build.
    tram_image = (
        _tram_image_mod.tram_image.add_local_python_source("modallabs", "trammotion")
        if _tram_image_mod is not None else None
    )

    # HF_HOME is set at RUNTIME, not as an image layer, and that is not a style choice.
    # OBSERVED 2026-08-04: chaining `.env({"HF_HOME": ...})` here raised
    #   InvalidError: An image tried to run a build step after using `image.add_local_*`
    # because the proven definition already ends with `.add_local_python_source(
    # "tram_image")`, and .env() is a BUILD step. Mount layers may stack on mount layers;
    # build steps may not follow them. Caught by actually hydrating the image on the CPU
    # builder rather than by reading the code.
    #
    # It is needed at all because _TRAM_COMMON mounts the hf-cache volume: without the var
    # the mount is decorative, and any HuggingFace fetch would land in the container's
    # ephemeral cache and be re-fetched on an H100 every cold start. Secret.from_dict is
    # Modal's documented way to inject env vars per function; nothing secret is in it.
    #
    # Nothing else from the deleted inline image's env block is carried over:
    # PYOPENGL_PLATFORM and MPLBACKEND were there for pyrender and matplotlib, and
    # build_check.py imports both successfully in the proven image without them
    # (CORE_IMPORTS lines 49 and 68, rc 0).
    _tram_env = modal.Secret.from_dict({"HF_HOME": _HF_CACHE_MOUNT})

    # Routed on the run's own type, not on its timeout: these lanes differ by
    # IMAGE and by VOLUME, which _lane_for cannot see. A single explicit key,
    # never a heuristic.
    # ---- TRELLIS.2-4B image-to-3D (TheExperiment) --------------------------
    #
    # Its own image because the upstream setup.sh compiles SIX CUDA extensions
    # (flash-attn, nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm) against a
    # pinned CUDA 12.4 / torch 2.6.0 toolchain. THE BUILD IS THE RISK IN THIS
    # LANE, NOT THE INFERENCE: measured upstream generation on H100 is ~3 s at
    # 512^3, ~17 s at 1024^3, ~60 s at 1536^3, so the container is dominated by
    # hydration and the 4B weight load, which is why weights get a volume.
    #
    # UNPROVEN AT TIME OF WRITING: this image has never been built. setup.sh is
    # invoked WITHOUT --new-env (we are not creating a conda env inside a Modal
    # image); if it hard-depends on conda the build fails loudly here, which is
    # the correct outcome -- not a silent fallback to a partial install.
    # ---- TRELLIS.2-4B image-to-3D (TheExperiment) --------------------------
    #
    # PORTED FROM THE PROVEN RECIPE at ~/furni/trellis/modal_app.py, which has
    # served TRELLIS-1 since 2026-07-30. TRELLIS.2 is a different codebase (Dec
    # 2025, six CUDA extensions, three built from source) but every structural
    # lesson in that file applies, and ignoring them cost six failed builds here:
    #
    #  1. LAYER ORDER: cheap CPU layers first, so a failure in the expensive GPU
    #     layer never re-runs them. Each attempt below re-ran the whole compile
    #     because it was one monolithic step.
    #  2. `bash -lc`, never a bare string: Modal runs run_commands under /bin/sh
    #     (dash), and POSIX `.` takes a FILENAME ONLY, silently discarding
    #     `--basic --flash-attn ...`. setup.sh printed its usage and exited 0.
    #  3. BUILD GPU == SERVE GPU. Their note, verbatim: "building on one arch and
    #     serving on another is how you get a container that imports fine and
    #     then dies inside the first kernel launch." H100/sm_90 both sides.
    #  4. VERIFY IN THE BUILD, on a GPU. "Fail the BUILD, not the first request."
    #  5. EMPTY HF_HOME IN THE LAST LAYER. Modal refuses to mount a Volume over a
    #     non-empty directory, and HF_HOME is a build-time env, so anything that
    #     touches it makes the runtime /weights mount fail. Not yet hit here --
    #     read from their file, not from our own outage.
    #
    # MEASURED HERE, and the reason the compile carries no GPU while the import
    # does: the GPU-less build compiled all seven CUDA objects at sm_90 and
    # installed every wheel, then died on IMPORT --
    #   o_voxel -> flex_gemm -> kernels.triton -> @triton_autotune at module
    #   scope -> triton driver._create_driver
    #   -> RuntimeError: 0 active drivers ([]). There should only be one.
    # Triton resolves its backend when flex_gemm is IMPORTED. So: toolkit to
    # build, driver to import. The device is attached only where it is required.
    trellis2_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive",
              "PYTHONUNBUFFERED": "1",
              # setup.sh never sets this, so it governs what nvcc emits. 9.0 is
              # H100. Add 8.0 if this lane is ever routed to A100.
              "TORCH_CUDA_ARCH_LIST": "9.0",
              "MAX_JOBS": "8"})
        .apt_install(
            "git", "build-essential", "ninja-build", "cmake",
            "libgl1", "libglib2.0-0", "libx11-6", "ca-certificates", "curl",
            # clang is a LINK dependency, not a compile one. Modal's add_python
            # ships a Python built with clang, so sysconfig reports CXX=clang++
            # and distutils links the extension .so with it. nvcc compiled all
            # seven objects with gcc and succeeded; the link then died on
            # `clang++: No such file or directory`, taking out o-voxel, cumesh
            # and flex-gemm identically. The error names a compiler, so it reads
            # as a toolchain mismatch when it is an absent binary.
            "clang",
        )
        .pip_install(
            "torch==2.6.0", "torchvision==0.21.0",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        # setup.sh uses --no-build-isolation, so the AMBIENT env must carry the
        # build backend; with isolation pip provisions it itself. o-voxel died on
        # `error: invalid command 'bdist_wheel'`, which is `wheel` missing.
        .pip_install("wheel", "setuptools", "packaging", "ninja")
        .run_commands(
            "git clone -b main --recursive "
            "https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2",
            "cd /opt/TRELLIS.2 && echo TRELLIS2_BUILD_SHA && git rev-parse HEAD",
            # setup.sh's "No supported GPU found" is a PLATFORM probe, quoted
            # from its source: `if command -v nvidia-smi > /dev/null`. It tests
            # that the BINARY EXISTS, never queries a device, and never sets
            # TORCH_CUDA_ARCH_LIST. The CUDA devel base has no nvidia-smi because
            # that ships with the DRIVER at container runtime. On the GPU layer
            # below the real one is present; this stub only covers CPU layers.
            "printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/nvidia-smi "
            "&& chmod +x /usr/local/bin/nvidia-smi",
        )
        # ---- the expensive layer: six CUDA extensions, no GPU needed ---------
        .run_commands(
            "bash -lc 'cd /opt/TRELLIS.2 && . ./setup.sh --basic --flash-attn "
            "--nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm'",
        )
        # ---- verification, GPU-gated, seconds ---------------------------------
        # Separate layer ON PURPOSE (lesson 1): a failure here must not re-run the
        # compile above. Two checks, not one, because they fail for unrelated
        # reasons -- a single message already misattributed an import failure as
        # "setup.sh built no wheels" while the wheels were the part that worked.
        .run_commands(
            # NOT `nvidia-smi`: the CPU layers above put a STUB at
            # /usr/local/bin/nvidia-smi that exits 0 silently, and /usr/local/bin
            # precedes /usr/bin, so the stub would answer and "prove" a device
            # that may not be attached. torch asks the driver directly and cannot
            # be satisfied by a shim.
            "bash -lc 'python -c \"import torch; assert torch.cuda.is_available(), "
            "\\\"NO CUDA DEVICE on the verification layer\\\"; "
            "print(\\\"CUDA_OK\\\", torch.cuda.get_device_name(0))\"'",
            # PRINT WHAT IS INSTALLED BEFORE ASSERTING ANYTHING. The previous
            # version asserted `import flexgemm` and `import cumesh` -- names I
            # guessed from the setup.sh FLAGS rather than read from the code. The
            # real module is `flex_gemm`, visible in attempt 7's own traceback
            # (o_voxel/postprocess.py line 9: `from flex_gemm.ops.grid_sample
            # import grid_sample_3d`). A guessed name turns a working install
            # into ModuleNotFoundError and reads as a build failure.
            "bash -lc 'pip list --format=freeze | grep -iE \"voxel|gemm|mesh|nvdiff|flash\" "
            "|| true'",
            # o_voxel transitively imports flex_gemm, so this one line proves
            # both, and it is the module to_glb() is actually reached through.
            "bash -lc 'python -c \"import o_voxel, flex_gemm; "
            "print(\\\"EXTENSIONS_IMPORT_OK\\\")\"'",
            # trellis2 itself is NEVER pip-installed: setup.sh installs deps and
            # extensions, and the package is a source tree upstream runs from its
            # own directory. PYTHONPATH is set for the container by .env() below,
            # but .env() applies to LATER steps, so it must be inline here.
            "bash -lc 'cd /opt/TRELLIS.2 && python -c \"from trellis2.pipelines "
            "import Trellis2ImageTo3DPipeline; print(\\\"FULL_IMPORT_OK\\\")\"'",
            gpu=_REMOTE_GPU,
        )
        .pip_install("pyyaml", "pillow", "trimesh")
        .env({"PYTHONPATH": "/opt/TRELLIS.2",
              "HF_HOME": "/weights",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        # LAST, and deliberately so (lesson 5): Modal will not mount the weights
        # Volume over a non-empty /weights, and HF_HOME above is a build env.
        .run_commands("rm -rf /weights && mkdir -p /weights")
        .add_local_python_source("modallabs")
    )
    # create_if_missing=False for views ON PURPOSE, same rule as every other
    # staged volume: a typo must fail at launch, not mount an empty volume that
    # the trainer then reconstructs as nothing on a hot H100. Stage it first:
    #   scripts/stage_theexperiment_views.sh
    trellis2_views_volume = modal.Volume.from_name(
        "theexperiment-views", create_if_missing=False)
    # Weights DO get create_if_missing: the 4B download is idempotent and paid
    # once, and an empty weights volume self-heals on the first run.
    trellis2_weights_volume = modal.Volume.from_name(
        "trellis2-weights", create_if_missing=True)

    # TRELLIS.2's image conditioning encoder is DINOv3
    # (facebook/dinov3-vitl16-pretrain-lvd1689m), which is a GATED HuggingFace
    # repo. Without a token the pipeline dies at from_pretrained with a 401
    # GatedRepoError AFTER the container is hot -- measured 185.4 s of H100
    # before it raised. The Secret already exists (credentials.py:43, carrying
    # HF_TOKEN) and is used by other lanes; it was simply never mounted here.
    #
    # A TOKEN IS NECESSARY AND NOT SUFFICIENT. Gating is per-account access, so
    # the account behind the token must ALSO have accepted the model's terms at
    # huggingface.co. If it has not, this mount changes the error text and
    # nothing else.
    _trellis2_hf = modal.Secret.from_name("huggingface-secret")

    _TRELLIS2_COMMON = dict(
        image=trellis2_image,
        gpu=_REMOTE_GPU,          # H100. >=24 GB is the upstream floor.
        secrets=[_trellis2_hf],
        volumes={"/runs": runs_volume,
                 "/views": trellis2_views_volume,
                 "/weights": trellis2_weights_volume},
    )

    @app.function(timeout=_LANES["short"], **_TRELLIS2_COMMON)
    def _remote_trellis2(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min TRELLIS.2 lane. One weight load, N arms -- the load dominates,
        so arms that share it are nearly free. Worst case $2.75 at the table."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    @app.function(timeout=_LANES["medium"], **_TRELLIS2_COMMON)
    def _remote_trellis2_medium(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """90-min TRELLIS.2 lane, for a 1536^3 sweep across several view sets.
        BILL_SAFETY: the timeout IS the worst-case bill -- $8.25 here."""
        return _remote_body(run_cfg, run_id, resume, _LANES["medium"])

    # ------------------------------------------------------------ Kimodo lane
    # Kimodo text-to-motion (nv-tlabs, Apache-2.0 code). Separate image: the
    # training image carries neither kimodo nor the LLM2Vec stack.
    #
    # A10G, NOT H100, deliberately. The denoiser is 282M parameters; the only heavy
    # component is the LLM2Vec/Llama-3-8B text encoder, and TEXT_ENCODER_DEVICE=cpu
    # keeps that off the GPU entirely (<3 GB). BILL_SAFETY: the lane timeout IS the
    # worst-case bill, so H100 would price a two-clip generation like a training run.
    # A10G brief = $0.18 worst case at the table rate; H100 brief would be ~5x that.
    #
    # The 16 GB Llama pull lands on the SHARED HF CACHE VOLUME, so it is paid once
    # across every future run rather than per launch.
    kimodo_image = (
        modal.Image.debian_slim(python_version="3.11")
        # cmake + a C++ toolchain are for MotionCorrection, which is a CMake
        # EXTENSION (its setup.py raises "CMake must be installed to build this
        # package"). debian_slim carries neither.
        .apt_install("git", "cmake", "build-essential")
        .pip_install("torch", "huggingface_hub", "hf_transfer")
        .run_commands(
            "git clone --depth 1 https://github.com/nv-tlabs/kimodo.git /kimodo",
            # `pip install -e .` FAILS to build upstream (MEASURED 2026-08-30 locally).
            # Install the declared dependency list and run from source on PYTHONPATH.
            "pip install hydra-core omegaconf 'numpy>=1.23' scipy 'transformers==5.1.0' "
            "'peft>=0.18' einops tqdm pydantic filelock trimesh pillow bvhio safetensors boto3",
            # MotionCorrection ships INSIDE the repo as its own package and the
            # default postprocess path REQUIRES it -- measured 2026-08-30, a run
            # reached postprocess and died with "the motion_correction package is
            # not installed". It is what cleans foot skating, which Kimodo names as
            # a stated limitation, so install it rather than passing --no-postprocess.
            # NOT `-e`: the editable path fails to build for BOTH upstream packages
            # (measured locally 2026-08-30 for the root package, and on the Modal
            # builder for this one). A plain install works and is what we want in an
            # image anyway -- nothing here is being edited in place.
            "pip install /kimodo/MotionCorrection",
        )
        .env({
            "PYTHONPATH": "/kimodo",
            "HF_HOME": _HF_CACHE_MOUNT,
            "TEXT_ENCODER_DEVICE": "cpu",
        })
        .add_local_python_source("modallabs")
    )

    _KIMODO_COMMON = dict(
        image=kimodo_image,
        gpu="A10G",
        secrets=[modal.Secret.from_name(_creds.HF_SECRET_NAME)],
        volumes={"/runs": runs_volume, _HF_CACHE_MOUNT: hf_cache_volume},
    )

    @app.function(timeout=_LANES["brief"], **_KIMODO_COMMON)
    def _remote_kimodo_brief(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """10-min Kimodo lane, A10G. Worst case $0.18 at the A10G table rate.
        Generation is seconds; the cold cost is the one-time Llama pull onto the
        shared HF cache volume."""
        return _remote_body(run_cfg, run_id, resume, _LANES["brief"])

    @app.function(timeout=_LANES["short"], **_KIMODO_COMMON)
    def _remote_kimodo_short(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min Kimodo lane, A10G, for a cold cache or a long multi-clip batch.
        Worst case $0.55."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])


    # ---- comfy_sheet: ComfyUI reference sheets on an H100 ----------------------------------
    #
    # One weight load, N arms. Krea2 int8 is 12,866 MB resident (MEASURED on an M5) and the load
    # dominates the container, so arms that share it are nearly free -- trellis2_recon's argument,
    # applied. 105 sheets as 105 containers pays that load 105 times.
    comfy_sheet_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
        .run_commands(
            # TORCH VERSION IS LOAD-BEARING, and pinning the cu124 index was a MEASURED mistake.
            # That index tops out well below what ComfyUI needs, and the build died in
            # comfy_kitchen/backends/eager/na.py: torch.library.custom_op could not infer a schema
            # for `kernel_size: list[int]`, because older torch.library only accepts typing.List.
            # The local M5 that DOES run this stack has torch 2.14.0, which pip gives from the
            # DEFAULT index -- and on Linux the default wheel is already the CUDA build, so naming
            # an index bought nothing and cost the build. No index pin.
            #
            # It is still not the CPU wheel: a CPU ComfyUI starts happily and samples at a rate
            # indistinguishable from a hang, which is a failure mode that bills for an hour, so the
            # build asserts CUDA is compiled in below.
            "pip install --upgrade pip",
            "pip install torch torchvision",
            "python -c \"import torch; print('TORCH', torch.__version__);"
            " assert torch.version.cuda, 'CPU-only torch wheel: this would sample at a rate that"
            " looks like a hang while billing'; print('CUDA', torch.version.cuda)\"",
            "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI",
            "pip install -r /opt/ComfyUI/requirements.txt",
            "git clone --depth 1 https://github.com/ethanfel/ComfyUI-Krea2TextEncoder.git "
            "/opt/ComfyUI/custom_nodes/ComfyUI-Krea2TextEncoder",
            "pip install --upgrade gguf",
        )
        # BUILD-TIME PROOF that the custom node actually REGISTERS, and that init_extra_nodes
        # really ran. The check is a FILE (lanes/verify_nodes.py) rather than an inline python -c:
        # the inline form needed four levels of shell and Python quote escaping, and one build
        # failure was partly the quoting hiding a real fault. See that file for both measured
        # failures it guards.
        #
        # gpu=_REMOTE_GPU is NOT optional. Modal builds on CPU, and ComfyUI's init_extra_nodes()
        # initialises model management, which probes CUDA and raises "Found no NVIDIA driver"
        # before a single node registers. The trellis2 image carries gpu= on its own verification
        # step for exactly this reason -- the pattern was already in the repo, and reading it was
        # not the same as applying it. The build pays seconds of GPU to prove the container is
        # sound before a 24 GB weight pull and a queued batch.
        .add_local_file('/Users/molyneaux/Game1AssetPipeline/lanes/verify_nodes.py', "/opt/verify_nodes.py", copy=True)
        .run_commands("python /opt/verify_nodes.py", gpu=_REMOTE_GPU)
        # Reference plates -> ComfyUI's input folder. A graph that wires LoadImage to a plate the
        # container does not have is refused by preflight before a sampler step, which is correct
        # and also useless if the plate is simply never shipped. Baked into the image so the set
        # is versioned with the code that references it.
        .add_local_dir('/Users/molyneaux/Game1AssetPipeline/reference_targets', "/opt/ComfyUI/input", copy=True)
        .pip_install("huggingface_hub", "pyyaml")
        .env({"COMFY_ROOT": "/opt/ComfyUI",
              "COMFY_WEIGHTS": "/comfy_weights",
              "HF_HOME": "/comfy_weights/hf",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        # LAST, and deliberately so -- the same lesson the trellis2 image records: Modal will not
        # mount a Volume over a non-empty directory, and HF_HOME above is a BUILD env that creates
        # one.
        .run_commands("rm -rf /comfy_weights && mkdir -p /comfy_weights")
        .add_local_python_source("modallabs")
    )

    # create_if_missing=True: a weights cache is idempotent and self-heals on the first run, unlike
    # a staged INPUT volume (theexperiment-views) where a typo must fail at launch rather than
    # mount empty on a hot GPU.
    comfy_weights_volume = modal.Volume.from_name("game1-comfy-weights", create_if_missing=True)

    _comfy_hf = modal.Secret.from_name("huggingface-secret")

    _COMFY_SHEET_COMMON = dict(
        image=comfy_sheet_image,
        gpu=_REMOTE_GPU,               # H100. The model is 12.9 GB; 24 GB would also fit, and
                                       # whether a smaller card is CHEAPER per sheet is unmeasured
                                       # -- see configs/krea2_sheet_probe.yaml, which refuses to
                                       # copy the heroshot A10G conclusion across from a raytracer.
        secrets=[_comfy_hf],
        volumes={"/runs": runs_volume, "/comfy_weights": comfy_weights_volume},
    )

    @app.function(timeout=_LANES["short"], **_COMFY_SHEET_COMMON)
    def _remote_comfy_sheet(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """30-min sheet lane. Worst case $1.97 on the verified $0.001097/s rate."""
        return _remote_body(run_cfg, run_id, resume, _LANES["short"])

    @app.function(timeout=_LANES["medium"], **_COMFY_SHEET_COMMON)
    def _remote_comfy_sheet_medium(run_cfg: dict, run_id: str, resume: bool) -> dict:
        """90-min sheet lane: the FIRST run, which pays a 24 GB weight pull before it samples.
        BILL_SAFETY: the timeout IS the worst-case bill -- $5.92 here."""
        return _remote_body(run_cfg, run_id, resume, _LANES["medium"])

    _TYPE_LANE_FNS = {
        "wan_vace_shot": _remote_wan,
        "longcat_avatar": _remote_longcat,
        "heroshot_take": _remote_heroshot,
        "trellis2_recon": _remote_trellis2,
        "kimodo_motion": _remote_kimodo_brief,
        "comfy_sheet": _remote_comfy_sheet,
    }
    # (type, lane) -> fn, consulted BEFORE _TYPE_LANE_FNS. Lets a type-routed run whose
    # max_runtime_sec fits a smaller lane get a container that actually honours it, so
    # the gate's price and the real timeout cannot diverge. A type absent here falls
    # back to its single entry above.
    _TYPE_LANE_FNS_BY_LANE = {
        ("longcat_avatar", "short"): _remote_longcat_short,
        ("longcat_avatar", "medium"): _remote_longcat,
        ("heroshot_take", "brief"): _remote_heroshot_brief,
        ("heroshot_take", "short"): _remote_heroshot,
        ("trellis2_recon", "short"): _remote_trellis2,
        ("trellis2_recon", "medium"): _remote_trellis2_medium,
        ("kimodo_motion", "brief"): _remote_kimodo_brief,
        ("kimodo_motion", "short"): _remote_kimodo_short,
        ("comfy_sheet", "short"): _remote_comfy_sheet,
        ("comfy_sheet", "medium"): _remote_comfy_sheet_medium,
    }

    if tram_image is not None:
        # create_if_missing=False ON PURPOSE, same rule as the Wan and LongCat volumes: a
        # typo must fail at launch, not mount an empty volume discovered 5.95 GB short on
        # a hot GPU.
        #
        # WEIGHTS ARE STAGED, PINNED, AND NEVER FETCHED HERE. 5.95 GB (ViTDet/SAM 2.56 GB,
        # VIMO 2.79 GB, camcalib 301 MB, DEVA 277 MB, droid 16 MB) live on this volume,
        # each pinned by URL + sha256 in trammotion/config.py and recorded in
        # /tram_weights/WEIGHTS_MANIFEST.json. The trainer verifies them and REFUSES on
        # missing / short / off-pin instead of downloading. Fetching them from inside this
        # container would bill H100 seconds ($0.001097/s) for network I/O, every cold start.
        #   modal run scripts/stage_weights.py              # CPU, no GPU allocated
        #   modal run scripts/stage_weights.py --audit-only # rewrites the manifest
        tram_weights_volume = modal.Volume.from_name(
            "tram-motion-weights", create_if_missing=False
        )

        _TRAM_COMMON = dict(
            image=tram_image,
            gpu=_REMOTE_GPU,
            secrets=[_tram_env],
            volumes={"/runs": runs_volume, _HF_CACHE_MOUNT: hf_cache_volume,
                     "/tram_weights": tram_weights_volume},
        )

        @app.function(timeout=_LANES["short"], **_TRAM_COMMON)
        def _remote_tram(run_cfg: dict, run_id: str, resume: bool) -> dict:
            """TRAM motion-solve lane, 30 min. One epoch is one clip. The DEFAULT for this
            type: the celebration clips are ~5.5 s / 164 frames each, and BILL_SAFETY.md's
            rule is the smallest lane that fits."""
            return _remote_body(run_cfg, run_id, resume, _LANES["short"])

        @app.function(timeout=_LANES["medium"], **_TRAM_COMMON)
        def _remote_tram_medium(run_cfg: dict, run_id: str, resume: bool) -> dict:
            """90-min TRAM lane, for solving several clips in one container so the ViTDet /
            SAM / DEVA / DROID weights load once instead of once per clip."""
            return _remote_body(run_cfg, run_id, resume, _LANES["medium"])

        _TYPE_LANE_FNS["tram_motion"] = _remote_tram
        _TYPE_LANE_FNS_BY_LANE[("tram_motion", "short")] = _remote_tram
        _TYPE_LANE_FNS_BY_LANE[("tram_motion", "medium")] = _remote_tram_medium

    @app.local_entrypoint()
    def main(
        config: str,
        resume: bool = False,
        dry_run: bool = False,
    ) -> None:
        cfg = _load_orchestrator_cfg(config)
        if dry_run:
            # Should be UNREACHABLE under `modal run`: _dry_run_short_circuit() exits at
            # import time, before any image is hydrated. Reaching here means the guard did
            # not match (a new invocation form), and images have ALREADY been built by the
            # time this prints. Kept as a backstop, and it says so rather than pretending
            # the preview was free.
            print("[modallabs] WARNING: the dry-run reached local_entrypoint, so the "
                  "import-time short-circuit did not fire and every lane image was "
                  "hydrated first. The preview below is correct; getting to it was not "
                  "free. Fix _dry_run_short_circuit() for this invocation form.")
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
        # Modal v1.4.2 fixes the function's (gpu, timeout) at decoration
        # time -- we verify EVERY run requests the same (gpu, timeout) so
        # we don't silently dispatch a non-H100 run to the H100 function.
        # If you need a mixed-(gpu, timeout) sweep, add a second
        # @app.function with the other tuple and dispatch on the values
        # here. (Modal v1.4+ has no per-call override API for fan-out.)
        # GPU still must match: every lane is pinned to _REMOTE_GPU, so a run
        # asking for different silicon has nowhere to go. The TIMEOUT no longer
        # has to match -- it selects a lane instead.
        gpu_mismatches = []
        routing = []
        for rc in runs:
            gpu = _gpu_for_run(rc)
            rtype = str(rc.get("type") or "")
            # heroshot_take dispatches on (gpu, lane) -- the one type with
            # declared non-H100 variants (see _HEROSHOT_FNS). Routed BEFORE the
            # homogeneous-GPU check so an L40S/A10G/L4 chunk run has somewhere
            # to go; an undeclared pair refuses here, pre-spawn, for free.
            if rtype == "heroshot_take":
                lane = _lane_for(_max_runtime_sec(rc))  # raises if no lane fits
                hfn = _HEROSHOT_FNS.get((gpu, lane))
                if hfn is None:
                    raise RuntimeError(
                        f"modallabs/modal: {rc.get('name')!r} asks for gpu={gpu!r} "
                        f"-> lane {lane!r} ({_max_runtime_sec(rc)}s), but heroshot_take "
                        f"declares only {sorted(_HEROSHOT_FNS)}. For a LANE problem: do "
                        "NOT buy a longer lane for a render -- split the frame window "
                        "into more chunks (every worker already receives the full "
                        "--frames, so chunk boundaries cost nothing). For a GPU "
                        "problem: add a module-level @app.function variant "
                        "deliberately -- each (gpu, timeout) pair is additional "
                        "worst-case billing exposure."
                    )
                # Stamp the staleness pins BEFORE the run is queued. Refusing
                # here is free; refusing in the container costs a boot; not
                # refusing at all costs a fleet of black-sky frames that pass
                # every cross-worker check.
                _heroshot_pin_run(rc)
                routing.append((rc, lane, hfn))
                continue
            # kimodo_motion is A10G-PINNED at the @app.function level, so it is
            # routed BEFORE the homogeneous-GPU check for the same reason heroshot
            # is: its lane exists, it simply is not on _REMOTE_GPU. The pin is
            # deliberate -- the denoiser is 282M and TEXT_ENCODER_DEVICE=cpu keeps
            # the 8B encoder off the card, so H100 would price a two-clip
            # generation like a training run for no gain.
            if rtype == "kimodo_motion":
                lane = _lane_for(_max_runtime_sec(rc))  # raises if no lane fits
                kfn = _TYPE_LANE_FNS_BY_LANE.get((rtype, lane))
                if kfn is None:
                    raise RuntimeError(
                        f"modallabs/modal: {rc.get('name')!r} asks for lane {lane!r} "
                        f"({_max_runtime_sec(rc)}s), but kimodo_motion declares only "
                        f"{sorted(k[1] for k in _TYPE_LANE_FNS_BY_LANE if k[0] == 'kimodo_motion')}. "
                        "Generation is seconds; the only reason to want a longer lane "
                        "is a COLD HF cache volume (~16 GB Llama pull). Once that is "
                        "warm, use brief."
                    )
                if gpu not in ("A10G", "auto"):
                    raise RuntimeError(
                        f"modallabs/modal: {rc.get('name')!r} asks for gpu={gpu!r}, but "
                        "the kimodo_motion lanes are declared A10G-only. Add a module-"
                        "level @app.function variant deliberately -- each (gpu, timeout) "
                        "pair is additional worst-case billing exposure."
                    )
                routing.append((rc, lane, kfn))
                continue
            if gpu != _REMOTE_GPU:
                gpu_mismatches.append({"name": rc.get("name"), "requested_gpu": gpu})
                continue
            lane = _lane_for(_max_runtime_sec(rc))  # raises if no lane fits
            fn = _TYPE_LANE_FNS_BY_LANE.get((rtype, lane)) or _TYPE_LANE_FNS.get(rtype)
            # A type that REQUIRES its own image must never fall through to _LANE_FNS,
            # which is the generic training image. Without this, a tram_motion run whose
            # lane failed to declare (see the TRAM image resolution above) would be
            # dispatched to a container with no TRAM checkout, no DROID-SLAM and no
            # weights volume -- and would burn its H100 discovering that.
            if fn is None and rtype in _TYPED_LANES_REQUIRED:
                raise RuntimeError(
                    f"modallabs/modal: {rc.get('name')!r} is a {rtype!r} run but no "
                    f"{rtype!r} lane is declared in this process. It will NOT be silently "
                    "dispatched to the generic image. For tram_motion this means the "
                    "proven image definition was not found -- see the TRAM_IMAGE_DEF "
                    "message on stderr at import."
                )
            # (heroshot_take never reaches here -- routed above on (gpu, lane),
            # where its staleness pins are stamped and its brief/short-only
            # refusal lives.)
            if fn is not None and _max_runtime_sec(rc) > _LANES["medium"]:
                raise RuntimeError(
                    f"modallabs/modal: {rc.get('name')!r} is a {rtype!r} run asking for "
                    f"{_max_runtime_sec(rc)}s; the {rtype!r} image is only declared up to "
                    f"the medium lane ({_LANES['medium']}s). Declare a long variant "
                    "deliberately -- it is 4h of H100 worst case per container."
                )
            routing.append((rc, lane, fn))
        if gpu_mismatches:
            raise RuntimeError(
                "modallabs/modal: every lane is pinned to "
                f"{_REMOTE_GPU}; {len(gpu_mismatches)} run(s) request different "
                f"silicon: {gpu_mismatches}. Fix: homogenize cfg.modal.gpu, OR add "
                "module-level @app.function variants for the other GPU and extend "
                "_LANE_FNS. Note each new (gpu, timeout) pair is additional "
                "worst-case billing exposure."
            )

        # Re-assert the ceiling immediately before spend. The dry-run gate ran
        # against the config; this catches a config mutated in between, and it
        # is the last point at which nothing has been billed yet.
        # Price what will ACTUALLY launch. `cfg` still holds the runs that
        # --resume filtered out; charging for them here gates a 1-run resume on
        # the whole board's worst case.
        _total_now, _ = estimate_total_cost_usd(cfg_after_resume)
        _cap_now = _max_total_usd()
        if _total_now > _cap_now:
            raise RuntimeError(
                f"modallabs/modal: worst-case ${_total_now:.2f} exceeds the ceiling "
                f"${_cap_now:.2f} (absolute cap ${_ABSOLUTE_MAX_USD:.2f}). Refusing to "
                "launch. Lower max_runtime_sec, reduce the run count, or raise "
                "MODALLABS_MAX_USD deliberately -- it cannot exceed the absolute cap."
            )

        by_lane: Dict[str, int] = {}
        for _rc, lane, _fn in routing:
            by_lane[lane] = by_lane.get(lane, 0) + 1
        print(f"[modallabs] lane routing: {by_lane} "
              f"(worst case ${_total_now:.2f} vs ceiling ${_cap_now:.2f}; "
              f"L4 dead-man switch at {_NO_PROGRESS_KILL_SEC}s idle)")

        futures = []
        for rc, lane, fn in routing:
            # An explicit per-type function wins over the lane's default; `fn` is
            # None for every run that is not a Wan run, so nothing else changes.
            target = fn if fn is not None else _LANE_FNS[lane]
            futures.append((rc.get("name"), target.spawn(rc, run_id, resume)))

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
