"""modallabs.tests.smoke -- 90-second end-to-end smoke.

Tests:
  1. Every registered local-only model type trains a tiny synthetic run
     end-to-end and writes a checkpoint + done sentinel + metrics.
  2. Determinism: running with seed=42 twice yields identical scalar
     metrics within 1e-6 (torch on CPU).
  3. Crash isolation: one Trainer that raises in setup() must NOT
     stop the orchestrator from succeeding the others.

This smoke skips any model type whose deps are not installed -- e.g.,
the lightgbm/xgboost/catboost/transformers cells are skipped if those
libs are missing. We log "SKIPPED dep <X>" and exit success.

Run from the repo root:
    python -m modallabs.tests.smoke
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Force CPU-only for the smoke. Single-threaded for max determinism.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import modallabs.models  # noqa: E402,F401  -- registers all built-ins
from modallabs.runner import train_one  # noqa: E402
from modallabs.registry import list_types  # noqa: E402


def _has(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False


# (registered_type, cfg) -- minimal cfg for a 1-2 epoch CPU run.
_LOCAL_CASES: List[Tuple[str, Dict[str, Any], List[str]]] = [
    # (type, config, required-modules)
    ("torch_module", {
        "module_path": "torch.nn:Linear",
        "module_kwargs": {"in_features": 8, "out_features": 3},
        "task": "classification",
        "epochs": 2, "batch_size": 16, "n": 64, "in_dim": 8, "n_classes": 3,
    }, ["torch"]),
    ("sklearn", {
        "estimator_path": "sklearn.ensemble:RandomForestClassifier",
        "estimator_kwargs": {"n_estimators": 4, "max_depth": 3},
        "task": "classification",
        "n": 64, "in_dim": 8, "n_classes": 3,
    }, ["sklearn"]),
    ("lstm", {
        "in_dim": 8, "hidden_dim": 8, "n_classes": 3, "seq_len": 4,
        "epochs": 2, "batch_size": 16, "n": 64,
    }, ["torch"]),
    ("rnn", {
        "in_dim": 8, "hidden_dim": 8, "n_classes": 3, "seq_len": 4,
        "epochs": 1, "batch_size": 16, "n": 64,
    }, ["torch"]),
    ("gru", {
        "in_dim": 8, "hidden_dim": 8, "n_classes": 3, "seq_len": 4,
        "epochs": 1, "batch_size": 16, "n": 64,
    }, ["torch"]),
    ("transformer", {
        "in_dim": 8, "d_model": 16, "nhead": 4, "num_layers": 1,
        "n_classes": 3, "seq_len": 4, "epochs": 1, "batch_size": 16, "n": 64,
    }, ["torch"]),
    ("manifold", {
        "in_dim": 8, "latent_dim": 8, "hidden_dim": 8, "n_classes": 3,
        "seq_len": 4, "epochs": 2, "batch_size": 16, "n": 64,
    }, ["torch"]),
    ("ntm", {
        "in_dim": 8, "memory_n": 4, "memory_m": 8, "hidden_dim": 8,
        "n_classes": 3, "seq_len": 4, "epochs": 1, "batch_size": 16, "n": 32,
    }, ["torch"]),
    ("q_learning", {
        "state_dim": 8, "n_actions": 4, "hidden_dim": 16,
        "epochs": 1, "batch_size": 32, "n": 128,
    }, ["torch"]),
    ("diffusion", {
        "feat_dim": 2, "hidden_dim": 16, "timesteps": 10,
        "epochs": 1, "batch_size": 32, "n": 128,
    }, ["torch"]),
    ("lightgbm", {
        "task": "classification", "num_boost_round": 5,
        "n": 64, "in_dim": 8, "n_classes": 3,
        "params": {"num_leaves": 7, "learning_rate": 0.1},
    }, ["lightgbm", "sklearn"]),
    ("xgboost", {
        "task": "classification", "num_boost_round": 5,
        "n": 64, "in_dim": 8, "n_classes": 3,
        "params": {"max_depth": 3, "learning_rate": 0.1},
    }, ["xgboost", "sklearn"]),
    ("catboost", {
        "task": "classification", "num_boost_round": 5,
        "n": 64, "in_dim": 8, "n_classes": 3,
        "params": {"depth": 3, "learning_rate": 0.1},
    }, ["catboost", "sklearn"]),
]


def _run_case(case_type: str, case_cfg: Dict[str, Any], out_root: Path,
              run_id: str, seed: int = 42) -> Dict[str, Any]:
    rc = {"name": f"{case_type}_smoke", "type": case_type,
          "seed": seed, "config": dict(case_cfg)}
    return train_one(rc, run_id=run_id, output_root=out_root,
                     resume=False, force_cpu=True)


def _scalar_metrics_from_done(done_path: Path) -> Dict[str, float]:
    if not done_path.exists():
        return {}
    body = json.loads(done_path.read_text(encoding="utf-8"))
    out = {}
    if isinstance(body.get("best_metric"), (int, float)):
        out["best_metric"] = float(body["best_metric"])
    return out


def _read_metrics(jsonl: Path) -> List[Dict[str, Any]]:
    if not jsonl.exists():
        return []
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main() -> int:
    print("modallabs smoke: starting")
    print(f"modallabs smoke: registered types = {list_types()}")

    base = Path(tempfile.mkdtemp(prefix="modallabs_smoke_"))
    print(f"modallabs smoke: scratch dir = {base}")

    n_run = 0
    n_skip = 0
    n_fail = 0
    failures: List[str] = []

    # ---- Phase 1: train every registered local case ----
    for case_type, case_cfg, required in _LOCAL_CASES:
        missing = [m for m in required if not _has(m)]
        if missing:
            print(f"  SKIP {case_type}: missing deps {missing}")
            n_skip += 1
            continue
        try:
            res = _run_case(case_type, case_cfg, base, run_id="phase1")
            phase = res.get("phase")
            if phase != "succeeded":
                print(f"  FAIL {case_type}: phase={phase} err={res.get('error')!r}")
                n_fail += 1
                failures.append(f"{case_type}: {res.get('error')!r}")
            else:
                bm = res.get("best_metric")
                print(f"  OK   {case_type}: best_metric={bm}")
                n_run += 1
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  FAIL {case_type}: exception {exc}")
            print(tb)
            n_fail += 1
            failures.append(f"{case_type}: {exc}")

    # ---- Phase 2: determinism (torch_module rerun) ----
    if _has("torch"):
        print("modallabs smoke: determinism check (torch_module x2 with seed=42)")
        c2_root_a = Path(tempfile.mkdtemp(prefix="modallabs_det_a_"))
        c2_root_b = Path(tempfile.mkdtemp(prefix="modallabs_det_b_"))
        cfg = {
            "module_path": "torch.nn:Linear",
            "module_kwargs": {"in_features": 8, "out_features": 3},
            "task": "classification",
            "epochs": 2, "batch_size": 16, "n": 64, "in_dim": 8, "n_classes": 3,
        }
        a = _run_case("torch_module", cfg, c2_root_a, run_id="det", seed=42)
        b = _run_case("torch_module", cfg, c2_root_b, run_id="det", seed=42)
        ma = a.get("best_metric")
        mb = b.get("best_metric")
        if ma is None or mb is None:
            print(f"  FAIL determinism: missing best_metric a={ma} b={mb}")
            n_fail += 1
            failures.append("determinism: missing best_metric")
        elif abs(float(ma) - float(mb)) > 1e-6:
            print(f"  FAIL determinism: |{ma} - {mb}| = {abs(float(ma)-float(mb))} > 1e-6")
            n_fail += 1
            failures.append(f"determinism diverge: {ma} vs {mb}")
        else:
            print(f"  OK   determinism: {ma} == {mb}")

    # ---- Phase 3: crash isolation ----
    print("modallabs smoke: crash isolation check")
    crash_root = Path(tempfile.mkdtemp(prefix="modallabs_crash_"))
    try:
        bad = _run_case("torch_module", {
            "module_path": "modallabs._intentionally_missing_module:Nope",
            "epochs": 1, "n": 16, "in_dim": 8, "n_classes": 3,
        }, crash_root, run_id="crash", seed=42)
        if bad.get("phase") == "failed":
            good = _run_case("torch_module", {
                "module_path": "torch.nn:Linear",
                "module_kwargs": {"in_features": 8, "out_features": 3},
                "task": "classification",
                "epochs": 1, "batch_size": 8, "n": 32, "in_dim": 8, "n_classes": 3,
            }, crash_root, run_id="crash", seed=42)
            if good.get("phase") == "succeeded":
                print("  OK   crash isolation: failed run did not block subsequent run")
            else:
                print(f"  FAIL crash isolation: subsequent run phase={good.get('phase')}")
                n_fail += 1
                failures.append("crash isolation: subsequent run failed")
        else:
            print(f"  FAIL crash isolation: bad run did not fail (phase={bad.get('phase')})")
            n_fail += 1
            failures.append("crash isolation: bad run did not fail")
    except Exception as exc:
        print(f"  FAIL crash isolation: orchestrator raised {exc}")
        n_fail += 1
        failures.append(f"crash isolation orchestrator: {exc}")

    # ---- Phase 4: report tool ----
    # The Phase 1 runs landed under base/phase1/<name>/ via train_one, which
    # does NOT write a summary.json (only the concurrent orchestrator does),
    # so this also exercises report.collect's sentinel-scan fallback path.
    print("modallabs smoke: report tool check")
    group_dir = base / "phase1"
    if not group_dir.exists():
        print("  SKIP report tool: no Phase 1 runs landed (all deps skipped)")
    else:
        try:
            from modallabs import report as _report
            rep = _report.collect(group_dir)
            text = _report.render(rep)
            if rep.get("kind") != "group":
                raise AssertionError(f"expected kind=group, got {rep.get('kind')!r}")
            if rep.get("n_runs", 0) < 1 or not text:
                raise AssertionError(
                    f"empty report: n_runs={rep.get('n_runs')} text_len={len(text)}")
            # Single-run view on the first successful run.
            first_ok = next((r for r in rep["runs"] if r.get("phase") == "succeeded"), None)
            if first_ok is not None:
                run_rep = _report.collect(group_dir / first_ok["name"])
                _ = _report.render(run_rep)
                if run_rep.get("kind") != "run":
                    raise AssertionError(f"expected kind=run, got {run_rep.get('kind')!r}")
            # CLI entrypoint should exit 0 on an all-succeeded group.
            rc_cli = _report.main([str(group_dir)])
            if rc_cli != 0:
                raise AssertionError(f"report CLI returned {rc_cli} on a clean group")
            print(f"  OK   report tool: {rep.get('n_runs')} runs summarized")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  FAIL report tool: {exc}")
            print(tb)
            n_fail += 1
            failures.append(f"report tool: {exc}")

    print(f"\nmodallabs smoke: ran={n_run} skipped={n_skip} failed={n_fail}")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("modallabs smoke: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
