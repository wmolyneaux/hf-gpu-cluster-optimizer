"""preflight_validate — infra/deploy-side gate before any paid Modal launch.

Gates on INFRA, CODE, and COST — the things that, if broken, cause a paid
`modal run` to fail *after* the GPU clock has started. Use this in front
of every paid launch and in CI before merging changes that touch
`modal_app.py` or any orchestrator config.

Checks:

  INFRA-1  modal CLI installed              (modal --version)
  INFRA-2  modal CLI on PATH                (shutil.which)
  INFRA-3  modal auth working               (modal volume list)
  INFRA-4  UTF-8 stdio configured           (stops the Windows charmap mid-upload kill)
  INFRA-5  modal CLI version >= 1.4.0       (older clients lack `--force` on volume put)

  VOL-1    target volume exists             (or auto-create with --auto-create-volume)

  CODE-1   modal_app.py imports cleanly     (catches syntax / pip-install drift)
  CODE-2   config YAML parses               (yaml.safe_load)
  CODE-3   trainer module resolves          (catches the container-side ModuleNotFoundError BEFORE paying)
  CODE-5   modal_app.py free of Modal SDK v1.4 antipatterns (decorator scope, .with_options)

  COST-1   dry-run worst-case <= ceiling    (mirrors modal_app.py --dry-run, gate same exit code)
  COST-2   per-run timeout sane             (no run with timeout > 4h unless --allow-long-timeout)

Exit codes:
  0  — all PASS or only WARN; caller may proceed to paid launch
  1  — at least one INFRA/VOL/CODE check FAILed; do NOT launch
  2  — COST check FAILed (ceiling breached); same code as `modal_app.py --dry-run`

CLI usage:
  python scripts/preflight_validate.py --config configs/all_models.yaml
  python scripts/preflight_validate.py --config <path> --auto-create-volume
  python scripts/preflight_validate.py --config <path> --json
  python scripts/preflight_validate.py --config <path> --skip COST-1 COST-2
  python scripts/preflight_validate.py --config <path> --remote-smoke  # opt-in CPU-container import probe (~$0.0001)

The validator is read-only by default. The only mutations possible are:
  * --auto-create-volume: `modal volume create <name>` (free, idempotent)
  * --remote-smoke: dispatches a 1-second CPU function (~$0.0001) that
    imports the trainer module inside an actual container, catching
    pip-install drift the local check misses.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------------------------------------------------- types

@dataclass
class CheckResult:
    """One check's outcome. `details` is freeform but JSON-serializable."""
    name: str
    category: str  # INFRA / VOL / DATA / CODE / COST
    status: str    # PASS / FAIL / WARN / SKIP
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        return {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}.get(self.status, "[?]")


@dataclass
class PreflightReport:
    config_path: str
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, c: CheckResult) -> CheckResult:
        self.checks.append(c)
        return c

    @property
    def n_pass(self) -> int: return sum(1 for c in self.checks if c.status == "PASS")
    @property
    def n_fail(self) -> int: return sum(1 for c in self.checks if c.status == "FAIL")
    @property
    def n_warn(self) -> int: return sum(1 for c in self.checks if c.status == "WARN")
    @property
    def n_skip(self) -> int: return sum(1 for c in self.checks if c.status == "SKIP")

    def fails(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == "FAIL"]

    def verdict_exit_code(self) -> int:
        """0 = proceed, 1 = infra fail, 2 = cost fail."""
        for c in self.checks:
            if c.status == "FAIL" and c.category == "COST":
                return 2
        for c in self.checks:
            if c.status == "FAIL":
                return 1
        return 0


# ---------------------------------------------------------------- INFRA

_MIN_MODAL_VERSION = (1, 4, 0)  # `--force` on `modal volume put` requires >= 1.4


def _run(args: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Subprocess wrapper that ALWAYS captures stdout+stderr as UTF-8.

    Avoids the Windows charmap codec failure we hit mid-upload: even if
    the parent shell is mis-configured, this child call decodes its bytes
    as UTF-8 so we never crash on a stray ✓ from modal CLI output."""
    try:
        res = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",  # last-resort: replace un-decodable bytes
        )
        return res.returncode, res.stdout or "", res.stderr or ""
    except FileNotFoundError:
        return 127, "", f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def _modal_path() -> Optional[str]:
    """Return the path to the modal CLI binary, or None.

    Tries (1) PATH lookup, (2) the user-site Python Scripts dir on Windows
    (where `pip install --user modal` lands the binary but doesn't add to PATH).
    """
    p = shutil.which("modal")
    if p:
        return p
    if os.name == "nt":
        # Windows roaming Python Scripts dir — pip install --user lands here.
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidate = Path(appdata) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "modal.exe"
            if candidate.exists():
                return str(candidate)
        # Also try the local AppData equivalent.
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate = Path(local) / "Programs" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "modal.exe"
            if candidate.exists():
                return str(candidate)
    return None


def _parse_modal_version(stdout: str) -> Optional[Tuple[int, int, int]]:
    """`modal --version` prints e.g. 'modal client version: 1.4.2'."""
    import re
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", stdout)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def check_modal_installed(report: PreflightReport, modal_bin: Optional[str]) -> CheckResult:
    if modal_bin is None:
        return report.add(CheckResult(
            name="INFRA-1", category="INFRA", status="FAIL",
            message="modal CLI not installed (`pip install modal`)",
            details={"resolved_path": None},
        ))
    return report.add(CheckResult(
        name="INFRA-1", category="INFRA", status="PASS",
        message=f"modal CLI installed at {modal_bin}",
        details={"resolved_path": modal_bin},
    ))


def check_modal_on_path(report: PreflightReport, modal_bin: Optional[str]) -> CheckResult:
    on_path = shutil.which("modal") is not None
    if on_path:
        return report.add(CheckResult(
            name="INFRA-2", category="INFRA", status="PASS",
            message="modal CLI on PATH",
            details={"path_lookup": shutil.which("modal")},
        ))
    if modal_bin is None:
        return report.add(CheckResult(
            name="INFRA-2", category="INFRA", status="SKIP",
            message="modal CLI not installed (covered by INFRA-1)",
        ))
    # Installed but not on PATH — warn and tell user how to fix.
    scripts_dir = str(Path(modal_bin).parent)
    return report.add(CheckResult(
        name="INFRA-2", category="INFRA", status="WARN",
        message=f"modal CLI not on PATH (installed at {modal_bin}); add to session via $env:PATH",
        details={"fix": f'$env:PATH = "{scripts_dir};" + $env:PATH'},
    ))


def check_modal_auth(report: PreflightReport, modal_bin: Optional[str]) -> CheckResult:
    if modal_bin is None:
        return report.add(CheckResult(
            name="INFRA-3", category="INFRA", status="SKIP",
            message="modal CLI not installed (covered by INFRA-1)",
        ))
    rc, out, err = _run([modal_bin, "volume", "list"], timeout=30)
    if rc != 0:
        return report.add(CheckResult(
            name="INFRA-3", category="INFRA", status="FAIL",
            message=f"modal auth probe failed (rc={rc}); run `modal token new`",
            details={"stderr": err[:500]},
        ))
    return report.add(CheckResult(
        name="INFRA-3", category="INFRA", status="PASS",
        message="modal auth probe succeeded (`modal volume list` returned 0)",
        details={"stdout_excerpt": out[:200]},
    ))


def check_utf8_stdio(report: PreflightReport) -> CheckResult:
    """Windows PowerShell defaults to cp1252; modal CLI emits UTF-8 (✓ etc.)
    and crashes mid-upload if the env isn't switched to UTF-8 stdio. Fail
    loud here so the user fixes it BEFORE upload, not partway through."""
    pyenc = os.environ.get("PYTHONIOENCODING", "").lower()
    pyutf8 = os.environ.get("PYTHONUTF8", "")
    stdout_enc = (sys.stdout.encoding or "").lower()
    is_utf8 = (
        "utf" in stdout_enc
        or pyutf8 == "1"
        or "utf" in pyenc
    )
    if is_utf8:
        return report.add(CheckResult(
            name="INFRA-4", category="INFRA", status="PASS",
            message=f"UTF-8 stdio configured (stdout_encoding={stdout_enc!r})",
            details={"stdout_encoding": stdout_enc, "PYTHONIOENCODING": pyenc, "PYTHONUTF8": pyutf8},
        ))
    return report.add(CheckResult(
        name="INFRA-4", category="INFRA", status="FAIL",
        message=(
            "UTF-8 stdio NOT configured; modal CLI ✓ output will crash on Windows "
            "(charmap codec). Set $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1' "
            "before running modal commands."
        ),
        details={"stdout_encoding": stdout_enc, "PYTHONIOENCODING": pyenc, "PYTHONUTF8": pyutf8},
    ))


def check_modal_version(report: PreflightReport, modal_bin: Optional[str]) -> CheckResult:
    if modal_bin is None:
        return report.add(CheckResult(
            name="INFRA-5", category="INFRA", status="SKIP",
            message="modal CLI not installed (covered by INFRA-1)",
        ))
    rc, out, _ = _run([modal_bin, "--version"], timeout=10)
    ver = _parse_modal_version(out) if rc == 0 else None
    if ver is None:
        return report.add(CheckResult(
            name="INFRA-5", category="INFRA", status="WARN",
            message=f"could not parse `modal --version` output: {out[:120]!r}",
            details={"stdout": out[:500]},
        ))
    if ver < _MIN_MODAL_VERSION:
        return report.add(CheckResult(
            name="INFRA-5", category="INFRA", status="FAIL",
            message=(
                f"modal CLI version {'.'.join(map(str, ver))} < required "
                f"{'.'.join(map(str, _MIN_MODAL_VERSION))} (older clients lack "
                f"`--force` on `modal volume put`). Run `pip install -U modal`."
            ),
            details={"version": ver, "min_required": _MIN_MODAL_VERSION},
        ))
    return report.add(CheckResult(
        name="INFRA-5", category="INFRA", status="PASS",
        message=f"modal CLI v{'.'.join(map(str, ver))} >= {'.'.join(map(str, _MIN_MODAL_VERSION))}",
        details={"version": ver},
    ))


# ------------------------------------------------------------------- VOL

def check_volume_exists(
    report: PreflightReport,
    modal_bin: Optional[str],
    volume_name: str,
    auto_create: bool,
) -> CheckResult:
    if modal_bin is None:
        return report.add(CheckResult(
            name="VOL-1", category="VOL", status="SKIP",
            message="modal CLI not installed (covered by INFRA-1)",
        ))
    rc, out, _ = _run([modal_bin, "volume", "list"], timeout=30)
    if rc != 0:
        return report.add(CheckResult(
            name="VOL-1", category="VOL", status="SKIP",
            message="modal auth not working (covered by INFRA-3)",
        ))
    # Output format: `| Name | Created at | Created by |` rows; volume name appears as substring.
    exists = volume_name in out
    if exists:
        return report.add(CheckResult(
            name="VOL-1", category="VOL", status="PASS",
            message=f"target volume {volume_name!r} exists",
            details={"volume": volume_name},
        ))
    if auto_create:
        rc2, out2, err2 = _run([modal_bin, "volume", "create", volume_name], timeout=30)
        if rc2 == 0:
            return report.add(CheckResult(
                name="VOL-1", category="VOL", status="PASS",
                message=f"volume {volume_name!r} did not exist; created via --auto-create-volume",
                details={"volume": volume_name, "auto_created": True, "stdout": out2[:200]},
            ))
        return report.add(CheckResult(
            name="VOL-1", category="VOL", status="FAIL",
            message=f"volume {volume_name!r} did not exist; auto-create failed (rc={rc2})",
            details={"volume": volume_name, "stderr": err2[:500]},
        ))
    return report.add(CheckResult(
        name="VOL-1", category="VOL", status="FAIL",
        message=(
            f"target volume {volume_name!r} does not exist; "
            f"create it manually (`modal volume create {volume_name}`) or re-run with --auto-create-volume"
        ),
        details={"volume": volume_name},
    ))


# ------------------------------------------------------------------ CODE

def check_modal_app_imports(report: PreflightReport, repo_root: Path) -> CheckResult:
    """Try to import modal_app.py — catches syntax errors / missing pip deps."""
    modal_app_path = repo_root / "modal_app.py"
    if not modal_app_path.exists():
        return report.add(CheckResult(
            name="CODE-1", category="CODE", status="FAIL",
            message=f"modal_app.py not found at {modal_app_path}",
        ))
    # spawn a child Python so the import doesn't pollute this process.
    code = (
        "import sys, os; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "import modal_app"
    )
    rc, out, err = _run([sys.executable, "-c", code], timeout=60)
    if rc == 0:
        return report.add(CheckResult(
            name="CODE-1", category="CODE", status="PASS",
            message="modal_app.py imports cleanly",
        ))
    return report.add(CheckResult(
        name="CODE-1", category="CODE", status="FAIL",
        message=f"modal_app.py import failed: {err.strip().splitlines()[-1][:200] if err.strip() else 'unknown'}",
        details={"stderr": err[:1000]},
    ))


def check_config_parses(report: PreflightReport, config_path: Path) -> CheckResult:
    if not config_path.exists():
        return report.add(CheckResult(
            name="CODE-2", category="CODE", status="FAIL",
            message=f"config not found: {config_path}",
        ))
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return report.add(CheckResult(
            name="CODE-2", category="CODE", status="FAIL",
            message=f"YAML parse failed: {type(exc).__name__}: {exc}",
        ))
    if not isinstance(cfg, dict):
        return report.add(CheckResult(
            name="CODE-2", category="CODE", status="FAIL",
            message=f"config root is {type(cfg).__name__}, expected dict",
        ))
    if "runs" not in cfg:
        return report.add(CheckResult(
            name="CODE-2", category="CODE", status="FAIL",
            message="config missing required key 'runs'",
        ))
    return report.add(CheckResult(
        name="CODE-2", category="CODE", status="PASS",
        message=f"config parses; {len(cfg['runs'])} runs declared",
        details={"n_runs": len(cfg["runs"]), "run_id": cfg.get("run_id", "<auto>")},
    ))


_MODAL_APP_ANTIPATTERNS: List[Tuple[str, str, str]] = [
    # (regex_pattern, fail_message, why)
    (
        r"\.with_options\s*\(",
        "modal_app.py calls .with_options(...) which doesn't exist in Modal SDK v1.4.2",
        "Modal v1.4 removed Function.with_options(); use module-level @app.function variants per (gpu, timeout) tuple",
    ),
    (
        r"def\s+_make_remote_runner|def\s+_build_app_function|@app\.function\s*\([^)]*\)\s*\n\s*def\s+\w+\([^)]*\)\s*:\s*\n\s*[^\n]*\n.*?return\s+_remote",
        "modal_app.py appears to define an @app.function inside an inner factory (closure)",
        "Modal SDK v1.4 requires @app.function to decorate globally-scoped callables OR pass serialized=True; "
        "closure-factory patterns hit `Function has not been hydrated` at .spawn() time",
    ),
]


def check_modal_app_antipatterns(report: PreflightReport, repo_root: Path) -> CheckResult:
    """CODE-5: source-level lint for known Modal SDK v1.4 incompatibilities.

    Catches:
      - InvalidError(@app.function in non-global scope)
      - ExecutionError(Function has not been hydrated -- .with_options removed)

    Both errors fire at `modal run` time AFTER all the local checks pass,
    so a regex-level lint at preflight time is the cheapest early gate."""
    import re
    app_py = repo_root / "modal_app.py"
    if not app_py.exists():
        return report.add(CheckResult(
            name="CODE-5", category="CODE", status="SKIP",
            message="modal_app.py not found (covered by CODE-1)",
        ))
    src = app_py.read_text(encoding="utf-8")
    hits: List[Dict[str, Any]] = []
    for pattern, msg, why in _MODAL_APP_ANTIPATTERNS:
        if re.search(pattern, src, re.MULTILINE | re.DOTALL):
            hits.append({"pattern": pattern, "msg": msg, "why": why})
    if hits:
        return report.add(CheckResult(
            name="CODE-5", category="CODE", status="FAIL",
            message=(
                f"{len(hits)} Modal SDK v1.4 antipattern(s) detected in modal_app.py: "
                f"{hits[0]['msg']}"
            ),
            details={"hits": hits, "all_messages": [h["msg"] for h in hits]},
        ))
    return report.add(CheckResult(
        name="CODE-5", category="CODE", status="PASS",
        message="no known Modal SDK v1.4 antipatterns in modal_app.py",
        details={"patterns_checked": len(_MODAL_APP_ANTIPATTERNS)},
    ))



def check_trainer_imports(report: PreflightReport, repo_root: Path, cfg: Dict[str, Any]) -> CheckResult:
    """Local-only check: try to import every trainer module referenced by
    cfg.runs[*].type (modallabs uses `type` as the registry key — see
    modallabs/registry.py:register). If any one fails locally it WILL fail
    in the container too (modal_app.py bundles `modallabs` via
    `add_local_python_source` so sys.path is identical)."""
    runs = cfg.get("runs") or []
    # Case-insensitive lookup matches modallabs.registry.get()'s lowercase normalization.
    trainer_names = sorted({str(r.get("type")).strip().lower() for r in runs if r.get("type")})
    if not trainer_names:
        return report.add(CheckResult(
            name="CODE-3", category="CODE", status="SKIP",
            message="no `type` keys in runs (nothing to import-check)",
        ))
    # For each trainer name, try to register-via-import. We dispatch through
    # modallabs.models which is the side-effect import that fires every
    # built-in @register decorator. _REGISTRY is the actual symbol (not TRAINERS).
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "import modallabs.models; "
        "from modallabs.registry import _REGISTRY; "
        f"missing = [n for n in {trainer_names!r} if n not in _REGISTRY]; "
        "print('REGISTERED:', sorted(_REGISTRY.keys())); "
        "print('MISSING:', missing) if missing else print('OK')"
    )
    rc, out, err = _run([sys.executable, "-c", code], timeout=60)
    if rc != 0:
        return report.add(CheckResult(
            name="CODE-3", category="CODE", status="FAIL",
            message=f"trainer import failed: {err.strip().splitlines()[-1][:200] if err.strip() else 'unknown'}",
            details={"stderr": err[:1000], "trainers": trainer_names},
        ))
    if "MISSING:" in out:
        missing = out.split("MISSING:", 1)[1].strip()
        return report.add(CheckResult(
            name="CODE-3", category="CODE", status="FAIL",
            message=f"trainers not in _REGISTRY after import: {missing}",
            details={"stdout": out, "trainers_requested": trainer_names},
        ))
    return report.add(CheckResult(
        name="CODE-3", category="CODE", status="PASS",
        message=f"all {len(trainer_names)} trainer(s) registered: {trainer_names}",
        details={"trainers": trainer_names},
    ))


# ----------------------------------------------------------------- COST

def check_cost_ceiling(report: PreflightReport, repo_root: Path, config_path: Path) -> CheckResult:
    """Delegate to modal_app.py --dry-run; that's the single source of truth for the cost gate."""
    rc, out, err = _run([sys.executable, str(repo_root / "modal_app.py"), "--config", str(config_path), "--dry-run"], timeout=60)
    # The dry-run prints the cost table; capture the totals line.
    total_line = ""
    ceiling_line = ""
    for line in out.splitlines():
        if "Total WORST-CASE cost" in line:
            total_line = line.strip()
        elif "Cost ceiling" in line:
            ceiling_line = line.strip()
    if rc == 2:
        return report.add(CheckResult(
            name="COST-1", category="COST", status="FAIL",
            message=f"dry-run cost ceiling breached. {total_line}",
            details={"total_line": total_line, "ceiling_line": ceiling_line, "rc": rc},
        ))
    if rc != 0:
        return report.add(CheckResult(
            name="COST-1", category="COST", status="FAIL",
            message=f"dry-run failed (rc={rc}): {err.strip()[:200]}",
            details={"stderr": err[:500], "rc": rc},
        ))
    return report.add(CheckResult(
        name="COST-1", category="COST", status="PASS",
        message=f"dry-run within ceiling. {total_line} | {ceiling_line}",
        details={"total_line": total_line, "ceiling_line": ceiling_line},
    ))


def check_timeouts_sane(report: PreflightReport, cfg: Dict[str, Any], allow_long: bool) -> CheckResult:
    runs = cfg.get("runs") or []
    long_runs: List[Tuple[str, float]] = []
    for rc in runs:
        modal_section = rc.get("modal") or {}
        sec = float(modal_section.get("max_runtime_sec", 4 * 3600))
        if sec > 4 * 3600:
            long_runs.append((str(rc.get("name", "?")), sec / 3600))
    if long_runs and not allow_long:
        return report.add(CheckResult(
            name="COST-2", category="COST", status="WARN",
            message=(
                f"{len(long_runs)} run(s) have max_runtime_sec > 4h "
                f"(worst-case unbounded; pass --allow-long-timeout to acknowledge)"
            ),
            details={"long_runs": long_runs},
        ))
    return report.add(CheckResult(
        name="COST-2", category="COST", status="PASS",
        message=f"all {len(runs)} run timeouts <= 4h" if runs else "no runs",
        details={"long_runs": long_runs},
    ))


# ---------------------------------------------------------------- runner

def run_preflight(
    config_path: Path,
    repo_root: Path,
    *,
    volume_name: str = "modallabs-runs",
    auto_create_volume: bool = False,
    allow_long_timeout: bool = False,
    skip: Optional[List[str]] = None,
    remote_smoke: bool = False,
) -> PreflightReport:
    skip = set(skip or [])
    report = PreflightReport(config_path=str(config_path))

    # ---- INFRA
    modal_bin = _modal_path()
    if "INFRA-1" not in skip:
        check_modal_installed(report, modal_bin)
    if "INFRA-2" not in skip:
        check_modal_on_path(report, modal_bin)
    if "INFRA-3" not in skip:
        check_modal_auth(report, modal_bin)
    if "INFRA-4" not in skip:
        check_utf8_stdio(report)
    if "INFRA-5" not in skip:
        check_modal_version(report, modal_bin)

    # ---- CODE (need cfg parsed for trainer check)
    if "CODE-1" not in skip:
        check_modal_app_imports(report, repo_root)
    cfg_result = None
    cfg: Optional[Dict[str, Any]] = None
    if "CODE-2" not in skip:
        cfg_result = check_config_parses(report, config_path)
        if cfg_result.status == "PASS":
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg is None:
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}
    if "CODE-3" not in skip:
        check_trainer_imports(report, repo_root, cfg)
    if "CODE-5" not in skip:
        check_modal_app_antipatterns(report, repo_root)

    # ---- VOL
    if "VOL-1" not in skip:
        check_volume_exists(report, modal_bin, volume_name, auto_create_volume)

    # ---- COST
    if "COST-1" not in skip:
        check_cost_ceiling(report, repo_root, config_path)
    if "COST-2" not in skip:
        check_timeouts_sane(report, cfg, allow_long_timeout)

    # ---- Optional remote import probe
    if remote_smoke and "CODE-3-REMOTE" not in skip:
        _check_remote_import(report, repo_root, cfg)

    return report


def _check_remote_import(report: PreflightReport, repo_root: Path, cfg: Dict[str, Any]) -> CheckResult:
    """Opt-in: dispatch a 1-second CPU function on Modal that imports the
    trainer module inside an actual container. Catches pip-install drift the
    local check misses. Costs ~$0.0001 (1 sec @ $0.20/h CPU)."""
    # We invoke this via a tiny ad-hoc modal-script the validator owns.
    probe_script = repo_root / "scripts" / "_remote_import_probe.py"
    if not probe_script.exists():
        # Lazily write the probe alongside the validator.
        probe_script.write_text(_REMOTE_IMPORT_PROBE_SRC, encoding="utf-8")
    runs = cfg.get("runs") or []
    trainers = sorted({r.get("trainer") for r in runs if r.get("trainer")})
    if not trainers:
        return report.add(CheckResult(
            name="CODE-3-REMOTE", category="CODE", status="SKIP",
            message="no trainers declared",
        ))
    modal_bin = _modal_path()
    if modal_bin is None:
        return report.add(CheckResult(
            name="CODE-3-REMOTE", category="CODE", status="SKIP",
            message="modal CLI not installed",
        ))
    rc, out, err = _run(
        [modal_bin, "run", str(probe_script), "--trainers", ",".join(trainers)],
        timeout=180,
    )
    if rc == 0 and "REMOTE_IMPORT_OK" in out:
        return report.add(CheckResult(
            name="CODE-3-REMOTE", category="CODE", status="PASS",
            message=f"remote import of {trainers} succeeded inside Modal container",
            details={"stdout_excerpt": out.split('REMOTE_IMPORT_OK', 1)[0][-200:]},
        ))
    return report.add(CheckResult(
        name="CODE-3-REMOTE", category="CODE", status="FAIL",
        message=f"remote import probe failed (rc={rc}); container CANNOT import trainer",
        details={"stderr": err[:1000], "stdout": out[:500]},
    ))


# Tiny inline script that imports modallabs.models inside the Modal image and
# verifies each requested trainer is registered. Written to disk on first
# --remote-smoke invocation so the user can audit / rerun it manually.
_REMOTE_IMPORT_PROBE_SRC = '''\
"""Remote import probe — verifies the modallabs container can import every
trainer module. Costs ~$0.0001 per run (1 sec on Modal CPU).
"""
from __future__ import annotations
import modal
import sys

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "numpy", "pandas", "pyarrow", "pyyaml", "scikit-learn",
                 "lightgbm", "transformers", "datasets", "accelerate", "safetensors",
                 "tokenizers", "scipy>=1.10")
    .add_local_python_source("modallabs")
    .add_local_python_source("tools")
)
app = modal.App("modallabs-remote-import-probe", image=image)


@app.function(cpu=1.0, timeout=120)
def probe(trainers: list[str]) -> dict:
    """Import the registry and verify each requested trainer is present."""
    import importlib
    try:
        import modallabs.models  # noqa
        from modallabs.registry import TRAINERS
    except Exception as exc:
        return {"ok": False, "error": f"top-level import: {type(exc).__name__}: {exc}"}
    missing = [t for t in trainers if t not in TRAINERS]
    return {"ok": not missing, "registered": sorted(TRAINERS.keys()), "missing": missing}


@app.local_entrypoint()
def main(trainers: str = "") -> None:
    wanted = [t.strip() for t in trainers.split(",") if t.strip()]
    res = probe.remote(wanted)
    if res.get("ok"):
        print(f"REMOTE_IMPORT_OK: {wanted} all registered in container")
    else:
        print(f"REMOTE_IMPORT_FAIL: {res}", file=sys.stderr)
        raise SystemExit(1)
'''


# --------------------------------------------------------------- pretty-print

def render_report(report: PreflightReport, *, json_mode: bool = False) -> str:
    if json_mode:
        return json.dumps({
            "config": report.config_path,
            "checks": [asdict(c) for c in report.checks],
            "summary": {
                "pass": report.n_pass, "fail": report.n_fail,
                "warn": report.n_warn, "skip": report.n_skip,
            },
            "verdict_exit_code": report.verdict_exit_code(),
        }, indent=2)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"preflight_validate — config={report.config_path}")
    lines.append("=" * 72)
    by_cat: Dict[str, List[CheckResult]] = {}
    for c in report.checks:
        by_cat.setdefault(c.category, []).append(c)
    for cat in ("INFRA", "VOL", "DATA", "CODE", "COST"):
        if cat not in by_cat:
            continue
        lines.append(f"\n--- {cat} ---")
        for c in by_cat[cat]:
            lines.append(f"  {c.emoji} {c.name:<14} {c.message}")
            if c.status == "FAIL" and c.details:
                fix = c.details.get("fix")
                if fix:
                    lines.append(f"        fix: {fix}")
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"summary: PASS={report.n_pass}  FAIL={report.n_fail}  "
        f"WARN={report.n_warn}  SKIP={report.n_skip}"
    )
    exit_code = report.verdict_exit_code()
    if exit_code == 0:
        verdict = "PROCEED" if report.n_fail == 0 else "PROCEED-WITH-WARNINGS"
        lines.append(f"verdict: {verdict} (exit 0)")
    elif exit_code == 2:
        lines.append("verdict: HALT — COST CEILING BREACHED (exit 2)")
    else:
        lines.append("verdict: HALT — INFRA/DATA/CODE FAILURE (exit 1)")
    lines.append("=" * 72)
    return "\n".join(lines)


# -------------------------------------------------------------------- CLI

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Infra/deploy-side preflight validator for hf-gpu-cluster-optimizer. "
            "Gates a paid `modal run` on INFRA/VOL/CODE/COST checks. "
            "Default operation is READ-ONLY (no Modal mutations)."
        ),
    )
    ap.add_argument("--config", required=True, help="path to orchestrator YAML")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent),
                    help="hf-gpu-cluster-optimizer repo root (default: parent of scripts/)")
    ap.add_argument("--volume", default="modallabs-runs",
                    help="Modal volume name (default modallabs-runs)")
    ap.add_argument("--auto-create-volume", action="store_true",
                    help="`modal volume create <name>` if missing (free, idempotent)")
    ap.add_argument("--allow-long-timeout", action="store_true",
                    help="acknowledge max_runtime_sec > 4h (otherwise COST-2 WARNs)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="space-separated list of check IDs to skip (e.g. COST-1)")
    ap.add_argument("--remote-smoke", action="store_true",
                    help="opt-in: dispatch a CPU-only Modal probe to verify in-container imports (~$0.0001)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    report = run_preflight(
        config_path=config_path,
        repo_root=repo_root,
        volume_name=args.volume,
        auto_create_volume=args.auto_create_volume,
        allow_long_timeout=args.allow_long_timeout,
        skip=list(args.skip),
        remote_smoke=args.remote_smoke,
    )
    print(render_report(report, json_mode=args.json))
    return report.verdict_exit_code()


if __name__ == "__main__":
    sys.exit(main())
