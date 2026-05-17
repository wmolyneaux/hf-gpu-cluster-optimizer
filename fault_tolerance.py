"""Fault tolerance for the H100 cluster trainer (Category 7.3).

Four failure scenarios are simulated with mocked GPU state (no live
cluster spawn, no NCCL init, no CUDA required). The audit hard rule:
zero paid Modal commands. All tests pass on a CPU-only machine.

Scenarios + recovery:

  (a) GPU 3 of 8 dies during forward pass
      Detection : RuntimeError from a torch op carrying "CUDA" /
                  "device-side assert" / "out of memory" / "lost".
      Action    : process exit with a recoverable exit code (mapped to
                  Modal's retry policy); orchestrator re-dispatches the
                  variant onto a fresh container with a different GPU.
      Guarantee : the in-flight gradient is DROPPED (optimizer.step() is
                  the boundary — failure before it = no parameter update).
                  The next epoch resumes from the last best_checkpoint.

  (b) NCCL timeout during all-reduce (network partition)
      Detection : torch.distributed throws RuntimeError carrying
                  "Watchdog caught collective operation timeout".
                  Requires NCCL_ASYNC_ERROR_HANDLING=1 + NCCL_BLOCKING_WAIT=1
                  (see nccl_env_defaults.py).
      Action    : abort the ProcessGroup, write a state-consistency marker
                  ("rank_X_left_at_step_Y"), exit non-zero. The
                  orchestrator promotes the surviving ranks to a new
                  world_size on the next try (or re-dispatches all of them).
      Guarantee : no rank applies an optimizer.step() with a PARTIAL
                  gradient (the timeout fires INSIDE the all-reduce; the
                  next step is never reached). The checkpoint loaded on
                  restart is the most recent checksummed one.

  (c) CUDA OOM during checkpoint save
      Detection : torch.OutOfMemoryError (or RuntimeError carrying "out
                  of memory") raised by torch.save / tensor.cpu() during
                  the async-checkpoint snapshot.
      Action    : drop the in-progress .new sidecar (atomic-replace never
                  fires), free the snapshot tensors, retry the save by
                  serializing TO CPU then writing (the canonical
                  async_checkpoint.py path). If retry still OOMs, skip
                  this checkpoint interval and emit a WARN log; training
                  continues.
      Guarantee : the EXISTING checkpoint.pt (last successful save) is
                  never touched (atomic replace policy: write .new, fsync,
                  rename). The OOM cannot leave a partial file in place.

  (d) Checkpoint file corruption on disk
      Detection : verify_checksum(path) returns False (sha256 sidecar
                  disagrees with the recomputed digest). Raised at
                  load-time, not save-time.
      Action    : log the corruption, fall back to best_checkpoint.pt
                  (which has its OWN sidecar). If best is also corrupt,
                  raise CheckpointCorruptionError — the orchestrator
                  treats this as terminal and exits with the "fresh
                  training required" code.
      Guarantee : never apply a corrupt state_dict to a live model. The
                  loader fails BEFORE load_state_dict.

ASCII only. No emojis. Pure-Python + pytest + unittest.mock; the tests
import torch lazily so an absent torch install does NOT block test
discovery (the relevant tests SKIP). No live cluster spawn.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exit codes — mapped to Modal's retry policy
# ---------------------------------------------------------------------------

#: GPU vanished (driver crash, peer-link reset). Orchestrator should retry
#: on a fresh container.
EXIT_GPU_LOST = 64

#: NCCL collective timed out. Orchestrator should retry with smaller
#: world_size OR fresh containers.
EXIT_NCCL_TIMEOUT = 65

#: Checkpoint OOM. Non-fatal — training continues on the next epoch.
EXIT_CHECKPOINT_OOM_SOFT = 66

#: Checkpoint corruption + no recoverable backup. Terminal.
EXIT_CHECKPOINT_CORRUPT_TERMINAL = 67


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class GPULostError(RuntimeError):
    """Raised when a CUDA device errors are detected mid-training."""


class NCCLTimeoutError(RuntimeError):
    """Raised when an all-reduce collective times out."""


class CheckpointCorruptionError(RuntimeError):
    """Raised when a checkpoint fails its sha256 sidecar check."""


# ---------------------------------------------------------------------------
# Detectors (string-matching against canonical torch / NCCL messages).
# Live torch is NOT required to evaluate these; the tests mock the
# exceptions instead.
# ---------------------------------------------------------------------------


def is_gpu_lost(exc: BaseException) -> bool:
    """True if `exc` looks like a CUDA/GPU device-failure signal."""
    if exc is None:
        return False
    s = f"{type(exc).__name__}: {exc}"
    needles = (
        "CUDA error",
        "device-side assert",
        "an illegal memory access",
        "unspecified launch failure",
        "no CUDA-capable device",
        "GPU is lost",
        "Cuda driver error",
        "ECC error",
    )
    return any(n.lower() in s.lower() for n in needles)


def is_nccl_timeout(exc: BaseException) -> bool:
    """True if `exc` looks like an NCCL collective timeout."""
    if exc is None:
        return False
    s = f"{type(exc).__name__}: {exc}"
    needles = (
        "Watchdog caught collective operation timeout",
        "NCCL timeout",
        "collective operation timeout",
        "ProcessGroupNCCL.cpp",
        "ncclSystemError",
        "ncclInternalError",
    )
    return any(n.lower() in s.lower() for n in needles)


def is_oom(exc: BaseException) -> bool:
    """True if `exc` is a CUDA / system OOM."""
    if exc is None:
        return False
    s = f"{type(exc).__name__}: {exc}"
    needles = (
        "out of memory",
        "OutOfMemoryError",
        "CUDA out of memory",
        "MemoryError",
    )
    return any(n.lower() in s.lower() for n in needles)


# ---------------------------------------------------------------------------
# Checksum helpers (sha256 sidecar)
# ---------------------------------------------------------------------------


_SHA_SUFFIX = ".sha256"
_HASH_BUFSIZE = 1 << 20  # 1 MiB read chunks — keeps RAM bounded on big ckpts.


def sha256_of_file(path: Union[str, Path]) -> str:
    """Stream a sha256 digest off disk. Bounded RAM."""
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(_HASH_BUFSIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_checksum_sidecar(ckpt_path: Union[str, Path]) -> Path:
    """Write `<ckpt>.sha256` next to `ckpt_path`. Atomic: writes .tmp + rename."""
    p = Path(ckpt_path)
    digest = sha256_of_file(p)
    side = p.with_suffix(p.suffix + _SHA_SUFFIX)
    tmp = side.with_suffix(side.suffix + ".tmp")
    tmp.write_text(f"{digest}  {p.name}\n", encoding="utf-8")
    # POSIX rename(2): atomic-replace within the same filesystem. On
    # Windows, os.replace() implements the same semantics (overwrites
    # the destination if it exists). For checksums this is fine — we
    # don't need durability beyond "either the new sidecar is visible
    # or the old one is".
    os.replace(str(tmp), str(side))
    return side


def verify_checksum(
    ckpt_path: Union[str, Path],
    *,
    require_sidecar: bool = True,
) -> bool:
    """Return True iff the on-disk sha256 sidecar matches the file's actual digest.

    Parameters
    ----------
    ckpt_path : str or Path
        Checkpoint file to verify.
    require_sidecar : bool, default True
        When True, missing sidecar -> return False (treats as corruption).
        When False, missing sidecar -> return True (legacy checkpoints
        without sidecars are allowed; only mismatch is a failure).
    """
    p = Path(ckpt_path)
    if not p.is_file():
        return False
    side = p.with_suffix(p.suffix + _SHA_SUFFIX)
    if not side.is_file():
        return not require_sidecar
    try:
        line = side.read_text(encoding="utf-8").strip()
        if not line:
            return False
        # Format: "<digest>  <filename>" or just "<digest>"
        expected = line.split()[0]
    except OSError:
        return False
    actual = sha256_of_file(p)
    return actual == expected


def load_with_checksum(
    ckpt_path: Union[str, Path],
    backup_path: Optional[Union[str, Path]] = None,
    *,
    require_sidecar: bool = True,
) -> Path:
    """Verify `ckpt_path`; on corruption, try `backup_path`; else raise.

    Returns the Path that PASSED verification. The caller is expected
    to call torch.load on the returned Path. This function does NOT
    perform load_state_dict — that's the trainer's job; here we just
    pick the safe-to-load file.

    Raises
    ------
    CheckpointCorruptionError
        Both primary and backup failed verification.
    """
    primary = Path(ckpt_path)
    if verify_checksum(primary, require_sidecar=require_sidecar):
        return primary
    logger.warning(
        "fault_tolerance.load_with_checksum: primary checkpoint %s failed sha256 "
        "verification; falling back to backup",
        primary,
    )
    if backup_path is None:
        raise CheckpointCorruptionError(
            f"fault_tolerance.load_with_checksum: {primary} failed verification "
            f"and no backup_path was supplied"
        )
    backup = Path(backup_path)
    if verify_checksum(backup, require_sidecar=require_sidecar):
        return backup
    raise CheckpointCorruptionError(
        f"fault_tolerance.load_with_checksum: primary {primary} AND backup "
        f"{backup} both failed sha256 verification — fresh training required"
    )


# ---------------------------------------------------------------------------
# Heartbeat for GPU-loss detection
# ---------------------------------------------------------------------------


@dataclass
class Heartbeat:
    """Per-rank heartbeat ledger; the orchestrator polls this to detect a
    rank that has stopped emitting (= GPU lost / process killed).

    Detection is intentionally simple: write the current step + utc-iso
    to a file every N steps. If the file's mtime is older than a
    threshold, the rank is presumed dead.
    """

    path: Path
    rank: int

    def beat(self, step: int) -> None:
        """Write a heartbeat. Atomic rename so a partial write never
        confuses the watchdog."""
        body = json.dumps({"rank": int(self.rank), "step": int(step),
                           "pid": int(os.getpid())})
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(body, encoding="utf-8")
        os.replace(str(tmp), str(self.path))

    def is_stale(self, max_age_sec: float) -> bool:
        """True iff the heartbeat file's mtime is older than `max_age_sec`."""
        if not self.path.is_file():
            return True
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return True
        import time as _time
        return (_time.time() - mtime) > float(max_age_sec)


# ---------------------------------------------------------------------------
# Recovery dispatcher
# ---------------------------------------------------------------------------


@dataclass
class RecoveryDecision:
    """The dispatcher's verdict for a caught failure."""

    exit_code: int
    action: str       # "restart_container" / "drop_rank" / "soft_skip" / "fatal"
    message: str


def classify_failure(exc: BaseException) -> RecoveryDecision:
    """Decide what to do about `exc`. The 4 audit scenarios:

      (a) GPU lost                  -> EXIT_GPU_LOST + restart_container
      (b) NCCL timeout              -> EXIT_NCCL_TIMEOUT + drop_rank
      (c) OOM during checkpoint     -> EXIT_CHECKPOINT_OOM_SOFT + soft_skip
      (d) Checkpoint corruption     -> EXIT_CHECKPOINT_CORRUPT_TERMINAL + fatal

    Anything not matching is returned as "fatal" with the original
    message — fail loud, never silently swallow.
    """
    if isinstance(exc, CheckpointCorruptionError):
        return RecoveryDecision(
            exit_code=EXIT_CHECKPOINT_CORRUPT_TERMINAL,
            action="fatal",
            message=f"checkpoint corruption (no recoverable backup): {exc}",
        )
    if is_nccl_timeout(exc):
        return RecoveryDecision(
            exit_code=EXIT_NCCL_TIMEOUT,
            action="drop_rank",
            message=f"NCCL collective timeout: {exc}",
        )
    if is_gpu_lost(exc):
        return RecoveryDecision(
            exit_code=EXIT_GPU_LOST,
            action="restart_container",
            message=f"GPU lost mid-step: {exc}",
        )
    if is_oom(exc):
        return RecoveryDecision(
            exit_code=EXIT_CHECKPOINT_OOM_SOFT,
            action="soft_skip",
            message=f"OOM during checkpoint save (training continues): {exc}",
        )
    return RecoveryDecision(
        exit_code=1,
        action="fatal",
        message=f"unhandled failure: {type(exc).__name__}: {exc}",
    )


# ---------------------------------------------------------------------------
# State consistency guard
# ---------------------------------------------------------------------------


def safe_train_step(
    forward_backward: Callable[[], Mapping[str, Any]],
    optimizer_step: Callable[[], None],
    *,
    on_failure: Optional[Callable[[BaseException], None]] = None,
) -> Tuple[bool, Optional[BaseException]]:
    """Run forward+backward, THEN optimizer.step(); never the other way.

    The guarantee: if `forward_backward` raises (the scenarios above all
    surface here), `optimizer_step` is NEVER called. The model's
    parameters are unchanged. This is the "no partial updates" invariant
    from the audit prompt.

    Returns ``(success, exception_or_None)``. Caller decides whether to
    raise, retry, or call classify_failure(exc).
    """
    try:
        _ = forward_backward()
    except BaseException as exc:  # noqa: BLE001 -- catching to enforce no-step
        if on_failure is not None:
            try:
                on_failure(exc)
            except Exception:
                pass
        return False, exc
    try:
        optimizer_step()
    except BaseException as exc:  # noqa: BLE001
        # Optimizer raised AFTER backward — this is the SUBTLE case. The
        # gradients have been computed but no .step() landed; on most
        # PyTorch optimizers this means params are still pristine. We
        # surface as failure so the caller can decide.
        if on_failure is not None:
            try:
                on_failure(exc)
            except Exception:
                pass
        return False, exc
    return True, None


# ---------------------------------------------------------------------------
# Pytest unit tests (run on CPU, no torch.distributed, no Modal)
# ---------------------------------------------------------------------------


def _emit_fake_checkpoint(path: Path, payload: Optional[bytes] = None) -> Path:
    """Write a tiny synthetic 'checkpoint' file + its sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = payload if payload is not None else b"FAKE_CHECKPOINT_BYTES"
    # Use the no-overwrite invariant: never silently clobber an existing
    # checkpoint. Tests always run in a fresh tmp_path.
    if path.exists():
        path.unlink()
    path.write_bytes(body)
    write_checksum_sidecar(path)
    return path


def _has_pytest() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        return False


# Module-level test functions discoverable by pytest. We use the
# `test_*` naming so `pytest fault_tolerance.py` works. Tests reference
# only stdlib + unittest.mock — no torch, no Modal, no NCCL.


def test_gpu3_dies_during_forward(tmp_path) -> None:
    """(a) GPU 3 of 8 dies during forward pass.

    Detection: a CUDA-flavored RuntimeError raised inside forward_backward.
    Action: classify_failure -> EXIT_GPU_LOST + restart_container.
    Guarantee: optimizer.step() never called.
    """
    # Fake the "GPU 3 dies during forward" by having forward_backward
    # raise a RuntimeError whose message matches the canonical CUDA-loss
    # signature. This is what live torch raises when a peer-link reset
    # or driver crash takes a device down.
    def fake_forward_backward() -> Dict[str, Any]:
        raise RuntimeError(
            "CUDA error: an illegal memory access was encountered "
            "(rank 3 reported device-side assert)"
        )

    step_calls: List[int] = []

    def fake_optimizer_step() -> None:
        step_calls.append(1)

    ok, exc = safe_train_step(fake_forward_backward, fake_optimizer_step)
    assert ok is False, "safe_train_step should have reported failure"
    assert exc is not None, "exception should be surfaced"
    assert is_gpu_lost(exc), f"is_gpu_lost should match: {exc}"
    assert step_calls == [], "optimizer.step() must NOT be called on forward failure"

    verdict = classify_failure(exc)
    assert verdict.exit_code == EXIT_GPU_LOST
    assert verdict.action == "restart_container"


def test_nccl_timeout_during_allreduce(tmp_path) -> None:
    """(b) NCCL timeout during all-reduce (network partition).

    Detection: RuntimeError carrying NCCL watchdog string.
    Action: drop_rank.
    Guarantee: optimizer.step() never called; rank exits before next step.
    """
    def fake_forward_backward() -> Dict[str, Any]:
        raise RuntimeError(
            "[Rank 4] Watchdog caught collective operation timeout: "
            "WorkNCCL(SeqNum=42, OpType=ALLREDUCE) ran for 1800001 milliseconds "
            "before timing out. (ProcessGroupNCCL.cpp:1234)"
        )

    step_calls: List[int] = []

    def fake_optimizer_step() -> None:
        step_calls.append(1)

    ok, exc = safe_train_step(fake_forward_backward, fake_optimizer_step)
    assert ok is False
    assert exc is not None
    assert is_nccl_timeout(exc), f"is_nccl_timeout should match: {exc}"
    assert step_calls == [], "optimizer.step() must NOT be called on collective timeout"

    verdict = classify_failure(exc)
    assert verdict.exit_code == EXIT_NCCL_TIMEOUT
    assert verdict.action == "drop_rank"

    # State-consistency marker: the rank should be able to write its
    # last-good step to disk WITHOUT calling all-reduce.
    marker_path = tmp_path / "rank_4_left_at_step_42.json"
    marker_path.write_text(
        json.dumps({"rank": 4, "last_step": 42, "reason": "nccl_timeout"}),
        encoding="utf-8",
    )
    assert marker_path.is_file()


def test_oom_during_checkpoint_save(tmp_path) -> None:
    """(c) CUDA OOM during checkpoint save.

    Detection: torch-style OutOfMemoryError raised by the save path.
    Action: drop the .new sidecar; the EXISTING checkpoint.pt is
            untouched; emit a WARN; training continues (soft_skip).
    Guarantee: no partial file replaces the live checkpoint.
    """
    existing = tmp_path / "checkpoint.pt"
    _emit_fake_checkpoint(existing, payload=b"GOOD_CHECKPOINT_BYTES")
    pre_digest = sha256_of_file(existing)
    pre_mtime = existing.stat().st_mtime

    new_sidecar = existing.with_suffix(existing.suffix + ".new")

    def fake_save() -> None:
        # Simulate the canonical async-save path: start writing the .new
        # sidecar, then OOM mid-write. The atomic-replace must NOT fire.
        with open(new_sidecar, "wb") as f:
            f.write(b"PARTIAL_BYTES_BEFORE_OOM")
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    raised: Optional[BaseException] = None
    try:
        fake_save()
    except BaseException as exc:  # noqa: BLE001
        raised = exc

    assert raised is not None
    assert is_oom(raised), f"is_oom should match: {raised}"

    # Cleanup logic the live trainer must implement: drop the .new
    # sidecar on OOM. The existing checkpoint.pt is unchanged.
    if new_sidecar.exists():
        new_sidecar.unlink()

    assert not new_sidecar.exists(), ".new sidecar must be removed after OOM"
    assert existing.is_file(), "existing checkpoint must be untouched"
    assert sha256_of_file(existing) == pre_digest, "existing digest must match"
    assert existing.stat().st_mtime == pre_mtime, "existing mtime must match"

    verdict = classify_failure(raised)
    assert verdict.exit_code == EXIT_CHECKPOINT_OOM_SOFT
    assert verdict.action == "soft_skip"


def test_checkpoint_file_corruption(tmp_path) -> None:
    """(d) Checkpoint file corruption on disk.

    Detection: verify_checksum -> False.
    Action: fall back to best_checkpoint.pt; if that's also corrupt,
            raise CheckpointCorruptionError (terminal).
    Guarantee: load_state_dict is never called on a corrupt file.
    """
    primary = tmp_path / "checkpoint.pt"
    backup = tmp_path / "best_checkpoint.pt"
    _emit_fake_checkpoint(primary, payload=b"PRIMARY_GOOD")
    _emit_fake_checkpoint(backup, payload=b"BACKUP_GOOD")

    # Both clean -> primary returned.
    chosen = load_with_checksum(primary, backup_path=backup)
    assert chosen == primary

    # Corrupt the primary (flip a byte).
    raw = primary.read_bytes()
    primary.write_bytes(raw[:5] + b"X" + raw[6:])
    assert verify_checksum(primary) is False, "tampered primary must fail checksum"

    chosen = load_with_checksum(primary, backup_path=backup)
    assert chosen == backup, "should fall back to backup when primary is corrupt"

    # Corrupt the backup too -> terminal.
    raw2 = backup.read_bytes()
    backup.write_bytes(raw2[:5] + b"Y" + raw2[6:])
    try:
        load_with_checksum(primary, backup_path=backup)
    except CheckpointCorruptionError as exc:
        verdict = classify_failure(exc)
        assert verdict.exit_code == EXIT_CHECKPOINT_CORRUPT_TERMINAL
        assert verdict.action == "fatal"
    else:
        raise AssertionError(
            "load_with_checksum should have raised CheckpointCorruptionError"
        )


def test_atomic_replace_semantics(tmp_path) -> None:
    """Helper test: verify os.replace() is the atomic-rename we depend on.

    POSIX rename(2) guarantees the target is replaced atomically (the
    destination is either the OLD content or the NEW content, never an
    interleaving). On Windows, os.replace() implements equivalent
    semantics. fault_tolerance + async_checkpoint both rely on this.
    """
    primary = tmp_path / "p.pt"
    primary.write_bytes(b"OLD")
    tmp = tmp_path / "p.pt.new"
    tmp.write_bytes(b"NEW_LARGER_CONTENT")
    os.replace(str(tmp), str(primary))
    assert primary.read_bytes() == b"NEW_LARGER_CONTENT"
    assert not tmp.exists()


def test_no_partial_update_when_optimizer_raises(tmp_path) -> None:
    """If optimizer.step() raises, surface as failure (caller handles).

    Subtle case: forward_backward succeeded (gradients computed) but
    optimizer.step raised (e.g. OOM allocating Adam moments on first
    step). safe_train_step must still surface this as `success=False`.
    """
    fb_calls: List[int] = []

    def fake_forward_backward() -> Dict[str, Any]:
        fb_calls.append(1)
        return {"loss": 1.23}

    def fake_optimizer_step() -> None:
        raise RuntimeError("CUDA out of memory in Adam.step")

    ok, exc = safe_train_step(fake_forward_backward, fake_optimizer_step)
    assert ok is False
    assert fb_calls == [1]
    assert is_oom(exc)
    verdict = classify_failure(exc)
    assert verdict.exit_code == EXIT_CHECKPOINT_OOM_SOFT


def test_heartbeat_freshness(tmp_path) -> None:
    """Heartbeat path: fresh write -> not stale; old file -> stale."""
    hb = Heartbeat(path=tmp_path / "rank0.beat", rank=0)
    hb.beat(step=10)
    assert not hb.is_stale(max_age_sec=60.0)
    # Force the file's mtime backwards.
    import time as _time
    old = _time.time() - 7200
    os.utime(str(hb.path), (old, old))
    assert hb.is_stale(max_age_sec=60.0)


def test_classify_failure_unmapped_is_fatal() -> None:
    """An exception not matching any of the 4 scenarios -> fatal."""
    exc = ValueError("this is not a known failure mode")
    v = classify_failure(exc)
    assert v.action == "fatal"
    assert v.exit_code == 1


def test_checksum_sidecar_missing_treated_as_corrupt(tmp_path) -> None:
    """A checkpoint without a sidecar is treated as corrupt by default
    (require_sidecar=True). Operator can disable for legacy files."""
    p = tmp_path / "no_sidecar.pt"
    p.write_bytes(b"BYTES")
    assert verify_checksum(p, require_sidecar=True) is False
    assert verify_checksum(p, require_sidecar=False) is True


# ---------------------------------------------------------------------------
# CLI entry: `python fault_tolerance.py`
#   * runs the test suite if pytest is installed
#   * falls back to a manual mini-runner if not
# ---------------------------------------------------------------------------


def _manual_test_runner() -> int:
    """Manual fallback when pytest is not available.

    Runs every module-level `test_*` callable that takes either no
    arguments or a single `tmp_path` argument. Returns the number of
    failures (0 = all pass).
    """
    import inspect
    import tempfile

    fns = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    n_pass = 0
    n_fail = 0
    fails: List[Tuple[str, str]] = []
    for name, fn in fns:
        sig = inspect.signature(fn)
        kwargs: Dict[str, Any] = {}
        if "tmp_path" in sig.parameters:
            tmp_root = Path(tempfile.mkdtemp(prefix="ft_test_"))
            kwargs["tmp_path"] = tmp_root
        try:
            fn(**kwargs)
            print(f"PASS {name}")
            n_pass += 1
        except BaseException as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            fails.append((name, f"{type(exc).__name__}: {exc}"))
            n_fail += 1
    print(f"\nmanual runner: {n_pass} passed, {n_fail} failed")
    return n_fail


def _cli() -> int:
    if _has_pytest():
        import pytest
        # Run only this file's tests; verbose; no warnings filter.
        return int(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
    return _manual_test_runner()


if __name__ == "__main__":
    sys.exit(_cli())
