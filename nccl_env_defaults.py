"""NCCL environment defaults for 8 x H100 NVLink clusters.

Single-source-of-truth for the NCCL env vars `torch.distributed` reads at
init time. Idempotent; never overwrites a value the caller has already
set. Safe to call from CPU-only contexts (no torch import is required).

Usage::

    from modallabs.nccl_env_defaults import apply_nccl_defaults
    apply_nccl_defaults(single_node=True)         # intra-node (one container, 8 GPUs)
    apply_nccl_defaults(single_node=False)        # multi-node DDP
    apply_nccl_defaults(single_node=True, debug="INFO")  # triage mode

The current ``modal_app.py`` runs one variant per GPU (no
``torch.distributed``) so NCCL never initializes; this module exists so
that the **first** DDP variant landed on top of the harness doesn't trip
on a misconfig. Defaults below are bandwidth-optimal for 8 x H100 NVLink
with the in-network NVSwitch SHARP reduce enabled.

ASCII only. No emojis. Pure-Python; no torch / numpy / pandas imports.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping, Optional


# Default values keyed by (env-var-name). The function applies them only
# when the env var is NOT already set, so a caller's explicit override
# always wins.
#
# Block 1: intra-node NVLink-only (single container, one or more H100s
# bound by NVLink/NVSwitch -- typically @app.function(gpu="H100:8") on Modal).
_INTRA_NODE_DEFAULTS: Dict[str, str] = {
    # NVLink SHARP (in-network reduce via NVSwitch 3). Halves small-payload
    # latency. Default off in NCCL <2.18; explicit ON here.
    "NCCL_NVLS_ENABLE": "1",
    # Force P2P over NVLink (never accidental PCIe fallback).
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_P2P_DISABLE": "0",
    # Skip IB probe inside single-node containers (saves init time + avoids
    # spurious "IB not found" warnings).
    "NCCL_IB_DISABLE": "1",
    # Filter the bootstrap interface (avoid loopback / docker bridge).
    "NCCL_SOCKET_IFNAME": "^lo,docker",
    # Make all-reduce timeouts surface as Python exceptions instead of
    # silent hangs. Required for the fault-tolerance handler in
    # fault_tolerance.py to fire (see Category 7.3).
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_BLOCKING_WAIT": "1",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    # Default debug level. Operator switches to INFO via the `debug=` kwarg.
    "NCCL_DEBUG": "WARN",
}

# Block 2: extra defaults that apply when crossing IB (multi-node).
_MULTI_NODE_EXTRAS: Dict[str, str] = {
    "NCCL_IB_DISABLE": "0",  # override the single-node default
    # Pin all 8 ConnectX-7 HCAs so NCCL doesn't pick a subset.
    "NCCL_IB_HCA": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
    # RoCEv2 GID index 3 (Mellanox default on Modal-equivalent fabrics).
    "NCCL_IB_GID_INDEX": "3",
}


# Hard cap on NCCL_TIMEOUT_SECONDS — keep aligned with modal_app.py's
# _REMOTE_TIMEOUT_SEC (1800s). The function clamps to whichever is
# smaller of (caller's requested timeout, modal-known timeout).
_DEFAULT_NCCL_TIMEOUT_SEC: int = 1800


# Acceptable NCCL_DEBUG levels (validated to fail loud on typos).
_DEBUG_LEVELS = frozenset(("VERSION", "WARN", "INFO", "TRACE"))


def _modal_timeout_hint() -> Optional[int]:
    """Best-effort read of the Modal-known per-function timeout.

    Modal sets MODAL_TIMEOUT_SEC inside the container; if the operator
    has overridden the @app.function(timeout=...) value, we honor it
    so NCCL's timeout never outlives the container's hard kill.

    Returns None when not running under Modal (the caller will fall
    back to _DEFAULT_NCCL_TIMEOUT_SEC).
    """
    raw = os.environ.get("MODAL_TIMEOUT_SEC")
    if not raw:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def apply_nccl_defaults(
    *,
    single_node: bool = True,
    debug: str = "WARN",
    timeout_sec: Optional[int] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Apply the bandwidth-optimal NCCL env vars for 8 x H100 NVLink.

    Parameters
    ----------
    single_node : bool, default True
        When True, apply only the intra-node NVLink defaults. Set False
        to also apply the multi-node IB extras (NCCL_IB_HCA,
        NCCL_IB_GID_INDEX, NCCL_IB_DISABLE=0).
    debug : str, default "WARN"
        One of "VERSION" / "WARN" / "INFO" / "TRACE". Anything else
        raises ValueError so a typo doesn't silently downgrade to WARN.
    timeout_sec : int, optional
        NCCL all-reduce timeout. Defaults to min(MODAL_TIMEOUT_SEC,
        1800) so the NCCL timeout NEVER exceeds the Modal container's
        hard kill. Negative or zero values raise ValueError.
    extra : Mapping[str, str], optional
        Per-call additions / overrides. Applied LAST so the caller can
        force a value that disagrees with the defaults (we still respect
        the "don't overwrite already-set env vars" rule for OS-level
        env, but `extra` always wins over `_INTRA_NODE_DEFAULTS`).

    Returns
    -------
    Dict[str, str]
        The full set of env vars this call effectively wrote, keyed by
        var name. Useful for logging and for the fault_tolerance.py
        recovery probe (it cross-checks the timeout used).

    Raises
    ------
    ValueError
        On unknown `debug` level or non-positive `timeout_sec`.
    """
    if debug not in _DEBUG_LEVELS:
        raise ValueError(
            f"apply_nccl_defaults: debug={debug!r} not in {sorted(_DEBUG_LEVELS)}"
        )
    # Resolve the timeout, clamped to the Modal-known container timeout.
    if timeout_sec is None:
        modal_hint = _modal_timeout_hint()
        if modal_hint is not None:
            timeout_sec = min(modal_hint, _DEFAULT_NCCL_TIMEOUT_SEC)
        else:
            timeout_sec = _DEFAULT_NCCL_TIMEOUT_SEC
    if not isinstance(timeout_sec, int) or timeout_sec <= 0:
        raise ValueError(
            f"apply_nccl_defaults: timeout_sec={timeout_sec!r} must be a positive int"
        )

    # Build the effective dict. Order: intra-node defaults first, then
    # multi-node extras (which may flip NCCL_IB_DISABLE), then the
    # timeout + debug overrides, then any caller `extra`.
    effective: Dict[str, str] = dict(_INTRA_NODE_DEFAULTS)
    effective["NCCL_DEBUG"] = debug
    effective["NCCL_TIMEOUT_SECONDS"] = str(int(timeout_sec))
    if not single_node:
        effective.update(_MULTI_NODE_EXTRAS)
    if extra:
        for k, v in extra.items():
            effective[str(k)] = str(v)

    # Apply to os.environ, but NEVER overwrite an existing value (operator
    # override is sacred). Track which keys we actually set.
    actually_set: Dict[str, str] = {}
    for k, v in effective.items():
        if k not in os.environ:
            os.environ[k] = v
            actually_set[k] = v
    return actually_set


def diff_against_env(effective: Mapping[str, str]) -> Dict[str, str]:
    """Return the subset of `effective` that does NOT match os.environ.

    Used by fault_tolerance tests to verify the recovery path didn't
    silently overwrite an operator's explicit env var.
    """
    out: Dict[str, str] = {}
    for k, v in effective.items():
        cur = os.environ.get(k)
        if cur != str(v):
            out[k] = f"want={v!r} have={cur!r}"
    return out


def all_known_env_keys() -> Iterable[str]:
    """Iterable of every env var name this module manages.

    Useful for the preflight validator: print the active set so an
    operator can confirm what NCCL will see.
    """
    keys: set = set(_INTRA_NODE_DEFAULTS.keys())
    keys.update(_MULTI_NODE_EXTRAS.keys())
    keys.add("NCCL_TIMEOUT_SECONDS")
    return sorted(keys)


__all__ = [
    "apply_nccl_defaults",
    "diff_against_env",
    "all_known_env_keys",
]
