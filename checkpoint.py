"""modallabs — checkpoint save / load helpers.

Format detection by file extension:
  .pt / .pth                 -> torch.save / torch.load
  .joblib                    -> joblib.dump / joblib.load
  <dir>                      -> HuggingFace save_pretrained / from_pretrained
  .safetensors               -> safetensors.torch.save_file / load_file

Trainers call torch.save / joblib.dump / model.save_pretrained directly;
this module only provides path-resolution + completion-sentinel helpers
so the orchestrator can detect "this run already finished" without
depending on framework-specific load.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


SENTINEL = ".modallabs_done"


def write_done(run_dir: Path, payload: Optional[Dict[str, Any]] = None) -> None:
    """Write the per-run completion sentinel.

    Presence of run_dir / SENTINEL means the run completed without
    raising. The orchestrator's --resume mode treats sentineled runs
    as skip-with-success.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    body = dict(payload or {})
    body.setdefault("modallabs_version", 1)
    (run_dir / SENTINEL).write_text(
        json.dumps(body, indent=2),
        encoding="utf-8",
    )


def is_done(run_dir: Path) -> bool:
    return (Path(run_dir) / SENTINEL).exists()


def remove_done(run_dir: Path) -> None:
    p = Path(run_dir) / SENTINEL
    if p.exists():
        p.unlink()


def best_checkpoint_path(run_dir: Path, suffix: str = ".pt") -> Path:
    return Path(run_dir) / f"best_checkpoint{suffix}"


def final_checkpoint_path(run_dir: Path, suffix: str = ".pt") -> Path:
    return Path(run_dir) / f"checkpoint{suffix}"


# Suffixes a Trainer might land on disk. Sklearn -> .joblib, LightGBM ->
# .txt, XGBoost -> .json, CatBoost -> .cbm, HF -> directory (no suffix),
# torch -> .pt / .pth / .safetensors. final_checkpoint_path returns .pt
# by default, but the orchestrator advertises this path back to callers
# in `result["checkpoint_path"]`. If the Trainer actually wrote to a
# different suffix, the advertised path is wrong. _resolve_checkpoint
# checks every plausible suffix + the bare-name directory and returns
# the first that exists, falling back to the original path so the
# pre-existing contract (caller gets a Path-typed string) holds.
_CKPT_SUFFIXES = (".pt", ".pth", ".joblib", ".txt", ".json", ".cbm", ".safetensors")


def _resolve_existing_checkpoint(p: Path) -> Path:
    """If `p` doesn't exist, try common Trainer suffixes + bare-name dir."""
    p = Path(p)
    if p.exists():
        return p
    bare = p.with_suffix("")
    if bare.exists():  # HF save_pretrained dir
        return bare
    for suf in _CKPT_SUFFIXES:
        cand = bare.with_suffix(suf)
        if cand.exists():
            return cand
    return p  # fall back to caller's expected path even if missing


__all__ = [
    "SENTINEL",
    "write_done",
    "is_done",
    "remove_done",
    "best_checkpoint_path",
    "final_checkpoint_path",
    "_resolve_existing_checkpoint",
]
