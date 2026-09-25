"""Stage Hunyuan3D 2.1 weights onto the worldclaw-hunyuan-weights volume.

Mirrors scripts/stage_wan_weights.py: a datacenter-side transfer (HF / GitHub
-> Modal volume) in a CPU-only container. The file table, the revisions and the
sha256 pins are imported from models/hunyuan3d_asset.py, the single source of
truth the trainer checks against.

NOTHING BILLABLE RUNS BY DEFAULT:
    python scripts/stage_hunyuan_weights.py
        prints the plan; never contacts Modal.
    modal run scripts/stage_hunyuan_weights.py
        prints the plan and exits BEFORE any remote call (no container).

The owner's command (a CPU container, ~24 GB of downloads; bills a little CPU
time, and the volume's storage is the owner's):
    modal volume create worldclaw-hunyuan-weights     # once; nothing creates it for you
    modal run scripts/stage_hunyuan_weights.py --confirm

What it writes under /hy3d_models:
    tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/     shape  (HY3DGEN_MODELS layout)
    hf_cache/models--tencent--Hunyuan3D-2.1/      paint  (HF cache; refs/main = pin)
    facebook/dinov2-giant/                        paint  (local dir, pinned revision)
    aux/RealESRGAN_x4plus.pth                     paint
    u2net/u2net.onnx                              background removal
    STAGED.json                                   written LAST, only if every pinned
                                                  sha256 verified; the trainer refuses
                                                  to start a worker without it

Idempotent: a file already present at its pinned size is hashed, not
re-downloaded. A sha256 mismatch raises and STAGED.json is not written.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

try:
    import modal
    _HAS_MODAL = True
except Exception:  # plain `python` without the SDK still prints the plan
    modal = None  # type: ignore
    _HAS_MODAL = False

from modallabs.models import hunyuan3d_asset as H

VOLUME_NAME = "worldclaw-hunyuan-weights"
MOUNT = str(H._MODELS)  # /hy3d_models


def plan_text() -> str:
    rows = [f"Hunyuan3D 2.1 weight staging plan -> volume {VOLUME_NAME!r} at {MOUNT}",
            f"  weights {H._WEIGHTS_REPO} @ {H._WEIGHTS_REVISION}",
            f"  dino    {H._DINO_REPO} @ {H._DINO_REVISION}",
            f"  code pin (image, not staged) {H._HY3D_REPO} @ {H._HY3D_COMMIT}"]
    total = 0
    for key, rel, size, sha, need in H._WEIGHT_FILES:
        total += size
        rows.append(f"  {need:<5} {size:>14d}  {rel}  sha256={sha or '(size only)'}")
    rows.append(f"  total {total} bytes ({total / 2**30:.1f} GiB)")
    rows.append("To stage (CPU container, owner only):")
    rows.append(f"  modal volume create {VOLUME_NAME}   # once")
    rows.append("  modal run scripts/stage_hunyuan_weights.py --confirm")
    return "\n".join(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_url(url: str, dest: Path, size: int) -> None:
    if dest.is_file() and dest.stat().st_size == size:
        print(f"SKIP (exists): {dest}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"DOWNLOADING {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
    os.replace(tmp, dest)


if _HAS_MODAL:
    app = modal.App("worldclaw-hunyuan-weight-staging")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("huggingface_hub==0.30.2", "hf_transfer==0.1.9")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_python_source("modallabs")
    )
    # create_if_missing=False ON PURPOSE: a typo'd name must fail at launch, and
    # nothing in this repo creates the volume implicitly.
    weights_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    @app.function(image=image, volumes={MOUNT: weights_volume}, timeout=5400, cpu=4)
    def stage() -> list:
        from huggingface_hub import snapshot_download

        root = Path(MOUNT)
        snapshot_download(repo_id=H._WEIGHTS_REPO, revision=H._WEIGHTS_REVISION,
                          allow_patterns=[f"{H._SHAPE_SUBFOLDER}/*"],
                          local_dir=str(root / "tencent" / "Hunyuan3D-2.1"))
        snap = snapshot_download(repo_id=H._WEIGHTS_REPO, revision=H._WEIGHTS_REVISION,
                                 allow_patterns=[f"{H._PAINT_SUBFOLDER}/*"],
                                 cache_dir=str(root / H._HF_CACHE_REL))
        if Path(snap).name != H._WEIGHTS_REVISION:
            raise RuntimeError(f"paint snapshot resolved to {snap}, not the pin")
        # Upstream multiview_utils calls snapshot_download(repo_id) with no
        # revision; offline, that resolves refs/main. Point it at the pin.
        refs = root / H._HF_CACHE_REL / "models--tencent--Hunyuan3D-2.1" / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(H._WEIGHTS_REVISION, encoding="utf-8")
        snapshot_download(repo_id=H._DINO_REPO, revision=H._DINO_REVISION,
                          allow_patterns=["config.json", "model.safetensors",
                                          "preprocessor_config.json"],
                          local_dir=str(root / "facebook" / "dinov2-giant"))
        sizes = {rel: size for _k, rel, size, _s, _n in H._WEIGHT_FILES}
        _fetch_url(H._REALESRGAN_URL, root / "aux" / "RealESRGAN_x4plus.pth",
                   sizes["aux/RealESRGAN_x4plus.pth"])
        _fetch_url(H._U2NET_URL, root / "u2net" / "u2net.onnx", sizes["u2net/u2net.onnx"])

        files = {}
        bad = []
        for key, rel, size, sha, _need in H._WEIGHT_FILES:
            p = root / rel
            if not p.is_file():
                bad.append(f"{rel}: missing")
                continue
            actual = p.stat().st_size
            digest = _sha256(p)
            print(f"{actual:>14d}  {digest}  {rel}", flush=True)
            if actual != size:
                bad.append(f"{rel}: {actual} bytes, pinned {size}")
            if sha is not None and digest != sha:
                bad.append(f"{rel}: sha256 {digest}, pinned {sha}")
            files[rel] = {"bytes": actual, "sha256": digest}
        if bad:
            weights_volume.commit()
            raise RuntimeError("staging verification FAILED; STAGED.json not written:\n  "
                               + "\n  ".join(bad))
        (root / H._STAGED_MANIFEST_REL).write_text(json.dumps({
            "weights_repo": H._WEIGHTS_REPO, "weights_revision": H._WEIGHTS_REVISION,
            "dino_repo": H._DINO_REPO, "dino_revision": H._DINO_REVISION,
            "hy3d_commit": H._HY3D_COMMIT, "files": files,
            "staged_at_unix": int(time.time()),
        }, indent=2), encoding="utf-8")
        weights_volume.commit()
        print("COMMIT OK", flush=True)
        return sorted((rel, rec["bytes"]) for rel, rec in files.items())

    @app.local_entrypoint()
    def main(confirm: bool = False) -> None:
        print(plan_text())
        if not confirm:
            print("\nNot staging: re-run with --confirm to start the CPU container.")
            return
        for name, size in stage.remote():
            print(f"{size:>14d}  {name}")
        print("STAGING COMPLETE")


if __name__ == "__main__":
    print(plan_text())
    sys.exit(0)
