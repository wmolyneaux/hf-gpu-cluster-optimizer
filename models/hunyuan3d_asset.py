"""hunyuan3d_asset -- Hunyuan3D 2.1 image-to-3D asset batch (WorldClaw lane WC-HF).

One Modal run = one batch of asset images; one epoch = one asset. Input per
asset: a PNG cut from Game1's pipelines/sheet (flat grey background), a name, a
seed, and options (texture on/off, target face count). Output per asset, under
<run_dir>/assets/<name>/: a GLB (+ the painted OBJ/texture maps when textured)
and manifest.json.

Pins (the image in modal_app.py and scripts/stage_hunyuan_weights.py read these
constants; they are the single source of truth):
  code    : Tencent-Hunyuan/Hunyuan3D-2.1 @ _HY3D_COMMIT
  weights : tencent/Hunyuan3D-2.1 @ _WEIGHTS_REVISION (sha256 per LFS file)
  paint   : facebook/dinov2-giant @ _DINO_REVISION, RealESRGAN_x4plus.pth
  bg      : rembg u2net.onnx (sha256 pinned; its md5 equals rembg 2.0.65's own pin)

LICENCE: every manifest carries _LICENCE. Hunyuan3D output is NOT cleared for
shipped assets (PLAN_WORLDCLAW 8.3a: the Community License excludes the EU, the
UK and South Korea, and 5(c) forbids displaying Output there). Blockout and
internal use only until counsel decides.

Standing controls, mirrored from wan_vace_shot:
  C-003 heartbeat   : every poll tick (worker startup and each asset) rewrites
                      output_dir/heartbeat.txt, so the L4 dead-man switch (600 s
                      no-write kill in modal_app) never kills a healthy asset.
  C-004 provenance  : setup() reads every input PNG ONCE, sha256-hashes those
                      bytes and stages that exact copy for the worker BEFORE the
                      weight check, the worker spawn or any model load. The
                      manifest records the hashes, the pinned code/weights and
                      every generation parameter.
  C-005 determinism : the worker runs with PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG
                      and NVIDIA_TF32_OVERRIDE pinned, torch deterministic
                      algorithms (warn_only) and a per-asset torch.Generator seeded
                      from the asset's seed, unless config determinism: false.
                      The paint stage seeds itself with 0 inside upstream code
                      (multiview_utils.forward_one); the manifest says so.

Stub mode (config stub: true) runs the whole control path -- validation, input
hashing, the worker subprocess and its job protocol, heartbeats, output
verification, manifests, checkpoint -- with no GPU, no weights and no Hunyuan
code: the worker writes a one-triangle GLB instead of calling the model.

Resume: an asset whose manifest says done, whose input hash and parameter
fingerprint match, and whose every output still hashes to the recorded value is
REUSED without GPU work. Anything else found in its directory is moved aside to
<name>.stale.<n>/ (never deleted) and regenerated.

Cost knobs (modal_app.py): this type is pinned to gpu L40S on the `short` lane
(1800 s). Set runs[*].modal.max_runtime_sec <= 1800; a larger request is
refused, a different explicit gpu is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modallabs.base import Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult
from modallabs.registry import register

# ----------------------------------------------------------------- pins
_HY3D_REPO = "Tencent-Hunyuan/Hunyuan3D-2.1"
_HY3D_COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"  # main, 2025-10-17
_WEIGHTS_REPO = "tencent/Hunyuan3D-2.1"
_WEIGHTS_REVISION = "0b94677654c57bb9a6b6845cd7b704ccf551d327"  # main, 2025-10-17
_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"
_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"
_DINO_REPO = "facebook/dinov2-giant"
_DINO_REVISION = "611a9d42f2335e0f921f1e313ad3c1b7178d206d"
_REALESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                   "v0.1.0/RealESRGAN_x4plus.pth")
_U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

_LICENCE = ("Tencent Hunyuan 3D 2.1 Community License -- output NOT cleared for shipped "
            "assets pending counsel (PLAN_WORLDCLAW 8.3a); blockout / internal use")

# Container paths (modal_app.py hunyuan_image + the worldclaw-hunyuan-weights volume).
_HY3D_DIR = Path("/hunyuan3d")
_MODELS = Path("/hy3d_models")
_HF_CACHE_REL = "hf_cache"
_STAGED_MANIFEST_REL = "STAGED.json"

# Every weight file the lane reads: (key, path relative to _MODELS, bytes,
# sha256 or None, needed_for). Sizes and LFS sha256 read from the HF API
# (?blobs=true) at the pinned revisions and from the two GitHub release assets
# (streamed and hashed 2026-09-24). sha256 is verified by the staging script at
# stage time (CPU, cheap) and recorded in STAGED.json; the trainer checks the
# sizes and STAGED.json before it starts a worker, so a GPU is never spent
# hashing 20 GB.
_PAINT_SNAPSHOT = (f"{_HF_CACHE_REL}/models--tencent--Hunyuan3D-2.1/snapshots/"
                   f"{_WEIGHTS_REVISION}/{_PAINT_SUBFOLDER}")
_WEIGHT_FILES: Tuple[Tuple[str, str, int, Optional[str], str], ...] = (
    ("shape_ckpt", f"tencent/Hunyuan3D-2.1/{_SHAPE_SUBFOLDER}/model.fp16.ckpt",
     7366389768, "6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e", "shape"),
    ("shape_config", f"tencent/Hunyuan3D-2.1/{_SHAPE_SUBFOLDER}/config.yaml",
     2078, None, "shape"),
    ("u2net", "u2net/u2net.onnx",
     175997641, "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491", "shape"),
    ("paint_unet", f"{_PAINT_SNAPSHOT}/unet/diffusion_pytorch_model.bin",
     3925293863, "675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8", "paint"),
    ("paint_text_encoder", f"{_PAINT_SNAPSHOT}/text_encoder/pytorch_model.bin",
     1361671895, "c3e254d7b61353497ea0be2c4013df4ea8f739ee88cffa0ba58cd085459ed565", "paint"),
    ("paint_image_encoder", f"{_PAINT_SNAPSHOT}/image_encoder/model.safetensors",
     1264217240, "ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a", "paint"),
    ("paint_vae", f"{_PAINT_SNAPSHOT}/vae/diffusion_pytorch_model.bin",
     334707217, "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc", "paint"),
    ("paint_model_index", f"{_PAINT_SNAPSHOT}/model_index.json", 617, None, "paint"),
    ("dino_giant", "facebook/dinov2-giant/model.safetensors",
     4546005432, "917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f", "paint"),
    ("dino_config", "facebook/dinov2-giant/config.json", 548, None, "paint"),
    ("dino_preprocessor", "facebook/dinov2-giant/preprocessor_config.json", 436, None, "paint"),
    ("realesrgan", "aux/RealESRGAN_x4plus.pth",
     67040989, "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", "paint"),
)

# ----------------------------------------------------------------- config
_REQUIRED = ("assets", "epochs", "package_root",
             "shape_steps", "guidance_scale", "octree_resolution",
             "paint_max_views", "paint_resolution")
_OPTIONAL = ("stub", "determinism", "startup_timeout_sec", "asset_timeout_sec",
             "poll_sec", "min_side_px", "num_chunks",
             "stub_delay_sec", "stub_fail_asset")  # stub_* only with stub: true
_ASSET_KEYS = ("name", "image", "seed", "texture", "target_faces")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
_OCTREE_OK = (256, 384, 512)
_PAINT_RES_OK = (512, 768)
# Upstream's paint path remeshes to 40000 faces (simplify_mesh_utils
# mesh_simplify_trimesh target_count) before texturing, so a textured asset can
# never honour a larger target. Refused rather than silently decimated.
_PAINT_MAX_FACES = 40000
_MAX_INPUT_BYTES = 64 * 1024 * 1024
_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# C-005: pinned env for the worker subprocess (same set as wan_vace_shot).
_DETERMINISM_ENV = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
}

_WORKER = Path(__file__).with_name("hunyuan3d_worker.py")


class Hunyuan3DAssetError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _png_info(data: bytes, label: str) -> Dict[str, Any]:
    """Width, height, colour type from the IHDR chunk. Refuses non-PNG input."""
    if len(data) < 33 or data[:8] != _PNG_SIG or data[12:16] != b"IHDR":
        raise Hunyuan3DAssetError(f"{label}: not a PNG (sheet cells must be PNG)")
    width, height = struct.unpack(">II", data[16:24])
    colour_type = data[25]
    return {"width": int(width), "height": int(height),
            "png_colour_type": int(colour_type),
            "has_alpha_channel": colour_type in (4, 6)}


def _glb_check(path: Path) -> None:
    """Header check for binary glTF 2.0: magic, version, declared length."""
    with path.open("rb") as f:
        head = f.read(12)
    if len(head) < 12:
        raise Hunyuan3DAssetError(f"output GLB truncated: {path}")
    magic, version, length = struct.unpack("<4sII", head)
    if magic != b"glTF" or version != 2:
        raise Hunyuan3DAssetError(f"output is not a glTF 2.0 binary: {path}")
    if length != path.stat().st_size:
        raise Hunyuan3DAssetError(
            f"GLB header length {length} != file size {path.stat().st_size}: {path}")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@register("hunyuan3d_asset")
class Hunyuan3DAssetTrainer(Trainer):

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self._assets: List[Dict[str, Any]] = [dict(a) for a in config["assets"]]
        self._cursor = 0
        self._proc: Optional[subprocess.Popen] = None
        self._worker_log: Optional[Any] = None
        self._setup_obj: Optional[TrainerSetup] = None
        self._results: List[Dict[str, Any]] = []
        self._epoch_metrics: List[Dict[str, float]] = []
        # C-004: asset name -> {file, sha256, bytes, width, height, ...}; filled
        # by _hash_inputs() at the very top of setup().
        self._input_sha: Dict[str, Dict[str, Any]] = {}
        self._staged_input: Dict[str, Path] = {}
        self._reusable: Dict[str, Dict[str, Any]] = {}
        self._worker_ready: Dict[str, Any] = {}
        self._jobs_dir: Optional[Path] = None

    # ------------------------------------------------------------- 1/9
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Hunyuan3DAssetTrainer":
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise Hunyuan3DAssetError(f"hunyuan3d_asset config missing keys: {missing}")
        unknown = sorted(set(config) - set(_REQUIRED) - set(_OPTIONAL))
        if unknown:
            raise Hunyuan3DAssetError(f"hunyuan3d_asset config has unknown keys: {unknown}")
        stub = config.get("stub", False)
        if not isinstance(stub, bool):
            raise Hunyuan3DAssetError(f"stub must be true/false, got {stub!r}")
        for k in ("stub_delay_sec", "stub_fail_asset"):
            if k in config and not stub:
                raise Hunyuan3DAssetError(f"{k} is a stub-only test knob; refused with stub: false")
        det = config.get("determinism", True)
        if not isinstance(det, bool):
            raise Hunyuan3DAssetError(f"determinism must be true/false, got {det!r}")

        assets = config["assets"]
        if not isinstance(assets, list) or not assets:
            raise Hunyuan3DAssetError("assets must be a non-empty list")
        if not _is_int(config["epochs"]) or int(config["epochs"]) != len(assets):
            raise Hunyuan3DAssetError(
                f"one epoch is one asset: epochs={config['epochs']!r} != n_assets={len(assets)}")

        def _int_in(key: str, lo: int, hi: int, default: Optional[int] = None) -> None:
            v = config.get(key, default)
            if not _is_int(v) or not lo <= v <= hi:
                raise Hunyuan3DAssetError(f"{key} must be an int in [{lo}, {hi}], got {v!r}")

        def _num_in(key: str, lo: float, hi: float, default: Optional[float] = None) -> None:
            v = config.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
                raise Hunyuan3DAssetError(f"{key} must be a number in [{lo}, {hi}], got {v!r}")

        _int_in("shape_steps", 1, 100)
        _num_in("guidance_scale", 0.0, 20.0)
        if config["octree_resolution"] not in _OCTREE_OK or isinstance(config["octree_resolution"], bool):
            raise Hunyuan3DAssetError(
                f"octree_resolution must be one of {_OCTREE_OK}, got {config['octree_resolution']!r}")
        _int_in("paint_max_views", 6, 9)
        if config["paint_resolution"] not in _PAINT_RES_OK or isinstance(config["paint_resolution"], bool):
            raise Hunyuan3DAssetError(
                f"paint_resolution must be one of {_PAINT_RES_OK}, got {config['paint_resolution']!r}")
        _int_in("startup_timeout_sec", 1, 1500, 900)
        _int_in("asset_timeout_sec", 1, 1500, 600)
        _num_in("poll_sec", 0.01, 1.0, 1.0)
        _int_in("min_side_px", 64, 8192, 256)
        _int_in("num_chunks", 1000, 200000, 8000)
        if "stub_delay_sec" in config:
            _num_in("stub_delay_sec", 0.0, 60.0)

        pkg = config["package_root"]
        if not isinstance(pkg, str) or not pkg:
            raise Hunyuan3DAssetError(f"package_root must be a non-empty path string, got {pkg!r}")

        seen = set()
        for a in assets:
            if not isinstance(a, dict):
                raise Hunyuan3DAssetError(f"asset entries must be mappings, got {a!r}")
            miss = [k for k in _ASSET_KEYS if k not in a]
            if miss:
                raise Hunyuan3DAssetError(f"asset {a.get('name')!r} missing keys: {miss}")
            extra = sorted(set(a) - set(_ASSET_KEYS))
            if extra:
                raise Hunyuan3DAssetError(f"asset {a.get('name')!r} has unknown keys: {extra}")
            name = a["name"]
            if not isinstance(name, str) or not _NAME_RE.match(name):
                raise Hunyuan3DAssetError(
                    f"asset name {name!r} must match {_NAME_RE.pattern} (it becomes a directory)")
            if name in seen:
                raise Hunyuan3DAssetError(f"duplicate asset name {name!r}")
            seen.add(name)
            img = a["image"]
            if (not isinstance(img, str) or not img or img.startswith(("/", "\\"))
                    or re.match(r"^[A-Za-z]:", img)
                    or ".." in re.split(r"[\\/]", img)):
                raise Hunyuan3DAssetError(
                    f"asset {name!r} image must be a relative path inside package_root, got {img!r}")
            if not img.lower().endswith(".png"):
                raise Hunyuan3DAssetError(f"asset {name!r} image must be a .png, got {img!r}")
            if not _is_int(a["seed"]) or not 0 <= a["seed"] <= 2**32 - 1:
                raise Hunyuan3DAssetError(f"asset {name!r} seed must be an int in [0, 2^32-1]")
            if not isinstance(a["texture"], bool):
                raise Hunyuan3DAssetError(f"asset {name!r} texture must be true/false")
            tf = a["target_faces"]
            if not _is_int(tf) or not 500 <= tf <= 500000:
                raise Hunyuan3DAssetError(
                    f"asset {name!r} target_faces must be an int in [500, 500000], got {tf!r}")
            if a["texture"] and tf > _PAINT_MAX_FACES:
                raise Hunyuan3DAssetError(
                    f"asset {name!r}: texture: true caps target_faces at {_PAINT_MAX_FACES} "
                    f"(upstream remeshes to that before painting); got {tf}")
        if "stub_fail_asset" in config and config["stub_fail_asset"] not in seen:
            raise Hunyuan3DAssetError("stub_fail_asset must name an asset in the batch")
        return cls(config)

    # ------------------------------------------------------------- helpers
    def _needs_paint(self) -> bool:
        return any(bool(a["texture"]) for a in self._assets)

    def _params(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "shape": {"steps": int(cfg["shape_steps"]),
                      "guidance_scale": float(cfg["guidance_scale"]),
                      "octree_resolution": int(cfg["octree_resolution"]),
                      "num_chunks": int(cfg.get("num_chunks", 8000))},
            "paint": {"max_views": int(cfg["paint_max_views"]),
                      "resolution": int(cfg["paint_resolution"]),
                      "seed": 0,
                      "seed_note": "fixed to 0 by upstream hy3dpaint multiview_utils.forward_one"},
            "determinism": bool(cfg.get("determinism", True)),
        }

    def _pins(self) -> Dict[str, Any]:
        return {
            "code": {"repo": _HY3D_REPO, "commit": _HY3D_COMMIT},
            "weights": {"repo": _WEIGHTS_REPO, "revision": _WEIGHTS_REVISION,
                        "shape_subfolder": _SHAPE_SUBFOLDER,
                        "paint_subfolder": _PAINT_SUBFOLDER},
            "dino": {"repo": _DINO_REPO, "revision": _DINO_REVISION},
            "aux": {"realesrgan": _REALESRGAN_URL, "u2net": _U2NET_URL},
            "files": {k: {"path": rel, "bytes": size, "sha256": sha, "for": need}
                      for k, rel, size, sha, need in _WEIGHT_FILES},
        }

    def _fingerprint(self, asset: Dict[str, Any]) -> str:
        """Everything that determines an asset's bytes. Reuse requires equality."""
        return _sha256_bytes(_canon({
            "stub": bool(self.config.get("stub", False)),
            "pins": self._pins(),
            "params": self._params(),
            "asset": {k: asset[k] for k in ("name", "seed", "texture", "target_faces")},
            "input_sha256": self._input_sha[asset["name"]]["sha256"],
        }).encode("ascii"))

    def _asset_dir(self, name: str) -> Path:
        assert self._setup_obj is not None
        return self._setup_obj.output_dir / "assets" / name

    def _log(self, msg: str) -> None:
        if self._setup_obj is not None:
            self._setup_obj.log_fn(f"hunyuan3d_asset: {msg}")

    # ------------------------------------------------------------- controls
    def _hash_inputs(self, pkg: Path) -> None:
        """C-004: read each input PNG once, hash THOSE bytes, stage THAT copy.

        The worker reads only the staged copy, so the bytes hashed are the bytes
        the model sees (no window for the source to change between hash and use).
        Always strict: a missing or malformed input refuses the run before spend.
        """
        assert self._setup_obj is not None
        if not pkg.is_dir():
            raise Hunyuan3DAssetError(f"package_root not found: {pkg}")
        staged_dir = self._setup_obj.output_dir / "inputs"
        staged_dir.mkdir(parents=True, exist_ok=True)
        min_side = int(self.config.get("min_side_px", 256))
        for a in self._assets:
            src = pkg / a["image"]
            if not src.is_file():
                raise Hunyuan3DAssetError(f"cannot hash missing input for {a['name']!r}: {src}")
            size = src.stat().st_size
            if size > _MAX_INPUT_BYTES:
                raise Hunyuan3DAssetError(
                    f"input for {a['name']!r} is {size} bytes (> {_MAX_INPUT_BYTES}): {src}")
            data = src.read_bytes()
            info = _png_info(data, f"input for {a['name']!r} ({src})")
            if min(info["width"], info["height"]) < min_side:
                raise Hunyuan3DAssetError(
                    f"input for {a['name']!r} is {info['width']}x{info['height']}; "
                    f"short side must be >= {min_side} px")
            sha = _sha256_bytes(data)
            dst = staged_dir / f"{a['name']}__{sha[:16]}.png"
            if not dst.exists() or _sha256_file(dst) != sha:
                tmp = dst.with_suffix(".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
            if _sha256_file(dst) != sha:
                raise Hunyuan3DAssetError(f"staged input copy does not hash back: {dst}")
            self._input_sha[a["name"]] = {"file": a["image"], "sha256": sha,
                                          "bytes": len(data), **info}
            self._staged_input[a["name"]] = dst
        self._log(f"C-004 hashed {len(self._input_sha)} input(s) before any spend")

    def _check_weights(self) -> None:
        """Every needed weight staged at its pinned size, with a STAGED.json whose
        sha256 entries equal the pins (written by scripts/stage_hunyuan_weights.py
        after it hashed the files). Fails at minute 0, before the worker starts."""
        need = {"shape"} | ({"paint"} if self._needs_paint() else set())
        staged_manifest = _MODELS / _STAGED_MANIFEST_REL
        if not staged_manifest.is_file():
            raise Hunyuan3DAssetError(
                f"{staged_manifest} missing: the weights volume was never staged. "
                "Run: modal run scripts/stage_hunyuan_weights.py --confirm")
        staged = json.loads(staged_manifest.read_text(encoding="utf-8"))
        if staged.get("weights_revision") != _WEIGHTS_REVISION:
            raise Hunyuan3DAssetError(
                f"STAGED.json revision {staged.get('weights_revision')!r} != pinned "
                f"{_WEIGHTS_REVISION}; re-stage deliberately")
        staged_files = staged.get("files", {})
        for key, rel, size, sha, needed_for in _WEIGHT_FILES:
            if needed_for not in need:
                continue
            p = _MODELS / rel
            if not p.is_file():
                raise Hunyuan3DAssetError(f"weight not staged on volume: {p}")
            actual = p.stat().st_size
            if actual != size:
                raise Hunyuan3DAssetError(f"weight {p} is {actual} bytes, pinned {size}")
            if sha is not None:
                rec = staged_files.get(rel) or {}
                if rec.get("sha256") != sha:
                    raise Hunyuan3DAssetError(
                        f"STAGED.json sha256 for {rel} is {rec.get('sha256')!r}, pinned {sha}")
            self._log(f"weight ok: {key} {rel} ({actual} bytes)")
        refs_main = (_MODELS / _HF_CACHE_REL / "models--tencent--Hunyuan3D-2.1" / "refs" / "main")
        if "paint" in need:
            got = refs_main.read_text(encoding="utf-8").strip() if refs_main.is_file() else None
            if got != _WEIGHTS_REVISION:
                raise Hunyuan3DAssetError(
                    f"{refs_main} is {got!r}; upstream paint resolves 'main' offline, so it "
                    f"must name the pin {_WEIGHTS_REVISION}")

    def _heartbeat(self, note: str) -> None:
        """C-003: refresh output_dir/heartbeat.txt so the L4 dead-man switch
        (600 s no-write kill) sees forward progress during model load and long
        assets. A failed write is logged LOUDLY: the guard is then live."""
        if self._setup_obj is None:
            return
        try:
            hb = self._setup_obj.output_dir / "heartbeat.txt"
            hb.parent.mkdir(parents=True, exist_ok=True)
            hb.write_text(f"{time.time():.3f} {note}\n", encoding="utf-8")
        except OSError as exc:
            self._setup_obj.log_fn(
                f"HEARTBEAT WRITE FAILED ({exc}); L4 dead-man switch is live")

    def _find_reusable(self) -> None:
        """Assets finished by an earlier attempt of this run, verified byte for byte."""
        for a in self._assets:
            d = self._asset_dir(a["name"])
            mf = d / "manifest.json"
            if not mf.is_file():
                continue
            why = None
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                if m.get("status") != "done":
                    why = f"status {m.get('status')!r}"
                elif m.get("fingerprint") != self._fingerprint(a):
                    why = "input or parameters changed"
                elif not m.get("outputs") or not any(
                        o.get("file") == m.get("primary") for o in m["outputs"]):
                    why = "manifest lists no primary output"
                else:
                    for o in m.get("outputs", []):
                        p = d / o["file"]
                        if not p.is_file() or _sha256_file(p) != o["sha256"]:
                            why = f"output {o['file']} missing or altered"
                            break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                why = f"unreadable manifest ({type(exc).__name__}: {exc})"
            if why is None:
                self._reusable[a["name"]] = m
                self._log(f"reuse {a['name']}: verified outputs from an earlier attempt")
            else:
                self._log(f"NOT reusing {a['name']}: {why}; it will be regenerated")

    def _move_aside(self, name: str) -> None:
        """Never delete: a stale or partial asset dir is renamed <name>.stale.<n>."""
        d = self._asset_dir(name)
        if not d.exists():
            return
        n = 1
        while (d.parent / f"{name}.stale.{n}").exists():
            n += 1
        dst = d.parent / f"{name}.stale.{n}"
        os.replace(d, dst)
        self._log(f"moved stale {d} -> {dst}")

    # ------------------------------------------------------------- worker
    def _worker_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if not self.config.get("stub"):
            env.update({
                "HY3DGEN_MODELS": str(_MODELS),
                "HF_HUB_CACHE": str(_MODELS / _HF_CACHE_REL),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "U2NET_HOME": str(_MODELS / "u2net"),
            })
        if bool(self.config.get("determinism", True)):
            env.update(_DETERMINISM_ENV)
        else:
            for k in _DETERMINISM_ENV:
                env.pop(k, None)
        return env

    def _worker_tail(self) -> str:
        assert self._setup_obj is not None
        p = self._setup_obj.output_dir / "worker.log"
        try:
            return p.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            return "<worker.log unreadable>"

    def _start_worker(self) -> None:
        assert self._setup_obj is not None
        out = self._setup_obj.output_dir
        self._jobs_dir = out / "worker_jobs"
        if self._jobs_dir.exists():
            n = 1
            while (out / f"worker_jobs.stale.{n}").exists():
                n += 1
            os.replace(self._jobs_dir, out / f"worker_jobs.stale.{n}")
        self._jobs_dir.mkdir(parents=True)
        cfg = self.config
        spec = {
            "stub": bool(cfg.get("stub", False)),
            "stub_delay_sec": float(cfg.get("stub_delay_sec", 0.0)),
            "stub_fail_asset": cfg.get("stub_fail_asset"),
            "determinism": bool(cfg.get("determinism", True)),
            "needs_paint": self._needs_paint(),
            "repo_dir": str(_HY3D_DIR),
            "models_dir": str(_MODELS),
            "shape_model": {"repo": _WEIGHTS_REPO, "subfolder": _SHAPE_SUBFOLDER},
            "paint": {**self._params()["paint"],
                      "cfg_path": str(_HY3D_DIR / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"),
                      "dino_path": str(_MODELS / "facebook" / "dinov2-giant"),
                      "realesrgan_path": str(_MODELS / "aux" / "RealESRGAN_x4plus.pth")},
            "shape": self._params()["shape"],
            "jobs_dir": str(self._jobs_dir),
            "parent_pid": os.getpid(),
            "poll_sec": float(cfg.get("poll_sec", 1.0)),
        }
        spec_path = self._jobs_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        argv = [sys.executable, str(_WORKER), "--spec", str(spec_path)]
        self._worker_log = (out / "worker.log").open("a", encoding="utf-8")
        self._log(f"starting worker ({'STUB' if spec['stub'] else 'Hunyuan3D-2.1'}); "
                  f"determinism={'on ' + str(sorted(_DETERMINISM_ENV)) if spec['determinism'] else 'OFF'}")
        self._proc = subprocess.Popen(
            argv, env=self._worker_env(), cwd=str(self._jobs_dir),
            stdout=self._worker_log, stderr=subprocess.STDOUT)
        ready = self._jobs_dir / "ready.json"
        failed = self._jobs_dir / "startup_error.json"
        t0 = time.time()
        limit = float(cfg.get("startup_timeout_sec", 900))
        poll = float(cfg.get("poll_sec", 1.0))
        while True:
            if ready.exists():
                self._worker_ready = json.loads(ready.read_text(encoding="utf-8"))
                break
            if failed.exists():
                err = json.loads(failed.read_text(encoding="utf-8"))
                self._stop_worker()
                raise Hunyuan3DAssetError(
                    f"worker failed during startup: {err.get('error')}\n{err.get('traceback', '')[-3000:]}")
            if self._proc.poll() is not None:
                code = self._proc.returncode
                self._stop_worker()
                raise Hunyuan3DAssetError(
                    f"worker exited during startup (code {code}):\n{self._worker_tail()}")
            if time.time() - t0 > limit:
                self._stop_worker()
                raise Hunyuan3DAssetError(f"worker not ready within startup_timeout_sec={limit:.0f}")
            self._heartbeat("worker startup")  # C-003
            time.sleep(poll)
        self._heartbeat("worker ready")
        self._log(f"worker ready in {time.time() - t0:.1f}s: {self._worker_ready.get('device')}")

    def _stop_worker(self) -> None:
        if self._jobs_dir is not None:
            try:
                (self._jobs_dir / "shutdown").write_text("1", encoding="utf-8")
            except OSError:
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=20)
        self._proc = None
        if self._worker_log is not None:
            try:
                self._worker_log.close()
            except OSError:
                pass
            self._worker_log = None

    # ------------------------------------------------------------- 2/9
    def setup(self, setup: TrainerSetup) -> None:
        self._setup_obj = setup
        cfg = self.config
        # C-004 FIRST: nothing below (weights, worker, model load) runs until
        # every input has been read, hashed and staged.
        self._hash_inputs(Path(cfg["package_root"]))
        self._find_reusable()
        for a in self._assets:
            if a["name"] not in self._reusable:
                self._move_aside(a["name"])
        todo = [a["name"] for a in self._assets if a["name"] not in self._reusable]
        if not todo:
            self._log("every asset reused; no worker started, no weights read")
            return
        if cfg.get("stub"):
            self._log("STUB mode: no weights, no GPU, no Hunyuan code")
        else:
            if not str(setup.device).startswith("cuda"):
                raise Hunyuan3DAssetError(
                    f"hunyuan3d_asset needs a CUDA device, got {setup.device!r}; there is "
                    "no CPU path (use stub: true for a no-GPU run)")
            self._check_weights()
        try:
            self._start_worker()
        except BaseException:
            self._stop_worker()
            raise

    # ------------------------------------------------------------- 3-6/9
    def train_iter(self) -> Iterable[Any]:
        if self._cursor >= len(self._assets):
            raise Hunyuan3DAssetError(
                f"epoch cursor {self._cursor} past {len(self._assets)} assets")
        return iter([self._assets[self._cursor]])

    def eval_iter(self) -> Iterable[Any]:
        return iter(())  # QA is Game1's audit + the owner's review, offline

    def train_step(self, batch: Any) -> TrainerStepResult:
        assert self._setup_obj is not None
        asset = dict(batch)
        name = asset["name"]
        if name in self._reusable:
            m = self._reusable[name]
            metrics = {"asset_sec": 0.0, "reused": 1.0,
                       "faces": float(m.get("mesh", {}).get("faces", 0)),
                       "vertices": float(m.get("mesh", {}).get("vertices", 0)),
                       "glb_bytes": float(next((o["bytes"] for o in m["outputs"]
                                                if o["file"] == m["primary"]), 0))}
            self._epoch_metrics.append(metrics)
            self._results.append({"asset": name, "reused": True,
                                  "manifest": str(self._asset_dir(name) / "manifest.json")})
            self._heartbeat(f"reused {name}")
            return TrainerStepResult(metrics=metrics, n_examples=1)
        try:
            return self._generate(asset)
        except BaseException:
            # The runner does not call teardown() on failure; stop the worker here.
            self._stop_worker()
            raise

    def _generate(self, asset: Dict[str, Any]) -> TrainerStepResult:
        assert self._setup_obj is not None and self._jobs_dir is not None
        if self._proc is None or self._proc.poll() is not None:
            raise Hunyuan3DAssetError(f"worker is not running:\n{self._worker_tail()}")
        name = asset["name"]
        out_dir = self._asset_dir(name)
        out_dir.mkdir(parents=True, exist_ok=False)
        t0 = time.time()
        job = {"name": name, "image": str(self._staged_input[name]), "out_dir": str(out_dir),
               "seed": int(asset["seed"]), "texture": bool(asset["texture"]),
               "target_faces": int(asset["target_faces"]),
               "has_alpha_channel": bool(self._input_sha[name]["has_alpha_channel"])}
        tmp = self._jobs_dir / f"{name}.job.tmp"
        tmp.write_text(json.dumps(job), encoding="utf-8")
        os.replace(tmp, self._jobs_dir / f"{name}.job.json")
        result_p = self._jobs_dir / f"{name}.result.json"
        error_p = self._jobs_dir / f"{name}.error.json"
        limit = float(self.config.get("asset_timeout_sec", 600))
        poll = float(self.config.get("poll_sec", 1.0))
        while True:
            if result_p.exists():
                result = json.loads(result_p.read_text(encoding="utf-8"))
                break
            if error_p.exists():
                err = json.loads(error_p.read_text(encoding="utf-8"))
                raise Hunyuan3DAssetError(
                    f"asset {name!r} failed in the worker: {err.get('error')}\n"
                    f"{err.get('traceback', '')[-3000:]}")
            if self._proc.poll() is not None:
                raise Hunyuan3DAssetError(
                    f"worker died during asset {name!r} (code {self._proc.returncode}):\n"
                    f"{self._worker_tail()}")
            if time.time() - t0 > limit:
                raise Hunyuan3DAssetError(
                    f"asset {name!r} exceeded asset_timeout_sec={limit:.0f}")
            self._heartbeat(f"generating {name}")  # C-003
            time.sleep(poll)
        dt = time.time() - t0

        primary = out_dir / result["primary"]
        if not primary.is_file() or primary.stat().st_size == 0:
            raise Hunyuan3DAssetError(f"worker reported {primary} but it is missing or empty")
        outputs = []
        for p in sorted(out_dir.rglob("*")):
            if not p.is_file() or p.name == "manifest.json":
                continue
            if p.suffix.lower() == ".glb":
                _glb_check(p)
            outputs.append({"file": p.relative_to(out_dir).as_posix(),
                            "sha256": _sha256_file(p), "bytes": p.stat().st_size})
        if not any(o["file"] == result["primary"] for o in outputs):
            raise Hunyuan3DAssetError(f"primary output {result['primary']} not among outputs")
        manifest = {
            "trainer": "hunyuan3d_asset",
            "status": "done",
            "licence": _LICENCE,
            "asset": name,
            "stub": bool(self.config.get("stub", False)),
            "fingerprint": self._fingerprint(asset),
            "input": self._input_sha[name],
            "seed": int(asset["seed"]),
            "texture": bool(asset["texture"]),
            "target_faces": int(asset["target_faces"]),
            "params": self._params(),
            "pins": self._pins(),
            "determinism_env": (dict(_DETERMINISM_ENV)
                                if bool(self.config.get("determinism", True)) else None),
            "worker": {"device": self._worker_ready.get("device"),
                       "env_observed": self._worker_ready.get("env_observed"),
                       "torch_deterministic": self._worker_ready.get("torch_deterministic")},
            "background": result.get("background"),
            "mesh": result.get("mesh", {}),
            "primary": result["primary"],
            "outputs": outputs,
            "asset_sec": dt,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        metrics = {"asset_sec": dt, "reused": 0.0,
                   "faces": float(manifest["mesh"].get("faces", 0)),
                   "vertices": float(manifest["mesh"].get("vertices", 0)),
                   "glb_bytes": float(primary.stat().st_size)}
        self._epoch_metrics.append(metrics)
        self._results.append({"asset": name, "reused": False,
                              "manifest": str(out_dir / "manifest.json")})
        self._heartbeat(f"done {name}")
        self._log(f"asset done: {name} in {dt:.1f}s -> {primary}")
        return TrainerStepResult(metrics=metrics, n_examples=1)

    def eval_step(self, batch: Any) -> TrainerStepResult:  # pragma: no cover
        raise Hunyuan3DAssetError("eval_step should be unreachable: eval_iter is empty")

    # ------------------------------------------------------------- 7-9/9
    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        m = self._epoch_metrics[-1] if self._epoch_metrics else {}
        self._cursor = int(epoch) + 1
        return TrainerEpochResult(train_metrics=m, val_metrics={},
                                  is_best=False, monitor_value=m.get("asset_sec"))

    def save_checkpoint(self, path: Path) -> None:
        """Batch manifest; never overwrites (mirrors wan_vace_shot C-7)."""
        p = Path(path).with_suffix(".json")
        n = 1
        while p.exists():
            p = Path(path).with_suffix(f".{n}.json")
            n += 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "trainer": "hunyuan3d_asset",
            "licence": _LICENCE,
            "config": {k: v for k, v in self.config.items() if k != "assets"},
            "assets": self._results,
            "inputs": self._input_sha,  # C-004
            "params": self._params(),
            "pins": self._pins(),
        }, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path).with_suffix(".json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self._results = list(data.get("assets", []))

    def num_epochs(self) -> int:
        return len(self._assets)

    def teardown(self) -> None:
        self._stop_worker()
