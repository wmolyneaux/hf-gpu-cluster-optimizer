"""hunyuan3d_worker -- the process that holds Hunyuan3D 2.1 for one asset batch.

Spawned by models/hunyuan3d_asset.py (never imported by the registry). It loads
the models ONCE, then serves jobs through files in the jobs directory:

    spec.json              written by the trainer before spawn
    ready.json             written here when the models are loaded
    startup_error.json     written here if loading failed (then exit 1)
    <name>.job.json        written by the trainer (atomic rename)
    <name>.result.json     written here on success (atomic rename)
    <name>.error.json      written here on failure (atomic rename)
    shutdown               written by the trainer; exit 0

Files, not pipes, so a chatty model can never fill a pipe buffer and deadlock
the trainer's poll loop (which is also where the C-003 heartbeat lives). No
threads: the process exits when told to, when its parent disappears, or when
the trainer terminates it.

The real path follows the pinned upstream API (Tencent-Hunyuan/Hunyuan3D-2.1 @
82920d64, demo.py + gradio_app.py): Hunyuan3DDiTFlowMatchingPipeline for shape,
FloaterRemover / DegenerateFaceRemover / FaceReducer for cleanup, and
Hunyuan3DPaintPipeline for PBR texture. Two upstream behaviours are NOT
inherited, on purpose:
  * demo.py converts the input to RGBA before testing for RGB, so it never
    removes the background; a flat-grey sheet cell would become a grey slab.
    Here an input without real alpha always goes through rembg, and the result
    is checked to contain an object.
  * convert_obj_to_glb swallows every exception and returns False. Here its
    return value and the file are checked, and a failure raises.

ASCII only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

_DETERMINISM_KEYS = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE")


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _stub_glb(seed: int) -> bytes:
    """A valid one-triangle glTF 2.0 binary whose bytes depend only on `seed`."""
    s = float(seed % 997 + 1) / 997.0
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s, 0.0)
    gltf = {
        "asset": {"version": "2.0", "generator": "modallabs hunyuan3d_worker STUB"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(positions)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0.0, 0.0, 0.0], "max": [s, s, 0.0]}],
    }
    js = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode("ascii")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(positions)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(positions), b"BIN\x00") + positions)


# ------------------------------------------------------------------ stub
def _load_stub(spec: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    delay = float(spec.get("stub_delay_sec") or 0.0)
    fail = spec.get("stub_fail_asset")
    time.sleep(delay)

    def run(job: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(delay)
        if job["name"] == fail:
            raise RuntimeError(f"STUB failure injected for {job['name']!r}")
        out = Path(job["out_dir"])
        name = job["name"]
        glb = _stub_glb(int(job["seed"]))
        if job["texture"]:
            (out / f"{name}_shape.glb").write_bytes(glb)
        (out / f"{name}.glb").write_bytes(glb)
        return {"primary": f"{name}.glb", "mesh": {"faces": 1, "vertices": 3},
                "background": "input-alpha" if job["has_alpha_channel"] else "stub-no-rembg"}

    return run


# ------------------------------------------------------------------ real
def _load_real(spec: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    repo = Path(spec["repo_dir"])
    for sub in (repo, repo / "hy3dpaint", repo / "hy3dshape"):
        if not sub.is_dir():
            raise RuntimeError(f"Hunyuan3D-2.1 checkout missing {sub}; the image is wrong")
        sys.path.insert(0, str(sub))

    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the worker; there is no CPU path")
    if spec["determinism"]:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover
    from hy3dshape.rembg import BackgroundRemover
    import trimesh

    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        spec["shape_model"]["repo"], subfolder=spec["shape_model"]["subfolder"],
        use_safetensors=False, variant="fp16", device="cuda", dtype=torch.float16)
    rembg = BackgroundRemover()
    floater, degenerate, reducer = FloaterRemover(), DegenerateFaceRemover(), FaceReducer()

    paint_pipe = None
    convert_obj_to_glb = None
    if spec["needs_paint"]:
        from torchvision_fix import apply_fix
        if not apply_fix():
            raise RuntimeError("torchvision functional_tensor compatibility fix failed")
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
        from DifferentiableRenderer.mesh_utils import convert_obj_to_glb as _c2g
        convert_obj_to_glb = _c2g
        p = spec["paint"]
        conf = Hunyuan3DPaintConfig(int(p["max_views"]), int(p["resolution"]))
        conf.multiview_cfg_path = p["cfg_path"]
        conf.custom_pipeline = str(repo / "hy3dpaint" / "hunyuanpaintpbr")
        conf.dino_ckpt_path = p["dino_path"]
        conf.realesrgan_ckpt_path = p["realesrgan_path"]
        paint_pipe = Hunyuan3DPaintPipeline(conf)

    shape = spec["shape"]

    def _prepare(job: Dict[str, Any]):
        img = Image.open(job["image"])
        img.load()
        if job["has_alpha_channel"]:
            rgba = img.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"))
            if alpha.min() < 255:
                return rgba, "input-alpha"
        rgba = rembg(img.convert("RGB"))
        alpha = np.asarray(rgba.getchannel("A"))
        frac = float((alpha > 127).mean())
        if frac < 0.005:
            raise RuntimeError(f"background removal found no object (foreground {frac:.4f})")
        if frac > 0.995:
            raise RuntimeError(f"background removal removed nothing (foreground {frac:.4f})")
        return rgba, "rembg-u2net"

    def run(job: Dict[str, Any]) -> Dict[str, Any]:
        seed = int(job["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        out = Path(job["out_dir"])
        name = job["name"]
        image, background = _prepare(job)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        meshes = shape_pipe(image=image, num_inference_steps=int(shape["steps"]),
                            guidance_scale=float(shape["guidance_scale"]),
                            octree_resolution=int(shape["octree_resolution"]),
                            num_chunks=int(shape["num_chunks"]), generator=gen,
                            output_type="trimesh", enable_pbar=False)
        mesh = meshes[0] if meshes else None
        if mesh is None:
            raise RuntimeError("shape pipeline returned no mesh (surface extraction failed)")
        mesh = floater(mesh)
        mesh = degenerate(mesh)
        mesh = reducer(mesh, max_facenum=int(job["target_faces"]))
        if job["texture"]:
            shape_glb = out / f"{name}_shape.glb"
            mesh.export(str(shape_glb))
            obj = out / f"{name}.obj"
            paint_pipe(mesh_path=str(shape_glb), image_path=image,
                       output_mesh_path=str(obj), use_remesh=True, save_glb=False)
            if not obj.is_file():
                raise RuntimeError(f"paint pipeline wrote no OBJ at {obj}")
            glb = out / f"{name}.glb"
            if not convert_obj_to_glb(str(obj), str(glb)) or not glb.is_file():
                raise RuntimeError(f"convert_obj_to_glb failed for {obj} (upstream hides the error)")
            final = trimesh.load(str(glb), force="mesh")
        else:
            glb = out / f"{name}.glb"
            mesh.export(str(glb))
            final = mesh
        return {"primary": glb.name, "background": background,
                "mesh": {"faces": int(len(final.faces)), "vertices": int(len(final.vertices))}}

    return run


# ------------------------------------------------------------------ loop
def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    jobs = Path(spec["jobs_dir"])
    parent = int(spec["parent_pid"])
    poll = float(spec.get("poll_sec", 1.0))
    try:
        run = _load_stub(spec) if spec["stub"] else _load_real(spec)
        ready: Dict[str, Any] = {
            "env_observed": {k: os.environ.get(k) for k in _DETERMINISM_KEYS},
            "pid": os.getpid(),
        }
        if spec["stub"]:
            ready.update({"device": "stub (no GPU)", "torch_deterministic": None})
        else:
            import torch
            ready.update({"device": torch.cuda.get_device_name(0),
                          "torch": torch.__version__,
                          "torch_deterministic": torch.are_deterministic_algorithms_enabled()})
        _write_atomic(jobs / "ready.json", ready)
    except BaseException as exc:  # noqa: BLE001 -- reported to the trainer, then exit 1
        _write_atomic(jobs / "startup_error.json",
                      {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
        return 1

    done = set()
    while True:
        if (jobs / "shutdown").exists():
            return 0
        if os.getppid() != parent:
            print("hunyuan3d_worker: parent is gone; exiting", flush=True)
            return 3
        for job_path in sorted(jobs.glob("*.job.json")):
            name = job_path.name[: -len(".job.json")]
            if name in done:
                continue
            done.add(name)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            t0 = time.time()
            try:
                result = run(job)
                result["sec"] = time.time() - t0
                _write_atomic(jobs / f"{name}.result.json", result)
            except BaseException as exc:  # noqa: BLE001 -- reported, the trainer raises
                print(traceback.format_exc(), flush=True)
                _write_atomic(jobs / f"{name}.error.json",
                              {"error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
