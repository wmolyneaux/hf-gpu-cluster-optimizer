"""Tests for the hunyuan3d_asset lane (WorldClaw WC-HF). CPU only, no Modal calls.

Mirrors the wan_vace_shot lane's controls: config refusals, stub mode end to
end, C-004 provenance hashed before any spend, C-003 heartbeat, C-005
determinism env, the licence field, and the modal_app pinning (L40S / short).

Run from the directory that CONTAINS the package directory named `modallabs`:
    python -m pytest modallabs/tests/test_hunyuan3d_asset.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Dict

import pytest

import modallabs.models  # noqa: F401 -- fires the registry
from modallabs.models import hunyuan3d_asset as H
from modallabs.registry import get as registry_get
from modallabs.runner import train_one

PKG_ROOT = Path(H.__file__).resolve().parent.parent  # the package directory


# ------------------------------------------------------------------ helpers
def make_png(path: Path, w: int = 64, h: int = 64, alpha: bool = False, shade: int = 128) -> None:
    """Stdlib-only PNG writer (flat grey, optionally RGBA with a transparent corner)."""
    ch = 4 if alpha else 3
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            px = [shade, shade, shade]
            if alpha:
                px.append(0 if (x < 4 and y < 4) else 255)
            row += bytes(px)
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw)
                     + chunk(b"IEND", b""))
    assert ch in (3, 4)


def base_cfg(pkg: Path, **over: Any) -> Dict[str, Any]:
    make_png(pkg / "cells" / "oak.png")
    make_png(pkg / "cells" / "lamp.png", shade=90)
    make_png(pkg / "cells" / "fern.png", alpha=True)
    cfg: Dict[str, Any] = {
        "stub": True,
        "package_root": str(pkg),
        "epochs": 3,
        "shape_steps": 50, "guidance_scale": 5.0, "octree_resolution": 384,
        "paint_max_views": 6, "paint_resolution": 512,
        "min_side_px": 64, "poll_sec": 0.05,
        "startup_timeout_sec": 60, "asset_timeout_sec": 60,
        "assets": [
            {"name": "oak", "image": "cells/oak.png", "seed": 11, "texture": True,
             "target_faces": 20000},
            {"name": "lamp", "image": "cells/lamp.png", "seed": 12, "texture": True,
             "target_faces": 8000},
            {"name": "fern", "image": "cells/fern.png", "seed": 13, "texture": False,
             "target_faces": 60000},
        ],
    }
    cfg.update(over)
    return cfg


def run(cfg: Dict[str, Any], out: Path, run_id: str = "r", name: str = "batch") -> Dict[str, Any]:
    rc = {"name": name, "type": "hunyuan3d_asset", "seed": 0, "config": copy.deepcopy(cfg)}
    return train_one(rc, run_id=run_id, output_root=out, resume=False, force_cpu=True)


_REAL_POPEN = subprocess.Popen


def spy_worker_spawn(monkeypatch, calls, on_spawn=None, allow=False):
    """Record every spawn of the hunyuan3d worker. Other subprocesses (the
    runner's `git rev-parse` provenance probe) pass straight through."""
    def spy(*a, **k):
        argv = a[0] if a else k.get("args")
        if isinstance(argv, (list, tuple)) and str(H._WORKER) in [str(x) for x in argv]:
            calls.append(argv)
            if on_spawn is not None:
                on_spawn()
            if not allow:
                raise AssertionError("worker spawn was not expected here")
        return _REAL_POPEN(*a, **k)
    monkeypatch.setattr(H.subprocess, "Popen", spy)


def manifest(out: Path, asset: str, run_id: str = "r", name: str = "batch") -> Dict[str, Any]:
    return json.loads((out / run_id / name / "assets" / asset / "manifest.json")
                      .read_text(encoding="utf-8"))


# ------------------------------------------------------------------ registry
def test_registered():
    assert registry_get("hunyuan3d_asset") is H.Hunyuan3DAssetTrainer


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("key", H._REQUIRED)
def test_refuses_each_missing_required_key(tmp_path, key):
    cfg = base_cfg(tmp_path)
    del cfg[key]
    with pytest.raises(H.Hunyuan3DAssetError, match=f"missing keys: .*{key}"):
        H.Hunyuan3DAssetTrainer.from_config(cfg)


@pytest.mark.parametrize("key", H._ASSET_KEYS)
def test_refuses_each_missing_asset_key(tmp_path, key):
    cfg = base_cfg(tmp_path)
    del cfg["assets"][0][key]
    with pytest.raises(H.Hunyuan3DAssetError, match="missing keys"):
        H.Hunyuan3DAssetTrainer.from_config(cfg)


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c.update(epochs=2), "one epoch is one asset"),
    (lambda c: c.update(bogus=1), "unknown keys"),
    (lambda c: c["assets"][0].update(textrue=True), "unknown keys"),
    (lambda c: c["assets"][1].update(name="oak"), "duplicate asset name"),
    (lambda c: c["assets"][0].update(name="../escape"), "must match"),
    (lambda c: c["assets"][0].update(name="Oak"), "must match"),
    (lambda c: c["assets"][0].update(image="/abs/oak.png"), "relative path"),
    (lambda c: c["assets"][0].update(image="C:/abs/oak.png"), "relative path"),
    (lambda c: c["assets"][0].update(image="cells/../../oak.png"), "relative path"),
    (lambda c: c["assets"][0].update(image="cells/oak.jpg"), "must be a .png"),
    (lambda c: c["assets"][0].update(seed=-1), "seed must be"),
    (lambda c: c["assets"][0].update(seed=True), "seed must be"),
    (lambda c: c["assets"][0].update(texture="yes"), "texture must be"),
    (lambda c: c["assets"][0].update(target_faces=40001), "caps target_faces"),
    (lambda c: c["assets"][2].update(target_faces=10), "target_faces must be"),
    (lambda c: c.update(octree_resolution=300), "octree_resolution"),
    (lambda c: c.update(paint_resolution=1024), "paint_resolution"),
    (lambda c: c.update(paint_max_views=12), "paint_max_views"),
    (lambda c: c.update(shape_steps=0), "shape_steps"),
    (lambda c: c.update(stub="true"), "stub must be"),
    (lambda c: c.update(determinism="no"), "determinism must be"),
    (lambda c: c.update(stub=False, stub_delay_sec=1.0), "stub-only"),
    (lambda c: c.update(stub_fail_asset="nope"), "stub_fail_asset"),
    (lambda c: c.update(assets=[], epochs=0), "non-empty"),
])
def test_refusals_fail_loudly(tmp_path, mutate, match):
    cfg = base_cfg(tmp_path)
    mutate(cfg)
    with pytest.raises(H.Hunyuan3DAssetError, match=match):
        H.Hunyuan3DAssetTrainer.from_config(cfg)


def test_refusal_surfaces_through_runner_as_failed(tmp_path):
    cfg = base_cfg(tmp_path / "pkg")
    cfg["assets"][0]["texture"] = "yes"
    res = run(cfg, tmp_path / "out")
    assert res["phase"] == "failed"
    assert "texture must be" in res["error"]
    assert not (tmp_path / "out" / "r" / "batch" / ".modallabs_done").exists()


def test_refuses_non_png_and_small_input(tmp_path):
    cfg = base_cfg(tmp_path / "pkg")
    (tmp_path / "pkg" / "cells" / "oak.png").write_bytes(b"GIF89a not a png at all" * 4)
    res = run(cfg, tmp_path / "out")
    assert res["phase"] == "failed" and "not a PNG" in res["error"]
    cfg = base_cfg(tmp_path / "pkg2", min_side_px=256)
    res = run(cfg, tmp_path / "out2")
    assert res["phase"] == "failed" and "short side must be >= 256" in res["error"]


def test_real_mode_refuses_cpu_device(tmp_path):
    cfg = base_cfg(tmp_path / "pkg", stub=False)
    res = run(cfg, tmp_path / "out")
    assert res["phase"] == "failed"
    assert "needs a CUDA device" in res["error"]


# ------------------------------------------------------------------ stub e2e
def test_stub_end_to_end(tmp_path):
    cfg = base_cfg(tmp_path / "pkg")
    out = tmp_path / "out"
    res = run(cfg, out)
    assert res["phase"] == "succeeded", res.get("error")
    run_dir = out / "r" / "batch"
    assert (run_dir / ".modallabs_done").exists()
    ckpt = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["licence"] == H._LICENCE
    assert [a["asset"] for a in ckpt["assets"]] == ["oak", "lamp", "fern"]
    for name, textured in (("oak", True), ("lamp", True), ("fern", False)):
        m = manifest(out, name)
        files = {o["file"] for o in m["outputs"]}
        assert m["primary"] == f"{name}.glb" and f"{name}.glb" in files
        assert (f"{name}_shape.glb" in files) is textured
        glb = run_dir / "assets" / name / f"{name}.glb"
        magic, version, length = struct.unpack("<4sII", glb.read_bytes()[:12])
        assert (magic, version, length) == (b"glTF", 2, glb.stat().st_size)
        assert m["status"] == "done" and m["stub"] is True
    assert manifest(out, "fern")["background"] == "input-alpha"
    # the worker process is gone after teardown
    assert not list((run_dir / "worker_jobs").glob("*.error.json"))


def test_licence_field_in_every_manifest(tmp_path):
    out = tmp_path / "out"
    assert run(base_cfg(tmp_path / "pkg"), out)["phase"] == "succeeded"
    for name in ("oak", "lamp", "fern"):
        assert manifest(out, name)["licence"] == (
            "Tencent Hunyuan 3D 2.1 Community License -- output NOT cleared for shipped "
            "assets pending counsel (PLAN_WORLDCLAW 8.3a); blockout / internal use")


def test_manifest_pins_are_the_real_ones(tmp_path):
    out = tmp_path / "out"
    assert run(base_cfg(tmp_path / "pkg"), out)["phase"] == "succeeded"
    pins = manifest(out, "oak")["pins"]
    assert pins["code"] == {"repo": "Tencent-Hunyuan/Hunyuan3D-2.1",
                            "commit": "82920d643c0dc2f7bfd7255f45f62d386edfe60c"}
    assert pins["weights"]["repo"] == "tencent/Hunyuan3D-2.1"
    assert pins["weights"]["revision"] == "0b94677654c57bb9a6b6845cd7b704ccf551d327"
    assert pins["files"]["shape_ckpt"]["sha256"] == (
        "6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e")
    assert manifest(out, "oak")["params"]["paint"]["seed"] == 0


# ------------------------------------------------------------------ C-004
def test_c004_inputs_hashed_before_worker_spawn(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path / "pkg")
    seen = {}
    trainer_ref = {}
    orig_setup = H.Hunyuan3DAssetTrainer.setup

    def spy_setup(self, setup):
        trainer_ref["t"] = self
        return orig_setup(self, setup)

    def on_spawn():
        seen["hashed_at_spawn"] = dict(trainer_ref["t"]._input_sha)

    monkeypatch.setattr(H.Hunyuan3DAssetTrainer, "setup", spy_setup)
    calls = []
    spy_worker_spawn(monkeypatch, calls, on_spawn=on_spawn, allow=True)
    out = tmp_path / "out"
    assert run(cfg, out)["phase"] == "succeeded"
    assert len(calls) == 1  # one worker for the whole batch
    import hashlib
    expect = {n: hashlib.sha256((tmp_path / "pkg" / "cells" / f"{n}.png").read_bytes()).hexdigest()
              for n in ("oak", "lamp", "fern")}
    assert {n: v["sha256"] for n, v in seen["hashed_at_spawn"].items()} == expect
    for n in expect:
        assert manifest(out, n)["input"]["sha256"] == expect[n]


def test_c004_missing_input_refuses_before_any_spawn(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path / "pkg")
    (tmp_path / "pkg" / "cells" / "lamp.png").unlink()
    calls = []
    spy_worker_spawn(monkeypatch, calls)
    res = run(cfg, tmp_path / "out")
    assert res["phase"] == "failed" and "cannot hash missing input" in res["error"]
    assert calls == []


def test_c004_input_hash_precedes_weight_check(tmp_path, monkeypatch):
    """Real mode on a (faked) CUDA device: a missing input must be reported,
    not the missing weights -- hashing comes first. With inputs present, the
    weight check refuses and no worker is ever spawned."""
    calls = []
    spy_worker_spawn(monkeypatch, calls)
    monkeypatch.setattr(H, "_MODELS", tmp_path / "no_such_volume")
    cfg = base_cfg(tmp_path / "pkg", stub=False)
    (tmp_path / "pkg" / "cells" / "oak.png").unlink()
    t = H.Hunyuan3DAssetTrainer.from_config(cfg)
    from modallabs.base import TrainerSetup
    su = TrainerSetup(config=cfg, seed=0, device="cuda", output_dir=tmp_path / "o1",
                      log_fn=lambda m: None, metric_fn=lambda *a, **k: None)
    with pytest.raises(H.Hunyuan3DAssetError, match="cannot hash missing input"):
        t.setup(su)
    make_png(tmp_path / "pkg" / "cells" / "oak.png")
    t = H.Hunyuan3DAssetTrainer.from_config(cfg)
    su.output_dir = tmp_path / "o2"
    with pytest.raises(H.Hunyuan3DAssetError, match="never staged"):
        t.setup(su)
    assert len(t._input_sha) == 3  # hashed, then refused on weights
    assert calls == []


def test_staged_input_is_the_hashed_bytes(tmp_path):
    out = tmp_path / "out"
    assert run(base_cfg(tmp_path / "pkg"), out)["phase"] == "succeeded"
    import hashlib
    for p in (out / "r" / "batch" / "inputs").glob("*.png"):
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert p.name.endswith(f"__{sha[:16]}.png")


def test_weight_check_refuses_wrong_size_and_bad_staged_sha(tmp_path, monkeypatch):
    models = tmp_path / "vol"
    monkeypatch.setattr(H, "_MODELS", models)
    # Same table shape with tiny sizes: never allocate multi-GB files on the dev box.
    monkeypatch.setattr(H, "_WEIGHT_FILES", tuple(
        (k, rel, 10, ("0" * 64 if sha else None), need)
        for k, rel, _size, sha, need in H._WEIGHT_FILES))
    cfg = base_cfg(tmp_path / "pkg", stub=False)
    cfg["assets"] = [dict(cfg["assets"][2])]  # shape only
    cfg["epochs"] = 1
    t = H.Hunyuan3DAssetTrainer.from_config(cfg)
    (models).mkdir()
    (models / "STAGED.json").write_text(json.dumps(
        {"weights_revision": H._WEIGHTS_REVISION, "files": {}}), encoding="utf-8")
    with pytest.raises(H.Hunyuan3DAssetError, match="weight not staged"):
        t._check_weights()
    for _k, rel, size, _sha, need in H._WEIGHT_FILES:
        if need == "shape":
            p = models / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"w" * size)
    with pytest.raises(H.Hunyuan3DAssetError, match="STAGED.json sha256"):
        t._check_weights()
    (models / H._WEIGHT_FILES[0][1]).write_bytes(b"x")
    with pytest.raises(H.Hunyuan3DAssetError, match="bytes, pinned"):
        t._check_weights()
    (models / "STAGED.json").write_text(json.dumps(
        {"weights_revision": "not-the-pin", "files": {}}), encoding="utf-8")
    with pytest.raises(H.Hunyuan3DAssetError, match="re-stage deliberately"):
        t._check_weights()


# ------------------------------------------------------------------ C-003
def test_c003_heartbeat_written_during_waits(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path / "pkg", stub_delay_sec=0.4)
    notes = []
    orig = H.Hunyuan3DAssetTrainer._heartbeat

    def spy(self, note):
        notes.append(note)
        return orig(self, note)

    monkeypatch.setattr(H.Hunyuan3DAssetTrainer, "_heartbeat", spy)
    out = tmp_path / "out"
    assert run(cfg, out)["phase"] == "succeeded"
    assert notes.count("worker startup") >= 2
    for n in ("oak", "lamp", "fern"):
        assert notes.count(f"generating {n}") >= 2
    hb = (out / "r" / "batch" / "heartbeat.txt").read_text(encoding="utf-8")
    assert hb.split()[1:] == ["done", "fern"]


# ------------------------------------------------------------------ C-005
def test_c005_determinism_env_reaches_worker(tmp_path):
    out = tmp_path / "out"
    assert run(base_cfg(tmp_path / "pkg"), out)["phase"] == "succeeded"
    m = manifest(out, "oak")
    assert m["determinism_env"] == H._DETERMINISM_ENV
    assert m["worker"]["env_observed"] == H._DETERMINISM_ENV
    assert m["seed"] == 11 and m["params"]["determinism"] is True


def test_c005_determinism_off_is_explicit(tmp_path, monkeypatch):
    for k in H._DETERMINISM_ENV:
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "out"
    assert run(base_cfg(tmp_path / "pkg", determinism=False), out)["phase"] == "succeeded"
    m = manifest(out, "oak")
    assert m["determinism_env"] is None
    assert all(v is None for v in m["worker"]["env_observed"].values())


def test_c005_same_seed_same_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run(base_cfg(tmp_path / "pkg"), a)["phase"] == "succeeded"
    assert run(base_cfg(tmp_path / "pkg"), b)["phase"] == "succeeded"
    for n in ("oak", "lamp", "fern"):
        assert manifest(a, n)["outputs"] == manifest(b, n)["outputs"]
    assert manifest(a, "oak")["outputs"] != manifest(a, "lamp")["outputs"]


# ------------------------------------------------------------------ failures
def test_worker_failure_is_loud_and_leaves_no_worker(tmp_path):
    out = tmp_path / "out"
    res = run(base_cfg(tmp_path / "pkg", stub_fail_asset="lamp"), out)
    assert res["phase"] == "failed"
    assert "lamp" in res["error"] and "STUB failure injected" in res["error"]
    assert not (out / "r" / "batch" / ".modallabs_done").exists()
    assert manifest(out, "oak")["status"] == "done"
    assert not (out / "r" / "batch" / "assets" / "lamp" / "manifest.json").exists()


def test_asset_timeout_is_loud(tmp_path):
    cfg = base_cfg(tmp_path / "pkg", stub_delay_sec=3.0, asset_timeout_sec=1,
                   startup_timeout_sec=30)
    res = run(cfg, tmp_path / "out")
    assert res["phase"] == "failed" and "exceeded asset_timeout_sec=1" in res["error"]


def test_startup_timeout_is_loud(tmp_path):
    cfg = base_cfg(tmp_path / "pkg", stub_delay_sec=3.0, startup_timeout_sec=1)
    res = run(cfg, tmp_path / "out")
    assert res["phase"] == "failed" and "not ready within startup_timeout_sec=1" in res["error"]


# ------------------------------------------------------------------ resume
def test_rerun_reuses_verified_assets_and_regenerates_tampered(tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = base_cfg(tmp_path / "pkg")
    assert run(cfg, out)["phase"] == "succeeded"
    calls = []
    spy_worker_spawn(monkeypatch, calls)
    res = run(cfg, out)  # same run dir: everything verified -> no worker at all
    assert res["phase"] == "succeeded" and calls == []
    monkeypatch.undo()
    glb = out / "r" / "batch" / "assets" / "lamp" / "lamp.glb"
    glb.write_bytes(glb.read_bytes() + b"tamper")
    assert run(cfg, out)["phase"] == "succeeded"
    assert (out / "r" / "batch" / "assets" / "lamp.stale.1" / "lamp.glb").exists()
    run_dir = out / "r" / "batch"
    ckpts = sorted(p.name for p in run_dir.glob("checkpoint*.json"))
    assert ckpts == ["checkpoint.1.json", "checkpoint.2.json", "checkpoint.json"]  # never overwritten
    second = json.loads((run_dir / "checkpoint.1.json").read_text(encoding="utf-8"))
    assert all(a["reused"] for a in second["assets"])
    third = json.loads((run_dir / "checkpoint.2.json").read_text(encoding="utf-8"))
    reused = {a["asset"]: a["reused"] for a in third["assets"]}
    assert reused == {"oak": True, "lamp": False, "fern": True}


def test_changed_seed_is_not_reused(tmp_path):
    out = tmp_path / "out"
    cfg = base_cfg(tmp_path / "pkg")
    assert run(cfg, out)["phase"] == "succeeded"
    cfg["assets"][0]["seed"] = 99
    assert run(cfg, out)["phase"] == "succeeded"
    assert (out / "r" / "batch" / "assets" / "oak.stale.1").is_dir()
    assert manifest(out, "oak")["seed"] == 99


# ------------------------------------------------------------------ worker
def test_stub_glb_is_valid_gltf():
    from modallabs.models import hunyuan3d_worker as W
    data = W._stub_glb(7)
    magic, version, length = struct.unpack("<4sII", data[:12])
    assert (magic, version, length) == (b"glTF", 2, len(data))
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    assert jtype == b"JSON" and jlen % 4 == 0
    json.loads(data[20:20 + jlen])
    assert W._stub_glb(7) == data and W._stub_glb(8) != data


# ------------------------------------------------------------------ modal_app
def _load_modal_app():
    spec = importlib.util.spec_from_file_location("wc_modal_app", PKG_ROOT / "modal_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modal_app_pins_hunyuan_to_l40s_short():
    ma = _load_modal_app()
    rc = {"name": "b", "type": "hunyuan3d_asset", "modal": {"max_runtime_sec": 600},
          "config": {"epochs": 3}}
    assert ma._gpu_for_run(rc) == "L40S"
    assert ma._pinned_lane(rc) == "short"
    # Worst case is the whole lane the function enforces, not the smaller ask.
    assert ma._worst_case_runtime_sec(rc) == 1800
    total, _ = ma.estimate_total_cost_usd({"runs": [rc]})
    assert total == pytest.approx(2.00 * 1800 / 3600)
    with pytest.raises(RuntimeError, match="pinned to 'L40S'"):
        ma._gpu_for_run(dict(rc, modal={"gpu": "H100"}))
    with pytest.raises(RuntimeError, match="pinned to the 'short' lane"):
        ma._pinned_lane(dict(rc, modal={"max_runtime_sec": 1801}))
    # A non-pinned run is untouched.
    other = {"name": "x", "type": "wan_vace_shot", "modal": {"gpu": "H100",
                                                            "max_runtime_sec": 1800}}
    assert ma._pinned_lane(other) is None and ma._gpu_for_run(other) == "H100"


def test_modal_app_commit_pin_matches_trainer():
    src = (PKG_ROOT / "modal_app.py").read_text(encoding="utf-8")
    assert f'_HUNYUAN3D_COMMIT = "{H._HY3D_COMMIT}"' in src
    assert 'Volume.from_name(\n        "worldclaw-hunyuan-weights", create_if_missing=False)' in src
    if getattr(_load_modal_app(), "_HAS_MODAL", False):
        ma = _load_modal_app()
        assert ma._HUNYUAN3D_COMMIT == H._HY3D_COMMIT
        assert set(ma._TYPE_PINNED) <= set(ma._TYPE_LANE_FNS)


def test_dry_run_on_example_config_is_within_ceiling():
    cfg_path = PKG_ROOT / "configs" / "worldclaw_hunyuan.yaml"
    proc = subprocess.run([sys.executable, str(PKG_ROOT / "modal_app.py"), "--config",
                           str(cfg_path), "--dry-run"], capture_output=True, text=True,
                          timeout=120, env=dict(os.environ, MODALLABS_MAX_USD="25"))
    assert proc.returncode == 0, proc.stderr
    assert "gpu=L40S" in proc.stdout
    assert "Total WORST-CASE cost (every run hits its max_runtime_sec timeout): $1.00" in proc.stdout


def test_example_config_validates():
    import yaml
    cfg = yaml.safe_load((PKG_ROOT / "configs" / "worldclaw_hunyuan.yaml").read_text(encoding="utf-8"))
    (rc,) = cfg["runs"]
    assert rc["type"] == "hunyuan3d_asset" and len(rc["config"]["assets"]) == 3
    H.Hunyuan3DAssetTrainer.from_config(rc["config"])
    c = rc["config"]
    assert c["startup_timeout_sec"] + len(c["assets"]) * c["asset_timeout_sec"] <= 1800 - 120


def test_stage_script_prints_plan_without_modal():
    proc = subprocess.run([sys.executable, str(PKG_ROOT / "scripts" / "stage_hunyuan_weights.py")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "modal run scripts/stage_hunyuan_weights.py --confirm" in proc.stdout
    assert H._WEIGHTS_REVISION in proc.stdout
    assert "STAGING COMPLETE" not in proc.stdout
