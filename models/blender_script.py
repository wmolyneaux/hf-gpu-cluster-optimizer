"""modallabs.models.blender_script -- run a repo's own Blender script headless on a GPU, from a staged bundle.

One run = one frame window of one render. Written 2026-09-24 for game1CreatureMesh's dummy-attack clip, whose renders
had been running on the local Mac GPU against the standing rule ("never not use the harness", renders included:
memory modallabs-harness-policy). heroshot_take is the berkeley-usd take and nothing else; this lane is the generic
form: any script, any inputs, the same image (Blender 4.5.12 LTS, the version the Mac runs).

THE BUNDLE is a tar on the runs volume (`modal volume put modallabs-runs <local.tar> <bundle>`) holding the script and
every input it reads, at paths relative to the bundle root; the container untars it ONCE to local disk (heroshot's
measured lesson: thousands of small reads over the volume's FUSE mount cost minutes per cold container).

THE RUN: `blender -b --factory-startup --python-exit-code 1 --python <script> -- <args> [--frames A B]`, cwd = the
bundle root, Blender's stdout streamed to <output_dir>/blender.log (a write per frame keeps the L4 dead-man switch fed).

NO SILENT CLOAKING: refused at setup if the bundle, the script or Blender is missing; refused after the render if the
expected outputs are not all there (a no-op render must fail, never pass), or if `require_gpu` is set and the script's
log does not say it rendered on the GPU (a CPU fallback on a GPU container is a price paid for nothing).

config:
  bundle        path of the tar, relative to /runs            (required)
  script        path of the Blender script inside the bundle  (required)
  args          list of strings passed after `--`             (paths inside the bundle are relative to its root)
  frames        [A, B], 1-based inclusive, appended as --frames A B
  out_rel       directory (inside the bundle root) the script writes into; copied to <output_dir>/out
  expect_glob   pattern that must match `expect_count` files in out_rel (default "*.png"; count defaults to B-A+1)
  gpu_marker    a regex the log must match when require_gpu is set (default "cycles device: GPU")
  require_gpu   bool, default true
  stub          true: no Blender, no volume (tests)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from modallabs.base import Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult
from modallabs.registry import register

_BLENDER = os.environ.get("BLENDER_SCRIPT_BLENDER", "/opt/blender/blender")
_RUNS = Path(os.environ.get("BLENDER_SCRIPT_RUNS", "/runs"))
_WORK = Path("/tmp/blender_script")


class BlenderScriptError(RuntimeError):
    pass


@register("blender_script")
class BlenderScriptTrainer(Trainer):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self._setup_obj = None
        self._results = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BlenderScriptTrainer":
        return cls(config)

    def setup(self, setup: TrainerSetup) -> None:
        self._setup_obj = setup
        cfg = self.config
        if cfg.get("stub"):
            setup.log_fn("blender_script: STUB mode -- no Blender, no volume")
            return
        for k in ("bundle", "script"):
            if not cfg.get(k):
                raise BlenderScriptError(f"config.{k} is required")
        tar = _RUNS / cfg["bundle"]
        if not tar.is_file():
            raise BlenderScriptError(
                f"bundle {tar} is not on the volume. Stage it first: "
                f"modal volume put modallabs-runs <local.tar> {cfg['bundle']}")
        if _WORK.exists():
            shutil.rmtree(_WORK)
        _WORK.mkdir(parents=True)
        t0 = time.time()
        with tarfile.open(tar) as tf:
            tf.extractall(_WORK, filter="data")
        setup.log_fn(f"untarred {tar.stat().st_size >> 20} MiB to {_WORK} in {time.time() - t0:.1f}s")
        if not (_WORK / cfg["script"]).is_file():
            raise BlenderScriptError(f"script {cfg['script']} is not in the bundle")
        ver = subprocess.run([_BLENDER, "--version"], capture_output=True, text=True)
        if ver.returncode != 0:
            raise BlenderScriptError(f"blender --version failed: {ver.stderr[-400:]}")
        setup.log_fn(f"blender: {ver.stdout.splitlines()[0]}")

    def train_iter(self) -> Iterable[Any]:
        return iter([dict(self.config)])

    def eval_iter(self) -> Iterable[Any]:
        return iter(())

    def train_step(self, batch: Any) -> TrainerStepResult:
        assert self._setup_obj is not None
        cfg = dict(batch)
        out_root = Path(self._setup_obj.output_dir)
        if cfg.get("stub"):
            (out_root / "out").mkdir(parents=True, exist_ok=True)
            (out_root / "out" / "stub.txt").write_text("stub")
            m = {"frames": 0.0, "wall_sec": 0.0}
            self._results.append(m)
            return TrainerStepResult(metrics=m, n_examples=1)
        argv = [_BLENDER, "-b", "--factory-startup", "--python-exit-code", "1",
                "--python", str(_WORK / cfg["script"]), "--"] + [str(a) for a in cfg.get("args", [])]
        frames = cfg.get("frames")
        if frames:
            argv += ["--frames", str(int(frames[0])), str(int(frames[1]))]
        log_path = out_root / "blender.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._setup_obj.log_fn(" ".join(argv))
        t0 = time.time()
        with log_path.open("wb") as fh:
            rc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(_WORK)).wait()
        wall = time.time() - t0
        text = log_path.read_text(errors="replace")
        if rc != 0:
            raise BlenderScriptError(f"blender rc={rc} after {wall:.1f}s; log tail:\n{text[-3000:]}")
        if cfg.get("require_gpu", True) and not re.search(cfg.get("gpu_marker", r"cycles device: GPU"), text):
            raise BlenderScriptError(
                "the log does not show a GPU render (require_gpu). A CPU fallback on a GPU container is paid for and "
                f"slow; log tail:\n{text[-1500:]}")
        out_dir = _WORK / cfg.get("out_rel", "out")
        found = sorted(out_dir.glob(cfg.get("expect_glob", "*.png"))) if out_dir.exists() else []
        want = int(cfg.get("expect_count", (int(frames[1]) - int(frames[0]) + 1) if frames else 1))
        if len(found) != want:
            raise BlenderScriptError(f"expected {want} outputs matching {cfg.get('expect_glob', '*.png')} in "
                                     f"{cfg.get('out_rel', 'out')}, found {len(found)} (a no-op render fails here)")
        dst = out_root / "out"
        dst.mkdir(parents=True, exist_ok=True)
        for p in found:
            shutil.copy2(p, dst / p.name)
        m = {"frames": float(want), "wall_sec": round(wall, 1), "s_per_output": round(wall / max(want, 1), 2)}
        self._results.append(m)
        (out_root / "result.json").write_text(json.dumps(m, indent=1))
        self._setup_obj.log_fn(f"done: {m}")
        return TrainerStepResult(metrics=m, n_examples=1)

    def eval_step(self, batch: Any) -> TrainerStepResult:
        return TrainerStepResult(metrics={}, n_examples=0)

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        n = float(len(self._results))
        return TrainerEpochResult(train_metrics={"renders": n}, val_metrics={}, is_best=True, monitor_value=n)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "results.json").write_text(json.dumps(self._results, indent=1))

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path) / "results.json"
        if p.exists():
            self._results = json.loads(p.read_text())
