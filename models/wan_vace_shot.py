"""wan_vace_shot -- Wan 2.2 VACE-Fun A14B shot renderer via headless ComfyUI.

One Modal run = one shot batch; one epoch = one shot (SwarmCRP RENDER_CONTRACT
M-6/M-7). Ported per SwarmCRP production/wan_package/harness_patch.md sec.4.2;
the config contract mirrors production/wan_package/modal_job.WanShotTrainer.

Every ComfyUI node emitted here was schema-verified against commit
b08debceca73bd3732b731c571bd0b4710281310 (the commit the wan_image pins).

Tiers:
  draft : Lightning 4-step LoRA pair, cfg 1.0  (lora_high/lora_low set)
  finals: 30 steps, cfg 5.0, no LoRA           (loras absent)

The MoE two-expert split runs as a KSamplerAdvanced pair: the high-noise expert
takes steps [0, boundary_step), the low-noise expert [boundary_step, steps].

Standing controls (GATES.md registry, 2026-07-28):
  C-003 heartbeat   : the poll loop refreshes output_dir/heartbeat.txt every
                      tick so the L4 dead-man switch (600s no-write kill in
                      modal_app) never kills a healthy 30-step finals shot.
  C-004 provenance  : the take manifest records sha256 of every package input
                      (control, mask, reference), the full prompt/negative
                      text, model/lora filenames and sampler params -- a board
                      is falsifiable from its manifest alone.
  C-005 determinism : the ComfyUI subprocess runs with PYTHONHASHSEED,
                      CUBLAS_WORKSPACE_CONFIG, NVIDIA_TF32_OVERRIDE pinned and
                      --deterministic (torch.use_deterministic_algorithms,
                      warn_only) unless config determinism: false.
  masks (E: VACE ch.16-31): optional per-shot "control_masks" mp4 in the
                      package -> LoadVideo -> GetVideoComponents ->
                      ImageToMask(red) -> WanVaceToVideo control_masks, so the
                      mask half of the 32 VACE channels carries real signal
                      instead of constant 0.5.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modallabs.base import Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult
from modallabs.registry import register

_COMFY_DIR = Path("/comfyui")
_WAN_MODELS = Path("/wan_models")
_PORT = 8188
_URL = f"http://127.0.0.1:{_PORT}"
_INVOICE_USD_PER_SEC = 0.001097  # verified vs modal.com/pricing 2026-07-24 (MASTER_SPEC)

_REQUIRED = ("shots", "epochs", "package_root", "out_prefix",
             "steps", "cfg", "strength", "width", "height", "frames", "fps",
             "model_high", "model_low", "text_encoder", "vae")
_SHOT_KEYS = ("name", "control_video", "reference_image", "prompt", "negative_prompt", "seed")

# C-005: pinned env for the ComfyUI subprocess. PYTHONHASHSEED kills dict/set
# iteration-order drift; CUBLAS_WORKSPACE_CONFIG makes cuBLAS reductions
# deterministic (required by torch.use_deterministic_algorithms on CUDA);
# NVIDIA_TF32_OVERRIDE=0 stops TF32 from changing matmul numerics run-to-run.
_DETERMINISM_ENV = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
}


class WanShotError(RuntimeError):
    pass


def _shot_input_keys(shot: Dict[str, Any]) -> Tuple[str, ...]:
    """Package-file keys a shot actually uses. control_masks is optional."""
    keys: Tuple[str, ...] = ("control_video", "reference_image")
    if shot.get("control_masks"):
        keys += ("control_masks",)
    return keys


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@register("wan_vace_shot")
class WanVaceShotTrainer(Trainer):

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self._shots: List[Dict[str, Any]] = [dict(s) for s in config["shots"]]
        self._cursor = 0
        self._proc: Optional[subprocess.Popen] = None
        self._setup_obj: Optional[TrainerSetup] = None
        self._takes: List[Dict[str, Any]] = []
        self._epoch_metrics: List[Dict[str, float]] = []
        # C-004: package-relative path -> sha256, filled in setup().
        self._input_sha: Dict[str, str] = {}

    # ------------------------------------------------------------- 1/9
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WanVaceShotTrainer":
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise WanShotError(f"wan_vace_shot config missing keys: {missing}")
        shots = list(config["shots"])
        if int(config["epochs"]) != len(shots):
            raise WanShotError(
                f"one epoch is one shot: epochs={config['epochs']} != n_shots={len(shots)}")
        for s in shots:
            miss = [k for k in _SHOT_KEYS if k not in s]
            if miss:
                raise WanShotError(f"shot {s.get('name')!r} missing keys: {miss}")
        # draft tier needs both LoRAs or neither
        has_hi, has_lo = "lora_high" in config, "lora_low" in config
        if has_hi != has_lo:
            raise WanShotError("lora_high and lora_low must be set together")
        return cls(config)

    # ------------------------------------------------------------- 2/9
    def setup(self, setup: TrainerSetup) -> None:
        self._setup_obj = setup
        cfg = self.config

        if cfg.get("stub"):
            # smoke-test mode (PORTING.md step 3): no server, no weights, no GPU.
            setup.log_fn("wan_vace_shot: STUB mode — no ComfyUI, no weights")
            # C-004 still fires in stub mode: hash whatever package inputs
            # exist so the provenance control is testable without a GPU.
            self._hash_inputs(Path(cfg["package_root"]), strict=False)
            return

        # model files must exist BEFORE the server starts (fail at minute 0)
        names = [cfg["model_high"], cfg["model_low"], cfg["text_encoder"], cfg["vae"]]
        if "lora_high" in cfg:
            names += [cfg["lora_high"], cfg["lora_low"]]
        for n in names:
            p = _WAN_MODELS / n
            if not p.exists() or p.stat().st_size == 0:
                raise WanShotError(f"weight not staged on volume: {p}")
            setup.log_fn(f"weight ok: {n} ({p.stat().st_size} bytes)")

        # package inputs -> ComfyUI input dir
        pkg = Path(cfg["package_root"])
        if not pkg.is_dir():
            raise WanShotError(f"package_root not found: {pkg}")
        inp = _COMFY_DIR / "input"
        inp.mkdir(parents=True, exist_ok=True)
        for s in self._shots:
            for key in _shot_input_keys(s):
                src = pkg / s[key]
                if not src.exists():
                    raise WanShotError(f"package file missing: {src}")
                shutil.copy2(src, inp / src.name)
        self._hash_inputs(pkg, strict=True)  # C-004: hash before any spend

        # point ComfyUI's model search at the weights volume
        extra = _COMFY_DIR / "extra_model_paths.yaml"
        extra.write_text(
            "wan_volume:\n"
            f"  diffusion_models: {_WAN_MODELS}\n"
            f"  text_encoders: {_WAN_MODELS}\n"
            f"  vae: {_WAN_MODELS}\n"
            f"  loras: {_WAN_MODELS / 'loras'}\n",
            encoding="utf-8",
        )

        # C-005: determinism env + ComfyUI --deterministic (warn_only torch
        # deterministic algorithms). On by default; determinism: false opts out.
        determinism = bool(cfg.get("determinism", True))
        argv = ["python", "main.py", "--listen", "127.0.0.1", "--port", str(_PORT),
                "--extra-model-paths-config", str(extra)]
        env = dict(os.environ)
        if determinism:
            env.update(_DETERMINISM_ENV)
            argv.append("--deterministic")
            setup.log_fn(f"determinism armed: {sorted(_DETERMINISM_ENV)} + --deterministic")
        self._proc = subprocess.Popen(
            argv, cwd=str(_COMFY_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        t0 = time.time()
        while True:
            if self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                raise WanShotError(f"ComfyUI exited during startup:\n{out[-4000:]}")
            try:
                with urllib.request.urlopen(f"{_URL}/system_stats", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 300:
                raise WanShotError("ComfyUI server did not come up within 300s")
            self._heartbeat("startup")  # C-003
            time.sleep(2.0)
        setup.log_fn(f"ComfyUI up in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------- controls
    def _hash_inputs(self, pkg: Path, strict: bool) -> None:
        """C-004: sha256 every package input a shot names, cached by rel path.

        strict=True (real runs) raises on a missing file; strict=False (stub)
        records None so the manifest shows the hole instead of hiding it.
        """
        for s in self._shots:
            for key in _shot_input_keys(s):
                rel = s[key]
                if rel in self._input_sha:
                    continue
                src = pkg / rel
                if src.exists():
                    self._input_sha[rel] = _sha256_file(src)
                elif strict:
                    raise WanShotError(f"cannot hash missing package file: {src}")

    def _heartbeat(self, note: str) -> None:
        """C-003: refresh output_dir/heartbeat.txt so the L4 dead-man switch
        (600s no-write kill) sees forward progress on long shots. A failed
        write is logged LOUDLY: the guard is then live and may kill the run."""
        if self._setup_obj is None:
            return
        try:
            hb = self._setup_obj.output_dir / "heartbeat.txt"
            hb.parent.mkdir(parents=True, exist_ok=True)
            hb.write_text(f"{time.time():.0f} {note}\n", encoding="utf-8")
        except OSError as exc:
            self._setup_obj.log_fn(
                f"HEARTBEAT WRITE FAILED ({exc}); L4 dead-man switch is live")

    def _provenance(self) -> Dict[str, Any]:
        """C-004: everything needed to falsify a board from its manifest."""
        cfg = self.config
        steps = int(cfg["steps"])
        prov: Dict[str, Any] = {
            "stub": bool(cfg.get("stub", False)),
            "models": {k: cfg[k] for k in
                       ("model_high", "model_low", "text_encoder", "vae")},
            "loras": {k: cfg[k] for k in ("lora_high", "lora_low") if k in cfg},
            "sampler": {
                "sampler": str(cfg.get("sampler", "euler")),
                "scheduler": str(cfg.get("scheduler", "simple")),
                "steps": steps,
                "boundary_step": int(cfg.get("boundary_step", max(1, steps // 2))),
                "cfg": float(cfg["cfg"]),
                "shift": float(cfg.get("shift", 5.0)),
                "strength": float(cfg["strength"]),
                "lora_strength": (float(cfg.get("lora_strength", 1.0))
                                  if "lora_high" in cfg else None),
            },
            "use_reference": bool(cfg.get("use_reference", True)),
            "determinism": bool(cfg.get("determinism", True)),
            "shots": {},
        }
        for s in self._shots:
            prov["shots"][s["name"]] = {
                "prompt": s["prompt"],
                "negative_prompt": s["negative_prompt"],
                "seed": int(s["seed"]),
                "inputs": {key: {"file": s[key],
                                 "sha256": self._input_sha.get(s[key])}
                           for key in _shot_input_keys(s)},
            }
        return prov

    # ------------------------------------------------------------- graph
    def _graph(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        steps = int(cfg["steps"])
        boundary = int(cfg.get("boundary_step", max(1, steps // 2)))
        use_lora = "lora_high" in cfg
        seed = int(shot["seed"])
        g: Dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {
                "unet_name": cfg["model_high"], "weight_dtype": "default"}},
            "2": {"class_type": "UNETLoader", "inputs": {
                "unet_name": cfg["model_low"], "weight_dtype": "default"}},
            "5": {"class_type": "CLIPLoader", "inputs": {
                "clip_name": cfg["text_encoder"], "type": "wan", "device": "default"}},
            "6": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["vae"]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {
                "clip": ["5", 0], "text": shot["prompt"]}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {
                "clip": ["5", 0], "text": shot["negative_prompt"]}},
            "10": {"class_type": "LoadVideo", "inputs": {
                "file": Path(shot["control_video"]).name}},
            "11": {"class_type": "GetVideoComponents", "inputs": {"video": ["10", 0]}},
        }
        # reference_image is optional: feeding the blockout frame back as the
        # reference pulls the restyle toward the blockout. Omit it (config
        # use_reference: false) to give the prompt full stylistic control.
        use_ref = bool(cfg.get("use_reference", True))
        if use_ref:
            g["9"] = {"class_type": "LoadImage", "inputs": {
                "image": Path(shot["reference_image"]).name}}
        # Optional per-shot mask video: fills VACE channels 16-31 with real
        # signal (they are constant 0.5 when control_masks is absent). White
        # (red channel 1.0) = repaint, black = preserve.
        use_masks = bool(shot.get("control_masks"))
        if use_masks:
            g["21"] = {"class_type": "LoadVideo", "inputs": {
                "file": Path(shot["control_masks"]).name}}
            g["22"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["21", 0]}}
            g["23"] = {"class_type": "ImageToMask", "inputs": {
                "image": ["22", 0], "channel": "red"}}
        hi_model, lo_model = ["1", 0], ["2", 0]
        if use_lora:
            g["3"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": hi_model, "lora_name": Path(cfg["lora_high"]).name,
                "strength_model": float(cfg.get("lora_strength", 1.0))}}
            g["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": lo_model, "lora_name": Path(cfg["lora_low"]).name,
                "strength_model": float(cfg.get("lora_strength", 1.0))}}
            hi_model, lo_model = ["3", 0], ["4", 0]
        g["12"] = {"class_type": "ModelSamplingSD3", "inputs": {
            "model": hi_model, "shift": float(cfg.get("shift", 5.0))}}
        g["13"] = {"class_type": "ModelSamplingSD3", "inputs": {
            "model": lo_model, "shift": float(cfg.get("shift", 5.0))}}
        vace_inputs: Dict[str, Any] = {
            "positive": ["7", 0], "negative": ["8", 0], "vae": ["6", 0],
            "width": int(cfg["width"]), "height": int(cfg["height"]),
            "length": int(cfg["frames"]), "batch_size": 1,
            "strength": float(cfg["strength"]),
            "control_video": ["11", 0]}
        if use_ref:
            vace_inputs["reference_image"] = ["9", 0]
        if use_masks:
            vace_inputs["control_masks"] = ["23", 0]
        g["14"] = {"class_type": "WanVaceToVideo", "inputs": vace_inputs}
        g["15"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["12", 0], "add_noise": "enable", "noise_seed": seed,
            "steps": steps, "cfg": float(cfg["cfg"]),
            "sampler_name": str(cfg.get("sampler", "euler")),
            "scheduler": str(cfg.get("scheduler", "simple")),
            "positive": ["14", 0], "negative": ["14", 1],
            "latent_image": ["14", 2],
            "start_at_step": 0, "end_at_step": boundary,
            "return_with_leftover_noise": "enable"}}
        g["16"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["13", 0], "add_noise": "disable", "noise_seed": seed,
            "steps": steps, "cfg": float(cfg["cfg"]),
            "sampler_name": str(cfg.get("sampler", "euler")),
            "scheduler": str(cfg.get("scheduler", "simple")),
            "positive": ["14", 0], "negative": ["14", 1],
            "latent_image": ["15", 0],
            "start_at_step": boundary, "end_at_step": steps,
            "return_with_leftover_noise": "disable"}}
        g["17"] = {"class_type": "TrimVideoLatent", "inputs": {
            "samples": ["16", 0], "trim_amount": ["14", 3]}}
        g["18"] = {"class_type": "VAEDecode", "inputs": {
            "samples": ["17", 0], "vae": ["6", 0]}}
        g["19"] = {"class_type": "CreateVideo", "inputs": {
            "images": ["18", 0], "fps": float(cfg["fps"])}}
        g["20"] = {"class_type": "SaveVideo", "inputs": {
            "video": ["19", 0], "filename_prefix": f"wan/{shot['name']}",
            "format": "mp4", "codec": "h264"}}
        return g

    # ------------------------------------------------------------- 3-6/9
    def train_iter(self) -> Iterable[Any]:
        if self._cursor >= len(self._shots):
            raise WanShotError(f"epoch cursor {self._cursor} past {len(self._shots)} shots")
        return iter([self._shots[self._cursor]])

    def eval_iter(self) -> Iterable[Any]:
        return iter(())  # QA runs offline against Blender ground truth (R-0.2)

    def train_step(self, batch: Any) -> TrainerStepResult:
        assert self._setup_obj is not None
        shot = dict(batch)
        t0 = time.time()
        if self.config.get("stub"):
            out_dir = self._setup_obj.output_dir / "takes"
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{shot['name']}.stub"
            dst.write_bytes(b"stub")
            m = {"shot_sec": 0.0, "steps": float(self.config["steps"]),
                 "invoice_usd": 0.0, "video_bytes": 4.0}
            self._epoch_metrics.append(m)
            self._takes.append({"shot": shot["name"], "file": str(dst),
                                "bytes": 4, "sec": 0.0, "seed": shot["seed"]})
            return TrainerStepResult(metrics=m, n_examples=1)
        payload = json.dumps({"prompt": self._graph(shot)}).encode("utf-8")
        req = urllib.request.Request(
            f"{_URL}/prompt", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        if "prompt_id" not in resp:
            raise WanShotError(f"ComfyUI rejected the graph: {resp}")
        pid = resp["prompt_id"]

        deadline = time.time() + float(self.config.get("shot_timeout_sec", 2400))
        outputs: Dict[str, Any] = {}
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise WanShotError("ComfyUI server died mid-generation")
            with urllib.request.urlopen(f"{_URL}/history/{pid}", timeout=30) as r:
                hist = json.loads(r.read())
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = json.dumps(status.get("messages", []))[-3000:]
                    raise WanShotError(f"ComfyUI execution error for {shot['name']}: {msgs}")
                if entry.get("outputs"):
                    outputs = entry["outputs"]
                    break
            if time.time() > deadline:
                raise WanShotError(f"shot {shot['name']} exceeded shot_timeout_sec")
            self._heartbeat(f"render {shot['name']} pid={pid}")  # C-003
            time.sleep(3.0)

        # find the saved video
        video_rel: Optional[str] = None
        for node_out in outputs.values():
            for key in ("images", "video", "videos", "gifs"):
                for item in node_out.get(key, []) or []:
                    fn = item.get("filename", "")
                    if fn.endswith((".mp4", ".webm", ".mov")):
                        sub = item.get("subfolder", "")
                        video_rel = os.path.join(sub, fn) if sub else fn
        if video_rel is None:
            raise WanShotError(f"no video in outputs for {shot['name']}: {list(outputs)}")
        src = _COMFY_DIR / "output" / video_rel
        if not src.exists():
            raise WanShotError(f"ComfyUI reported {video_rel} but file is missing")
        out_dir = self._setup_obj.output_dir / "takes"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{shot['name']}.mp4"
        shutil.copy2(src, dst)

        dt = time.time() - t0
        m = {"shot_sec": dt, "steps": float(self.config["steps"]),
             "invoice_usd": dt * _INVOICE_USD_PER_SEC,
             "video_bytes": float(dst.stat().st_size)}
        self._epoch_metrics.append(m)
        self._takes.append({"shot": shot["name"], "file": str(dst),
                            "bytes": dst.stat().st_size, "sec": dt,
                            "seed": shot["seed"]})
        self._setup_obj.log_fn(f"take done: {shot['name']} in {dt:.1f}s -> {dst}")
        return TrainerStepResult(metrics=m, n_examples=1)

    def eval_step(self, batch: Any) -> TrainerStepResult:  # pragma: no cover
        raise WanShotError("eval_step should be unreachable: eval_iter is empty")

    # ------------------------------------------------------------- 7-9/9
    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        m = self._epoch_metrics[-1] if self._epoch_metrics else {}
        self._cursor = int(epoch) + 1
        return TrainerEpochResult(
            train_metrics=m, val_metrics={},
            is_best=False, monitor_value=m.get("shot_sec"))

    def save_checkpoint(self, path: Path) -> None:
        """Take manifest (C-6); never overwrites (C-7)."""
        p = Path(path).with_suffix(".json")
        n = 1
        while p.exists():
            p = Path(path).with_suffix(f".{n}.json")
            n += 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "trainer": "wan_vace_shot",
            "config": {k: v for k, v in self.config.items() if k != "shots"},
            "takes": self._takes,
            "provenance": self._provenance(),  # C-004
        }, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path).with_suffix(".json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self._takes = list(data.get("takes", []))

    def num_epochs(self) -> int:
        return len(self._shots)

    def teardown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except Exception:
                self._proc.kill()
        self._proc = None
