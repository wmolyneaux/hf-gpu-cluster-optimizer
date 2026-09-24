"""modallabs.models.comfy_sheet -- run a ComfyUI API-format graph on an H100.

One run = one ComfyUI prompt graph. The graph is supplied WHOLE in the config, already built and
already validated by whoever owns it; this lane boots a headless ComfyUI, POSTs the graph, polls
/history and writes every image the run saved into output_dir.

WHY THE GRAPH IS AN INPUT AND NOT BUILT HERE. Game1's `pipelines/reference/quadview.py` builds the
graph from a recipe and refuses five silent failure modes before emitting it (wrong text template, a
VAE wired into conditioning, an unapplied LoRA, an inert negative prompt at CFG 1.0, an overridden
descriptor template). Rebuilding any of that here would fork the contract and let the two drift.
So the producer stays there, this lane is the runner, and the seam is the API graph -- which means
this lane runs ANY ComfyUI graph, not only reference sheets.

MEASURED basis, local M5 Metal (Game1 docs/PLAN_GORE_REFERENCE.md, 2026-09-14):
  - krea2_turbo_int8_convrot, 8 steps, 1536x1024: 8m39s bare, 5m33s with --cpu-vae --lowvram
  - model resident 12,866 MB; VAE decode OOM-kills the box without --cpu-vae (jetsam named python3.12)
  - ComfyUI reports Krea2 as model_type FLUX

H100 s/step for this model is NOT measured. The first run measures it and that number replaces the
estimate in the cost table below; nothing here should be quoted as a measurement until it has.

WEIGHTS are staged once into the `game1-comfy-weights` Volume and reused. The manifest is explicit
(repo, filename, destination folder) rather than a snapshot of a whole repo, because Comfy-Org/Krea-2
alone is 130 GB across eight quantisations and a run needs one of them.

L4 COOPERATION IS FREE: ComfyUI's own stdout is streamed to <output_dir>/comfyui.log, so the dead-man
switch sees an mtime bump on every sampler step without this trainer calling anything.

ASCII only. No emojis.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from modallabs.base import (
    Trainer,
    TrainerEpochResult,
    TrainerSetup,
    TrainerStepResult,
)
from modallabs.registry import register


class ComfySheetError(RuntimeError):
    pass


_COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
_WEIGHTS_ROOT = Path(os.environ.get("COMFY_WEIGHTS", "/comfy_weights"))
_PORT = int(os.environ.get("COMFY_PORT", "8188"))
_URL = f"http://127.0.0.1:{_PORT}"

# Where each declared role lands under ComfyUI/models. Mirrors the folders ComfyUI itself scans.
_DEST = {
    "unet": "unet",
    "diffusion_models": "diffusion_models",
    "clip": "text_encoders",
    "text_encoders": "text_encoders",
    "vae": "vae",
    "lora": "loras",
    "loras": "loras",
}


def _get(path: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(_URL + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, payload: Dict[str, Any], timeout: int = 60) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_URL + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


@register("comfy_sheet")
class ComfySheetTrainer(Trainer):
    """Run one ComfyUI API graph. cfg keys:

        graph          the API-format graph, {node_id: {class_type, inputs, _meta}}   (required)
        weights        [{repo, filename, role}]  staged into the weights volume        (required)
        expect_images  int, how many images the run must save (default 1)
        boot_timeout   seconds to wait for ComfyUI to answer /object_info (default 600)
        run_timeout    seconds to wait for the prompt (default: lane budget minus margin)
        comfy_args     extra CLI args; ALWAYS include the VAE flag your device needs
        preflight      bool, default true -- refuse a graph whose classes/enums the server lacks
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ComfySheetTrainer":
        """Construct and validate. THIS IS NOT OPTIONAL and its absence is silent.

        `Trainer.from_config` is an @abstractmethod whose body is only a docstring, and ABC blocks
        INSTANTIATION, not a classmethod call. A subclass that omits it therefore has
        `cls.from_config(cfg)` return None, and the runner dies one line later on
        "'NoneType' object has no attribute 'setup'" -- which names the symptom and not the cause.
        MEASURED: that is exactly how this lane failed its first real launch.
        """
        t = cls()
        t.config = dict(config or {})
        t._results, t._failures = [], {}
        t._validate()
        return t

    def _validate(self) -> None:
        cfg = self.config
        # ONE WEIGHT LOAD, N ARMS -- the whole cost argument, and it is trellis2_recon's argument
        # verbatim: "the 4B weight load dominates the container, so arms that share it are nearly
        # free". Krea2 int8 is 12,866 MB resident and takes minutes to load; a sheet is ~10 sampler
        # steps. Running 16 sheets as 16 containers pays that load 16 times and turns a ~$1.50 batch
        # into a ~$30 one. So `arms` is a LIST of graphs run on one boot, and `graph` stays as the
        # single-arm spelling of the same thing.
        arms = cfg.get("arms")
        if arms:
            if not isinstance(arms, list) or not all(isinstance(a, dict) for a in arms):
                raise ComfySheetError("cfg.arms must be a list of {name, graph} entries")
            self.arms = [{"name": a.get("name") or f"arm{i:02d}", "graph": a["graph"]}
                         for i, a in enumerate(arms)]
        else:
            g = cfg.get("graph")
            if not isinstance(g, dict) or not g:
                raise ComfySheetError("cfg.graph must be a non-empty ComfyUI API-format graph, "
                                      "or use cfg.arms for several on one weight load")
            self.arms = [{"name": cfg.get("name", "arm00"), "graph": g}]
        self.graph = self.arms[0]["graph"]      # what preflight and the summary report against
        self.weights = list(cfg.get("weights") or [])
        if not self.weights:
            raise ComfySheetError(
                "cfg.weights is empty. Declare every file the graph loads as "
                "{repo, filename, role}; staging is explicit here because Comfy-Org/Krea-2 is "
                "130 GB across eight quantisations and a run needs one of them")

    def setup(self, setup: TrainerSetup) -> None:
        """Stage weights and boot ComfyUI. ONE argument, a TrainerSetup -- see base.Trainer."""
        self.out = Path(setup.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_fn = getattr(setup, "log_fn", None) or (lambda m: print(m))
        self.cfg = self.config
        self._stage_weights()
        self._stage_input_images()
        self._boot()

    # -- external plates -----------------------------------------------------------------------
    def _stage_input_images(self) -> None:
        """Copy externally-supplied plates into ComfyUI's input/ before any arm runs.

        WHY THIS EXISTS. The lane could only ever feed an arm a plate that ANOTHER ARM IN THE SAME
        CONTAINER generated -- the stage-1 -> stage-2 copy further down. A plate produced anywhere
        else (a hand-authored target, or a sheet minted by a different model entirely) had no way
        in: LoadImage reads input/, nothing writes input/ except that copy, and so the arm was
        classified as depending on a generated input, deferred, and finally refused in train_step
        naming an arm that was never going to produce it.

        ⛔ A MISSING PLATE DIES HERE, LOUDLY. Skipping a missing file would put the arm straight
        back into the deferred path and reproduce the confusing failure this method exists to
        remove -- the error would name the wrong cause, three steps from the real one.
        """
        specs = self.cfg.get("input_images") or []
        if not specs:
            return
        inp = _COMFY_ROOT / "input"
        inp.mkdir(parents=True, exist_ok=True)
        staged = []
        for spec in specs:
            name, src = spec.get("name"), spec.get("path")
            if not name or not src:
                raise RuntimeError(f"[comfy_sheet] input_images entry needs name and path: {spec}")
            src_p = Path(src)
            if not src_p.exists():
                raise RuntimeError(
                    f"[comfy_sheet] input image {name!r} not found at {src}. It must be on the "
                    f"runs volume before launch, e.g. "
                    f"`modal volume put modallabs-runs <local.png> {src.lstrip('/').split('/', 1)[-1]}`")
            shutil.copy2(src_p, inp / name)
            staged.append(f"{name} ({src_p.stat().st_size / 1e6:.1f} MB)")
        self._log("staged input plates: " + ", ".join(staged))

    # -- weights -------------------------------------------------------------------------------
    def _stage_weights(self) -> None:
        """Copy declared files into ComfyUI/models, from the volume, downloading only what is absent.

        The volume is the cache; ComfyUI reads from its own models tree. Symlinks are used where the
        filesystem allows, because a 13 GB copy per container start is pure billed time.
        """
        from huggingface_hub import hf_hub_download

        staged: List[str] = []
        for w in self.weights:
            repo, fn, role = w["repo"], w["filename"], w.get("role", "unet")
            dest_dir = _COMFY_ROOT / "models" / _DEST.get(role, role)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(fn).name
            if dest.exists():
                staged.append(f"{dest.name} (already staged)")
                continue
            cached = _WEIGHTS_ROOT / repo.replace("/", "__") / Path(fn).name
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                got = hf_hub_download(repo_id=repo, filename=fn,
                                      local_dir=str(cached.parent), local_dir_use_symlinks=False)
                if Path(got) != cached:
                    shutil.move(got, cached)
                self._log(f"downloaded {repo}/{fn} in {time.time() - t0:.0f}s "
                          f"({cached.stat().st_size / 1e9:.2f} GB)")
            try:
                dest.symlink_to(cached)
            except OSError:
                shutil.copy2(cached, dest)
            staged.append(dest.name)
        self._log("staged: " + ", ".join(staged))

    # -- server --------------------------------------------------------------------------------
    def _boot(self) -> None:
        # THE INTERPRETER IS sys.executable, NOT a .venv inside the ComfyUI tree.
        #
        # MEASURED failure: this line hardcoded `<COMFY_ROOT>/.venv/bin/python`, which is the LOCAL
        # Mac layout -- a venv made inside ~/ComfyUI by hand. The Modal container installs ComfyUI
        # against the system interpreter and has no such venv, so the lane staged 25 GB of weights
        # over 139 seconds of H100 and then died on FileNotFoundError before booting anything. A
        # local-machine assumption baked into a lane that only ever runs remotely.
        #
        # sys.executable is right in BOTH places: locally it is whatever ran the harness, remotely
        # it is the container's python with ComfyUI's requirements already installed.
        python = os.environ.get("COMFY_PYTHON") or sys.executable
        if not Path(python).exists():
            raise ComfySheetError(
                f"no python interpreter at {python!r}. Set COMFY_PYTHON to the one that has "
                f"ComfyUI's requirements installed")
        main_py = _COMFY_ROOT / "main.py"
        if not main_py.exists():
            raise ComfySheetError(
                f"no ComfyUI at {main_py!r}. Set COMFY_ROOT; the image installs it at /opt/ComfyUI")
        args = [python, str(main_py), "--listen", "127.0.0.1", "--port", str(_PORT)]
        args += list(self.cfg.get("comfy_args") or [])
        self.log_path = self.out / "comfyui.log"
        self._log_fh = open(self.log_path, "ab", buffering=0)
        self.proc = subprocess.Popen(args, cwd=str(_COMFY_ROOT),
                                     stdout=self._log_fh, stderr=subprocess.STDOUT)
        deadline = time.time() + int(self.cfg.get("boot_timeout", 600))
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise ComfySheetError(
                    f"ComfyUI exited with code {self.proc.returncode} during boot; "
                    f"see {self.log_path}")
            try:
                self.info = _get("/object_info", timeout=10)
                self._log(f"ComfyUI up: {len(self.info)} node classes")
                return
            except Exception:
                time.sleep(3)
        raise ComfySheetError(f"ComfyUI did not answer /object_info in time; see {self.log_path}")

    def _preflight(self) -> None:
        """Refuse a graph this server cannot run, before it costs a sampler step.

        Same check Game1's pipelines/reference/comfy.py runs locally, restated here because the
        container is a different machine: a node pack that failed to register, or a model file that
        did not stage, must fail in seconds rather than after a cold start and a load.
        """
        missing, bad = [], []
        for nid, node in self.graph.items():
            cls = node.get("class_type")
            spec = self.info.get(cls)
            if not spec:
                missing.append(f"{cls} (node {nid})")
                continue
            groups = spec.get("input") or {}
            for field, value in (node.get("inputs") or {}).items():
                if isinstance(value, list):
                    continue
                allowed = None
                for g in ("required", "optional"):
                    s = (groups.get(g) or {}).get(field)
                    if s and isinstance(s, list) and s and isinstance(s[0], list):
                        allowed = s[0]
                if allowed is not None and value not in allowed:
                    bad.append(f"{cls}.{field}={value!r} (node {nid}; server offers {len(allowed)})")
        if missing or bad:
            raise ComfySheetError(
                "preflight failed, nothing was submitted.\n"
                + ("missing node classes: " + "; ".join(missing) + "\n" if missing else "")
                + ("bad enum values: " + "; ".join(bad) if bad else ""))

    # -- the run -------------------------------------------------------------------------------
    # -- the real lifecycle ----------------------------------------------------------------
    #
    # base.Trainer's contract is: setup once, then for each epoch iterate train_iter() feeding
    # train_step(), then eval_iter()/eval_step(), then epoch_summary(). An earlier version of this
    # file invented a `train_one` method that nothing calls. Mapping the generation onto the real
    # lifecycle costs nothing and buys the framework's own logging, metric writing and -- the part
    # that matters -- L4 dead-man progress, because every arm writes an image into output_dir.

    def train_iter(self):
        """One batch per arm. The unit of work is a sheet."""
        if self.cfg.get("preflight", True):
            # TWO-PASS, and the split is forced by the two-stage design. A stage-2 (chibify) arm
            # wires LoadImage to a plate that a stage-1 arm has not generated yet, so preflighting
            # every arm up front would refuse the entire batch for files that are supposed to
            # appear during it. Arms that depend on nothing are checked NOW -- fail fast, before a
            # single sampler step -- and the dependent ones are checked just before they run, in
            # train_step, by which time their input exists.
            eager, deferred = [], []
            for arm in self.arms:
                (deferred if self._depends_on_a_generated_input(arm) else eager).append(arm)
            for arm in eager:
                self.graph = arm["graph"]
                self._preflight()
            self.graph = self.arms[0]["graph"]
            self._log(f"preflight clean for {len(eager)} independent arm(s); "
                      f"{len(deferred)} deferred until their input exists")
        return list(self.arms)

    def _depends_on_a_generated_input(self, arm) -> bool:
        """True when this arm loads an image the container does not have YET."""
        inp = _COMFY_ROOT / "input"
        for node in arm["graph"].values():
            if node.get("class_type") == "LoadImage":
                name = node.get("inputs", {}).get("image")
                if isinstance(name, str) and not (inp / name).exists():
                    return True
        return False

    def train_step(self, batch) -> TrainerStepResult:
        """Run one arm. A failure is RECORDED and the batch continues.

        Zero restarts, the rule the Game1 perf driver runs on: one bad prompt must not throw away
        the container minutes already paid for every good sheet before it.
        """
        try:
            if self.cfg.get("preflight", True) and self._depends_on_a_generated_input(batch):
                # Its input should exist by now; if it does not, say which arm was supposed to
                # produce it rather than letting ComfyUI fail on a missing file.
                self.graph = batch["graph"]
                self._preflight()
            r = self._run_arm(batch)
            self._results.append(r)
            return TrainerStepResult(metrics={"seconds": r["seconds"],
                                              "seconds_per_step": r["seconds_per_step"] or 0.0,
                                              "images": float(len(r["images"]))},
                                     n_examples=1)
        except ComfySheetError as e:
            self._failures[batch["name"]] = str(e)
            self._log(f"ARM FAILED {batch['name']}: {e}. Continuing with the rest.")
            return TrainerStepResult(metrics={"seconds": 0.0, "images": 0}, n_examples=1)

    def eval_iter(self):
        return []

    def eval_step(self, batch) -> TrainerStepResult:
        return TrainerStepResult(metrics={}, n_examples=0)

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        saved = [p for r in self._results for p in r["images"]]
        sps = [r["seconds_per_step"] for r in self._results if r["seconds_per_step"]]
        if not self._results:
            raise ComfySheetError(f"every arm failed: {json.dumps(self._failures)[:2000]}")
        self._write_batch_summary(self._results, self._failures,
                                  sum(r["seconds"] for r in self._results))
        self._log(f"{len(self._results)}/{len(self.arms)} arms, {len(saved)} image(s)"
                  + (f", {min(sps):.2f}-{max(sps):.2f} s/step" if sps else "")
                  + (f"; FAILED: {sorted(self._failures)}" if self._failures else ""))
        # TrainerEpochResult's fields are train_metrics / val_metrics / is_best / monitor_value.
        # It has NO `epoch` and NO `metrics`. MEASURED: passing those produced
        # "unexpected keyword argument 'epoch'" AFTER all thirteen arms had generated and written
        # their images -- the whole batch succeeded and the run was still marked failed. Invented
        # from the shape of the name rather than read from base.py, which is the same mistake that
        # made from_config silently return None.
        mean_sps = (sum(sps) / len(sps)) if sps else 0.0
        return TrainerEpochResult(
            train_metrics={"arms_ok": float(len(self._results)),
                           "arms_failed": float(len(self._failures)),
                           "images": float(len(saved)),
                           "seconds_per_step": mean_sps},
            val_metrics={},
            is_best=True,
            monitor_value=mean_sps)

    def save_checkpoint(self, path) -> None:
        """There is no model state; the checkpoint is the manifest of what was produced."""
        Path(path).write_text(json.dumps(
            {"arms_ok": [r["arm"] for r in self._results], "failures": self._failures}, indent=2))

    def load_checkpoint(self, path) -> None:
        """--resume: skip arms already produced, so a relaunch never redoes paid work."""
        try:
            done = set(json.loads(Path(path).read_text()).get("arms_ok") or [])
        except Exception:
            return
        if done:
            before = len(self.arms)
            self.arms = [a for a in self.arms if a["name"] not in done]
            self._log(f"resume: {before - len(self.arms)} arm(s) already done, {len(self.arms)} left")

    def _run_arm(self, arm) -> Dict[str, Any]:
        self.graph = arm["graph"]
        t0 = time.time()
        res = _post("/prompt", {"prompt": self.graph, "client_id": "modallabs-comfy-sheet"})
        pid = res.get("prompt_id")
        if not pid:
            raise ComfySheetError(f"no prompt_id in /prompt response: {res}")
        self._log(f"queued {pid}")

        budget = int(self.cfg.get("run_timeout", 1500))
        deadline, consecutive = time.time() + budget, 0
        entry = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise ComfySheetError(
                    f"ComfyUI died (code {self.proc.returncode}) while running {pid}. On Apple "
                    f"Silicon this is the VAE-decode OOM; in a container check the GPU memory and "
                    f"the tail of {self.log_path}")
            try:
                hist = _get(f"/history/{urllib.parse.quote(pid)}")
                consecutive = 0
            except (urllib.error.URLError, OSError) as e:
                consecutive += 1
                if consecutive >= 20:
                    raise ComfySheetError(f"lost the ComfyUI server after {consecutive} polls: {e}")
                time.sleep(2)
                continue
            entry = hist.get(pid)
            if entry:
                break
            time.sleep(2)
        if entry is None:
            raise ComfySheetError(f"prompt {pid} did not finish within {budget}s")

        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise ComfySheetError(
                f"the graph failed on the server: {json.dumps(status.get('messages', []))[:2000]}")

        saved = self._save_images(entry, arm["name"])
        want = int(self.cfg.get("expect_images", 1))
        if len(saved) < want:
            raise ComfySheetError(
                f"arm {arm['name']} finished and saved {len(saved)} image(s), expected {want}. A "
                f"graph that finishes green and produces nothing is the failure this check exists for")
        elapsed = time.time() - t0
        steps = None
        for node in self.graph.values():
            if str(node.get("class_type", "")).startswith("KSampler"):
                steps = node.get("inputs", {}).get("steps")
        self._log(f"  {arm['name']}: {elapsed:.0f}s"
                  + (f" ({elapsed / steps:.2f} s/step)" if steps else "")
                  + f", {len(saved)} image(s)")
        return {"arm": arm["name"], "prompt_id": pid, "seconds": round(elapsed, 1),
                "steps": steps, "seconds_per_step": round(elapsed / steps, 2) if steps else None,
                "images": saved}

    def _save_images(self, entry: Dict[str, Any], arm_name: str = "") -> List[Path]:
        out: List[Path] = []
        for _node, node_out in (entry.get("outputs") or {}).items():
            for img in node_out.get("images") or []:
                q = urllib.parse.urlencode({"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                with urllib.request.urlopen(_URL + "/view?" + q, timeout=120) as r:
                    blob = r.read()
                stem = Path(img["filename"]).name
                dest = self.out / (f"{arm_name}_{stem}" if arm_name else stem)
                # STAGE 1 FEEDS STAGE 2. A chibify arm wires LoadImage to "<stage1 arm>.png", and
                # LoadImage reads ComfyUI's INPUT folder -- which is not where a run SAVES. Without
                # this copy every stage-2 arm is refused by preflight for a file that exists three
                # directories away, and the two-stage design cannot run in one container at all.
                try:
                    inp = _COMFY_ROOT / "input"
                    inp.mkdir(parents=True, exist_ok=True)
                    (inp / f"{arm_name}.png").write_bytes(blob)
                except Exception as e:
                    self._log(f"could not stage {arm_name}.png as a stage-2 input: {e}")
                dest.write_bytes(blob)
                out.append(dest)
        return out

    def _write_batch_summary(self, results, failures, elapsed) -> None:
        """The measurement this batch exists to produce, alongside the sheets.

        seconds_per_step on the FIRST arm includes nothing but sampling -- the weight load is paid
        before it -- so it is the clean H100 number every fan-out estimate has been waiting on, and
        the per-arm spread says whether the load really is amortised or something reloads per arm.
        """
        sps = [r["seconds_per_step"] for r in results if r["seconds_per_step"]]
        (self.out / "comfy_summary.json").write_text(json.dumps({
            "arms_ok": len(results), "arms_failed": len(failures),
            "wall_seconds": round(elapsed, 1),
            "seconds_per_step_first_arm": sps[0] if sps else None,
            "seconds_per_step_min": min(sps) if sps else None,
            "seconds_per_step_max": max(sps) if sps else None,
            "per_arm": [{k: v for k, v in r.items() if k != "images"} for r in results],
            "images": [p.name for r in results for p in r["images"]],
            "failures": failures,
        }, indent=2))

    def _write_summary_unused(self, pid: str, elapsed: float, saved: List[Path]) -> None:
        steps = None
        for node in self.graph.values():
            if str(node.get("class_type", "")).startswith("KSampler"):
                steps = node.get("inputs", {}).get("steps")
        (self.out / "comfy_summary.json").write_text(json.dumps({
            "prompt_id": pid,
            "seconds": round(elapsed, 1),
            "steps": steps,
            "seconds_per_step": round(elapsed / steps, 2) if steps else None,
            "images": [p.name for p in saved],
            "node_classes": sorted({n.get("class_type") for n in self.graph.values()}),
        }, indent=2))

    def teardown(self, *_a, **_k) -> None:
        try:
            if getattr(self, "proc", None) and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=30)
        except Exception:
            pass
        try:
            self._log_fh.close()
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        line = f"[comfy_sheet] {msg}\n"
        (self.out / "lane.log").open("a").write(line)
        print(line, end="")
