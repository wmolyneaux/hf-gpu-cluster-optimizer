"""modallabs.models.heroshot_take -- Cycles take chunks for the berkeley-usd heroshot.

One run = one FRAME WINDOW of one take. The camera path depends on the TOTAL
frame count (place_rig.py: t = f/(n-1)), so every worker is told the full
--frames and only restricts what it WRITES via --frame-start/--frame-end --
both of which are place_rig digest inputs as of manifest v2. Fan N windows out
as N config runs and the harness's .spawn() concurrency, --dry-run pricing,
lane routing and the four BILL_SAFETY termination layers all apply unchanged.

MEASURED basis (berkeley-usd/docs/FAST-TAKE-PLAN.md):
  - local M5 Metal take: 753 f x 4.463 s/f = 56.5 min (walk270_v3, complete run)
  - chunk-context / standalone renders sit INSIDE the identical-command
    run-to-run band (60/72 px at +/-1 LSB vs control 74 of 589,824); a wrong
    --frames moves 99.48% of pixels, mean |diff| 45.87 -- hence _BAND below.
  - the landed --frame-start/--frame-end script reproduced walk270_v3 f_0400
    at 66 px, +/-1 LSB (regression run, 2026-08-12).

The trainer shells Blender; it does not import bpy. Blender's own stdout is
streamed to <output_dir>/blender_<tag>.log, so the L4 dead-man switch sees
progress at every frame without trainer cooperation.

PILOT MODE (config.pilot: true): one container renders the splice trio
(chunk [398,410), standalone [400,401), NEGATIVE --frames 401) plus nothing
else, band-compares them in-container, and reports steady-state s/f on H100 --
the one number the fleet sizing table is waiting on. The negative arm MUST
fail the band or the comparator is broken; both outcomes are asserted.

ASCII only. No emojis.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modallabs.base import (
    Trainer,
    TrainerEpochResult,
    TrainerSetup,
    TrainerStepResult,
)
from modallabs.registry import register


class HeroshotTakeError(RuntimeError):
    pass


_BLENDER = os.environ.get("HEROSHOT_BLENDER", "/opt/blender/blender")
_BUSD = Path(os.environ.get("BUSD_MOUNT", "/busd"))
# MEASURED on pilot r2 (2026-08-12): reading the campus tree straight off the
# volume FUSE mount cost ~430 s of scene setup per COLD invocation (chunk arm:
# 460.4 s wall for ~26 s of rendering) vs ~40 s warm (attempt 2: 43.8 s total)
# -- thousands of small texture files at per-file FUSE latency. The fix is one
# sequential read: stage_berkeley_take.sh puts busd_take.tar on the volume,
# and each container untars it ONCE to local disk. A brief-lane worker must
# fit L3's 480 s; 430 s of avoidable I/O does not.
_BUSD_TAR_NAME = "busd_take.tar"
_LOCAL_ROOT = Path("/tmp/busd")
_REQUIRED = ("retarget", "frames")
# Acceptance band, MEASURED on the identical-command control (74 px of 589,824
# at +/-1 LSB). The trap signature is 4 orders of magnitude away (99.48% px).
_BAND_FRAC = 0.0005          # <= 0.05% of pixels may differ ...
_BAND_MAXDIFF = 1            # ... and only by one 8-bit step.
_TIME_RE = re.compile(r"^Time: (\d{2}):(\d{2}\.\d{2}) \(Sav")
# Crash-retry bound. An L3/L4 watchdog kill is os._exit, which Modal treats as
# a container CRASH and reschedules -- fleet r1 (2026-08-12) looped
# kill -> fresh container -> recompile -> kill, billing every lap, until the
# app was stopped BY HAND ($1.4 actual vs $1.02 modelled). One retry is
# legitimate (preemption, transient infra, or a retry that CACHE-HITs prior
# work -- pilot r1's retry did exactly that); a THIRD container entering the
# same run dir means the setup systematically cannot finish, and burning
# another H100 proves nothing new. Raising here is an app-level exception:
# runner.train_one catches it, the function RETURNS phase=failed, and Modal
# does not reschedule -- the loop is cut.
_MAX_ATTEMPTS = 2
# The render script, and the one file whose staleness has already cost a run:
# berkeley-usd-take was staged with a place_rig.py whose sky Mapping rotation
# was (0,0,90) instead of (-90,0,90). MEASURED consequence (place_rig's own
# comment block): the camera ray lands in the sky map's below-horizon half and
# the sky renders EXACTLY 0.0 -- min = max = 0. Every worker reads the SAME
# stale script, so all eight agree and the cross-worker equality band PASSES.
# Determinism is not correctness, and the invoice arrives either way.
_PIN_REQUIRED = "tools/heroshot/place_rig.py"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _band_compare(a_png: Path, b_png: Path) -> Dict[str, float]:
    """Pixel band comparison; prints what it selected, never a bare verdict."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(a_png).convert("RGB"), np.int16)
    b = np.asarray(Image.open(b_png).convert("RGB"), np.int16)
    if a.shape != b.shape:
        raise HeroshotTakeError(f"shape mismatch {a.shape} vs {b.shape}: "
                                f"{a_png} vs {b_png}")
    d = np.abs(a - b)
    n_px = int(d.shape[0] * d.shape[1])
    n_diff = int((d.sum(2) > 0).sum())
    return {"n_px": n_px, "n_diff": n_diff, "frac": n_diff / n_px,
            "maxdiff": int(d.max()), "meandiff": float(d.mean())}


def _band_ok(r: Dict[str, float]) -> bool:
    return r["frac"] <= _BAND_FRAC and r["maxdiff"] <= _BAND_MAXDIFF


@register("heroshot_take")
class HeroshotTakeTrainer(Trainer):

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self._setup_obj: Optional[TrainerSetup] = None
        self._results: List[Dict[str, Any]] = []
        self._root: Path = _BUSD

    def _resolve_root(self, log_fn) -> Path:
        """Untar the input bundle to local disk once per container, else fall
        back to the raw volume mount (correct but ~430 s slower when cold)."""
        tar_p = _BUSD / _BUSD_TAR_NAME
        if not tar_p.exists():
            log_fn(f"no {_BUSD_TAR_NAME} on the volume; using FUSE mount directly "
                   f"(MEASURED ~430 s cold setup penalty)")
            return _BUSD
        # The marker records WHICH tar was untarred (size + mtime identity),
        # not merely THAT one was. MEASURED failure without this (2026-08-12,
        # heroshot_w270npr_L4_r1): Modal reused 17 warm containers from the
        # just-drained 1189 fleet, every one skipped the untar of a tar that
        # had been re-staged in between, and all 17 refused on "input not
        # staged" -- cheap only because the setup asserts fire before any GPU
        # sampling. Present is not current, for markers exactly like volumes.
        st = tar_p.stat()
        tar_id = f"{st.st_size}:{st.st_mtime_ns}"
        marker = _LOCAL_ROOT / ".untarred"
        stale = (not marker.exists()
                 or marker.read_text().strip() != tar_id)
        if stale:
            t0 = time.time()
            if _LOCAL_ROOT.exists():
                import shutil as _sh
                _sh.rmtree(_LOCAL_ROOT)
            _LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xf", str(tar_p), "-C", str(_LOCAL_ROOT)],
                           check=True)
            marker.write_text(tar_id + "\n")
            log_fn(f"untarred {st.st_size >> 20} MiB of inputs to "
                   f"{_LOCAL_ROOT} in {time.time() - t0:.1f}s (tar id {tar_id})")
        return _LOCAL_ROOT

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HeroshotTakeTrainer":
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise HeroshotTakeError(f"heroshot_take config missing keys: {missing}")
        cfg = dict(config)
        n = int(cfg["frames"])
        cfg.setdefault("frame_start", 0)
        cfg.setdefault("frame_end", n)
        a, b = int(cfg["frame_start"]), int(cfg["frame_end"])
        if not (0 <= a < b <= n):
            raise HeroshotTakeError(
                f"window [{a},{b}) invalid for frames={n}: need 0 <= start < end <= frames")
        cfg.setdefault("res", "1024x576")
        cfg.setdefault("samples", 96)
        # LOOK HAS NO DEFAULT. It is a place_rig DIGEST INPUT and therefore a PIXEL
        # input: `plain` and `magical` are different renders, not different qualities
        # of the same render. This line used to read cfg.setdefault("look", "plain"),
        # and a silently-defaulted `plain` has now cost TWO renders -- once before
        # 2026-08-12, and once this run when take_restyle.sh's own place_rig call
        # omitted the flag and would have discarded a paid magical fleet take.
        # A shot config must SAY what it renders. Unset is a refusal, not a default.
        if not cfg.get("look"):
            raise HeroshotTakeError(
                "config.look is unset. It is a pixel input and a digest input, so a "
                "default here silently changes what you render and what you cache. "
                "State it explicitly: look: magical (graded take + wildflowers) or "
                "look: plain. There is deliberately no fallback.")
        cfg.setdefault("stage", "usd/shots/shotHeroGlade.usda")
        cfg.setdefault("glb", "rig/rigged.glb")
        cfg.setdefault("pilot", False)
        # Staleness pins. modal_app._heroshot_pin_run stamps these from the
        # LAUNCHER's copy of the scripts; refusing here as well costs one
        # container boot and covers anything that reaches a container by
        # another route (a hand-written config, a resumed run, a direct
        # runner.train_one call).
        cfg.setdefault("expect_sha256", {})
        cfg.setdefault("allow_unpinned", False)
        if not isinstance(cfg["expect_sha256"], dict):
            raise HeroshotTakeError(
                f"expect_sha256 must be a dict of relpath -> sha256 hex, got "
                f"{type(cfg['expect_sha256']).__name__}")
        if not cfg["expect_sha256"] and not cfg["allow_unpinned"]:
            raise HeroshotTakeError(
                "REFUSING an unpinned heroshot run: config.expect_sha256 is empty, so "
                "nothing would notice if berkeley-usd-take still carried a stale "
                f"{_PIN_REQUIRED}. Launch through modal_app (it stamps the pins from "
                "the local tree), or set config.allow_unpinned: true deliberately.")
        if int(cfg.get("epochs", 1)) != 1:
            raise HeroshotTakeError("one epoch is one window: set epochs to 1")
        return cls(cfg)

    # ------------------------------------------------------------- staleness
    def _verify_pins(self, log_fn) -> None:
        """Compare the volume's scripts against the launcher's. REFUSE, do not
        warn: a warning in a fan-out of eight scrolls past and the frames still
        get rendered and paid for."""
        pins: Dict[str, str] = dict(self.config.get("expect_sha256") or {})
        if not pins:
            log_fn("STALENESS ASSERTION DISABLED (allow_unpinned): the volume's "
                   "render scripts are NOT being checked against the launcher's.")
            return
        bad, ok = [], []
        for rel, want in sorted(pins.items()):
            p = self._root / rel
            if not p.is_file():
                bad.append(f"{rel}: MISSING on the volume (launcher has {want[:12]})")
                continue
            got = _sha256_file(p)
            (ok if got == want else bad).append(
                f"{rel}: {got[:12]}" if got == want
                else f"{rel}: volume {got[:12]} != launcher {want[:12]}")
        for line in ok:
            log_fn(f"pin ok: {line}")
        if bad:
            raise HeroshotTakeError(
                "STALE INPUT VOLUME -- refusing before any GPU sampling.\n  "
                + "\n  ".join(bad)
                + "\nThe berkeley-usd-take volume does not match this launcher's "
                  "berkeley-usd tree. Re-stage it:\n"
                  "  scripts/stage_berkeley_take.sh <path to retarget_<shot>.json>\n"
                  "and note the tar is what workers read, so it must be rebuilt too "
                  "(the script does that). Rendering anyway would produce frames that "
                  "agree with each other and disagree with the shot -- which is how a "
                  "pre-fix place_rig put a black sky past every equality check.")

    # ------------------------------------------------------------------- setup
    def setup(self, setup: TrainerSetup) -> None:
        self._setup_obj = setup
        cfg = self.config
        if cfg.get("stub"):
            setup.log_fn("heroshot_take: STUB mode -- no Blender, no volume")
            return
        # Bound the L3/L4-kill -> Modal crash-retry loop (see _MAX_ATTEMPTS).
        # The run dir persists on the runs volume across attempts, so the
        # counter survives os._exit; a fresh run_id starts the count fresh.
        attempts_p = setup.output_dir / ".heroshot_attempts"
        try:
            n_prev = int(attempts_p.read_text().strip() or "0") \
                if attempts_p.exists() else 0
        except (OSError, ValueError):
            n_prev = 0
        if n_prev >= _MAX_ATTEMPTS:
            raise HeroshotTakeError(
                f"attempt {n_prev + 1} on this run dir: {n_prev} earlier "
                "container(s) entered and never finished (watchdog kill -> "
                "Modal crash-retry). A third container would fail the same "
                "way at the same price. Fix the cause (cold kernel cache? "
                "window too big for the lane?), then relaunch under a FRESH "
                f"run_id -- or delete {attempts_p} to deliberately re-arm.")
        attempts_p.write_text(f"{n_prev + 1}\n", encoding="utf-8")
        setup.log_fn(f"attempt {n_prev + 1}/{_MAX_ATTEMPTS} for this run dir")
        self._root = self._resolve_root(setup.log_fn)
        root = self._root
        # Fail at minute 0, before any GPU sampling: every input the render
        # will read must already be staged, and the binary must run.
        needed = [
            Path(_BLENDER),
            root / "tools/heroshot/place_rig.py",
            root / "tools/render/lut_repair.py",
            root / "tools/render/material_lut.json",
            root / cfg["retarget"],
            root / cfg["stage"],
            root / cfg["glb"],
            root / "textures/_sky/campusSky_2026-09-22T0910PDT.exr",
        ]
        for p in needed:
            if not p.exists() or (p.is_file() and p.stat().st_size == 0):
                raise HeroshotTakeError(f"input not staged: {p}")
            setup.log_fn(f"input ok: {p}")
        # PRESENT is not the same as CURRENT. Everything above passes on a
        # volume staged with last week's scripts.
        self._verify_pins(setup.log_fn)
        ver = subprocess.run([_BLENDER, "--version"], capture_output=True,
                             text=True, timeout=120)
        if ver.returncode != 0:
            raise HeroshotTakeError(f"blender --version rc={ver.returncode}: "
                                    f"{(ver.stdout + ver.stderr)[-500:]}")
        setup.log_fn(f"blender: {ver.stdout.splitlines()[0]}")

    # ------------------------------------------------------------------ render
    def _run_blender(self, tag: str, outdir: Path, frames: int,
                     fstart: int, fend: int) -> Dict[str, Any]:
        assert self._setup_obj is not None
        cfg = self.config
        root = self._root
        log_path = self._setup_obj.output_dir / f"blender_{tag}.log"
        argv = [
            _BLENDER, "-b", "--factory-startup",
            "-P", str(root / "tools/heroshot/place_rig.py"), "--",
            "--retarget", str(root / cfg["retarget"]),
            "--frames", str(frames),
            "--frame-start", str(fstart), "--frame-end", str(fend),
            "--res", str(cfg["res"]), "--samples", str(cfg["samples"]),
            "--look", str(cfg["look"]),
            "--stage", str(root / cfg["stage"]),
            "--glb", str(root / cfg["glb"]),
            "--resume",
            "--outdir", str(outdir),
        ]
        # STYLE CARD passthrough. place_rig.py:138 defaults --style-card to
        # ~/golden-rig/library/characters/hero_0454/style.json, which exists on the M5
        # and NOWHERE on a worker -- so any character other than hero_0454 either
        # crashed at the json.load or, worse, would have rendered against HER card.
        # Same failure class the char_key note below describes: a pixel-affecting input
        # the config can set and the lane silently drops. Resolved against the volume
        # root like every other path here.
        if str(cfg.get("style_card") or "").strip():
            argv += ["--style-card", str(root / str(cfg["style_card"]).strip())]
        # CHARACTER KEY passthrough (2026-08-12). Added because the flag existed in
        # place_rig, was in the shot config, and was SILENTLY DROPPED here -- the lane
        # simply had no reference to it, so every render came back with her at 18.76
        # luminance and nobody could see why the setting "did nothing".
        # A pixel-affecting flag that a config can set and the lane can ignore is the
        # same failure class as a default that wins silently. If it is in the config
        # it must reach the renderer or the run must say it did not.
        for _k, _flag in (("char_key", "--char-key"),
                          ("char_key_ring", "--char-key-ring"),
                          ("char_key_offset", "--char-key-offset"),
                          ("char_key_size", "--char-key-size")):
            if cfg.get(_k) not in (None, ""):
                argv += [_flag, str(cfg[_k])]
        # BOOLEAN passthroughs. Separate loop because these are FLAGS, not values --
        # `--guide-passes` takes no argument. This was lost once already: it lived in
        # a block I deleted as a duplicate of the npr passthrough, and the config
        # carried guide_passes: true through a whole $1.20 render that wrote no
        # guides. Same failure as char_key, one hour apart. A config key that the
        # lane does not read is indistinguishable from a key that does nothing.
        for _k, _flag in (("guide_passes", "--guide-passes"),):
            if cfg.get(_k):
                argv += [_flag]

        # ASSET GATE passthrough. The post-NPR place_rig (v2.3.2) refuses the
        # FUSED rigged.glb unless given a stated structural reason:
        # MEASURED 2026-08-12, rc=1 in 1.48 s against the freshly re-staged
        # volume. Deliberately NOT defaulted to some boilerplate string here --
        # that would cloak a gate whose entire purpose is to make rendering the
        # fused asset a choice somebody typed. Put the reason in the run config
        # (allow_fused: "...") where it is reviewed and recorded, or let the
        # render refuse.
        if str(cfg.get("allow_fused") or "").strip():
            argv += ["--allow-fused", str(cfg["allow_fused"]).strip()]
        # SHOT-CRITICAL passthrough (2026-08-12, runToCampanile NPR fleet).
        # All five are place_rig DIGEST inputs; every one has a default that is
        # WRONG for the NPR shot (npr=off, npr_mix=1.0, flower_extent="" which
        # re-derives the corridor from THIS take's travel and relocates all
        # 2200 wildflowers -- RUN-SHOT.md calls it "the trap"). Appended only
        # when the config carries a non-empty value, so every existing plain
        # config builds the byte-identical argv it always did. Values pass as
        # str() of what the YAML holds; flower_extent should be QUOTED in the
        # YAML so no float re-parse can move its digits between workers.
        for key, flag in (("npr", "--npr"), ("npr_mix", "--npr-mix"),
                          ("outline", "--outline"),
                          ("flower_extent", "--flower-extent"),
                          ("camera", "--camera")):
            v = cfg.get(key)
            if v is not None and str(v).strip() != "":
                argv += [flag, str(v).strip()]
        env = dict(os.environ)
        env["BUSD_ROOT"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        self._setup_obj.log_fn(f"[{tag}] {' '.join(argv)}")
        t0 = time.time()
        # Blender's stdout goes STRAIGHT to a file under the run dir: that is
        # what keeps the L4 dead-man switch fed (a write per frame), and the
        # full log survives for the s/f parse and the device assert below.
        with log_path.open("wb") as fh:
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env)
            rc = proc.wait()
        wall = time.time() - t0
        text = log_path.read_text(errors="replace")
        if rc != 0:
            raise HeroshotTakeError(
                f"[{tag}] blender rc={rc} after {wall:.1f}s; log tail:\n{text[-3000:]}")
        m = re.search(r"device probe: (GPU-\w+)", text)
        if not m:
            # place_rig refuses CPU itself (SystemExit -> rc!=0); no probe line
            # on rc==0 means the script changed under us -- refuse loudly.
            raise HeroshotTakeError(f"[{tag}] no 'device probe: GPU-*' line in log")
        device = m.group(1)
        times = [int(a) * 60 + float(b)
                 for a, b in (mm.groups() for mm in map(_TIME_RE.match,
                                                        text.splitlines()) if mm)]
        # first Time line is the silmask render, second is the first frame of
        # the process (carries kernel warmup); steady state is the rest.
        steady = times[2:] if len(times) > 2 else []
        # verify the window's frames actually exist -- glob is the truth, not rc
        missing = [k for k in range(fstart, fend)
                   if not (outdir / ("f_%04d.png" % k)).exists()]
        if missing:
            raise HeroshotTakeError(f"[{tag}] {len(missing)} window frames missing, "
                                    f"first {missing[:5]}")
        if not (outdir / "f_0000.png").exists():
            raise HeroshotTakeError(f"[{tag}] gate frame f_0000.png missing")
        man_p = outdir / "manifest.json"
        if not man_p.exists():
            raise HeroshotTakeError(f"[{tag}] manifest.json missing")
        man = json.loads(man_p.read_text())
        di = man.get("digest_inputs", {})
        if (di.get("frame_start"), di.get("frame_end")) != (fstart, fend):
            raise HeroshotTakeError(
                f"[{tag}] manifest window {di.get('frame_start')},{di.get('frame_end')} "
                f"!= requested {fstart},{fend}")
        # The pin again, but this time against the hash the RENDER ITSELF
        # certified. _verify_pins hashed the file at setup; this is what
        # place_rig actually executed and recorded (script_sha256 is one of its
        # own digest inputs), so it closes the gap between the check and the
        # exec -- and it fails a frame that has already been rendered rather
        # than shipping it into the assembly.
        want = (self.config.get("expect_sha256") or {}).get(_PIN_REQUIRED)
        if want and di.get("script_sha256") != want:
            raise HeroshotTakeError(
                f"[{tag}] place_rig certified script_sha256="
                f"{str(di.get('script_sha256'))[:12]} but the launcher pinned "
                f"{want[:12]}. The script changed between setup and exec, or the "
                "manifest is from a different script. These frames are NOT the "
                "shot that was launched -- discarding rather than assembling them.")
        r = {
            "tag": tag, "rc": rc, "wall_sec": round(wall, 1), "device": device,
            "digest": man.get("digest", "")[:16],
            "frames_written": fend - fstart,
            "s_f_steady": round(sum(steady) / len(steady), 3) if steady else None,
            "s_f_min": round(min(steady), 3) if steady else None,
            "s_f_max": round(max(steady), 3) if steady else None,
        }
        self._setup_obj.log_fn(f"[{tag}] done: {r}")
        return r

    # ---------------------------------------------------------------- lifecycle
    def train_iter(self) -> Iterable[Any]:
        return iter([dict(self.config)])

    def eval_iter(self) -> Iterable[Any]:
        return iter(())

    def train_step(self, batch: Any) -> TrainerStepResult:
        assert self._setup_obj is not None
        cfg = dict(batch)
        out_root = self._setup_obj.output_dir
        if cfg.get("stub"):
            (out_root / "take").mkdir(parents=True, exist_ok=True)
            (out_root / "take" / "stub.txt").write_text("stub")
            m = {"frames": 0.0, "wall_sec": 0.0}
            self._results.append({"tag": "stub", **m})
            return TrainerStepResult(metrics=m, n_examples=1)

        n = int(cfg["frames"])
        if cfg.get("pilot"):
            # -- splice trio + the s_f the fleet table is waiting on ----------
            chunk = self._run_blender("chunk", out_root / "take_chunk", n, 398, 410)
            alone = self._run_blender("alone", out_root / "take_alone", n, 400, 401)
            neg = self._run_blender("negative", out_root / "take_neg", 401, 400, 401)
            band_ac = _band_compare(out_root / "take_alone/f_0400.png",
                                    out_root / "take_chunk/f_0400.png")
            band_neg = _band_compare(out_root / "take_neg/f_0400.png",
                                     out_root / "take_alone/f_0400.png")
            band_gate = _band_compare(out_root / "take_alone/f_0000.png",
                                      out_root / "take_chunk/f_0000.png")
            self._setup_obj.log_fn(f"BAND alone-vs-chunk: {band_ac}")
            self._setup_obj.log_fn(f"BAND gate f_0000 cross-invocation: {band_gate}")
            self._setup_obj.log_fn(f"BAND NEGATIVE (wrong n): {band_neg}")
            if not _band_ok(band_ac):
                raise HeroshotTakeError(f"SPLICE BAND FAILED on this device: {band_ac}")
            if not _band_ok(band_gate):
                raise HeroshotTakeError(f"gate-frame band FAILED: {band_gate}")
            if _band_ok(band_neg) or band_neg["frac"] < 0.10:
                raise HeroshotTakeError(
                    f"NEGATIVE control passed the band -- comparator broken: {band_neg}")
            m = {
                "s_f_steady": float(chunk["s_f_steady"] or 0.0),
                "band_ac_frac": band_ac["frac"], "band_ac_max": band_ac["maxdiff"],
                "band_neg_frac": band_neg["frac"],
                "wall_sec": chunk["wall_sec"] + alone["wall_sec"] + neg["wall_sec"],
            }
            self._results += [chunk, alone, neg,
                              {"tag": "band_alone_vs_chunk", **band_ac},
                              {"tag": "band_gate_f0000", **band_gate},
                              {"tag": "band_negative", **band_neg}]
            return TrainerStepResult(metrics=m, n_examples=3)

        a, b = int(cfg["frame_start"]), int(cfg["frame_end"])
        r = self._run_blender("take", out_root / "take", n, a, b)
        self._results.append(r)
        m = {"frames": float(r["frames_written"]), "wall_sec": r["wall_sec"],
             "s_f_steady": float(r["s_f_steady"] or 0.0)}
        return TrainerStepResult(metrics=m, n_examples=r["frames_written"])

    def eval_step(self, batch: Any) -> TrainerStepResult:  # pragma: no cover
        raise HeroshotTakeError("eval_step unreachable: eval_iter is empty")

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        last = self._results[-1] if self._results else {}
        wall = last.get("wall_sec", 0.0)
        return TrainerEpochResult(train_metrics={"wall_sec": float(wall or 0.0)},
                                  val_metrics={}, is_best=False,
                                  monitor_value=float(wall or 0.0))

    def save_checkpoint(self, path: Path) -> None:
        p = Path(path).with_suffix(".json")
        i = 1
        while p.exists():
            p = Path(path).with_suffix(f".{i}.json")
            i += 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "trainer": "heroshot_take",
            "config": self.config,
            "results": self._results,
        }, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path).with_suffix(".json")
        if p.exists():
            self._results = list(json.loads(p.read_text()).get("results", []))

    def num_epochs(self) -> int:
        return 1

    def teardown(self) -> None:
        pass
