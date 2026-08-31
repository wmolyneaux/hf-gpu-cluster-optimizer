"""Kimodo text-to-motion (NVIDIA nv-tlabs) as a modallabs lane.

Generates 3D skeletal motion from text + kinematic constraints, retargets it onto
the 22-joint smpl22 rig the berkeley-usd / mocap-lane pipeline already speaks, and
writes a PROVENANCE SIDECAR that binds the action string to everything downstream.

WHY A LANE AND NOT A MODAL APP: settled `modallabs-harness-policy`. Any model goes
through this harness; a lane is added, an app is not written.

================================================================================
THE SKELETON, AND WHY THE BRIDGE IS FREE
================================================================================
Kimodo-SMPLX emits `local_rot_mats [T, 22, 3, 3]` on `SMPLXSkeleton22`.
`tram-motion/retarget/smpl_to_smpl22.py` requires `(F, 24, 3, 3)`.

MEASURED 2026-08-30, not assumed, against
kimodo/skeleton/definitions.py::SMPLXSkeleton22.bone_order_names_with_parents:

    NAMES   identical to SMPL24_NAMES[0:22]   : True
    PARENTS identical to SMPL24_PARENTS[0:22] : True

It is the SAME skeleton, truncated at the wrists. The two joints Kimodo omits
(22 left_hand, 23 right_hand) are exactly the two `BONE_MAP` already DROPS as
rotation sources (`UNMAPPED_SOURCE`); only their REST POSITIONS are read, and
those come from `rest_joints`, a static (24,3) array, never from the motion.

FALSIFIED, because "it should be a no-op" is not evidence: padding 22/23 with
identity vs with RANDOM rotations produced BIT-IDENTICAL retarget output
(max abs diff 0.000e+00) while a CONTROL perturbation of index 21 (right_wrist,
which BONE_MAP does read) moved the output by 1.999. Without that control,
"no difference" is indistinguishable from a metric that reads nothing at all.

================================================================================
THE TEXT ENCODER IS THE EXPENSIVE, GATED, CACHEABLE PART
================================================================================
The motion model is 282M parameters. The text encoder is LLM2Vec over
Llama-3-8B -- that is where the ~17 GB VRAM figure comes from, not the denoiser.

They are SEPARABLE: the encoder emits `[B, max_text_len, 4096]` and the diffusion
model consumes it. For a FIXED set of action strings the embedding can be
computed once and cached, after which generation needs no encoder at all. That
composes with the provenance contract below -- a pinned action string carries a
pinned embedding beside it, under the same digest.

`TEXT_ENCODER_DEVICE=cpu` drops the encoder off the GPU (<3 GB) at some speed
cost. `TEXT_ENCODERS_DIR` repoints base/peft at local paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.registry import register


#: Every HF repo this lane resolves, with a file that PROVES access.
#: Verified 2026-08-30 that each probe file exists in its repo.
#:   nvidia/Kimodo-SMPLX-RP-v1 .................. gated=auto   (accept terms)
#:   McGill-NLP/...-mntp ........................ ungated
#:   McGill-NLP/...-mntp-supervised ............. ungated (peft adapter)
#:   meta-llama/Meta-Llama-3-8B-Instruct ........ gated=MANUAL (Meta reviews)
#: The Llama repo is the one that bites: the McGill LLM2Vec repos are open but
#: load their BASE weights from it, so an unapproved account fails deep inside
#: text encoding rather than at load.
_REQUIRED_REPOS: Tuple[Tuple[str, str], ...] = (
    ("nvidia/Kimodo-SMPLX-RP-v1", "config.yaml"),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "config.json"),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "adapter_config.json"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "config.json"),
)


class KimodoError(RuntimeError):
    """Refusal from this lane. Always names what was expected and what was found."""


def preflight_hf_repos(token: Optional[str] = None) -> Dict[str, str]:
    """Resolve every required HF repo BEFORE any weight load. Raises on the first
    one that is unreachable.

    This lane pulls FOUR repos across three owners, two of them gated. Discovering
    that on a hot GPU is the expensive way to learn it -- see PREFLIGHT_HF_GATES.md,
    where the DINOv3 gate cost ~$0.28 and 185.4 s of H100 to report a 401.

    A FILE FETCH, NEVER model_info(). `HfApi.model_info` returns PUBLIC metadata
    for a gated repo and succeeds with NO TOKEN AT ALL. Metadata is the field
    adjacent to the signal; the signal is whether a file actually comes back.

    whoami() is checked FIRST and separately, because an invalid token and an
    unaccepted gate produce the SAME 401, and telling them apart is the difference
    between "mint a new token" and "accept the licence".
    """
    from huggingface_hub import HfApi, hf_hub_download

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out: Dict[str, str] = {}
    if not tok:
        raise KimodoError(
            "preflight: no HF_TOKEN in the environment. The lane must mount the "
            "`huggingface-secret` Secret (credentials.py). Repos needed: "
            + ", ".join(r for r, _ in _REQUIRED_REPOS))
    try:
        out["whoami"] = HfApi(token=tok).whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        raise KimodoError(
            f"preflight: the HF token is INVALID ({type(exc).__name__}: {exc}). "
            "This is NOT a gate problem -- a dead token and an unaccepted licence "
            "both 401 on the repo, and only whoami() separates them. Mint a read "
            "token and run: modal secret create huggingface-secret "
            "HF_TOKEN=hf_... --force") from exc

    for repo, probe_file in _REQUIRED_REPOS:
        try:
            hf_hub_download(repo, probe_file, token=tok)
            out[repo] = "OK"
        except Exception as exc:  # noqa: BLE001
            hint = ""
            if repo.startswith("meta-llama/"):
                hint = (" This repo is gated=MANUAL: Meta REVIEWS the request, it "
                        "is not granted on click-through. Request access and wait "
                        "for approval before relaunching.")
            raise KimodoError(
                f"preflight: {repo} is not accessible to account {out['whoami']!r} "
                f"({type(exc).__name__}: {str(exc)[:200]}). Accept the terms at "
                f"https://huggingface.co/{repo} while signed in as that account."
                f"{hint} Refusing before any weight load rather than discovering "
                f"it on a hot GPU.") from exc
    return out


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pad_22_to_24(rot22):
    """Kimodo (T,22,3,3) -> smpl_to_smpl22's required (T,24,3,3).

    Indices 22/23 are filled with IDENTITY. Proven a no-op by falsification
    (see the module docstring): randomising them leaves the retarget output
    bit-identical, while perturbing index 21 moves it. Do not "simplify" this
    into a reshape -- the shape guard in `source_global` is what catches a
    skeleton that is not actually SMPL-ordered.
    """
    import numpy as np
    r = np.asarray(rot22, float)
    if r.ndim != 4 or r.shape[1:] != (22, 3, 3):
        raise KimodoError(
            f"expected Kimodo local_rot_mats (T,22,3,3), got {r.shape}. If this is "
            f"a SOMA model its skeleton is SOMASkeleton30 and this bridge does NOT "
            f"apply -- SOMA needs its own 30->22 map.")
    out = np.zeros((r.shape[0], 24, 3, 3), dtype=float)
    out[:, :22] = r
    out[:, 22] = np.eye(3)
    out[:, 23] = np.eye(3)
    return out


@register("kimodo_motion")
class KimodoMotionTrainer(Trainer):
    """Text -> motion -> smpl22 retarget, with the action string bound to the record."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.clips: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self._out = Path(".")
        self._log = print

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KimodoMotionTrainer":
        return cls(config)

    # ---- setup ------------------------------------------------------------

    def setup(self, setup: TrainerSetup) -> None:
        # DIRECT ATTRIBUTE ACCESS, NOT getattr-with-default. The first version used
        # getattr(setup, "out_dir", ".") -- the field is actually `output_dir`, so
        # the typo fell through to the container CWD and every NPZ was written to
        # ephemeral disk and lost when the container died. The run still reported
        # SUCCEEDED. A defaulted lookup turns a contract mismatch into a silent
        # wrong answer; an attribute error is the loud failure we want.
        self._out = Path(setup.output_dir)
        self.device = setup.device
        self.seed = int(setup.seed)
        self._log = setup.log_fn

        # 1. GATES FIRST. Cheapest failure class, most expensive to find late.
        self.results["hf_preflight"] = preflight_hf_repos()

        # 2. Declare the clips. Each carries its OWN action string; that string is
        #    the thing the whole provenance contract hangs off.
        clips = self.cfg.get("clips") or []
        if not clips:
            raise KimodoError("cfg.clips is empty -- nothing to generate.")
        for c in clips:
            if not c.get("action"):
                raise KimodoError(f"clip {c.get('name')!r} has no `action` string.")
        self.clips = [dict(c) for c in clips]

        # 3. THE BINDING CHECK. The action string that drives Kimodo MUST be the
        #    same one the style card composes into the routed action span. If they
        #    diverge, the take carries motion A while the prompt paints motion B --
        #    and because the hero is RELEASED from the guide (~0.30) and the action
        #    span is routed to hero_cov at elaborate_in 1.0, THE WORDS WIN. The
        #    render would look like a stylizer defect rather than a prompt desync.
        card = self.cfg.get("style_card_action")
        if card is not None:
            for c in self.clips:
                if c["action"].strip() != str(card).strip():
                    raise KimodoError(
                        "ACTION DESYNC. cfg.style_card_action and clip "
                        f"{c['name']!r}.action are not the same string.\n"
                        f"  card:  {str(card)[:120]!r}\n"
                        f"  clip:  {c['action'][:120]!r}\n"
                        "One action string, authored once, used twice. Refusing "
                        "rather than generating motion the prompt will overrule.")

    # ---- the work ---------------------------------------------------------

    def train_iter(self) -> Iterable[Any]:
        return list(self.clips)

    def train_step(self, batch: Any) -> TrainerStepResult:
        """Generate one clip, retarget it, and write its provenance sidecar."""
        import numpy as np

        clip = batch
        name = clip["name"]
        outdir = self._out / name
        outdir.mkdir(parents=True, exist_ok=True)
        npz = outdir / f"{name}.npz"

        model = self.cfg.get("model", "kimodo-smplx-rp")
        duration = float(clip.get("duration_s", 3.5))
        seed = int(clip.get("seed", self.seed))

        cmd = [
            os.environ.get("KIMODO_PY", "python"), "-m", "kimodo.scripts.generate",
            clip["action"],
            "--model", model,
            "--duration", str(duration),
            "--seed", str(seed),
            "--output", str(npz),
        ]
        if clip.get("constraints"):
            cmd += ["--constraints", str(clip["constraints"])]
        if clip.get("diffusion_steps"):
            cmd += ["--diffusion_steps", str(int(clip["diffusion_steps"]))]
        # ESCAPE HATCH, not a default. The postprocess path needs the in-repo
        # MotionCorrection package and is what cleans FOOT SKATING -- a stated
        # Kimodo limitation. Skipping it produces motion that will slide.
        if self.cfg.get("no_postprocess"):
            cmd += ["--no-postprocess"]

        env = dict(os.environ)
        env.setdefault("TEXT_ENCODER_DEVICE", self.cfg.get("text_encoder_device", "cpu"))
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0 or not npz.exists():
            raise KimodoError(
                f"kimodo generate failed for clip {name!r} (rc={proc.returncode}).\n"
                f"stderr tail: {proc.stderr[-800:]}")

        d = np.load(npz)
        rot22 = d["local_rot_mats"]
        frames = int(rot22.shape[0])

        # THE SIDECAR. Everything needed to prove what produced this motion, and
        # to prove the action string is the same one the stylizer will be told.
        sidecar = {
            "clip": name,
            "action": clip["action"],
            "action_md5": _md5(clip["action"]),
            "style_card_md5": self.cfg.get("style_card_md5"),
            "style_card_action_md5": (_md5(str(self.cfg["style_card_action"]))
                                      if self.cfg.get("style_card_action") is not None
                                      else None),
            "model": model,
            "model_revision": self.cfg.get("model_revision"),
            "skeleton": "SMPLXSkeleton22",
            "skeleton_equals_smpl24_prefix": True,   # measured, see module docstring
            "fps": 30,
            "frames": frames,
            "duration_s": duration,
            "seed": seed,
            "constraints": clip.get("constraints"),
            "diffusion_steps": clip.get("diffusion_steps"),
            "text_encoder": {
                "base": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                "peft": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                "device": env.get("TEXT_ENCODER_DEVICE"),
            },
            "npz_sha256": _sha256_file(npz),
            "keys": sorted(list(d.keys())),
            "hf_preflight": self.results.get("hf_preflight"),
        }
        (outdir / f"{name}.provenance.json").write_text(
            json.dumps(sidecar, indent=1), encoding="utf-8")

        # Frames guard: Kimodo maxes at 300 (10 s @ 30 fps). A silently short clip
        # would be discovered only at composite time.
        want = int(round(duration * 30))
        if abs(frames - want) > 2:
            self._log(f"  WARN {name}: asked {want} frames, got {frames}")

        self.results[name] = {"frames": frames, "npz": str(npz),
                              "action_md5": sidecar["action_md5"]}
        return TrainerStepResult(metrics={"frames": float(frames)}, n_examples=1)

    # ---- framework plumbing ----------------------------------------------

    def eval_iter(self) -> Iterable[Any]:
        return []

    def eval_step(self, batch: Any) -> TrainerStepResult:
        return TrainerStepResult(metrics={}, n_examples=0)

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        n = float(len(self.clips))
        return TrainerEpochResult(
            train_metrics={"clips": n}, val_metrics={}, is_best=True,
            monitor_value=n)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "results.json").write_text(
            json.dumps(self.results, indent=1), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path) / "results.json"
        if p.exists():
            self.results = json.loads(p.read_text(encoding="utf-8"))
