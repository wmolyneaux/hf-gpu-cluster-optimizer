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

#: The motion checkpoint each accepted `model` name resolves to, and the skeleton its
#: generate CLI WRITES. A name not in this table is refused: the lane must know the
#: skeleton before it can write the sidecar honestly.
#:   LICENCES, read on each Hugging Face model card (2026-09-23 / 2026-09-24):
#:   Kimodo-SMPLX-RP-v1 ....... "for non-commercial research use only"
#:   Kimodo-SOMA-RP-v1, v1.1 .. NVIDIA Open Model License, "ready for commercial use"; ungated
#:   SOMA OUTPUT: the model runs on SOMASkeleton30, but kimodo_model.py converts its output to
#:   somaskel77 ("Convert SOMA output to somaskel77 for external API") before the CLI saves it,
#:   so the npz carries 77 joints. The 30-joint subset is exported beside it (soma30_export).
_MODEL_REPOS: Dict[str, Tuple[str, str, int]] = {
    "kimodo-smplx-rp": ("nvidia/Kimodo-SMPLX-RP-v1", "SMPLXSkeleton22", 22),
    "Kimodo-SMPLX-RP-v1": ("nvidia/Kimodo-SMPLX-RP-v1", "SMPLXSkeleton22", 22),
    "Kimodo-SOMA-RP-v1": ("nvidia/Kimodo-SOMA-RP-v1", "somaskel77", 77),
    "Kimodo-SOMA-RP-v1.1": ("nvidia/Kimodo-SOMA-RP-v1.1", "somaskel77", 77),
}


def model_repo(model: str) -> Tuple[str, str, int]:
    """(HF repo, skeleton the CLI writes, joint count) for a `model` name, or a refusal."""
    if model not in _MODEL_REPOS:
        raise KimodoError(
            f"model {model!r} is not in _MODEL_REPOS {sorted(_MODEL_REPOS)}. Add it with the "
            "skeleton its CLI writes (read kimodo/model/kimodo_model.py) and its licence.")
    return _MODEL_REPOS[model]


def required_repos(model: str) -> Tuple[Tuple[str, str], ...]:
    """The repos a run of `model` resolves: its own checkpoint plus the three text-encoder repos."""
    return ((model_repo(model)[0], "config.yaml"),) + _REQUIRED_REPOS[1:]


class KimodoError(RuntimeError):
    """Refusal from this lane. Always names what was expected and what was found."""


def preflight_hf_repos(token: Optional[str] = None,
                       repos: Optional[Tuple[Tuple[str, str], ...]] = None) -> Dict[str, str]:
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
    repos = _REQUIRED_REPOS if repos is None else repos
    out: Dict[str, str] = {}
    if not tok:
        raise KimodoError(
            "preflight: no HF_TOKEN in the environment. The lane must mount the "
            "`huggingface-secret` Secret (credentials.py). Repos needed: "
            + ", ".join(r for r, _ in repos))
    try:
        out["whoami"] = HfApi(token=tok).whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        raise KimodoError(
            f"preflight: the HF token is INVALID ({type(exc).__name__}: {exc}). "
            "This is NOT a gate problem -- a dead token and an unaccepted licence "
            "both 401 on the repo, and only whoami() separates them. Mint a read "
            "token and run: modal secret create huggingface-secret "
            "HF_TOKEN=hf_... --force") from exc

    for repo, probe_file in repos:
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


#: SOMA hand ends: in SOMASkeleton30 their parent is the HAND, in somaskel77 it is the last
#: finger joint (posed with the relaxed-hands rest), so their 30-joint FK legitimately differs.
_SOMA30_HAND_ENDS = ("LeftHandThumbEnd", "LeftHandMiddleEnd", "RightHandThumbEnd", "RightHandMiddleEnd")


def soma30_export(npz77: Path, stem: Path) -> Dict[str, Any]:
    """Write the 30-joint SOMA teacher beside a somaskel77 clip, with its skeleton definition.

    `<stem>.soma30.npz`: local_rot_mats, global_rot_mats (T,30,3,3), posed_joints (T,30,3),
    root_positions (T,3), foot_contacts (T,4) [L_heel, L_toe, R_heel, R_toe] (the CLI's six
    channels minus its two toe-end copies). `<stem>.skeleton.json`: names, parent indices,
    neutral joints (metres) for somaskel30 and somaskel77, the 30-in-77 index map, and the axes
    MEASURED from the neutral joints (never assumed).

    BIND-PROOF: the 30-joint local rotations are re-run through somaskel30's own FK and compared
    with the CLI's 77-joint posed joints at every body joint. A wrong slice or key shows as a
    metre-scale error; the hand ends are reported apart (their chain differs, see above).
    """
    import numpy as np
    import torch
    from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77

    s30, s77 = SOMASkeleton30(), SOMASkeleton77()
    sl = s30.get_skel_slice(s77)
    d = np.load(npz77)
    loc77 = np.asarray(d["local_rot_mats"], float)
    if loc77.ndim != 4 or loc77.shape[1:] != (77, 3, 3):
        raise KimodoError(f"{npz77.name}: expected somaskel77 local_rot_mats (T,77,3,3), got {loc77.shape}")
    root = np.asarray(d["root_positions"], float)
    loc30 = loc77[:, sl]
    # Kimodo's skeleton buffers are float32: FK in float32, compare in float64
    g30, p30, _ = s30.fk(torch.from_numpy(loc30.astype(np.float32)), torch.from_numpy(root.astype(np.float32)))
    g30, p30 = g30.double(), p30.double()
    g30, p30 = g30.numpy(), p30.numpy()
    p77 = np.asarray(d["posed_joints"], float)[:, sl]
    body = [i for i, n in enumerate(s30.bone_order_names) if n not in _SOMA30_HAND_ENDS]
    err_body = float(np.abs(p30[:, body] - p77[:, body]).max())
    if err_body > 1e-3:
        raise KimodoError(
            f"{npz77.name}: SOMA-30 FK disagrees with the CLI's posed joints by {err_body:.4f} m at a "
            "body joint. The 30-in-77 slice or the npz keys are not what this export assumes.")
    out = {"local_rot_mats": loc30, "global_rot_mats": g30, "posed_joints": p30, "root_positions": root}
    if "foot_contacts" in d.files:
        fc = np.asarray(d["foot_contacts"])
        out["foot_contacts"] = fc[:, [0, 1, 3, 4]] if fc.shape[-1] == 6 else fc
    np.savez_compressed(f"{stem}.soma30.npz", **out)

    n30 = s30.neutral_joints.numpy()
    ix = s30.bone_index

    def _skel(s):
        return {"name": s.name, "joints": list(s.bone_order_names),
                "parents": [int(p) for p in s.joint_parents.tolist()],
                "neutral_joints_m": np.round(s.neutral_joints.numpy(), 6).tolist()}
    axes = {
        "up": "+Y" if n30[ix["Head"], 1] > n30[ix["LeftFoot"], 1] else "-Y",
        "forward": "+Z" if (n30[ix["LeftEye"], 2] + n30[ix["RightEye"], 2]) / 2 > n30[ix["Head"], 2] else "-Z",
        "character_left": "+X" if n30[ix["LeftArm"], 0] > 0 else "-X",
        "measured_from": "neutral joints: Head above LeftFoot, eyes ahead of Head, LeftArm x sign",
    }
    skel = {"teacher": _skel(s30), "cli_output": _skel(s77), "teacher_slice_in_cli_output": [int(i) for i in sl],
            "axes": axes, "fps": 30, "rest": "identity local rotations = the neutral joints (a T-pose)",
            "fk_bind_proof_max_err_m": {"body_joints": err_body,
                                        "hand_ends": float(np.abs(p30 - p77).max())}}
    Path(f"{stem}.skeleton.json").write_text(json.dumps(skel, indent=1), encoding="utf-8")
    return {"frames": int(loc30.shape[0]), "fk_err_body_m": err_body, "axes": axes}


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

        # 1. GATES FIRST. Cheapest failure class, most expensive to find late. The model
        #    name must be known (its skeleton decides the sidecar) and its OWN checkpoint is
        #    the one probed: a SOMA run must not pass on the SMPL-X repo's access.
        model = self.cfg.get("model", "kimodo-smplx-rp")
        model_repo(model)
        self.results["hf_preflight"] = preflight_hf_repos(repos=required_repos(model))

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
        _, skeleton, n_joints = model_repo(model)
        # A LIST of durations is the CLI's multi-prompt form: the action holds one sentence
        # per segment ("walks forward. slams its fist down.") and the segments are generated
        # as ONE motion with transitions. A scalar is the single-prompt form, as before.
        dur = clip.get("duration_s", 3.5)
        durations = [float(x) for x in dur] if isinstance(dur, (list, tuple)) else [float(dur)]
        duration = sum(durations)
        n_samples = int(clip.get("num_samples", 1))
        seed = int(clip.get("seed", self.seed))

        cmd = [
            os.environ.get("KIMODO_PY", "python"), "-m", "kimodo.scripts.generate",
            clip["action"],
            "--model", model,
            "--duration", " ".join(str(x) for x in durations),
            "--seed", str(seed),
            "--output", str(npz),
        ]
        if n_samples > 1:
            # one model load, several samples: the CLI then writes <outdir>/<name>/<name>_NN.npz
            cmd += ["--num_samples", str(n_samples)]
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
        npzs = ([npz] if n_samples == 1
                else [outdir / name / f"{name}_{i:02d}.npz" for i in range(n_samples)])
        missing = [p.name for p in npzs if not p.exists()]
        if proc.returncode != 0 or missing:
            raise KimodoError(
                f"kimodo generate failed for clip {name!r} (rc={proc.returncode}, missing "
                f"{missing}).\nstderr tail: {proc.stderr[-800:]}")

        d = np.load(npzs[0])
        rot22 = d["local_rot_mats"]
        frames = int(rot22.shape[0])
        for p in npzs:
            got = np.load(p)["local_rot_mats"].shape
            if len(got) != 4 or got[1] != n_joints:
                raise KimodoError(
                    f"{p.name}: model {model!r} should write {skeleton} ({n_joints} joints), the "
                    f"npz has local_rot_mats {got}. Refusing to label it.")
        exports = ({p.stem: soma30_export(p, p.with_suffix("")) for p in npzs}
                   if skeleton == "somaskel77" else None)

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
            "model_repo": model_repo(model)[0],
            "model_revision": self.cfg.get("model_revision"),
            "skeleton": skeleton,
            # measured for SMPLXSkeleton22 only (see module docstring); meaningless for SOMA
            "skeleton_equals_smpl24_prefix": True if skeleton == "SMPLXSkeleton22" else None,
            "teacher_skeleton": "somaskel30" if exports else skeleton,
            "soma30_exports": exports,
            "fps": 30,
            "frames": frames,
            "duration_s": duration,
            "durations_s": durations,
            "num_samples": n_samples,
            "sample_npz_sha256": {p.name: _sha256_file(p) for p in npzs},
            "seed": seed,
            "constraints": clip.get("constraints"),
            "diffusion_steps": clip.get("diffusion_steps"),
            "text_encoder": {
                "base": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                "peft": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                "device": env.get("TEXT_ENCODER_DEVICE"),
            },
            "npz_sha256": _sha256_file(npzs[0]),
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

        self.results[name] = {"frames": frames, "npz": str(npzs[0]), "npzs": [str(p) for p in npzs],
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
