"""trellis2_recon -- image-to-3D reconstruction with microsoft/TRELLIS.2-4B.

One Modal run = one weight load = N reconstruction ARMS. That is the whole cost
argument: the 4B weight load dominates the container, so arms that share it are
nearly free, and arms that do not share it are not comparable anyway.

An ARM is (view_set x resolution). Two kinds:

  PUBLIC   -- pipeline.run(image), single PIL image. Documented, guaranteed.
  MULTIVIEW -- EXPERIMENTAL. `get_cond` is typed
               `Union[torch.Tensor, list[Image.Image]]`, so the conditioning
               stage takes a LIST, but no public method orchestrates one. This
               lane drives the stages directly and REFUSES with the available
               attribute names if the pipeline's internals do not match what it
               expects. It never guesses a flow model.

  ACCEPTING A LIST IS NOT PROOF THE MODEL FUSES THE LIST. It may batch them or
  average embeddings, and either would look like success while silently
  ignoring every view after the first. The comparison that settles it is a
  1-view arm and an N-view arm on ONE weight load, scored by reprojection
  against held-out views -- which is why both kinds exist here rather than the
  multiview path replacing the public one.

Inputs are matted RGBA PNGs staged on the views volume by
scripts/stage_theexperiment_views.sh. Alpha matters: the subject must already be
cut out, because a background is conditioning too.

ASCII only. No emojis.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modallabs.base import Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult
from modallabs.registry import register

_VIEWS_DIR = Path("/views")          # volume mount, see modal_app _TRELLIS2_COMMON
_WEIGHTS_DIR = Path("/weights")      # HF cache volume; the 4B download is paid once
_MODEL_ID = "microsoft/TRELLIS.2-4B"
_VALID_RES = (512, 1024, 1536)

#: Every HF repo the pipeline resolves, with a file that PROVES access.
#: DINOv3 is TRELLIS.2's image-conditioning encoder and is gated=manual.
_REQUIRED_REPOS = (
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "config.json"),
    ("microsoft/TRELLIS.2-4B", "pipeline.json"),
)


class Trellis2Error(RuntimeError):
    """Refusal from this lane. Always names what was expected and what was found."""


def preflight_hf_repos(token: Optional[str] = None) -> Dict[str, str]:
    """Resolve every required HF repo BEFORE any weight load. Raises on the first
    one that is unreachable.

    WHY, measured: the DINOv3 gate raised a 401 GatedRepoError **185.4 s into a
    hot H100** -- inside from_pretrained, after the container booted, the image
    hydrated and the run was billed. An access problem is the cheapest class of
    failure to detect and one of the most expensive to discover late.

    A FILE FETCH, NEVER model_info(). `HfApi.model_info` returns PUBLIC metadata
    for a gated repo and succeeds with NO TOKEN AT ALL -- the first version of
    this check reported "OK gated=manual" for DINOv3 while every download was
    401ing. Metadata is the field adjacent to the signal; the signal is whether a
    file actually comes back.

    whoami() is checked FIRST and separately, because an invalid token and an
    unaccepted gate produce the SAME 401 on the repo, and telling them apart is
    the difference between "mint a new token" and "accept the licence".
    """
    import os
    from huggingface_hub import HfApi, hf_hub_download

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out: Dict[str, str] = {}
    if not tok:
        raise Trellis2Error(
            "preflight: no HF_TOKEN in the environment. The lane must mount the "
            "`huggingface-secret` Secret (credentials.py:43). Repos needed: "
            + ", ".join(r for r, _ in _REQUIRED_REPOS))
    try:
        out["whoami"] = HfApi(token=tok).whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        raise Trellis2Error(
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
            raise Trellis2Error(
                f"preflight: {repo} is not accessible to account {out['whoami']!r} "
                f"({type(exc).__name__}: {str(exc)[:200]}). If it is gated, accept "
                f"the terms at https://huggingface.co/{repo} while signed in as "
                f"that account. Refusing before any weight load rather than "
                f"discovering it on a hot GPU.") from exc
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@register("trellis2_recon")
class Trellis2ReconTrainer(Trainer):

    # ---- construction -----------------------------------------------------

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.arms: List[Dict[str, Any]] = []
        self.pipeline = None
        self.results: Dict[str, Any] = {}
        self._log = print
        self._out = Path(".")
        self._seed = 42

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Trellis2ReconTrainer":
        view_sets = config.get("view_sets")
        if not isinstance(view_sets, dict) or not view_sets:
            raise Trellis2Error(
                "cfg.view_sets must be a non-empty mapping "
                "{arm_name: [view_stem, ...]}, e.g. {single_0553: [IMG_0553]}"
            )
        res = config.get("resolutions") or [1024]
        if not isinstance(res, (list, tuple)) or not res:
            raise Trellis2Error("cfg.resolutions must be a non-empty list")
        bad = [r for r in res if int(r) not in _VALID_RES]
        if bad:
            raise Trellis2Error(
                f"cfg.resolutions {bad} not in {list(_VALID_RES)}. TRELLIS.2 is "
                "trained at these; an arbitrary value is not a quality dial."
            )
        for name, views in view_sets.items():
            if not isinstance(views, (list, tuple)) or not views:
                raise Trellis2Error(f"view_set {name!r} is empty")
        return cls(config)

    # ---- lifecycle --------------------------------------------------------

    def setup(self, setup: TrainerSetup) -> None:
        self._log = setup.log_fn
        self._out = Path(setup.output_dir)
        self._seed = int(setup.seed)
        self._out.mkdir(parents=True, exist_ok=True)

        # Resolve and HASH every input before any GPU work. A missing view is a
        # config error and must not surface as a CUDA traceback 12 minutes into
        # a paid container.
        suffix = str(self.cfg.get("view_suffix", "_sam.png"))
        views_dir = Path(self.cfg.get("views_dir", _VIEWS_DIR))
        manifest: Dict[str, Dict[str, Any]] = {}
        for arm_name, stems in self.cfg["view_sets"].items():
            paths = []
            for stem in stems:
                p = views_dir / f"{stem}{suffix}"
                if not p.exists():
                    have = sorted(q.name for q in views_dir.glob("*.png"))[:20]
                    raise Trellis2Error(
                        f"view {p} not staged. view_set={arm_name!r} wants "
                        f"{stem!r}. Volume has: {have}"
                    )
                paths.append(p)
                manifest.setdefault(stem, {
                    "path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size})
            for r in self.cfg.get("resolutions", [1024]):
                self.arms.append({
                    "arm": f"{arm_name}@{int(r)}", "view_set": arm_name,
                    "paths": paths, "resolution": int(r),
                    "multiview": len(paths) > 1,
                })
        (self._out / "input_manifest.json").write_text(json.dumps(manifest, indent=1))
        self._log(f"[trellis2] {len(self.arms)} arm(s), {len(manifest)} distinct view(s)")

        # PREFLIGHT FIRST, always. Seconds of CPU on a container already up,
        # against 185 s of H100 before a 401.
        pf = preflight_hf_repos()
        self._log(f"[trellis2] preflight OK as {pf.get('whoami')!r}: "
                  + ", ".join(f"{k}={v}" for k, v in pf.items() if k != "whoami"))

        # ONE weight load for every arm. This is the cost model of the lane.
        t0 = time.time()
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_HOME", str(_WEIGHTS_DIR))
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except Exception as exc:  # noqa: BLE001
            raise Trellis2Error(
                f"trellis2 is not importable in this image ({type(exc).__name__}: "
                f"{exc}). The lane's image must have run setup.sh with "
                "--flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm."
            ) from exc
        # REMBG IS CONSTRUCTED EAGERLY, AND WE MUST NEVER USE IT.
        # Trellis2ImageTo3DPipeline.from_pretrained builds `pipeline.rembg_model`
        # while loading, BEFORE anything consults `preprocess_image` (which this
        # trainer reads far below, at inference time). On 2026-08-24 that killed a
        # run at 101s: rembg's BiRefNet pulls the GATED repo briaai/RMBG-2.0, which
        # is a third gated dependency beyond the two the handoff named -- so a gate
        # check written against a KNOWN list could not see it.
        #
        # Requesting access would be the wrong fix. When preprocess_image is false
        # the inputs are ALREADY MATTED (TheExperiment stages `*_matte.png`, six SAM
        # mattes inspected by eye and locked by the harness); handing them to a
        # background remover would re-matte them with a worse tool and silently
        # replace the root input of every downstream reconstruction.
        #
        # So when preprocessing is off, the rembg class is replaced by a stub that
        # RAISES if anything calls it. Loud, not silent: a no-op stub would let a
        # future `preprocess_image: true` sail through with images that were never
        # background-removed, which reads as success and is not.
        if not bool(self.cfg.get("preprocess_image", True)):
            try:
                from trellis2.pipelines import rembg as _rembg

                class _RembgDisabled:
                    def __init__(self, *a, **k):
                        pass

                    def __call__(self, *a, **k):
                        raise Trellis2Error(
                            "rembg was disabled because preprocess_image=false, but "
                            "something called it. Either the inputs are NOT pre-matted "
                            "(then set preprocess_image: true and obtain access to "
                            "briaai/RMBG-2.0), or a code path ignores the flag."
                        )

                _stubbed = [n for n in dir(_rembg)
                            if isinstance(getattr(_rembg, n, None), type)
                            and not n.startswith("_")]
                for _n in _stubbed:
                    setattr(_rembg, _n, _RembgDisabled)
                self._log("[trellis2] preprocess_image=false -> rembg disabled "
                          f"(stubbed: {_stubbed}); inputs are taken as already matted")
            except ImportError:
                self._log("[trellis2] preprocess_image=false and no rembg module "
                          "present -- nothing to disable")

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            str(self.cfg.get("model_id", _MODEL_ID)))
        self._shim_dinov3_layer()
        self.pipeline.cuda()
        self._log(f"[trellis2] weights loaded in {time.time()-t0:.1f}s")

    def _shim_dinov3_layer(self) -> None:
        """Give DINOv3 the `.layer` attribute TRELLIS.2 iterates over.

        `trellis2/modules/image_feature_extractor.py:86` runs
            for i, layer_module in enumerate(self.model.layer):
        against a transformers version whose DINOv3ViTModel has no top-level
        `.layer`, so the run dies at get_cond with AttributeError (MEASURED
        2026-08-24, 106s in).

        WHERE THE BLOCKS ACTUALLY LIVE, measured on transformers 5.15.1 by
        building the model from its real config rather than guessing:
            children  = embeddings, rope_embeddings, model (DINOv3ViTEncoder), norm
            ModuleList = model.layer, len 24, of DINOv3ViTLayer
            .layer at top level: False      .encoder: False
        So the standard ViT guess (`encoder.layer`) is WRONG here -- the blocks
        sit under a child literally named `model`.

        This SEARCHES rather than hardcoding `model.layer`, because the image's
        transformers version need not match the probe's. It accepts only an
        unambiguous answer: exactly one ModuleList whose length equals the
        config's num_hidden_layers. Anything else RAISES, because silently
        binding to the wrong block list would produce a plausible mesh from the
        wrong features -- success-shaped and wrong.
        """
        import torch

        # Patch the CLASS, not an instance. The DINOv3 model is held by TRELLIS.2's
        # image_feature_extractor, not by the pipeline's own __dict__ -- an earlier
        # instance walk found nothing and the run failed identically (MEASURED:
        # traceback moved 285 -> 347, proving the shim ran and simply missed).
        # A property on the class resolves wherever the instance lives, and torch's
        # nn.Module.__getattr__ only fires when normal lookup fails, so a real
        # property takes precedence over it.
        cls = None
        try:
            from transformers.models.dinov3_vit.modeling_dinov3_vit import (
                DINOv3ViTModel as cls,
            )
        except Exception:                                   # noqa: BLE001
            import transformers
            for _n in dir(transformers):
                _o = getattr(transformers, _n, None)
                if isinstance(_o, type) and _n == "DINOv3ViTModel":
                    cls = _o
                    break
        if cls is None:
            self._log("[trellis2] dinov3 shim: DINOv3ViTModel not importable; "
                      "nothing to patch")
            print("[trellis2] dinov3 shim: DINOv3ViTModel not importable", flush=True)
            return
        if "layer" in vars(cls):
            self._log("[trellis2] dinov3 shim: .layer already defined; no patch")
            return

        def _blocks(self):
            depth = getattr(getattr(self, "config", None), "num_hidden_layers", None)
            cands = [(n, m) for n, m in self.named_modules()
                     if isinstance(m, torch.nn.ModuleList)
                     and (depth is None or len(m) == depth)]
            if len(cands) != 1:
                raise Trellis2Error(
                    f"cannot bind DINOv3 `.layer`: expected exactly one ModuleList of "
                    f"depth {depth}, found {[(n, len(mm)) for n, mm in cands]}. "
                    "Refusing to guess -- binding the wrong block list yields a "
                    "plausible mesh from the wrong features."
                )
            return cands[0][1]

        cls.layer = property(_blocks)
        msg = ("[trellis2] dinov3 shim: bound %s.layer -> the unique ModuleList "
               "matching num_hidden_layers" % cls.__name__)
        self._log(msg)
        print(msg, flush=True)          # _log goes to the run log; print reaches stdout

    def train_iter(self) -> Iterable[Dict[str, Any]]:
        return iter(self.arms)

    def eval_iter(self) -> Iterable[Any]:
        return iter(())

    def eval_step(self, batch: Any) -> TrainerStepResult:
        # No eval split: an arm's quality is judged by REPROJECTION against the
        # views it did not see, which happens locally against the returned GLB.
        # A no-op here rather than a fake metric -- a number invented in this
        # method would be the exact kind of green that proves nothing.
        raise Trellis2Error("trellis2_recon has no eval split; eval_iter is empty")

    # ---- the work ---------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> TrainerStepResult:
        from PIL import Image
        arm = batch["arm"]
        res = batch["resolution"]
        imgs = [Image.open(p).convert("RGBA") for p in batch["paths"]]
        t0 = time.time()

        if batch["multiview"]:
            mesh = self._run_multiview(imgs, res)
        else:
            mesh = self.pipeline.run(
                imgs[0], num_samples=1, seed=self._seed,
                max_num_tokens=int(self.cfg.get("max_num_tokens", 49152)),
            )[0]

        secs = time.time() - t0
        glb = self._out / f"{arm.replace('@','_at_')}.glb"
        self._export_glb(mesh, glb)

        nfaces, nverts = self._mesh_counts(mesh)
        rec = {"arm": arm, "resolution": res, "n_views": len(imgs),
               "multiview": batch["multiview"], "seconds": round(secs, 2),
               "glb": str(glb), "glb_bytes": glb.stat().st_size if glb.exists() else 0,
               "faces": nfaces, "vertices": nverts,
               "views": [Path(p).name for p in batch["paths"]]}
        self.results[arm] = rec
        self._log(f"[trellis2] {arm}: {secs:.1f}s faces={nfaces} verts={nverts} "
                  f"-> {glb.name}")
        return TrainerStepResult(metrics={"seconds": secs, "faces": float(nfaces or 0)},
                                 n_examples=1, extras=rec)

    def _run_multiview(self, imgs: List[Any], res: int):
        """EXPERIMENTAL multi-view: drive the public stages with a list cond.

        Refuses rather than guesses. `run()` selects flow models internally via
        `pipeline_type`; this lane will not reimplement that selection from
        assumption, so if the expected attributes are absent it says exactly
        which names it looked for and what the object actually carries.
        """
        pipe = self.pipeline
        for meth in ("get_cond", "sample_sparse_structure", "sample_shape_slat",
                     "sample_tex_slat", "decode_latent"):
            if not hasattr(pipe, meth):
                raise Trellis2Error(
                    f"multiview arm: pipeline has no {meth!r}. TRELLIS.2's stage "
                    f"API has drifted. Present: {sorted(a for a in dir(pipe) if not a.startswith('_'))}"
                )
        models = getattr(pipe, "models", None)
        if not isinstance(models, dict):
            raise Trellis2Error(
                "multiview arm: pipeline.models is not a dict, so the flow models "
                f"cannot be resolved by name (got {type(models).__name__}). "
                "Refusing rather than guessing which module is the shape flow."
            )
        shape_key = self.cfg.get("shape_flow_key")
        tex_key = self.cfg.get("tex_flow_key")
        if shape_key not in models or tex_key not in models:
            raise Trellis2Error(
                "multiview arm needs cfg.shape_flow_key and cfg.tex_flow_key naming "
                f"entries of pipeline.models. Available keys: {sorted(models)}. "
                "These are NOT guessed: picking the wrong flow model would produce "
                "a plausible mesh from the wrong network and read as success."
            )

        prep = bool(self.cfg.get("preprocess_image", True))
        if prep and hasattr(pipe, "preprocess_image"):
            imgs = [pipe.preprocess_image(im) for im in imgs]
        cond = pipe.get_cond(imgs, res)
        coords = pipe.sample_sparse_structure(cond, res, num_samples=1)
        shape = pipe.sample_shape_slat(cond, models[shape_key], coords)
        tex = pipe.sample_tex_slat(cond, models[tex_key], shape)
        out = pipe.decode_latent(shape, tex, res)
        return out[0] if isinstance(out, (list, tuple)) else out

    def _export_glb(self, mesh: Any, dst: Path) -> None:
        tex = int(self.cfg.get("texture_size", 4096))
        try:
            from o_voxel.postprocess import to_glb
        except Exception as exc:  # noqa: BLE001
            raise Trellis2Error(
                f"o_voxel.postprocess.to_glb unavailable ({type(exc).__name__}: {exc}); "
                "the image did not build --o-voxel."
            ) from exc
        # to_glb's SIGNATURE VARIES between TRELLIS.2 builds. On 2026-08-24 a run
        # that had already spent 340s of H100 producing a real mesh was thrown away
        # by `to_glb() missing 5 required positional arguments: faces, attr_volume,
        # coords, attr_layout, aabb` -- $1.41 of reconstruction discarded by an
        # export mismatch. Never again: bind by INTROSPECTION, and if that fails,
        # DUMP THE MESH before raising so the compute survives the bug.
        import inspect
        import pickle

        try:
            _sig = inspect.signature(to_glb)
            _params = list(_sig.parameters.values())
        except (TypeError, ValueError):
            _params = []

        # MEASURED 2026-08-24 against this build. to_glb wants
        #   (vertices, faces, attr_volume, coords, attr_layout, aabb, ...)
        # and the mesh is a MeshWithVoxel carrying
        #   attrs, coords, faces, layout, origin, vertices, voxel_shape, voxel_size
        # Three of the six are not same-named: two are pure ALIASES, and `aabb` is
        # DERIVED from the voxel grid. The derivation is verified below against the
        # vertex bounds rather than trusted -- a wrong aabb would rescale or shift
        # the texture bake and still export a plausible GLB.
        _ALIAS = {"attr_volume": "attrs", "attr_layout": "layout",
                  "grid_size": "voxel_shape"}

        def _derive_aabb(m):
            """Build aabb from the voxel grid, then PROVE it against the vertices.

            The first attempt assumed voxel_shape was a 3-vector of grid dims and
            died on `operands could not be broadcast together with shapes (3,) (5,)`
            -- it is 5-D (MEASURED 2026-08-24), so the grid dims are a SLICE of it,
            not the whole thing. Rather than guess which slice, enumerate the
            defensible candidates and keep only those that actually contain the
            mesh. Ambiguity is reported, never resolved by preference order: a
            wrong aabb does not crash, it rescales the texture bake and exports a
            plausible GLB, which is worse than an error.
            """
            import numpy as _np
            to_np = lambda x: (x.detach().cpu().numpy() if hasattr(x, "detach")
                               else _np.asarray(x))
            org = getattr(m, "origin", None)
            shp = getattr(m, "voxel_shape", None)
            vsz = getattr(m, "voxel_size", None)
            if org is None or shp is None or vsz is None:
                return None, "origin/voxel_shape/voxel_size not all present"
            try:
                org = to_np(org).astype(float).reshape(-1)
                shp = to_np(shp).astype(float).reshape(-1)
                vsz = to_np(vsz).astype(float).reshape(-1)
                v = to_np(getattr(m, "vertices")).astype(float)
            except Exception as exc:                      # noqa: BLE001
                return None, f"could not read grid fields ({type(exc).__name__}: {exc})"

            vlo, vhi = v.min(axis=0), v.max(axis=0)
            tol = float(_np.max(vsz)) * 2.0
            dims = {"first3": shp[:3], "last3": shp[-3:]}
            if getattr(m, "coords", None) is not None:    # voxel indices actually used
                try:
                    c = to_np(m.coords).astype(float)
                    if c.ndim == 2 and c.shape[1] >= 3:
                        dims["coords_extent"] = c[:, -3:].max(axis=0) + 1.0
                except Exception:                          # noqa: BLE001
                    pass

            ok = {}
            for name, d in dims.items():
                d = _np.asarray(d, float).reshape(-1)
                if d.size != 3 or org.size != 3:
                    continue
                lo = org
                hi = org + d * (vsz if vsz.size in (1, 3) else vsz[:1])
                if (not _np.any(vlo < lo - tol)) and (not _np.any(vhi > hi + tol)):
                    ok[name] = (lo, hi)

            diag = (f"origin={org.tolist()} voxel_shape={shp.tolist()} "
                    f"voxel_size={vsz.tolist()} vertex_bounds=[{vlo.tolist()}, "
                    f"{vhi.tolist()}] candidates_tested={list(dims)} tol={tol:.4g}")
            if len(ok) != 1:
                return None, (f"aabb is AMBIGUOUS: {len(ok)} of {len(dims)} candidates "
                              f"contain the mesh ({list(ok)}). {diag}")
            name, (lo, hi) = next(iter(ok.items()))
            return (_np.stack([lo, hi]),
                    f"origin + {name}*voxel_size, VERIFIED to contain vertices ({diag})")

        _kw, _missing, _notes = {}, [], []
        if _params:
            for p in _params:
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.name in ("texture_size", "tex_size"):
                    _kw[p.name] = tex
                elif hasattr(mesh, p.name):
                    _kw[p.name] = getattr(mesh, p.name)
                elif p.name in _ALIAS and hasattr(mesh, _ALIAS[p.name]):
                    _kw[p.name] = getattr(mesh, _ALIAS[p.name])
                    _notes.append(f"{p.name}<-{_ALIAS[p.name]}")
                elif p.name == "aabb":
                    _v, _why = _derive_aabb(mesh)
                    if _v is None:
                        _missing.append(f"aabb ({_why})")
                    else:
                        _kw["aabb"] = _v
                        _notes.append(f"aabb<-{_why}")
                elif p.default is p.empty:
                    _missing.append(p.name)

        if _missing:
            dump = dst.with_suffix(".mesh.pkl")
            try:
                dump.parent.mkdir(parents=True, exist_ok=True)
                with open(dump, "wb") as fh:
                    pickle.dump(mesh, fh, protocol=4)
                saved = f"mesh dumped to {dump} ({dump.stat().st_size} bytes)"
            except Exception as exc:                        # noqa: BLE001
                saved = f"mesh could NOT be dumped ({type(exc).__name__}: {exc})"

            # LAST RESORT before losing the run: the mesh carries its own
            # serializer. It may not write a GLB, but ANY faithful geometry file
            # beats paying for the reconstruction twice. Reported loudly, never
            # silently substituted -- the caller must know it did not get a GLB.
            if hasattr(mesh, "save"):
                for _ext in (".glb", ".ply", ".obj", ""):
                    _alt = dst.with_suffix(_ext) if _ext else dst.with_suffix(".mesh")
                    try:
                        mesh.save(str(_alt))
                        if _alt.exists() and _alt.stat().st_size > 0:
                            _n = (f"[trellis2] to_glb unsatisfiable; mesh.save() wrote "
                                  f"{_alt} ({_alt.stat().st_size} bytes) as a FALLBACK. "
                                  f"This is NOT a to_glb export -- no texture bake.")
                            self._log(_n)
                            print(_n, flush=True)
                            saved += f"; mesh.save() fallback -> {_alt}"
                            break
                    except Exception:                       # noqa: BLE001
                        continue

            raise Trellis2Error(
                "to_glb%s cannot be satisfied from this mesh: missing %s. "
                "mesh is %s with attributes %s. %s -- the reconstruction itself "
                "SUCCEEDED; only the export binding is wrong, so re-export offline "
                "rather than paying for the mesh twice."
                % (_sig if _params else "(signature unavailable)", _missing,
                   type(mesh).__name__,
                   sorted(a for a in dir(mesh) if not a.startswith("_"))[:40], saved)
            )

        _m = ("[trellis2] to_glb bound: %s%s"
              % (sorted(_kw), ("  [" + "; ".join(_notes) + "]") if _notes else ""))
        self._log(_m)
        print(_m, flush=True)
        glb = to_glb(**_kw) if _kw else to_glb(mesh, texture_size=tex)
        if hasattr(glb, "export"):
            glb.export(str(dst))
        elif isinstance(glb, (bytes, bytearray)):
            dst.write_bytes(glb)
        else:
            raise Trellis2Error(
                f"to_glb returned {type(glb).__name__}, which is neither a trimesh "
                "scene nor bytes; refusing to invent an export path."
            )

    @staticmethod
    def _mesh_counts(mesh: Any):
        for f_attr, v_attr in (("faces", "vertices"), ("faces", "verts")):
            f, v = getattr(mesh, f_attr, None), getattr(mesh, v_attr, None)
            if f is not None and v is not None:
                try:
                    return int(len(f)), int(len(v))
                except TypeError:
                    pass
        return None, None

    # ---- summary ----------------------------------------------------------

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        (self._out / "arms.json").write_text(json.dumps(self.results, indent=1))
        faces = [r["faces"] for r in self.results.values() if r.get("faces")]
        best = max(faces) if faces else 0.0
        return TrainerEpochResult(
            train_metrics={"arms": float(len(self.results)),
                           "max_faces": float(best)},
            val_metrics={}, is_best=True, monitor_value=float(best))

    def save_checkpoint(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "arms.json").write_text(json.dumps(self.results, indent=1))

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path) / "arms.json"
        if p.exists():
            self.results = json.loads(p.read_text())

    def teardown(self) -> None:
        self.pipeline = None


__all__ = ["Trellis2ReconTrainer", "Trellis2Error"]
