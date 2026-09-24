"""modallabs.models -- import every concrete trainer module to fire registry.

Importing this package is enough to register every built-in trainer.
Each submodule is wrapped in try/except so a missing optional dep
(e.g. lightgbm) does not block the rest from registering.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _safe_import(modname: str) -> None:
    try:
        __import__(modname)
    except Exception as exc:  # noqa: BLE001 -- log + continue
        logger.debug("modallabs.models: skipping %s (%s: %s)",
                     modname, type(exc).__name__, exc)


# Pure-stdlib + numpy/pandas only (always succeed unless deps absent)
_safe_import("modallabs.models.generic_torch")
_safe_import("modallabs.models.generic_sklearn")
_safe_import("modallabs.models.lstm")
_safe_import("modallabs.models.rnn")
_safe_import("modallabs.models.transformer")
_safe_import("modallabs.models.manifold")
_safe_import("modallabs.models.ntm")
_safe_import("modallabs.models.q_learning")
_safe_import("modallabs.models.wan_vace_shot")
_safe_import("modallabs.models.diffusion")
# Cycles take chunks for the berkeley-usd heroshot; shells Blender, no bpy import.
_safe_import("modallabs.models.heroshot_take")
# Lives in the separate LongCatAvatar repo; skipped cleanly when not pip-installed.
_safe_import("modallabs.models.longcat_avatar")
# Lives in the separate tram-motion repo; skipped cleanly when not pip-installed.
_safe_import("modallabs.models.tram_motion")
# Kimodo text-to-motion (nv-tlabs); retargets onto smpl22 via tram-motion.
_safe_import("modallabs.models.kimodo_motion")
# Orpheus 3B voice clone; the tokenized dataset comes from the voicecraft repo.
_safe_import("modallabs.models.orpheus_voice")
# TRELLIS.2-4B image-to-3D for TheExperiment; heavy CUDA-extension image,
# skipped cleanly wherever trellis2 is not installed.
_safe_import("modallabs.models.trellis2_recon")
_safe_import("modallabs.models.orpheus_tts")
# A repo's own Blender script on a GPU from a staged bundle (game1CreatureMesh renders); shells Blender, no bpy.
_safe_import("modallabs.models.blender_script")

# Optional ML libs
_safe_import("modallabs.models.lightgbm")
_safe_import("modallabs.models.xgboost")
_safe_import("modallabs.models.catboost")

# HuggingFace transformers (heavy import; do last)
_safe_import("modallabs.models.hf_transformer")
_safe_import("modallabs.models.comfy_sheet")
