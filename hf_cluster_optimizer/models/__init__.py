"""hf_cluster_optimizer.models -- import every concrete trainer module to fire registry.

Importing this package is enough to register every built-in trainer.
Each submodule is wrapped in try/except so a missing optional dep
(e.g. lightgbm) does not block the rest from registering.
"""
from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def _safe_import(modname: str) -> None:
    try:
        __import__(modname)
    except Exception as exc:  # noqa: BLE001 -- log + continue
        logger.debug("hf_cluster_optimizer.models: skipping %s (%s: %s)",
                     modname, type(exc).__name__, exc)


# Pure-stdlib + numpy/pandas only (always succeed unless deps absent)
_safe_import("hf_cluster_optimizer.models.generic_torch")
_safe_import("hf_cluster_optimizer.models.generic_sklearn")
_safe_import("hf_cluster_optimizer.models.lstm")
_safe_import("hf_cluster_optimizer.models.rnn")
_safe_import("hf_cluster_optimizer.models.transformer")
_safe_import("hf_cluster_optimizer.models.manifold")
_safe_import("hf_cluster_optimizer.models.ntm")
_safe_import("hf_cluster_optimizer.models.q_learning")
_safe_import("hf_cluster_optimizer.models.diffusion")

# Optional ML libs
_safe_import("hf_cluster_optimizer.models.lightgbm")
_safe_import("hf_cluster_optimizer.models.xgboost")
_safe_import("hf_cluster_optimizer.models.catboost")

# HuggingFace transformers (heavy import; do last)
_safe_import("hf_cluster_optimizer.models.hf_transformer")
