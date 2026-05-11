"""modallabs -- portable concurrent training harness for ML models.

Runs locally (subprocess pool) or on Modal Labs cloud through the same
code path. HuggingFace AutoModel support plus generic torch.nn.Module /
sklearn / boosting wrappers, with cost controls on the cloud path.

This package keeps a flat layout: submodules sit directly under the
`modallabs/` directory. Importing this package does NOT pull in torch,
transformers, or any optional ML lib -- those are imported lazily inside
the trainers that need them. Import `modallabs.models` to fire the
registry (registers every built-in trainer whose deps are installed).
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
