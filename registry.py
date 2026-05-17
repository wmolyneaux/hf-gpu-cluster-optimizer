"""modallabs — model-type registry.

Registers Trainer classes by their cfg `type` field. Decorator-based;
import side-effect populates the registry.

Usage:

    from modallabs.registry import register, get
    from modallabs.base import Trainer

    @register("lstm")
    class LSTMTrainer(Trainer):
        ...

    cls = get("lstm")  # returns LSTMTrainer
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from modallabs.base import Trainer


_REGISTRY: Dict[str, Type[Trainer]] = {}


def register(name: str) -> Callable[[Type[Trainer]], Type[Trainer]]:
    """Decorator: register a Trainer class under cfg `type=name`.

    Names are case-insensitive on lookup but stored case-sensitive.
    Re-registering an existing name raises ValueError to prevent
    accidental shadowing. Use force_register if you need to override
    (e.g. in tests).
    """
    def _wrap(cls: Type[Trainer]) -> Type[Trainer]:
        key = str(name).strip().lower()
        if key in _REGISTRY:
            raise ValueError(
                f"modallabs.registry: type {key!r} already registered "
                f"({_REGISTRY[key].__name__}); use force_register to override"
            )
        _REGISTRY[key] = cls
        return cls
    return _wrap


def force_register(name: str, cls: Type[Trainer]) -> None:
    """Register or overwrite. Use sparingly (tests, monkey-patching)."""
    _REGISTRY[str(name).strip().lower()] = cls


def get(name: str) -> Type[Trainer]:
    """Look up a Trainer class by cfg `type` name. Raises KeyError if missing."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"modallabs.registry: unknown model type {name!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_types() -> List[str]:
    """Sorted list of registered cfg `type` names."""
    return sorted(_REGISTRY.keys())


__all__ = ["register", "force_register", "get", "list_types"]
