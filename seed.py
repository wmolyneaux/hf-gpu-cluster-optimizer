"""modallabs — deterministic seed setup.

Set seeds across every common library so a Trainer started with seed=N
produces the same checkpoint every time.
"""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, strict_torch: bool = True) -> None:
    """Set every common-library seed deterministically.

    Args:
        seed: integer seed.
        strict_torch: if True, also set torch's deterministic algorithms
            and disable cuDNN benchmarking. Costs some throughput; ensures
            bit-exact repeatability across runs on the same GPU. Set to
            False for prod runs where you want reproducibility on the
            metrics but not bit-exact tensor equality.
    """
    seed = int(seed)
    # Note: PYTHONHASHSEED is read by CPython ONCE at interpreter startup
    # to seed string-/bytes-/datetime-hash randomization. Setting it here
    # has NO effect on the current process's hash randomization. We still
    # write the env var because spawn-context subprocesses (used by
    # concurrent_train's ProcessPoolExecutor) inherit the parent's env
    # and DO read PYTHONHASHSEED at their fresh interpreter startup, so
    # the seed propagates to children. Functional but subtle; documented
    # here so future readers don't "fix" it by deleting the assignment.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        if strict_torch:
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
            try:
                _torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except Exception:
        pass
    try:
        import transformers as _tf
        _tf.set_seed(seed)
    except Exception:
        pass


__all__ = ["set_global_seed"]
