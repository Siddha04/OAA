from __future__ import annotations

from collections.abc import Sequence

import oaa_cpp


def compiled() -> bool:
    """Return whether the OAA build contains the CUDA backend."""
    return bool(oaa_cpp.cuda_compiled())


def available() -> bool:
    """Return whether a usable CUDA device is currently visible."""
    return bool(oaa_cpp.cuda_available())


def vector_add(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    """Run element-wise vector addition through the active CUDA backend."""
    if len(lhs) != len(rhs):
        raise ValueError("lhs and rhs must have the same length")
    return [float(value) for value in oaa_cpp.cuda_vector_add(list(lhs), list(rhs))]
