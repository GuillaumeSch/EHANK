"""Minimal reconstruction of the helpers HH_Block imports from Fun.my_funs.
Only make_strictly_decreasing is used inside the backward step."""
import numpy as np
from numba import njit


@njit
def make_strictly_decreasing(x):
    """Enforce a strictly-decreasing sequence along the last axis.

    Marginal utility u'(c) must be decreasing in end-of-period assets for the
    EGM inversion / upper envelope to be well behaved. Where the raw array
    violates monotonicity (numerical noise near kinks), values are lowered to
    the minimum needed to restore strict monotonicity, leaving the rest intact.
    """
    out = x.copy()
    flat = out.reshape((-1, x.shape[-1]))
    n, m = flat.shape
    eps = 1e-12
    for i in range(n):
        for j in range(1, m):
            if flat[i, j] >= flat[i, j - 1]:
                flat[i, j] = flat[i, j - 1] - eps
    return flat.reshape(x.shape)
