"""Port of `poliastro.core.propagation.base`."""

import numpy as np

from ..._dispatch import Batch
from ..._lib import addr, fn

__all__ = ["func_twobody"]


def func_twobody(t0, u_, k):
    r"""Right-hand side of the two-body problem.

    .. math::

        \ddot{\vec{r}} = -\frac{\mu}{r^3}\vec{r}
    """
    # upstream's `u_` is a single 6-vector; an (n, 6) array is the batched form.
    u = np.asarray(u_, dtype=np.float64)
    flat = u.reshape(-1, 6)
    n = flat.shape[0]
    src = np.ascontiguousarray(flat)
    out = np.empty((n, 6), dtype=np.float64)
    fn("mp_func_twobody")(float(k), addr(src), addr(out), n)
    if u.ndim == 1:
        return out[0]
    return out.reshape(u.shape)
