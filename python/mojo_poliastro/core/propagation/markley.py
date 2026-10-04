"""Port of `poliastro.core.propagation.markley`."""

import numpy as np

from ..._dispatch import Batch
from ..._lib import addr, fn
from .farnocchia import _propagate

__all__ = ["markley_coe", "markley"]


def markley_coe(k, p, ecc, inc, raan, argp, nu, tof):
    r"""True anomaly after `tof` seconds, from classical orbital elements."""
    b = Batch(p, ecc, inc, raan, argp, nu, tof)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_markley_coe")(
        float(k), addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]),
        addr(b.inp[3]), addr(b.inp[4]), addr(b.inp[5]), addr(b.inp[6]),
        addr(out), b.n
    )
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def markley(k, r0, v0, tof):
    r"""State vector after `tof` seconds, from a state vector."""
    return _propagate("mp_markley", k, r0, v0, tof)
