"""Port of `poliastro.core.propagation.gooding`.

Upstream raises ``NotImplementedError`` for the parabolic and hyperbolic cases;
the kernel returns NaN there instead, because the C ABI cannot raise.
"""

import numpy as np

from ..._dispatch import Batch
from ..._lib import addr, fn
from .farnocchia import _propagate

__all__ = ["gooding_coe", "gooding"]


def gooding_coe(k, p, ecc, inc, raan, argp, nu, tof, numiter=150, rtol=1e-08):
    r"""True anomaly after `tof` seconds, from classical orbital elements."""
    b = Batch(p, ecc, inc, raan, argp, nu, tof)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_gooding_coe")(
        float(k), addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]),
        addr(b.inp[3]), addr(b.inp[4]), addr(b.inp[5]), addr(b.inp[6]),
        int(numiter), float(rtol), addr(out), b.n
    )
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def gooding(k, r0, v0, tof, numiter=150, rtol=1e-08):
    r"""State vector after `tof` seconds, from a state vector."""
    return _propagate("mp_gooding", k, r0, v0, tof, int(numiter), float(rtol))
