"""Port of `poliastro.core.earth_atmosphere.util`."""

import numpy as np

from ..._dispatch import Batch
from ..._lib import addr, fn

__all__ = [
    "geometric_to_geopotential", "z_to_h", "geopotential_to_geometric",
    "h_to_z", "gravity", "check_altitude",
]


def _run(symbol, arr, *scalars, tail=()):
    b = Batch(arr, tail=tail)
    out = np.empty((b.n,) + tail, dtype=np.float64)
    fn(symbol)(addr(b.inp[0]), *[float(s) for s in scalars], addr(out), b.n)
    out = out.reshape(b.shape + tail)
    if b.scalar and not tail:
        return float(out)
    if b.scalar:
        return out[0]
    return out


def geometric_to_geopotential(z, r0):
    """Geopotential altitude from a geometric altitude."""
    return _run("mp_geometric_to_geopotential", z, r0)


z_to_h = geometric_to_geopotential


def geopotential_to_geometric(h, r0):
    """Geometric altitude from a geopotential altitude."""
    return _run("mp_geopotential_to_geometric", h, r0)


h_to_z = geopotential_to_geometric


def gravity(z, g0, r0):
    """Gravity at a geometric altitude."""
    return _run("mp_gravity", z, g0, r0)


def check_altitude(alt, r0, geometric):
    """Resolve `alt` to ``(z, h)`` following `geometric`."""
    b = Batch(alt, tail=(2,))
    out = np.empty((b.n, 2), dtype=np.float64)
    fn("mp_check_altitude")(
        addr(b.inp[0]), float(r0), int(bool(geometric)), addr(out), b.n
    )
    out = out.reshape(b.shape + (2,))
    if b.scalar:
        return out[0], out[1]
    return out[..., 0], out[..., 1]
