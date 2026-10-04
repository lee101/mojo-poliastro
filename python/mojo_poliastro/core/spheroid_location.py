"""Port of `poliastro.core.spheroid_location`.

Upstream takes 3-vectors, not sequences of them, so a 1-D length-3 array is
one vector here too; an ``(n, 3)`` array is the batched form.
"""

import numpy as np

from .._dispatch import MixedBatch
from .._lib import addr, fn

__all__ = [
    "cartesian_cords", "f", "N", "tangential_vecs", "radius_of_curvature",
    "distance", "is_visible", "cartesian_to_ellipsoidal",
]


def _run(scalars, vectors, tail, call):
    b = MixedBatch(scalars, vectors, tail=tail)
    call(b)
    return b.out[0] if b.scalar else b.out


def cartesian_cords(_a, _c, _lon, _lat, _h):
    r"""Cartesian coordinates of a spheroid surface point."""
    return _run(
        (_a, _c, _lon, _lat, _h), (), (3,),
        lambda b: fn("mp_cartesian_cords")(
            addr(b.s[0]), addr(b.s[1]), addr(b.s[2]), addr(b.s[3]),
            addr(b.s[4]), addr(b.out), b.n
        ),
    )


def f(_a, _c):
    """Flattening of the spheroid."""
    return _run(
        (_a, _c), (), (),
        lambda b: fn("mp_f")(addr(b.s[0]), addr(b.s[1]), addr(b.out), b.n),
    )


def N(a, b, c, cartesian_cords):
    """Outward unit normal at a point on the spheroid."""
    return _run(
        (a, b, c), (cartesian_cords,), (3,),
        lambda b: fn("mp_N")(
            addr(b.s[0]), addr(b.s[1]), addr(b.s[2]), addr(b.v[0]),
            addr(b.out), b.n
        ),
    )


def tangential_vecs(N):
    """Orthonormal tangent vectors ``(u, v)`` of the spheroid at `N`."""
    b = MixedBatch((), (N,), tail=(6,))
    fn("mp_tangential_vecs")(addr(b.v[0]), addr(b.out), b.n)
    if b.scalar:
        return b.out[0, :3].copy(), b.out[0, 3:].copy()
    return b.out[..., :3], b.out[..., 3:]


def radius_of_curvature(_a, _c, _lat):
    """Radius of curvature of the spheroid at the given latitude."""
    return _run(
        (_a, _c, _lat), (), (),
        lambda b: fn("mp_radius_of_curvature")(
            addr(b.s[0]), addr(b.s[1]), addr(b.s[2]), addr(b.out), b.n
        ),
    )


def distance(cartesian_cords, px, py, pz):
    """Distance from a spheroid point to a point in cartesian coordinates."""
    p = np.stack(np.broadcast_arrays(px, py, pz), axis=-1)
    return _run(
        (), (cartesian_cords, p), (),
        lambda b: fn("mp_distance")(
            addr(b.v[0]), addr(b.v[1]), addr(b.out), b.n
        ),
    )


def is_visible(cartesian_cords, px, py, pz, N):
    """Whether a point is above the spheroid horizon at `cartesian_cords`."""
    p = np.stack(np.broadcast_arrays(px, py, pz), axis=-1)
    out = _run(
        (), (cartesian_cords, p, N), (),
        lambda b: fn("mp_is_visible")(
            addr(b.v[0]), addr(b.v[1]), addr(b.v[2]), addr(b.out), b.n
        ),
    )
    return bool(out) if np.ndim(out) == 0 else out.astype(bool)


def cartesian_to_ellipsoidal(_a, _c, x, y, z):
    """Geodetic latitude, longitude and height of a cartesian point."""
    b = MixedBatch((_a, _c, x, y, z), (), tail=(3,))
    fn("mp_cartesian_to_ellipsoidal")(
        addr(b.s[0]), addr(b.s[1]), addr(b.s[2]), addr(b.s[3]),
        addr(b.s[4]), addr(b.out), b.n
    )
    if b.scalar:
        return tuple(b.out[0].tolist())
    return tuple(b.out[..., j] for j in range(3))
