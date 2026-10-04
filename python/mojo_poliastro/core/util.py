"""Port of `poliastro.core.util` -- see `src/mojopoliastro/util.mojo`."""

import numpy as np

from .._dispatch import Batch
from .._lib import addr, fn

__all__ = ["rotation_matrix", "alinspace", "spherical_to_cartesian",
           "planetocentric_to_AltAz"]

TWO_PI = 2 * np.pi


def rotation_matrix(angle, axis):
    """Elementary rotation matrix about `axis` (0, 1 or 2) by `angle` radians."""
    b = Batch(angle)
    out = np.empty((b.n, 3, 3), dtype=np.float64)
    fn("mp_rotation_matrix")(addr(b.inp[0]), int(axis), addr(out), b.n)
    if b.scalar:
        return out[0]
    return out.reshape(b.shape + (3, 3))


def alinspace(start, stop=None, num=50, endpoint=True):
    """Increasing, evenly spaced angular values over a specified interval."""
    stop_arg = np.nan if stop is None else float(stop)
    # upstream builds `num + 1` samples and drops the last one when
    # `endpoint` is False; the kernel writes all `count` of them.
    count = int(num) if endpoint else int(num) + 1
    out = np.empty(count, dtype=np.float64)
    fn("mp_alinspace")(
        float(start), stop_arg, int(num), 1 if endpoint else 0, addr(out)
    )
    return out if endpoint else out[:-1]


def spherical_to_cartesian(v):
    r"""Cartesian coordinates from spherical coordinates (norm, colat, long).

    .. math::

       v = norm \cdot \begin{bmatrix}
       \sin(colat)\cos(long)\\
       \sin(colat)\sin(long)\\
       \cos(colat)\\
       \end{bmatrix}
    """
    return _vec_map("mp_spherical_to_cartesian", v)


def planetocentric_to_AltAz(theta, phi):
    r"""Transformation matrix from planetocentric to Altitude-Azimuth.

    .. math::
       t\_matrix = \begin{bmatrix}
       -\sin(theta) & \cos(theta) & 0\\
       -\sin(phi)\cdot\cos(theta) & -\sin(phi)\cdot\sin(theta) & \cos(phi)\\
       \cos(phi)\cdot\cos(theta) & \cos(phi)\cdot\sin(theta) & \sin(phi)
       \end{bmatrix}
    """
    b = Batch(theta, phi)
    out = np.empty((b.n, 3, 3), dtype=np.float64)
    fn("mp_planetocentric_to_AltAz")(
        addr(b.inp[0]), addr(b.inp[1]), addr(out), b.n
    )
    if b.scalar:
        return out[0]
    return out.reshape(b.shape + (3, 3))


def _vec_map(symbol, v):
    a = np.asarray(v, dtype=np.float64)
    scalar = a.ndim == 1
    flat = a.reshape(1, 3) if scalar else a.reshape(-1, 3)
    n = flat.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    fn(symbol)(addr(np.ascontiguousarray(flat)), addr(out), n)
    if scalar:
        return out[0]
    return out.reshape(a.shape)
