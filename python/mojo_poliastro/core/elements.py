"""Port of `poliastro.core.elements` -- see `src/mojopoliastro/elements.mojo`."""

import numpy as np

from .._lib import addr, f64, fn

__all__ = [
    "eccentricity_vector", "circular_velocity", "rv_pqw",
    "coe_rotation_matrix", "coe2rv", "coe2rv_many", "coe2mee", "rv2coe",
    "mee2coe", "mee2rv",
]


def _scalar_or_array(out, b, tail):
    if b.scalar:
        return out[0]
    return out.reshape(b.shape + tail)


def _k(k):
    if np.ndim(k) == 0:
        return float(k)
    raise TypeError("k must be a scalar; use coe2rv_many for per-orbit k")


def eccentricity_vector(k, r, v):
    r"""Eccentricity vector.

    .. math::

       \vec{e} = \frac{1}{\mu} \left(
           \left( v^{2} - \frac{\mu}{r}\right ) \vec{r}  - (\vec{r} \cdot \vec{v})\vec{v}
       \right)
    """
    return _vectors("mp_eccentricity_vector", _k(k), r, v)


def circular_velocity(k, a):
    r"""Circular velocity for a given body.

    .. math::

       v = \sqrt{\frac{\mu}{a}}
    """
    from .._dispatch import Batch
    b = Batch(a)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_circular_velocity")(_k(k), addr(b.inp[0]), addr(out), b.n)
    return _scalar_or_array(out, b, ())


def rv_pqw(k, p, ecc, nu):
    r"""Position and velocity vectors in the perifocal frame."""
    from .._dispatch import Batch
    b = Batch(p, ecc, nu)
    out = np.empty((b.n, 2, 3), dtype=np.float64)
    fn("mp_rv_pqw")(
        _k(k), addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(out), b.n
    )
    return _scalar_or_array(out, b, (2, 3))


def coe_rotation_matrix(inc, raan, argp):
    """Rotation matrix from the perifocal frame to the IJK frame."""
    from .._dispatch import Batch
    b = Batch(inc, raan, argp)
    out = np.empty((b.n, 3, 3), dtype=np.float64)
    fn("mp_coe_rotation_matrix")(
        addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(out), b.n
    )
    return _scalar_or_array(out, b, (3, 3))


def coe2rv(k, p, ecc, inc, raan, argp, nu):
    r"""Converts from classical orbital elements to state vectors.

    Returns ``(r_ijk, v_ijk)``, each of shape ``broadcast_shape + (3,)``.
    """
    from .._dispatch import Batch
    b = Batch(p, ecc, inc, raan, argp, nu)
    out = np.empty((b.n, 2, 3), dtype=np.float64)
    fn("mp_coe2rv")(
        _k(k), addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(b.inp[3]),
        addr(b.inp[4]), addr(b.inp[5]), addr(out), b.n
    )
    out = b.rows(out)
    if b.scalar:
        return out[0].copy(), out[1].copy()
    return out[..., 0, :], out[..., 1, :]


def coe2rv_many(k, p, ecc, inc, raan, argp, nu):
    r"""Converts from classical orbital elements to state vectors.

    Upstream's batched variant, which takes a per-orbit gravitational
    parameter and runs its loop under numba's ``parallel=True``.  Returns
    ``(rr, vv)``, each of shape ``broadcast_shape + (3,)``.
    """
    from .._dispatch import Batch
    b = Batch(k, p, ecc, inc, raan, argp, nu)
    out = np.empty((b.n, 2, 3), dtype=np.float64)
    fn("mp_coe2rv_many")(
        addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(b.inp[3]),
        addr(b.inp[4]), addr(b.inp[5]), addr(b.inp[6]), addr(out), b.n
    )
    return out[..., 0, :], out[..., 1, :]


def coe2mee(p, ecc, inc, raan, argp, nu):
    r"""Converts from classical orbital elements to modified equinoctial ones."""
    from .._dispatch import Batch
    b = Batch(p, ecc, inc, raan, argp, nu)
    out = np.empty((b.n, 6), dtype=np.float64)
    codes = np.empty(b.n, dtype=np.int64)
    status = fn("mp_coe2mee")(
        addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(b.inp[3]),
        addr(b.inp[4]), addr(b.inp[5]), addr(out), addr(codes), b.n
    )
    if status == -1:
        raise ValueError(
            "Cannot compute modified equinoctial set for 180 degrees orbit "
            "inclination due to `h` and `k` singularity."
        )
    out = b.rows(out)
    if b.scalar:
        return tuple(out.tolist())
    return tuple(out[..., j] for j in range(6))


def rv2coe(k, r, v, tol=1e-8):
    r"""Converts from state vectors to classical orbital elements."""
    from .._dispatch import Batch
    a = np.asarray(r, dtype=np.float64)
    b_ = np.asarray(v, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape[:-1], b_.shape[:-1])
    scalar = not shape
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    ra = f64(np.broadcast_to(a, shape + (3,)).reshape(-1))
    va = f64(np.broadcast_to(b_, shape + (3,)).reshape(-1))
    out = np.empty((n, 6), dtype=np.float64)
    fn("mp_rv2coe")(_k(k), addr(ra), addr(va), float(tol), addr(out), n)
    out = out.reshape(shape + (6,))
    if scalar:
        return tuple(out.reshape(6).tolist())
    return tuple(out[..., j] for j in range(6))


def mee2coe(p, f, g, h, k, L):
    r"""Converts from modified equinoctial to classical orbital elements."""
    from .._dispatch import Batch
    b = Batch(p, f, g, h, k, L)
    out = np.empty((b.n, 6), dtype=np.float64)
    fn("mp_mee2coe")(
        addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(b.inp[3]),
        addr(b.inp[4]), addr(b.inp[5]), addr(out), b.n
    )
    out = b.rows(out)
    if b.scalar:
        return tuple(out.tolist())
    return tuple(out[..., j] for j in range(6))


def mee2rv(p, f, g, h, k, L):
    """Position and velocity vector from modified equinoctial elements."""
    from .._dispatch import Batch
    b = Batch(p, f, g, h, k, L)
    out = np.empty((b.n, 2, 3), dtype=np.float64)
    fn("mp_mee2rv")(
        addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]), addr(b.inp[3]),
        addr(b.inp[4]), addr(b.inp[5]), addr(out), b.n
    )
    out = b.rows(out)
    if b.scalar:
        return out[0].copy(), out[1].copy()
    return out[..., 0, :], out[..., 1, :]


def _vectors(symbol, k, r, v):
    a = np.asarray(r, dtype=np.float64)
    b_ = np.asarray(v, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape[:-1], b_.shape[:-1])
    scalar = not shape
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    ra = f64(np.broadcast_to(a, shape + (3,)).reshape(-1))
    va = f64(np.broadcast_to(b_, shape + (3,)).reshape(-1))
    out = np.empty((n, 3), dtype=np.float64)
    fn(symbol)(k, addr(ra), addr(va), addr(out), n)
    out = out.reshape(shape + (3,))
    return out[0] if scalar else out
