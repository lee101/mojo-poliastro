"""Port of `poliastro.core.propagation.farnocchia`."""

import numpy as np

from ..._dispatch import Batch, map2
from ..._lib import addr, f64, fn

__all__ = [
    "S_x", "dS_x_alt", "d2S_x_alt", "D_to_M_near_parabolic",
    "M_to_D_near_parabolic", "delta_t_from_nu", "nu_from_delta_t",
    "farnocchia_coe", "farnocchia",
]


def _pair_with_atol(symbol, ecc, x, atol):
    b = Batch(ecc, x)
    out = np.empty(b.n, dtype=np.float64)
    fn(symbol)(addr(b.inp[0]), addr(b.inp[1]), float(atol), addr(out), b.n)
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def S_x(ecc, x, atol=1e-12):
    r"""The near-parabolic series :math:`S(x; e)`."""
    return _pair_with_atol("mp_S_x", ecc, x, atol)


def dS_x_alt(ecc, x, atol=1e-12):
    r"""Derivative of :func:`S_x` with respect to :math:`x`."""
    return _pair_with_atol("mp_dS_x_alt", ecc, x, atol)


def d2S_x_alt(ecc, x, atol=1e-12):
    r"""Second derivative of :func:`S_x` with respect to :math:`x`."""
    return _pair_with_atol("mp_d2S_x_alt", ecc, x, atol)


def D_to_M_near_parabolic(D, ecc):
    r"""Mean anomaly from parabolic anomaly, near a parabolic orbit."""
    return map2("mp_D_to_M_near_parabolic", D, ecc)


def M_to_D_near_parabolic(M, ecc, tol=1.48e-08, maxiter=50):
    r"""Parabolic anomaly from mean anomaly, near a parabolic orbit."""
    b = Batch(M, ecc)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_M_to_D_near_parabolic")(
        addr(b.inp[0]), addr(b.inp[1]), float(tol), int(maxiter), addr(out), b.n
    )
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def delta_t_from_nu(nu, ecc, k=1.0, q=1.0, delta=0.01):
    r"""Time since periapsis passage from the true anomaly."""
    b = Batch(nu, ecc)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_delta_t_from_nu")(
        addr(b.inp[0]), addr(b.inp[1]), float(k), float(q), float(delta),
        addr(out), b.n
    )
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def nu_from_delta_t(delta_t, ecc, k=1.0, q=1.0, delta=0.01):
    r"""True anomaly from the time since periapsis passage."""
    b = Batch(delta_t, ecc)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_nu_from_delta_t")(
        addr(b.inp[0]), addr(b.inp[1]), float(k), float(q), float(delta),
        addr(out), b.n
    )
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def farnocchia_coe(k, p, ecc, inc, raan, argp, nu, tof):
    r"""True anomaly after `tof` seconds, from classical orbital elements."""
    b = Batch(p, ecc, inc, raan, argp, nu, tof)
    out = np.empty(b.n, dtype=np.float64)
    fn("mp_farnocchia_coe")(
        float(k), addr(b.inp[0]), addr(b.inp[1]), addr(b.inp[2]),
        addr(b.inp[3]), addr(b.inp[4]), addr(b.inp[5]), addr(b.inp[6]),
        addr(out), b.n
    )
    return float(out[0]) if b.scalar else out.reshape(b.shape)


def farnocchia(k, r0, v0, tof):
    r"""State vector after `tof` seconds, from a state vector."""
    return _propagate("mp_farnocchia_rv", k, r0, v0, tof)


def _propagate(symbol, k, r0, v0, tof, *tail, status_is_error=False):
    a = np.asarray(r0, dtype=np.float64)
    b_ = np.asarray(v0, dtype=np.float64)
    t = np.asarray(tof, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape[:-1], b_.shape[:-1], t.shape)
    scalar = not shape
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    r = f64(np.broadcast_to(a, shape + (3,)).reshape(-1))
    v = f64(np.broadcast_to(b_, shape + (3,)).reshape(-1))
    tt = f64(np.broadcast_to(t, shape).reshape(-1))
    out = np.empty((n, 2, 3), dtype=np.float64)
    fn(symbol)(float(k), addr(r), addr(v), addr(tt), *tail, addr(out), n)
    out = out.reshape(shape + (2, 3))
    if scalar:
        return out[0].copy(), out[1].copy()
    return out[..., 0, :], out[..., 1, :]
