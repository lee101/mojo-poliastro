"""Port of `poliastro.core.iod` -- see `src/mojopoliastro/iod.mojo`.

Upstream signals every failure mode with a `raise`.  The Mojo kernels return a
status integer instead, because an `abi("C")` entry point cannot raise; this
module turns the status back into upstream's exception, with upstream's
message.
"""

import numpy as np

from .._lib import addr, f64, fn

__all__ = ["vallado", "izzo"]

_MULTIREV = (
    "Multi-revolution scenario not supported for Vallado. See issue "
    "https://github.com/poliastro/poliastro/issues/858"
)
_COLLINEAR = "Lambert solution cannot be computed for collinear vectors"
_INFEASIBLE = "No feasible solution, try lower M"


def _raise_for(status, k):
    if status == 0:
        return
    if status == -1:
        raise NotImplementedError(_MULTIREV)
    if status == -2:
        raise RuntimeError("Cannot compute orbit, phase angle is 180 degrees")
    if status == -3:
        raise RuntimeError("Maximum number of iterations reached")
    if status == -4:
        raise ValueError(_COLLINEAR)
    if status == -5:
        raise ValueError(_INFEASIBLE)
    raise RuntimeError("Lambert solver failed")


def _lambert(symbol, k, r0, r, tof, M, prograde, lowpath, numiter, rtol):
    a = np.asarray(r0, dtype=np.float64)
    b = np.asarray(r, dtype=np.float64)
    t = np.asarray(tof, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1], t.shape)
    scalar = not shape
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    ra = f64(np.broadcast_to(a, shape + (3,)).reshape(-1))
    rb = f64(np.broadcast_to(b, shape + (3,)).reshape(-1))
    ta = f64(np.broadcast_to(t, shape).reshape(-1))
    out = np.empty((n, 2, 3), dtype=np.float64)
    # One status slot per element: the batch is solved in parallel above the
    # kernel's threshold, and a worker cannot report back through a local.
    codes = np.empty(n, dtype=np.int64)
    status = fn(symbol)(
        float(k), addr(ra), addr(rb), addr(ta), int(M), int(bool(prograde)),
        int(bool(lowpath)), int(numiter), float(rtol), addr(out), addr(codes), n
    )
    _raise_for(status, k)
    out = out.reshape(shape + (2, 3))
    if scalar:
        return out[0].copy(), out[1].copy()
    return out[..., 0, :], out[..., 1, :]


def vallado(k, r0, r, tof, M, prograde, lowpath, numiter, rtol):
    r"""Solves the Lambert's problem.

    The algorithm returns the initial velocity vector and the final one.
    """
    return _lambert(
        "mp_vallado_iod", k, r0, r, tof, M, prograde, lowpath, numiter, rtol
    )


def izzo(k, r1, r2, tof, M, prograde, lowpath, numiter, rtol):
    """Solves the Lambert problem using the Izzo algorithm."""
    return _lambert(
        "mp_izzo", k, r1, r2, tof, M, prograde, lowpath, numiter, rtol
    )
