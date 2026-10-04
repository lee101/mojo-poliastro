"""Shared helpers for the parity tests.

Most of upstream `poliastro.core` is `@njit`-decorated and typed for scalars
only, so a vectorised Mojo call cannot be compared against one upstream call:
upstream has to be looped.  `us` does that, and `assert_close` compares the
two with a scale-relative tolerance.

Why 1e-9 and not 1e-15.  Mojo's `std.math` carries a fast approximation of the
transcendental functions: `sin`, `cos`, `tan`, `atan` and `atan2` come out
bit-identical to glibc, and most of the suite sits at 1e-16 or better, but
`pow`, `exp` and `log` differ, and Mojo's `pow` disagrees with glibc's by
around 1e-9 relative.  `M_to_D` evaluates `B + sqrt(1 + B**2)`, which
cancels catastrophically, so that 1e-9 becomes 1.7e-10 on its worst input
here.  Three places document their own, looser tolerance and pass it in: the
Jacchia77 composition recursion (two chained recurrences over up to 2500
altitude steps, each seeded by `exp`), the farnocchia `S_x` series, which is
compared against a re-derivation rather than against upstream's own truncated
sum, and the Lagrange coefficients.

`parity_report` prints the worst error each label actually reached at the end
of the session, so the headroom above is visible rather than asserted.  On
this build the worst case across all 347 labels is 4.4e-09 against its own
1e-07, and the worst against the 1e-9 default is 1.7e-10.
"""

import sys

import numpy as np

RTOL = 1e-9


def us(fn, *args):
    """Loop an upstream scalar-only kernel over a leading-axis batch."""
    arrs = [np.asarray(a, dtype=np.float64) for a in args]
    n = 1
    for a in arrs:
        if a.ndim:
            n = a.shape[0]
            break
    cols = []
    for a in arrs:
        if a.ndim == 0:
            cols.append(np.full(n, float(a)))
        else:
            cols.append(np.broadcast_to(a, (n,) + a.shape[1:]))
    rows = [fn(*[c[i] for c in cols]) for i in range(n)]
    if isinstance(rows[0], tuple):
        return tuple(
            np.stack([np.asarray(r[j], dtype=np.float64) for r in rows])
            for j in range(len(rows[0]))
        )
    return np.stack([np.asarray(r, dtype=np.float64) for r in rows])


def err(a, b):
    """Max |a - b| normalised by the scale of `b`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(b))) if b.size else 1.0)
    return float(np.max(d) / scale)


def assert_close(ours, theirs, rtol=RTOL, label=""):
    e = err(ours, theirs)
    # `parity_report` records the worst error each label reached so the
    # session can print it; it is optional because `bench` and ad-hoc scripts
    # import this module without the test tree on the path.
    report = sys.modules.get("parity_report")
    if report is not None:
        report.record(label, e, rtol)
    assert e <= rtol, f"{label}: relative error {e:.3e} exceeds {rtol:.3e}"
    return e
