"""mojo-poliastro against upstream poliastro on the same data.

    pixi run bench

Upstream is numba-jitted but scalar: most of `poliastro.core` cannot be
handed an array at all, so the "upstream" column is a Python loop over the
same batch.  That is the honest comparison for these kernels -- it is what a
caller of upstream actually pays.  Where upstream *can* take an array, it is
timed that way instead (noted per row).

Read the speedups with the dispatch cost in mind.  Measured on this machine, a
single call into an upstream jitted scalar kernel costs about 45 us, while the
arithmetic inside it costs a few nanoseconds: the "upstream" column is
dominated by the Python -> numba dispatch, not by the kernel.  A caller who
wrote their own `@njit` wrapper around upstream's kernels would amortise that
away and land far closer to this port's numbers, so the large factors below
are mostly what this port removes rather than arithmetic it outruns.  The
absolute times for both sides are the sound comparison; the ratio is not a
measure of kernel quality.

Reports the honest number, including the rows where upstream wins.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_poliastro  # noqa: E402  (loads the .so before numba's runtime)
import mojo_poliastro._math.special as our_special  # noqa: E402
import mojo_poliastro.core.angles as our_angles  # noqa: E402
import mojo_poliastro.core.elements as our_elements  # noqa: E402
import mojo_poliastro.core.iod as our_iod  # noqa: E402
import mojo_poliastro.core.propagation as our_prop  # noqa: E402
import poliastro._math.special as their_special  # noqa: E402
import poliastro.core.angles as their_angles  # noqa: E402
import poliastro.core.elements as their_elements  # noqa: E402
import poliastro.core.iod as their_iod  # noqa: E402
import poliastro.core.propagation as their_prop  # noqa: E402

# `poliastro.core.propagation.__init__` rebinds the package attribute
# `farnocchia` to the function it re-exports, so `import ... .farnocchia as x`
# binds the function.  Go through sys.modules for the module.
import poliastro.core.propagation.farnocchia  # noqa: E402

their_farnocchia = sys.modules["poliastro.core.propagation.farnocchia"]

K = 398600.4418


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _py(x):
    """A plain Python float, or the ndarray itself when the argument is a
    vector: upstream's `rv2coe` needs real arrays because it uses `@` and
    `np.cross`, and numba's dispatcher is markedly slower on np.float64."""
    return x if isinstance(x, np.ndarray) else float(x)


def loop(fn, *args):
    """What a caller of scalar-only upstream actually has to write."""
    n = max(np.asarray(a).shape[0] for a in args if np.ndim(a))
    cols = [np.broadcast_to(np.asarray(a, dtype=float), (n,) + np.shape(a)[1:])
            for a in args]

    def run():
        # Python floats, not numpy scalars: numba's dispatcher is markedly
        # slower on np.float64 for the nested-dispatch kernels, and handing it
        # the scalar a caller would write is the fair comparison.
        return [fn(*[_py(c[i]) for c in cols]) for i in range(n)]

    return run


CASES = []


def case(name, note="", counted=True):
    """`counted=False` keeps a row out of the total: it exists to show a
    reference-implementation quirk, not to measure the kernel."""
    def deco(fn):
        CASES.append((name, note, fn, counted))
        return fn
    return deco


def coe(n, seed=0):
    rng = np.random.default_rng(seed)
    return (
        np.full(n, 11000.0),
        rng.uniform(0.0, 0.8, n),
        rng.uniform(0.05, 3.0, n),
        rng.uniform(0.0, 6.0, n),
        rng.uniform(0.0, 6.0, n),
        rng.uniform(-3.0, 3.0, n),
    )


@case("nu_to_E (1M anomalies)", "upstream: Python loop")
def _():
    n = 1_000_000
    nu = np.linspace(-3.0, 3.0, n)
    ecc = np.full(n, 0.3)
    return (
        lambda: our_angles.nu_to_E(nu, ecc),
        loop(their_angles.nu_to_E, nu, ecc),
    )


@case("M_to_E Newton solve (1M)", "upstream: Python loop")
def _():
    n = 1_000_000
    M = np.linspace(-3.0, 3.0, n)
    ecc = np.full(n, 0.3)
    return (
        lambda: our_angles.M_to_E(M, ecc),
        loop(their_angles.M_to_E, M, ecc),
    )


@case("E_to_M (1M)", "upstream: Python loop")
def _():
    n = 1_000_000
    E = np.linspace(-3.0, 3.0, n)
    ecc = np.full(n, 0.3)
    return (
        lambda: our_angles.E_to_M(E, ecc),
        loop(their_angles.E_to_M, E, ecc),
    )


@case("stumpff_c2 (1M)", "upstream: Python loop")
def _():
    n = 1_000_000
    psi = np.linspace(-4.0, 4.0, n)
    return (
        lambda: our_special.stumpff_c2(psi),
        loop(their_special.stumpff_c2, psi),
    )


@case("hyp2f1b (200k)", "upstream: Python loop")
def _():
    n = 200_000
    x = np.linspace(-0.9, 0.9, n)
    return (
        lambda: our_special.hyp2f1b(x),
        loop(their_special.hyp2f1b, x),
    )


@case("coe2rv (1M orbits)", "upstream: Python loop")
def _():
    n = 1_000_000
    p, ecc, inc, raan, argp, nu = coe(n)
    return (
        lambda: our_elements.coe2rv(K, p, ecc, inc, raan, argp, nu),
        loop(their_elements.coe2rv, K, p, ecc, inc, raan, argp, nu),
    )


@case("coe2rv_many (200k orbits)", "ours: parallel, upstream: loop")
def _():
    n = 200_000
    p, ecc, inc, raan, argp, nu = coe(n, 1)
    k = np.full(n, K)

    def theirs():
        for i in range(n):
            their_elements.coe2rv(k[i], p[i], ecc[i], inc[i], raan[i], argp[i], nu[i])

    return (
        lambda: our_elements.coe2rv_many(k, p, ecc, inc, raan, argp, nu),
        theirs,
    )


@case("rv2coe (1M)", "upstream: Python loop")
def _():
    n = 1_000_000
    p, ecc, inc, raan, argp, nu = coe(n, 2)
    r, v = their_elements.coe2rv(K, p[0], ecc[0], inc[0], raan[0], argp[0], nu[0])
    r = np.ascontiguousarray(np.broadcast_to(np.asarray(r), (n, 3)))
    v = np.ascontiguousarray(np.broadcast_to(np.asarray(v), (n, 3)))
    return (
        lambda: our_elements.rv2coe(K, r, v),
        loop(their_elements.rv2coe, K, r, v),
    )


@case("coe2mee (1M)", "upstream: Python loop")
def _():
    n = 1_000_000
    p, ecc, inc, raan, argp, nu = coe(n, 3)
    return (
        lambda: our_elements.coe2mee(p, ecc, inc, raan, argp, nu),
        loop(their_elements.coe2mee, p, ecc, inc, raan, argp, nu),
    )


@case("mee2rv (500k)", "reference is upstream transcribed: upstream's own source is unrunnable", counted=False)
def _():
    n = 500_000
    rng = np.random.default_rng(4)
    p = np.full(n, 11000.0)
    f = rng.uniform(-0.3, 0.3, n)
    g = rng.uniform(-0.3, 0.3, n)
    h = np.full(n, 0.1)
    k = np.full(n, 0.2)
    L = np.linspace(0.0, 6.0, n)

    def theirs():
        for i in range(n):
            ref_mee2rv(float(p[i]), float(f[i]), float(g[i]), float(h[i]),
                       float(k[i]), float(L[i]))

    return (lambda: our_elements.mee2rv(p, f, g, h, k, L), theirs)


@case("Izzo Lambert (200k transfers)", "upstream: Python loop")
def _():
    n = 200_000
    rng = np.random.default_rng(5)
    r0 = rng.normal(size=(n, 3))
    r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
    r0 *= rng.uniform(7000.0, 42000.0, (n, 1))
    r1 = rng.normal(size=(n, 3))
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r1 *= rng.uniform(7000.0, 42000.0, (n, 1))
    tof = rng.uniform(600.0, 20000.0, n)

    def theirs():
        for i in range(n):
            their_iod.izzo(K, r0[i], r1[i], tof[i], 0, True, True, 35, 1e-8)

    return (lambda: our_iod.izzo(K, r0, r1, tof, 0, True, True, 35, 1e-8), theirs)


@case("Vallado Lambert (200k transfers)", "upstream: Python loop")
def _():
    # Vallado bisects psi in [-4 pi^2, 4 pi^2], so upstream only converges
    # for a window of time of flight; outside it upstream raises too.  The
    # benchmark stays inside the window rather than timing a failure.
    n = 200_000
    r0e = 7000.0
    period = 2 * np.pi * np.sqrt(r0e**3 / K)
    rng = np.random.default_rng(6)
    dnu = rng.uniform(0.05, 3.0, n)
    r0 = np.zeros((n, 3))
    r0[:, 0] = r0e
    r1 = np.stack([r0e * np.cos(dnu), r0e * np.sin(dnu), np.zeros(n)], -1)
    tof = rng.uniform(0.3, 0.45, n) * period

    def theirs():
        for i in range(n):
            their_iod.vallado(K, r0[i], r1[i], tof[i], 0, True, True, 100, 1e-8)

    return (
        lambda: our_iod.vallado(K, r0, r1, tof, 0, True, True, 100, 1e-8),
        theirs,
    )


def _state(n, tofs, seed=7):
    rng = np.random.default_rng(seed)
    p, ecc, inc, raan, argp, nu = coe(n, seed)
    r, v = their_elements.coe2rv(K, p[0], ecc[0], inc[0], raan[0], argp[0], nu[0])
    r = np.ascontiguousarray(np.broadcast_to(np.asarray(r), (n, 3)))
    v = np.ascontiguousarray(np.broadcast_to(np.asarray(v), (n, 3)))
    return r, v, tofs


@case("farnocchia propagate (500k)", "upstream: Python loop")
def _():
    n = 500_000
    r, v, tofs = _state(n, np.full(n, 600.0))
    return (
        lambda: our_prop.farnocchia(K, r, v, tofs),
        loop(their_prop.farnocchia, K, r, v, tofs),
    )


@case("markley propagate (500k)", "upstream: Python loop")
def _():
    n = 500_000
    r, v, tofs = _state(n, np.full(n, 600.0), 8)
    return (
        lambda: our_prop.markley(K, r, v, tofs),
        loop(their_prop.markley, K, r, v, tofs),
    )


@case("gooding propagate (500k)", "upstream: Python loop")
def _():
    n = 500_000
    r, v, tofs = _state(n, np.full(n, 600.0), 9)
    return (
        lambda: our_prop.gooding(K, r, v, tofs),
        loop(their_prop.gooding, K, r, v, tofs),
    )


@case("delta_t_from_nu (1M)", "upstream: Python loop, delta passed explicitly")
def _():
    n = 1_000_000
    nu = np.linspace(-1.5, 1.5, n)
    ecc = np.full(n, 0.5)
    return (
        lambda: our_prop.delta_t_from_nu(nu, ecc, K, 8000.0),
        loop(their_farnocchia.delta_t_from_nu, nu, ecc, K, 8000.0, 0.01),
    )


@case("delta_t_from_nu (1M, upstream default arg)", "numba default-arg dispatch dominates, not the kernel", counted=False)
def _():
    n = 1_000_000
    nu = np.linspace(-1.5, 1.5, n)
    ecc = np.full(n, 0.5)
    return (
        lambda: our_prop.delta_t_from_nu(nu, ecc, K, 8000.0),
        loop(their_farnocchia.delta_t_from_nu, nu, ecc, K, 8000.0),
    )


def ref_mee2rv(p, f, g, h, k, L):
    """Upstream's `mee2rv` transcribed; its own source cannot be executed."""
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w
    s2 = 1 + h**2 + k**2
    a2 = h**2 - k**2
    rx = (r / s2) * (np.cos(L) + a2**2 * np.cos(L) + 2 * h * k * np.sin(L))
    ry = (r / s2) * (np.sin(L) - a2**2 * np.sin(L) + 2 * h * k * np.cos(L))
    rz = (2 * r / s2) * (h * np.sin(L) - k * np.cos(L))
    vx = (-1 / s2) * np.sqrt(k / p) * (
        np.sin(L) + a2 * np.sin(L) - 2 * h * k * np.cos(L) + g
        - 2 * f * h * k + a2 * g
    )
    vy = (-1 / s2) * np.sqrt(k / p) * (
        -np.cos(L) + a2 * np.cos(L) + 2 * h * k * np.sin(L) - f
        + 2 * g * h * k + a2 * f
    )
    vz = (2 / s2) * np.sqrt(k / p) * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
    return np.array([rx, ry, rz]), np.array([vx, vy, vz])


def _host() -> None:
    """Everything a reader needs to judge the numbers below.

    The load average is here on purpose.  Both columns are wall clock, so a
    shared machine inflates them; the upstream column is almost pure
    interpreter dispatch and inflates proportionally more than this port's,
    which flatters the ratio.  A reader who knows the box was busy can read
    the row honestly; a reader who does not cannot.
    """
    import subprocess

    print(f"machine : {platform.platform()}")
    print(f"python  : {platform.python_version()}  numpy {np.__version__}")
    print(f"cpus    : {len(os.sched_getaffinity(0))} usable "
          f"(load average {os.getloadavg()[0]:.0f})")
    try:
        ver = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
        print(f"mojo    : {ver}")
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    _host()
    print()
    print("| kernel | mojo-poliastro | upstream poliastro | speedup | note |")
    print("| --- | ---: | ---: | ---: | --- |")
    total_ours = 0.0
    total_theirs = 0.0
    counted_rows = 0
    for name, note, build, counted in CASES:
        ours, theirs = build()
        t_ours = timeit(ours)
        t_theirs = timeit(theirs, repeat=1 if "1M" in name or "500k" in name else 3)
        if counted:
            total_ours += t_ours
            total_theirs += t_theirs
            counted_rows += 1
        speed = t_theirs / t_ours if t_ours > 0 else math.inf
        print(
            f"| {name} | {t_ours * 1e3:.2f} ms | {t_theirs * 1e3:.2f} ms "
            f"| {speed:.2f}x | {note} |"
        )
    print(
        f"| **total ({counted_rows} counted rows)** | **{total_ours * 1e3:.2f} ms** "
        f"| **{total_theirs * 1e3:.2f} ms** "
        f"| **{total_theirs / total_ours:.2f}x** | |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
