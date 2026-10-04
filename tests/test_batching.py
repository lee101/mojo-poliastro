"""Tests for the batched dispatch in `src/capi.mojo`.

Three things in the export layer are not exercised by the per-function parity
tests, because those all call with a handful of elements:

* the SIMD body and its scalar tail, which only differ when the batch length
  is not a whole number of vectors;
* the choice between the serial loop and `max.algorithm.parallelize`, which
  switches at two different row counts;
* the per-element status buffer the four status-returning exports use, since
  a parallel worker cannot report back through a captured local.

Every assertion here is a property of the batched layer, not of a kernel: the
SIMD body must reproduce the scalar kernel lane for lane, and the parallel and
serial paths must produce the same numbers as each other.
"""

import numpy as np
import pytest

from parity import assert_close

import mojo_poliastro._math.special as our_special
import mojo_poliastro.core.angles as our_angles
import mojo_poliastro.core.elements as our_elements
import mojo_poliastro.core.iod as our_iod

# `simd_width_of[DType.float64]()` on the build host.  The export layer gives
# each work item exactly W elements, so these are the lengths that leave a
# partial vector behind.  A build on a wider host would still pass, just with
# a different set of tail lengths, which is why the body is also checked at
# several unrelated lengths below.
TAILS = [1, 2, 3, 5, 7, 8, 9, 13]

ONE_ARG = {
    "stumpff_c2": our_special.stumpff_c2,
    "stumpff_c3": our_special.stumpff_c3,
    "hyp2f1b": our_special.hyp2f1b,
    "D_to_nu": our_angles.D_to_nu,
    "nu_to_D": our_angles.nu_to_D,
    "D_to_M": our_angles.D_to_M,
    "M_to_D": our_angles.M_to_D,
}
TWO_ARG = {
    "E_to_M": our_angles.E_to_M,
    "F_to_M": our_angles.F_to_M,
    "nu_to_E": our_angles.nu_to_E,
    "E_to_nu": our_angles.E_to_nu,
    "nu_to_F": our_angles.nu_to_F,
    "F_to_nu": our_angles.F_to_nu,
    "fp_angle": our_angles.fp_angle,
}


@pytest.mark.parametrize("name", sorted(ONE_ARG))
@pytest.mark.parametrize("n", TAILS)
def test_simd_tail_matches_scalar(name, n):
    """A partial final vector must be finished by the scalar kernel.

    The vector body runs `W` elements at a time and the last work item falls
    back to the scalar loop, so a length that is not a multiple of `W` is the
    only place a lane can be dropped or written twice.
    """
    rng = np.random.default_rng(11)
    if name == "hyp2f1b":
        # straddles the x >= 1 branch, which the vector body masks to inf
        x = np.concatenate([rng.uniform(-0.95, 0.95, max(n - 3, 0)), [1.0, 1.5, 0.0]])
        x = x[:n] if n <= 3 else x[rng.permutation(n)]
    elif name.startswith("stumpff"):
        # psi = 0 and psi = +-1 sit on the branch boundaries
        x = np.concatenate([rng.uniform(-6.0, 6.0, max(n - 4, 0)), [0.0, 1.0, -1.0, 1e-9]])
        x = x[:n] if n <= 4 else x[rng.permutation(n)]
    else:
        x = rng.uniform(-4.0, 4.0, n)
    got = ONE_ARG[name](x)
    # one element per call always takes the scalar tail, so it is the oracle
    want = np.array([ONE_ARG[name](x[i : i + 1])[0] for i in range(n)])
    assert got.shape == x.shape
    assert np.array_equal(got, want, equal_nan=True), f"{name} tail at n={n}"


@pytest.mark.parametrize("name", sorted(TWO_ARG))
@pytest.mark.parametrize("n", TAILS)
def test_simd_two_argument_tail(name, n):
    rng = np.random.default_rng(12)
    a = rng.uniform(-3.0, 3.0, n)
    b = rng.uniform(0.05, 0.9, n)
    got = TWO_ARG[name](a, b)
    want = np.array([TWO_ARG[name](a[i : i + 1], b[i : i + 1])[0] for i in range(n)])
    assert np.array_equal(got, want, equal_nan=True), f"{name} tail at n={n}"


def test_simd_and_scalar_agree_on_the_full_vector():
    """A whole number of vectors still has to match the scalar path.

    The vector body and the scalar kernel are separate implementations of the
    same arithmetic, so this is the check that the mask arithmetic in
    `stumpff_c2_v` / `hyp2f1b_v` picks the same branch the scalar `if` does.
    """
    n = 4096
    psi = np.concatenate([np.linspace(-6.0, 6.0, n - 4), [0.0, 1.0, -1.0, -1e-12]])
    for fn in (our_special.stumpff_c2, our_special.stumpff_c3):
        got = fn(psi)
        want = np.array([fn(psi[i : i + 1])[0] for i in range(n)])
        assert np.array_equal(got, want, equal_nan=True), fn.__name__

    x = np.concatenate([np.linspace(-0.95, 0.95, n - 3), [1.0, 0.999, -0.999]])
    got = our_special.hyp2f1b(x)
    want = np.array([our_special.hyp2f1b(np.array([v]))[0] for v in x])
    assert np.array_equal(got, want, equal_nan=True)
    # the x >= 1 lanes really are inf, and the blend that produces them must
    # not turn the finite lanes into NaN on the way
    assert np.isposinf(want[-3]) and np.isfinite(got[:-3]).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda a: our_special.stumpff_c2(a),
        lambda a: our_angles.E_to_M(a, np.full(a.shape, 0.4)),
        lambda a: our_angles.nu_to_E(a, np.full(a.shape, 0.4)),
        lambda a: our_special.hyp2f1b(a * 0.5),
    ],
)
@pytest.mark.parametrize("n", [4095, 4096, 4097, 8192, 131071, 131072, 131073, 200_000])
def test_parallel_and_serial_agree(call, n):
    """Both sides of the dispatch must return the same numbers.

    `PARALLEL_THRESHOLD` is 4096 and `PARALLEL_THRESHOLD_CHEAP` is 131072; the
    vectorised maps switch at the second one.  Whichever side a length lands
    on, the answer cannot change.  Whether the batched side is the
    `parallelize` branch at all is up to `_lib.worker_count`, which returns 1
    on a host whose cgroup has no room for a pool; either way it runs the same
    `work` item, and `tests/test_a_tight_pids_budget_selects_the_serial_branch`
    pins that decision separately.
    """
    rng = np.random.default_rng(13)
    a = rng.uniform(-1.0, 1.0, n)
    got = call(a)
    # one element at a time goes through the scalar kernel every time
    want = np.array([call(a[i : i + 1])[0] for i in range(n)])
    assert got.shape == (n,)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-15, equal_nan=True), f"n={n}"


def test_parallel_coe2rv_agrees_across_the_threshold():
    n = 5000
    p = np.full(n, 11000.0)
    ecc = np.linspace(0.0, 0.9, n)
    inc = np.linspace(0.05, 3.0, n)
    raan = np.linspace(0.0, 6.0, n)
    argp = np.linspace(0.0, 6.0, n)
    nu = np.linspace(-3.0, 3.0, n)
    r_many, v_many = our_elements.coe2rv_many(np.full(n, 398600.4418), p, ecc, inc, raan, argp, nu)
    r_one, v_one = our_elements.coe2rv(398600.4418, p, ecc, inc, raan, argp, nu)
    assert_close(r_many, r_one, label="coe2rv_many vs coe2rv r")
    assert_close(v_many, v_one, label="coe2rv_many vs coe2rv v")


def test_status_buffer_reports_a_failure_from_inside_a_parallel_batch():
    """One bad element in a large batch still has to raise.

    The status is accumulated per element into a caller-owned buffer and
    reduced afterwards, so a worker that fails deep inside a parallel batch
    cannot lose the code on the way out.
    """
    k = 398600.4418
    n = 6000  # above PARALLEL_THRESHOLD, so the failing element runs on a worker
    r0 = np.tile(np.array([7000.0, 0.0, 0.0]), (n, 1))
    r1 = np.tile(np.array([0.0, 9000.0, 0.0]), (n, 1))
    tof = np.full(n, 3600.0)
    # good batch first, to show the path itself is fine
    our_iod.izzo(k, r0, r1, tof, 0, True, True, 35, 1e-8)
    # a 180 degree phase angle is collinear and upstream raises ValueError
    bad = r1.copy()
    bad[n // 2] = -r0[n // 2]
    with pytest.raises(ValueError):
        our_iod.izzo(k, r0, bad, tof, 0, True, True, 35, 1e-8)


def test_status_buffer_agrees_with_the_serial_batch():
    """A scalar call and a batch call must raise the same way.

    The scalar call runs the serial branch of the export, the batch runs the
    parallel one; both go through the same status buffer.
    """
    k = 398600.4418
    r0 = np.array([7000.0, 0.0, 0.0])
    r1 = np.array([0.0, 9000.0, 0.0])
    tof = 3600.0
    v0, v = our_iod.izzo(k, r0, r1, tof, 0, True, True, 35, 1e-8)
    v0b, vb = our_iod.izzo(k, r0[None, :], r1[None, :], np.array([tof]), 0, True, True, 35, 1e-8)
    assert_close(v0[None, :], v0b, label="izzo scalar vs batch v0")
    assert_close(v[None, :], vb, label="izzo scalar vs batch v")
    with pytest.raises(ValueError):
        our_iod.izzo(k, r0, -r0, tof, 0, True, True, 35, 1e-8)
    with pytest.raises(ValueError):
        our_iod.izzo(k, r0[None, :], (-r0)[None, :], np.array([tof]), 0, True, True, 35, 1e-8)



def test_worker_count_respects_an_explicit_override(monkeypatch):
    """`MOJO_POLIASTRO_WORKERS` is the documented way to pin the pool."""
    from mojo_poliastro import _lib

    monkeypatch.setenv("MOJO_POLIASTRO_WORKERS", "4")
    monkeypatch.setattr(_lib, "_workers", None)
    assert _lib.worker_count() == 4
    monkeypatch.setenv("MOJO_POLIASTRO_WORKERS", "1")
    monkeypatch.setattr(_lib, "_workers", None)
    assert _lib.worker_count() == 1
    monkeypatch.setattr(_lib, "_workers", None)


def test_a_tight_pids_budget_selects_the_serial_branch(monkeypatch):
    """No pool is launched unless the cgroup has room for one.

    `parallelize` sizes its pool from the machine, so the launch needs room
    for roughly one thread per core.  It dies on `pthread_create failed`
    rather than on an exception, so the decision has to be made before the
    launch; a count of 1 is the signal for "run the serial loop".
    """
    from mojo_poliastro import _lib

    monkeypatch.delenv("MOJO_POLIASTRO_WORKERS", raising=False)
    monkeypatch.setattr(_lib, "_cpus", lambda: 36)
    monkeypatch.setattr(_lib, "_pids_budget", lambda: 10)
    monkeypatch.setattr(_lib, "_workers", None)
    assert _lib.worker_count() == 1
    monkeypatch.setattr(_lib, "_pids_budget", lambda: 36 * _lib._POOL_SLACK + 1)
    monkeypatch.setattr(_lib, "_workers", None)
    assert _lib.worker_count() == 32
    monkeypatch.setattr(_lib, "_pids_budget", lambda: None)
    monkeypatch.setattr(_lib, "_workers", None)
    assert _lib.worker_count() == 32
    monkeypatch.setattr(_lib, "_workers", None)


def test_the_serial_branch_returns_the_same_numbers(monkeypatch):
    """A worker count of 1 runs the same work item, one element at a time."""
    from mojo_poliastro import _lib

    a = np.random.default_rng(3).uniform(-1.0, 1.0, 150_000)
    want = our_angles.E_to_M(a, np.full(a.shape, 0.4))
    monkeypatch.setenv("MOJO_POLIASTRO_WORKERS", "1")
    monkeypatch.setattr(_lib, "_workers", None)
    got = our_angles.E_to_M(a, np.full(a.shape, 0.4))
    assert_close(got, want, label="E_to_M serial vs default")
    monkeypatch.setattr(_lib, "_workers", None)
