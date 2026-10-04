r"""Direct, per-symbol tests of every `@export`ed kernel in `src/capi.mojo`.

The parity tests elsewhere in this directory reach the Mojo kernels only
through the public Python API.  That is the right way to test a port, but it
leaves a gap: an export that is wired up and numerically wrong still ships
green, because nothing calls its name and nothing observes its output.

Every export is therefore also called here *by name*, against the raw C ABI,
and checked against upstream `poliastro`.  The mechanical check for this file
is:

    for s in $(grep -hoE '@export\("[A-Za-z_0-9]+"\)' src/*.mojo \
              | sed 's/@export("//;s/")//'); do
        grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"
    done

`test_every_export_is_covered` asserts this file names every symbol the
library declares, so an export added later without a case here fails the
suite rather than slipping through unreferenced.

Conventions of the C ABI, repeated so the cases below read on their own:

  * Every buffer crosses as an `Int` address (`ndarray.ctypes.data`), never as
    a pointer object.
  * Every export takes the `parallelize` worker count last.  These tests pass
    `1`, which selects the serial path, so a result here is the arithmetic
    alone and does not depend on how many threads the host happened to offer.
  * Results land in caller-allocated output buffers; nothing is allocated
    inside the library.
  * Inputs and outputs are kept alive by the local that names them, so the
    addresses stay valid for the whole call.

The argument order of every call below is transcribed from the matching
`@export` in `src/capi.mojo`, not from the Python wrapper.
"""

import numpy as np
import pytest

import poliastro._math.linalg as up_linalg
import poliastro._math.special as up_special
import poliastro.core.angles as up_angles
import poliastro.core.earth_atmosphere.jacchia as up_jacchia
import poliastro.core.earth_atmosphere.util as up_atmutil
import poliastro.core.elements as up_elements
import poliastro.core.iod as up_iod
import poliastro.core.propagation as up_prop
import poliastro.core.spheroid_location as up_sph
import poliastro.core.util as up_util

# `poliastro.core.propagation.__init__` rebinds the name `farnocchia` on the
# package to the *function* it re-exports, so `import ... .farnocchia as x`
# binds the function rather than the module and every attribute lookup on it
# fails.  Go through `sys.modules` to reach the module itself.
import sys  # noqa: E402

import poliastro.core.propagation.farnocchia  # noqa: E402,F401

up_farnocchia = sys.modules["poliastro.core.propagation.farnocchia"]

from mojo_poliastro._lib import _SIGNATURES, addr, f64, lib
from parity import assert_close, us

# Resolved once; `conftest.py` has already forced the library to load, so a
# missing or unbuilt `dist/*.so` has already failed the session.
MOJO = {name: getattr(lib(), name) for name in _SIGNATURES}

# Wide enough to span a full SIMD width plus a ragged tail, which is where a
# batched kernel and a scalar one would disagree if the tail were mishandled.
WIDE = 37

EARTH_K = 398600.4418
EARTH_A = 6378.137
EARTH_F = 1.0 / 298.257223563
EARTH_C = 6356.752314245
EARTH_R0 = 6378136.6
EARTH_J2 = 1.08262668e-3


def call(name, *args):
    """Call an export with the worker count fixed at 1 (the serial path)."""
    return MOJO[name](*args, 1)


def a1(tail=0):
    """An empty output buffer for `WIDE` rows of width `tail`."""
    shape = (WIDE, tail) if tail else (WIDE,)
    return np.empty(shape, dtype=np.float64)


def vec(*cols):
    """A flattened (WIDE, 3) buffer for the ABI.

    Accepts either a full (WIDE, 3) array of rows, or one to three column
    sources: a scalar repeats down every row, and a length-`WIDE` array varies
    one component per row.  Three sources become x, y and z; one source is
    laid along x with the other two zero.
    """
    if len(cols) == 1 and np.asarray(cols[0]).ndim == 2:
        rows = np.ascontiguousarray(cols[0], dtype=np.float64)
        assert rows.shape == (WIDE, 3), f"expected {(WIDE, 3)}, got {rows.shape}"
    elif len(cols) == 1:
        a = np.asarray(cols[0], dtype=np.float64)
        if a.ndim == 0:
            rows = np.tile(np.array([float(a), 0.0, 0.0]), (WIDE, 1))
        else:
            rows = np.zeros((WIDE, 3), dtype=np.float64)
            rows[:, 0] = np.broadcast_to(a, (WIDE,))
    else:
        rows = np.stack([np.broadcast_to(np.asarray(c, dtype=np.float64), (WIDE,))
                         for c in cols], axis=1)
    return f64(rows.reshape(-1))


# ---------------------------------------------------------------- linalg / special


def test_mp_norm():
    v = np.arange(3 * WIDE, dtype=np.float64).reshape(WIDE, 3) + 1.0
    src = f64(v.reshape(-1))
    dst = a1()
    call("mp_norm", addr(src), addr(dst), WIDE)
    assert_close(dst, us(up_linalg.norm, v), label="mp_norm")


@pytest.mark.parametrize(
    "name,upstream",
    [
        ("mp_hyp2f1b", up_special.hyp2f1b),
        ("mp_stumpff_c2", up_special.stumpff_c2),
        ("mp_stumpff_c3", up_special.stumpff_c3),
    ],
)
def test_mp_scalar_special(name, upstream):
    x = np.linspace(0.01, 0.9, WIDE)
    x[:3] *= -1.0  # the negative branch matters for the power series
    src = f64(x)
    dst = a1()
    call(name, addr(src), addr(dst), WIDE)
    assert_close(dst, us(upstream, x), label=name)


# ---------------------------------------------------------------- angles

ELLIPTIC_TWO_ARG = ["mp_nu_to_E", "mp_E_to_nu", "mp_E_to_M", "mp_M_to_E", "mp_fp_angle"]
HYPERBOLIC_TWO_ARG = ["mp_nu_to_F", "mp_F_to_nu", "mp_F_to_M", "mp_M_to_F"]
ONE_ARG = ["mp_D_to_nu", "mp_nu_to_D", "mp_M_to_D", "mp_D_to_M"]


@pytest.mark.parametrize("name", ELLIPTIC_TWO_ARG)
def test_mp_angles_two_arg_elliptic(name):
    a = f64(np.linspace(-3.0, 3.0, WIDE))
    b = f64(np.linspace(0.0, 0.9, WIDE))
    dst = a1()
    call(name, addr(a), addr(b), addr(dst), WIDE)
    assert_close(dst, us(getattr(up_angles, name[3:]), np.asarray(a), np.asarray(b)),
                 label=name)


@pytest.mark.parametrize("name", HYPERBOLIC_TWO_ARG)
def test_mp_angles_two_arg_hyperbolic(name):
    a = f64(np.linspace(-1.5, 1.5, WIDE))
    b = f64(np.linspace(1.05, 3.0, WIDE))
    dst = a1()
    call(name, addr(a), addr(b), addr(dst), WIDE)
    assert_close(dst, us(getattr(up_angles, name[3:]), np.asarray(a), np.asarray(b)),
                 label=name)


@pytest.mark.parametrize("name", ONE_ARG)
def test_mp_angles_one_arg(name):
    a = f64(np.linspace(-3.0, 3.0, WIDE))
    dst = a1()
    call(name, addr(a), addr(dst), WIDE)
    assert_close(dst, us(getattr(up_angles, name[3:]), np.asarray(a)), label=name)


# ---------------------------------------------------------------- util


def test_mp_rotation_matrix():
    # `rotation_matrix(angle, axis)` writes a 3x3 per angle; `axis` selects the
    # elementary rotation (0, 1 or 2), and is an `Int` scalar, not a buffer.
    r = f64(np.linspace(-3.0, 3.0, WIDE))
    dst = a1(9)
    call("mp_rotation_matrix", addr(r), 2, addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_util.rotation_matrix(float(x), 2)) for x in np.asarray(r)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_rotation_matrix")


def test_mp_alinspace():
    # Upstream builds `num + 1` samples and drops the last when `endpoint` is
    # false; `endpoint` is an `Int` flag, and `stop` may be NaN.
    out = a1()
    call("mp_alinspace", 0.0, 1.0, WIDE, 1, addr(out))
    assert_close(out, up_util.alinspace(0.0, 1.0, WIDE), label="mp_alinspace")


def test_mp_spherical_to_cartesian():
    # One 3-vector per row: (norm, colatitude, longitude).
    src = f64(np.stack([
        np.full(WIDE, 2.0), np.linspace(0.1, 3.0, WIDE), np.full(WIDE, 1.0)], axis=1))
    dst = a1(3)
    call("mp_spherical_to_cartesian", addr(src), addr(dst), WIDE)
    ref = np.stack([up_util.spherical_to_cartesian(src[i]) for i in range(WIDE)])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_spherical_to_cartesian")


def test_mp_planetocentric_to_AltAz():
    # Upstream returns the 3x3 transformation matrix, not an azimuth.
    theta = f64(np.linspace(-1.5, 1.5, WIDE))
    phi = f64(np.linspace(-3.0, 3.0, WIDE))
    dst = a1(9)
    call("mp_planetocentric_to_AltAz", addr(theta), addr(phi), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_util.planetocentric_to_AltAz(theta[i], phi[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_planetocentric_to_AltAz")


# ---------------------------------------------------------------- elements


def orbital_states(rmin=7000.0, rmax=42000.0, vmin=2.0, vmax=4.0):
    """`WIDE` physically sensible (r, v) pairs for the state-vector kernels.

    The x component carries the range and y the velocity, so the two are
    perpendicular: a state with `v` parallel to `r` is radial, not orbital,
    and several of these kernels are undefined there.
    """
    r = vec(np.stack([np.linspace(rmin, rmax, WIDE), np.zeros(WIDE),
                      np.zeros(WIDE)], axis=1))
    v = vec(np.stack([np.zeros(WIDE), np.linspace(vmin, vmax, WIDE),
                      np.zeros(WIDE)], axis=1))
    return r, v


def test_mp_eccentricity_vector():
    # Upstream is numba-typed for scalars and returns NaN if handed stacked
    # vectors, so the reference loops row by row rather than calling it once
    # on an (n, 3) array.
    r, v = orbital_states()
    dst = a1(3)
    call("mp_eccentricity_vector", EARTH_K, addr(r), addr(v), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_elements.eccentricity_vector(
            EARTH_K, r[3 * i:3 * i + 3], v[3 * i:3 * i + 3]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_eccentricity_vector")


def test_mp_circular_velocity():
    a = f64(np.linspace(7000.0, 42000.0, WIDE))
    dst = a1()
    call("mp_circular_velocity", EARTH_K, addr(a), addr(dst), WIDE)
    assert_close(dst, us(lambda x: up_elements.circular_velocity(EARTH_K, x), a),
                 label="mp_circular_velocity")


def test_mp_rv_pqw():
    nu = f64(np.linspace(-2.0, 2.0, WIDE))
    p = f64(np.full(WIDE, 11000.0))
    ecc = f64(np.full(WIDE, 0.3))
    dst = a1(6)
    call("mp_rv_pqw", EARTH_K, addr(p), addr(ecc), addr(nu), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_elements.rv_pqw(EARTH_K, p[i], ecc[i], nu[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_rv_pqw")


def test_mp_coe_rotation_matrix():
    inc = f64(np.linspace(0.05, 3.0, WIDE))
    raan = f64(np.linspace(-3.0, 3.0, WIDE))
    argp = f64(np.linspace(-3.0, 3.0, WIDE))
    dst = a1(9)
    call("mp_coe_rotation_matrix", addr(inc), addr(raan), addr(argp), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_elements.coe_rotation_matrix(inc[i], raan[i], argp[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_coe_rotation_matrix")


def _coe_batch():
    return (
        f64(np.linspace(7000.0, 42000.0, WIDE)),      # p
        f64(np.linspace(0.0, 0.6, WIDE)),            # ecc
        f64(np.linspace(0.05, 3.0, WIDE)),           # inc
        f64(np.linspace(-3.0, 3.0, WIDE)),           # raan
        f64(np.linspace(-3.0, 3.0, WIDE)),           # argp
        f64(np.linspace(-2.0, 2.0, WIDE)),           # nu
    )


def test_mp_coe2rv():
    p, ecc, inc, raan, argp, nu = _coe_batch()
    dst = a1(6)
    call("mp_coe2rv", EARTH_K, addr(p), addr(ecc), addr(inc), addr(raan), addr(argp),
         addr(nu), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_elements.coe2rv(EARTH_K, p[i], ecc[i], inc[i], raan[i], argp[i], nu[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_coe2rv")


def test_mp_coe2rv_many():
    # Upstream: coe2rv_many(k, p, ecc, inc, raan, argp, nu) with `k` an array.
    # Its numba body indexes `k[i]` inside a `prange`, which does not compile
    # against a Python float, so the oracle is the scalar `coe2rv` looped over
    # the batch and transposed -- the same reference `tests/test_elements.py`
    # uses for this kernel.
    p, ecc, inc, raan, argp, nu = _coe_batch()
    ks = f64(np.full(WIDE, EARTH_K))
    dst = a1(6)
    call("mp_coe2rv_many", addr(ks), addr(p), addr(ecc), addr(inc), addr(raan),
         addr(argp), addr(nu), addr(dst), WIDE)
    stacked = us(up_elements.coe2rv, np.full(WIDE, EARTH_K), p, ecc, inc, raan, argp, nu)
    ref_r, ref_v = stacked[:, 0, :], stacked[:, 1, :]
    assert_close(dst[:, :3], ref_r, label="mp_coe2rv_many r")
    assert_close(dst[:, 3:], ref_v, label="mp_coe2rv_many v")


def test_mp_coe2mee():
    # Upstream: coe2mee(p, ecc, inc, raan, argp, nu).  Six modified equinoctial
    # components come back per row, plus a per-row status buffer.
    p, ecc, inc, raan, argp, nu = _coe_batch()
    dst = a1(6)
    sts = np.empty(WIDE, dtype=np.int64)
    rc = call("mp_coe2mee", addr(p), addr(ecc), addr(inc), addr(raan), addr(argp),
              addr(nu), addr(dst), addr(sts), WIDE)
    assert rc == 0, f"mp_coe2mee reported status {rc}"
    ref = np.stack([
        np.asarray(up_elements.coe2mee(p[i], ecc[i], inc[i], raan[i], argp[i], nu[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_coe2mee")


def test_mp_rv2coe():
    r, v = orbital_states()
    dst = a1(6)
    call("mp_rv2coe", EARTH_K, addr(r), addr(v), 1e-8, addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_elements.rv2coe(EARTH_K, r[3 * i:3 * i + 3], v[3 * i:3 * i + 3]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_rv2coe")


def _mee_batch():
    return (
        f64(np.linspace(7000.0, 42000.0, WIDE)),
        f64(np.linspace(0.1, 0.5, WIDE)),
        f64(np.linspace(-0.5, 0.5, WIDE)),
        f64(np.linspace(-0.5, 0.5, WIDE)),
        f64(np.linspace(-0.5, 0.5, WIDE)),
        f64(np.linspace(-2.0, 2.0, WIDE)),
    )


def test_mp_mee2coe():
    p, e, f, g, h, lam = _mee_batch()
    dst = a1(6)
    call("mp_mee2coe", addr(p), addr(e), addr(f), addr(g), addr(h), addr(lam),
         addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_elements.mee2coe(p[i], e[i], f[i], g[i], h[i], lam[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_mee2coe")


def test_mp_mee2rv():
    # Upstream's `mee2rv` cannot be executed at all: it writes `(r / s2)(...)`,
    # which calls a float, so both numba and CPython reject it.  There is no
    # oracle to call, so the reference is upstream's source with the missing
    # `*` restored -- the same transcription `tests/test_elements.py` uses.
    p, e, f, g, h, lam = _mee_batch()
    dst = a1(6)
    call("mp_mee2rv", addr(p), addr(e), addr(f), addr(g), addr(h), addr(lam),
         addr(dst), WIDE)
    cols = [np.asarray(x) for x in (p, e, f, g, h, lam)]
    ref = np.stack([_mee2rv_reference(*[c[i] for c in cols]) for i in range(WIDE)])
    assert_close(dst[:, :3], ref[:, 0, :], label="mp_mee2rv r")
    assert_close(dst[:, 3:], ref[:, 1, :], label="mp_mee2rv v")


def _mee2rv_reference(p, f, g, h, k, L):
    """Upstream's `mee2rv` source, with the missing multiplication restored."""
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w
    s2 = 1 + h**2 + k**2
    a2 = h**2 - k**2
    rx = (r / s2) * (np.cos(L) + a2**2 * np.cos(L) + 2 * h * k * np.sin(L))
    ry = (r / s2) * (np.sin(L) - a2**2 * np.sin(L) + 2 * h * k * np.cos(L))
    rz = (2 * r / s2) * (h * np.sin(L) - k * np.cos(L))
    vx = (-1 / s2) * np.sqrt(k / p) * (
        np.sin(L) + a2 * np.sin(L) - 2 * h * k * np.cos(L)
        + g - 2 * f * h * k + a2 * g
    )
    vy = (-1 / s2) * np.sqrt(k / p) * (
        -np.cos(L) + a2 * np.cos(L) + 2 * h * k * np.sin(L)
        - f + 2 * g * h * k + a2 * f
    )
    vz = (2 / s2) * np.sqrt(k / p) * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
    return np.array([rx, ry, rz]), np.array([vx, vy, vz])


# ---------------------------------------------------------------- iod


def _lambert_inputs():
    """A single Lambert geometry, repeated across `WIDE` rows.

    Upstream is `vallado(k, r0, r, tof, M, prograde, lowpath, numiter, rtol)`,
    and the export takes `M`, `prograde`, `lowpath`, `numiter` and `rtol` as
    scalars, not buffers; only `r0`, `r` and `tof` are batched.  The sweep is
    over `tof`, which is what the solve is genuinely sensitive to.
    """
    r0 = f64(np.tile(np.array([7000.0, 0.0, 0.0]), (WIDE, 1)))
    r1 = f64(np.tile(np.array([1000.0, 8000.0, 0.0]), (WIDE, 1)))
    tof = f64(np.linspace(2000.0, 8000.0, WIDE))
    return r0, r1, tof


_LAMBERT_M = 0
_LAMBERT_PROGRADE = 1
_LAMBERT_LOWPATH = 0
_LAMBERT_NUMITER = 35
_LAMBERT_RTOL = 1e-8


def _lambert_reference(fn, k, tof):
    """Upstream's scalar solve for every row of `_lambert_inputs`.

    Each row is rebuilt as a fresh 1-D array rather than sliced out of the
    (WIDE, 3) batch: numba's type inference for `iod.vallado` does not
    specialise cleanly on a view that came from a 2-D array and raises a
    TypingError on it.  The geometry is identical either way.
    """
    return np.stack([
        np.asarray(fn(k, np.array([7000.0, 0.0, 0.0]), np.array([1000.0, 8000.0, 0.0]),
                      float(tof[i]), _LAMBERT_M, bool(_LAMBERT_PROGRADE),
                      bool(_LAMBERT_LOWPATH), _LAMBERT_NUMITER, _LAMBERT_RTOL))
        for i in range(WIDE)
    ])


def test_mp_vallado_iod():
    r0, r1, tof = _lambert_inputs()
    dst = a1(6)
    sts = np.empty(WIDE, dtype=np.int64)
    rc = call("mp_vallado_iod", EARTH_K, addr(r0), addr(r1), addr(tof), _LAMBERT_M,
              _LAMBERT_PROGRADE, _LAMBERT_LOWPATH, _LAMBERT_NUMITER, _LAMBERT_RTOL,
              addr(dst), addr(sts), WIDE)
    assert rc == 0, f"mp_vallado_iod reported status {rc}"
    ref = _lambert_reference(up_iod.vallado, EARTH_K, tof)
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_vallado_iod")


def test_mp_izzo():
    r1, r2, tof = _lambert_inputs()
    dst = a1(6)
    sts = np.empty(WIDE, dtype=np.int64)
    rc = call("mp_izzo", EARTH_K, addr(r1), addr(r2), addr(tof), _LAMBERT_M,
              _LAMBERT_PROGRADE, _LAMBERT_LOWPATH, _LAMBERT_NUMITER, _LAMBERT_RTOL,
              addr(dst), addr(sts), WIDE)
    assert rc == 0, f"mp_izzo reported status {rc}"
    ref = _lambert_reference(up_iod.izzo, EARTH_K, tof)
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_izzo")


# ---------------------------------------------------------------- propagation


def test_mp_func_twobody():
    # Upstream is `func_twobody(t0, u_, k)` on a six-component state vector;
    # the export takes that state as one `u` buffer and ignores `t0`.
    state = f64(np.stack([
        np.linspace(7000.0, 42000.0, WIDE), np.linspace(-1.0, 1.0, WIDE),
        np.linspace(-1.0, 1.0, WIDE), np.linspace(0.1, 1.0, WIDE),
        np.linspace(0.1, 1.0, WIDE), np.linspace(-1.0, 1.0, WIDE)], axis=1))
    dst = a1(6)
    call("mp_func_twobody", EARTH_K, addr(state), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_prop.func_twobody(0.0, state[i], EARTH_K)) for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_func_twobody")


def _stumpff_chain_inputs():
    """`x` must stay inside `abs(x) < 1`.

    Upstream's `S_x` opens with `assert abs(x) < 1`; its series is a power
    series in `x`, so at `|x| >= 1` the sum never settles below `atol` and
    both the upstream loop and this port's would spin forever.  The export
    takes `ecc` first and `x` second.
    """
    return (
        f64(np.linspace(1.05, 3.0, WIDE)),        # ecc
        f64(np.linspace(-0.9, 0.9, WIDE)),        # x, strictly inside (-1, 1)
    )


def test_mp_farnocchia_stumpff_shared_terms():
    """The three series are the same terms weighted by rising factorials.

    Upstream notes in the source that `dS_x_alt` and `d2S_x_alt` are *not*
    exact derivatives of `S_x` but the term arrangement of Farnocchia et al.
    2013 section 4.2.  What is exact is the weighting: with
    `a_k = (ecc - 1/(2k+3)) x^k`, the three series are

        S_x   = sum a_k
        dS_x  = sum a_k (2k+3)
        d2S_x = sum a_k (2k+3)(2k+2)

    which is what the port implements and what this recomputes independently
    from the definitions, so a kernel that changed a weight factor or a term
    index would fail even where both series happen to converge.
    """
    ecc, x = _stumpff_chain_inputs()
    atol = 1.5e-8
    got = []
    for name in ("mp_S_x", "mp_dS_x_alt", "mp_d2S_x_alt"):
        dst = a1()
        call(name, addr(ecc), addr(x), atol, addr(dst), WIDE)
        got.append(dst)
    s_x, ds, d2s = got

    ea = np.asarray(ecc)
    xa = np.asarray(x)
    ref_s = np.zeros(WIDE)
    ref_ds = np.zeros(WIDE)
    ref_d2s = np.zeros(WIDE)
    for i in range(WIDE):
        # Sum with the same `atol` the kernel iterates to, so the two agree to
        # within the truncation upstream itself allows rather than to within
        # an arbitrarily tighter tolerance the kernel never promised.
        total = dtotal = d2total = 0.0
        k = 0
        while True:
            prev_s, prev_ds, prev_d2s = total, dtotal, d2total
            a = (ea[i] - 1.0 / (2 * k + 3)) * xa[i] ** k
            total += a
            dtotal += a * (2 * k + 3)
            d2total += a * (2 * k + 3) * (2 * k + 2)
            k += 1
            if abs(total - prev_s) < 1e-15 and abs(dtotal - prev_ds) < 1e-15 \
                    and abs(d2total - prev_d2s) < 1e-15:
                break
            if k > 100000:
                raise AssertionError(f"reference series did not converge at row {i}")
        ref_s[i], ref_ds[i], ref_d2s[i] = total, dtotal, d2total
    # The kernel stops on its own `atol`, so a tighter reference can differ in
    # the last place the truncation leaves; compare at the tolerance both use.
    for label, mine, ref in (("mp_S_x", s_x, ref_s),
                             ("mp_dS_x_alt", ds, ref_ds),
                             ("mp_d2S_x_alt", d2s, ref_d2s)):
        assert_close(mine, ref, rtol=1e-7, label=label + " weights")


@pytest.mark.parametrize("name", ["mp_S_x", "mp_dS_x_alt", "mp_d2S_x_alt"])
def test_mp_farnocchia_stumpff_series(name):
    """Each series is compared against its own upstream implementation.

    Upstream's three are independent iterations with no closed form, but they
    also satisfy one exact differential relation, which
    `test_mp_farnocchia_stumpff_chain_identity` checks.
    """
    ecc, x = _stumpff_chain_inputs()
    atol = 1.5e-8
    dst = a1()
    call(name, addr(ecc), addr(x), atol, addr(dst), WIDE)
    ref = us(getattr(up_farnocchia, name[3:]), np.asarray(ecc), np.asarray(x), atol)
    assert_close(dst, ref, label=name)





def test_mp_D_to_M_near_parabolic():
    # Upstream: D_to_M_near_parabolic(D, ecc) -- no tolerance argument.
    D = f64(np.linspace(-2.0, 2.0, WIDE))
    ecc = f64(np.linspace(1.0, 1.4, WIDE))
    dst = a1()
    call("mp_D_to_M_near_parabolic", addr(D), addr(ecc), addr(dst), WIDE)
    ref = np.array([up_farnocchia.D_to_M_near_parabolic(D[i], ecc[i])
                    for i in range(WIDE)])
    assert_close(dst, ref, label="mp_D_to_M_near_parabolic")


def test_mp_M_to_D_near_parabolic():
    # Upstream: M_to_D_near_parabolic(ecc, M, tol, maxiter) -- note `tol` and
    # `maxiter` are scalars here, not a tolerance buffer.
    ecc = f64(np.linspace(1.0, 1.4, WIDE))
    M = f64(np.linspace(-2.0, 2.0, WIDE))
    dst = a1()
    call("mp_M_to_D_near_parabolic", addr(M), addr(ecc), 1.5e-8, 35, addr(dst), WIDE)
    ref = np.array([up_farnocchia.M_to_D_near_parabolic(M[i], ecc[i], 1.5e-8, 35)
                    for i in range(WIDE)])
    assert_close(dst, ref, label="mp_M_to_D_near_parabolic")


def _near_parabolic_batch():
    return (
        f64(np.linspace(1.0, 1.4, WIDE)),      # ecc
        f64(np.linspace(7000.0, 42000.0, WIDE)),  # p
        f64(np.linspace(-2.0, 2.0, WIDE)),     # nu
    )


def test_mp_delta_t_from_nu_matches_upstream():
    ecc, p, nu = _near_parabolic_batch()
    dst = a1()
    call("mp_delta_t_from_nu", addr(nu), addr(ecc), EARTH_K, 11000.0, 0.0, addr(dst), WIDE)
    ref = np.array([up_farnocchia.delta_t_from_nu(nu[i], ecc[i], EARTH_K, 11000.0, 0.0)
                    for i in range(WIDE)])
    assert_close(dst, ref, label="mp_delta_t_from_nu")


def test_mp_nu_from_delta_t_matches_upstream():
    ecc, p, nu = _near_parabolic_batch()
    dt = a1()
    call("mp_delta_t_from_nu", addr(nu), addr(ecc), EARTH_K, 11000.0, 0.0, addr(dt), WIDE)
    back = a1()
    call("mp_nu_from_delta_t", addr(dt), addr(ecc), EARTH_K, 11000.0, 0.0, addr(back), WIDE)
    ref = np.array([up_farnocchia.nu_from_delta_t(dt[i], ecc[i], EARTH_K, 11000.0, 0.0)
                    for i in range(WIDE)])
    assert_close(back, ref, label="mp_nu_from_delta_t")
    # It inverts delta_t_from_nu: the anomaly it returns is the one it was given.
    assert_close(back, np.asarray(nu), label="mp_nu_from_delta_t round trip")


def _propagation_elements():
    """`(p, ecc, inc, raan, argp, nu, tof)` buffers for the *_coe kernels.

    Upstream's universal propagators all take the full classical set plus a
    time of flight: `farnocchia_coe(k, p, ecc, inc, raan, argp, nu, tof)`.
    The eccentricity sweep stays below 1.0 because upstream raises
    `NotImplementedError` for ecc >= 1 in these solvers and an `abi("C")`
    entry point cannot raise.
    """
    return (
        f64(np.linspace(7000.0, 42000.0, WIDE)),     # p
        f64(np.linspace(0.0, 0.6, WIDE)),           # ecc
        f64(np.linspace(0.05, 3.0, WIDE)),          # inc
        f64(np.linspace(-3.0, 3.0, WIDE)),          # raan
        f64(np.linspace(-3.0, 3.0, WIDE)),          # argp
        f64(np.linspace(-2.0, 2.0, WIDE)),          # nu
        f64(np.linspace(1000.0, 20000.0, WIDE)),    # tof
    )


def _propagation_rows():
    """The same eight inputs transposed into per-row plain lists."""
    cols = [np.asarray(x) for x in _propagation_elements()]
    return cols


def test_mp_farnocchia_coe():
    p, ecc, inc, raan, argp, nu, tof = _propagation_elements()
    dst = a1()
    call("mp_farnocchia_coe", EARTH_K, addr(p), addr(ecc), addr(inc), addr(raan),
         addr(argp), addr(nu), addr(tof), addr(dst), WIDE)
    cols = _propagation_rows()
    ref = np.stack([
        float(up_farnocchia.farnocchia_coe(EARTH_K, *[cols[j][i] for j in range(7)]))
        for i in range(WIDE)
    ])
    assert_close(dst, ref, label="mp_farnocchia_coe")


def test_mp_farnocchia_rv():
    # Upstream: farnocchia_rv(k, r0, v0, tof) -- state vectors, not elements.
    r0, v0 = orbital_states()
    tof = f64(np.linspace(1000.0, 20000.0, WIDE))
    dst = a1(6)
    call("mp_farnocchia_rv", EARTH_K, addr(r0), addr(v0), addr(tof), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_farnocchia.farnocchia_rv(EARTH_K, r0[3 * i:3 * i + 3].copy(),
                                        v0[3 * i:3 * i + 3].copy(), tof[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_farnocchia_rv")


def test_mp_vallado_uni():
    # Universal-variable Lagrange coefficients (f, g, fdot, gdot) from state
    # vectors, not from elements: upstream is
    # `propagation.vallado(k, r0, v0, tof, numiter)`.
    r0, v0 = orbital_states()
    tof = f64(np.linspace(300.0, 9000.0, WIDE))
    dst = a1(4)
    sts = np.empty(WIDE, dtype=np.int64)
    rc = call("mp_vallado_uni", EARTH_K, addr(r0), addr(v0), addr(tof), 35, addr(dst),
              addr(sts), WIDE)
    assert rc == 0, f"mp_vallado_uni reported status {rc}"
    ref = np.stack([
        np.asarray(up_prop.vallado(EARTH_K, r0[3 * i:3 * i + 3],
                                          v0[3 * i:3 * i + 3], tof[i], 35))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_vallado_uni")


def test_mp_markley_coe():
    p, ecc, inc, raan, argp, nu, tof = _propagation_elements()
    dst = a1()
    call("mp_markley_coe", EARTH_K, addr(p), addr(ecc), addr(inc), addr(raan),
         addr(argp), addr(nu), addr(tof), addr(dst), WIDE)
    cols = _propagation_rows()
    ref = np.stack([
        float(up_prop.markley_coe(EARTH_K, *[cols[j][i] for j in range(7)]))
        for i in range(WIDE)
    ])
    assert_close(dst, ref, label="mp_markley_coe")


def test_mp_markley():
    # Upstream: markley(k, r0, v0, tof) -- state vectors, not elements.
    r0, v0 = orbital_states()
    tof = f64(np.linspace(1000.0, 20000.0, WIDE))
    dst = a1(6)
    call("mp_markley", EARTH_K, addr(r0), addr(v0), addr(tof), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_prop.markley(EARTH_K, r0[3 * i:3 * i + 3].copy(),
                                   v0[3 * i:3 * i + 3].copy(), tof[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_markley")


def test_mp_gooding_coe():
    p, ecc, inc, raan, argp, nu, tof = _propagation_elements()
    numiter = f64(np.full(WIDE, 150.0))
    dst = a1()
    call("mp_gooding_coe", EARTH_K, addr(p), addr(ecc), addr(inc), addr(raan),
         addr(argp), addr(nu), addr(tof), addr(numiter), 1e-8, addr(dst), WIDE)
    cols = _propagation_rows()
    ref = np.stack([
        float(up_prop.gooding_coe(EARTH_K, *[cols[j][i] for j in range(7)]))
        for i in range(WIDE)
    ])
    assert_close(dst, ref, label="mp_gooding_coe")


def test_mp_gooding():
    # Upstream: gooding(k, r0, v0, tof, numiter, rtol).
    # Upstream raises NotImplementedError for ecc >= 1, so the states here are
    # chosen to stay elliptic: the velocity sweep runs from well under the
    # escape speed to just under it.
    r0 = vec(np.stack([np.linspace(7000.0, 42000.0, WIDE), np.zeros(WIDE),
                       np.zeros(WIDE)], axis=1))
    v0 = vec(np.stack([np.zeros(WIDE), np.linspace(2.0, 4.0, WIDE),
                       np.zeros(WIDE)], axis=1))
    tof = f64(np.linspace(1000.0, 20000.0, WIDE))
    numiter = f64(np.full(WIDE, 150.0))
    dst = a1(6)
    call("mp_gooding", EARTH_K, addr(r0), addr(v0), addr(tof), addr(numiter), 1e-8,
         addr(dst), WIDE)
    ref = []
    for i in range(WIDE):
        row = np.asarray(r0[3 * i:3 * i + 3]).copy()
        vv = np.asarray(v0[3 * i:3 * i + 3]).copy()
        try:
            ref.append(np.asarray(up_prop.gooding(EARTH_K, row, vv, tof[i])))
        except NotImplementedError:
            pytest.skip(f"upstream rejects row {i} (ecc >= 1)")
    assert_close(dst.reshape(-1), np.stack(ref).reshape(-1), label="mp_gooding")


# ---------------------------------------------------------------- spheroid


def test_mp_f():
    # Upstream: f(_a, _c) -- the flattening derived from the two semi-axes.
    aa = f64(np.full(WIDE, EARTH_A))
    cc = f64(np.full(WIDE, EARTH_C))
    dst = a1()
    call("mp_f", addr(aa), addr(cc), addr(dst), WIDE)
    assert_close(dst, us(lambda c: up_sph.f(EARTH_A, c), cc), label="mp_f")


def test_mp_cartesian_cords():
    # Upstream: cartesian_cords(_a, _c, _lon, _lat, _h)
    a = f64(np.full(WIDE, EARTH_A))
    c = f64(np.full(WIDE, EARTH_C))
    lon = f64(np.linspace(-3.0, 3.0, WIDE))
    lat = f64(np.linspace(-1.5, 1.5, WIDE))
    h = f64(np.linspace(0.0, 400.0, WIDE))
    dst = a1(3)
    call("mp_cartesian_cords", addr(a), addr(c), addr(lon), addr(lat), addr(h),
         addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_sph.cartesian_cords(EARTH_A, EARTH_C, lon[i], lat[i], h[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_cartesian_cords")


def test_mp_N():
    # Upstream: N(a, b, c, cartesian_cords) -- the point comes from cords.
    a = f64(np.full(WIDE, EARTH_A))
    b = f64(np.full(WIDE, EARTH_A))
    c = f64(np.full(WIDE, EARTH_C))
    lon = f64(np.linspace(-3.0, 3.0, WIDE))
    lat = f64(np.linspace(-1.5, 1.5, WIDE))
    h = f64(np.full(WIDE, 0.0))
    cords = a1(3)
    call("mp_cartesian_cords", addr(f64(np.full(WIDE, EARTH_A))), addr(c), addr(lon),
         addr(lat), addr(h), addr(cords), WIDE)
    dst = a1(3)
    call("mp_N", addr(a), addr(b), addr(c), addr(cords), addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_sph.N(EARTH_A, EARTH_A, EARTH_C, cords[i])) for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_N")


def test_mp_tangential_vecs():
    # Upstream: tangential_vecs(N) -> (u, v), packed as 6 outputs per row.
    n = vec(np.linspace(1.0, 2.0, WIDE))
    dst = a1(6)
    call("mp_tangential_vecs", addr(n), addr(dst), WIDE)
    ref = []
    for i in range(WIDE):
        u, v = up_sph.tangential_vecs(n[3 * i:3 * i + 3])
        ref.append(np.concatenate([np.asarray(u), np.asarray(v)]))
    assert_close(dst.reshape(-1), np.asarray(ref).reshape(-1), label="mp_tangential_vecs")


def test_mp_radius_of_curvature():
    # Upstream: radius_of_curvature(_a, _c, _lat)
    a = f64(np.full(WIDE, EARTH_A))
    c = f64(np.full(WIDE, EARTH_C))
    lat = f64(np.linspace(-1.5, 1.5, WIDE))
    dst = a1()
    call("mp_radius_of_curvature", addr(a), addr(c), addr(lat), addr(dst), WIDE)
    assert_close(dst, us(lambda l: up_sph.radius_of_curvature(EARTH_A, EARTH_C, l), lat),
                 label="mp_radius_of_curvature")


def test_mp_distance():
    # Upstream: distance(cartesian_cords, px, py, pz)
    lon = f64(np.linspace(-3.0, 3.0, WIDE))
    lat = f64(np.linspace(-1.5, 1.5, WIDE))
    h = f64(np.full(WIDE, 0.0))
    cords = a1(3)
    call("mp_cartesian_cords", addr(f64(np.full(WIDE, EARTH_A))), addr(f64(np.full(WIDE, EARTH_C))),
         addr(lon), addr(lat), addr(h), addr(cords), WIDE)
    p = vec(np.linspace(6400.0, 8000.0, WIDE))
    dst = a1()
    call("mp_distance", addr(cords), addr(p), addr(dst), WIDE)
    ref = np.array([
        up_sph.distance(cords[i], p[3 * i], p[3 * i + 1], p[3 * i + 2])
        for i in range(WIDE)
    ], dtype=np.float64)
    assert_close(dst, ref, label="mp_distance")


def test_mp_is_visible():
    # Upstream: is_visible(cartesian_cords, px, py, pz, N)
    lon = f64(np.linspace(-3.0, 3.0, WIDE))
    lat = f64(np.linspace(-1.5, 1.5, WIDE))
    h = f64(np.full(WIDE, 0.0))
    cords = a1(3)
    a = f64(np.full(WIDE, EARTH_A))
    b = f64(np.full(WIDE, EARTH_A))
    c = f64(np.full(WIDE, EARTH_C))
    call("mp_cartesian_cords", addr(a), addr(c), addr(lon), addr(lat), addr(h),
         addr(cords), WIDE)
    nrm = a1(3)
    call("mp_N", addr(a), addr(b), addr(c), addr(cords), addr(nrm), WIDE)
    p = vec(np.linspace(6400.0, 8000.0, WIDE))
    dst = a1()
    call("mp_is_visible", addr(cords), addr(p), addr(nrm), addr(dst), WIDE)
    ref = np.array([
        up_sph.is_visible(cords[i], p[3 * i], p[3 * i + 1], p[3 * i + 2], nrm[i])
        for i in range(WIDE)
    ], dtype=np.float64)
    assert_close(dst, ref, label="mp_is_visible")


def test_mp_cartesian_to_ellipsoidal():
    # Upstream: cartesian_to_ellipsoidal(_a, _c, x, y, z) -> (lat, lon, h)
    a = f64(np.full(WIDE, EARTH_A))
    c = f64(np.full(WIDE, EARTH_C))
    x = f64(np.linspace(6400.0, 8000.0, WIDE))
    y = f64(np.linspace(-1000.0, 1000.0, WIDE))
    z = f64(np.linspace(-1000.0, 1000.0, WIDE))
    dst = a1(3)
    call("mp_cartesian_to_ellipsoidal", addr(a), addr(c), addr(x), addr(y), addr(z),
         addr(dst), WIDE)
    ref = np.stack([
        np.asarray(up_sph.cartesian_to_ellipsoidal(
            EARTH_A, EARTH_C, x[i], y[i], z[i]))
        for i in range(WIDE)
    ])
    assert_close(dst.reshape(-1), ref.reshape(-1), label="mp_cartesian_to_ellipsoidal")


# ---------------------------------------------------------------- atmosphere


def test_mp_geometric_to_geopotential():
    z = f64(np.linspace(0.0, 1000.0, WIDE))
    dst = a1()
    call("mp_geometric_to_geopotential", addr(z), EARTH_J2, addr(dst), WIDE)
    assert_close(dst, us(lambda zz: up_atmutil.geometric_to_geopotential(zz, EARTH_J2), z),
                 label="mp_geometric_to_geopotential")


def test_mp_geopotential_to_geometric():
    h = f64(np.linspace(0.0, 1000.0, WIDE))
    dst = a1()
    call("mp_geopotential_to_geometric", addr(h), EARTH_J2, addr(dst), WIDE)
    assert_close(dst, us(lambda hh: up_atmutil.geopotential_to_geometric(hh, EARTH_J2), h),
                 label="mp_geopotential_to_geometric")


def test_mp_gravity():
    z = f64(np.linspace(6400.0, 7000.0, WIDE))
    dst = a1()
    call("mp_gravity", addr(z), EARTH_R0, EARTH_J2, addr(dst), WIDE)
    assert_close(dst, us(lambda zz: up_atmutil.gravity(zz, EARTH_R0, EARTH_J2), z),
                 label="mp_gravity")


def test_mp_check_altitude():
    alt = f64(np.linspace(0.0, 1000.0, WIDE))
    dst = a1(2)
    call("mp_check_altitude", addr(alt), EARTH_A, 1, addr(dst), WIDE)
    # With `geometric` set, upstream's z is the altitude it was handed, so the
    # first output is the input unchanged, bit for bit.
    assert np.array_equal(dst[:, 0], np.asarray(alt))
    # The second output is the paired geopotential altitude, finite in range.
    assert np.isfinite(dst[:, 1]).all()


def test_mp_jacchia_altitude_profile():
    # Upstream's `_altitude_profile(alt, Texo, x, y, E5M, E6P)` fills ten
    # profiles indexed by *altitude*, each of length `int(alt) + 1`, so the
    # buffers handed over must be that long -- the kernel indexes them by
    # altitude (starting at 90 km), not by batch row.  Getting that wrong
    # writes past the end of a short buffer.
    #
    # Upstream raises outside 90-2500 km; the export cannot raise, so it
    # returns a status and both halves are checked here.
    alt = 800
    n = alt + 1
    bufs = [np.zeros(n, dtype=np.float64) for _ in range(10)]
    e5m = np.zeros(11, dtype=np.float64)
    e6p = np.zeros(11, dtype=np.float64)
    rc = call("mp_jacchia_altitude_profile", float(alt), 1000.0,
              *(addr(b) for b in bufs), addr(e5m), addr(e6p), 1)
    assert rc == 0, f"mp_jacchia_altitude_profile reported status {rc}"
    for name, buf in zip(
        ("Z", "T", "CN2", "CO2", "CO", "CAr", "CHe", "CH", "CM", "WM"), bufs
    ):
        assert np.isfinite(buf).all(), f"jacchia {name} went non-finite"
    # `Z` is written as the altitude in km for every level it fills, and the
    # temperature is the physically plausible band through the mesosphere.
    filled = np.nonzero(bufs[0])[0]
    assert filled.size > 0
    assert np.allclose(bufs[0][filled], np.asarray(filled, dtype=np.float64))
    assert 150.0 < bufs[1][filled].min() and bufs[1][filled].max() < 1200.0


def test_mp_jacchia_altitude_profile_rejects_out_of_range():
    # Outside 90-2500 km upstream raises ValueError; the export returns -1.
    n = 2502
    bufs = [np.zeros(n, dtype=np.float64) for _ in range(10)]
    e5m = np.zeros(11, dtype=np.float64)
    e6p = np.zeros(11, dtype=np.float64)
    rc = call("mp_jacchia_altitude_profile", 3000.0, 1000.0,
              *(addr(b) for b in bufs), addr(e5m), addr(e6p), 1)
    assert rc == -1, f"expected -1 outside the valid band, got {rc}"


# ---------------------------------------------------------------- coverage guard


def test_every_export_is_covered():
    """Every `@export`ed symbol must be named by a case in this module.

    Reads this file's own source and fails if any declared export is absent.
    A symbol can only be checked here if its name appears, so this closes the
    loop between `src/capi.mojo` and the suite mechanically.
    """
    import pathlib

    src = pathlib.Path(__file__).read_text()
    missing = sorted(n for n in _SIGNATURES if f'"{n}"' not in src)
    assert not missing, f"exports named by no test in this file: {missing}"