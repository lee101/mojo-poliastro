"""Parity for `poliastro.core.iod` and `poliastro.core.propagation`."""

import numpy as np
import pytest

from parity import assert_close, us

import sys

import mojo_poliastro.core.iod as our_iod
import mojo_poliastro.core.propagation as ours
import poliastro.core.elements as their_elements
import poliastro.core.iod as their_iod
import poliastro.core.propagation as their_prop

# `poliastro.core.propagation.__init__` rebinds the name `farnocchia` on the
# package to the *function* it re-exports, so `import ... .farnocchia as x`
# binds the function, not the module.  Go through sys.modules instead.
import poliastro.core.propagation.farnocchia  # noqa: E402

their_farnocchia = sys.modules["poliastro.core.propagation.farnocchia"]

K = 398600.4418
R0 = np.array([7000.0, 0.0, 0.0])
R1 = np.array([0.0, 9000.0, 0.0])
TOF = 3600.0


def test_izzo_scalar():
    v0_o, v_o = our_iod.izzo(K, R0, R1, TOF, 0, True, True, 35, 1e-8)
    v0_u, v_u = their_iod.izzo(K, R0, R1, TOF, 0, True, True, 35, 1e-8)
    assert_close(v0_o, v0_u, label="izzo v0")
    assert_close(v_o, v_u, label="izzo v")


def test_izzo_retrograde():
    v0_o, v_o = our_iod.izzo(K, R0, R1, TOF, 0, False, True, 35, 1e-8)
    v0_u, v_u = their_iod.izzo(K, R0, R1, TOF, 0, False, True, 35, 1e-8)
    assert_close(v0_o, v0_u, label="izzo retrograde v0")
    assert_close(v_o, v_u, label="izzo retrograde v")


def test_izzo_batch():
    rng = np.random.default_rng(7)
    n = 64
    r0 = rng.normal(size=(n, 3))
    r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
    r0 *= rng.uniform(7000.0, 42000.0, (n, 1))
    r1 = rng.normal(size=(n, 3))
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r1 *= rng.uniform(7000.0, 42000.0, (n, 1))
    tof = rng.uniform(600.0, 20000.0, n)
    v0_o, v_o = our_iod.izzo(K, r0, r1, tof, 0, True, True, 35, 1e-8)
    for i in range(n):
        v0_u, v_u = their_iod.izzo(K, r0[i], r1[i], tof[i], 0, True, True, 35, 1e-8)
        assert_close(v0_o[i], v0_u, label=f"izzo batch v0 {i}")
        assert_close(v_o[i], v_u, label=f"izzo batch v {i}")


def test_izzo_closes_the_transfer_orbit():
    # Round trip: propagate the initial Lambert velocity with farnocchia and
    # check the orbit returns to the target position at `tof`.
    v0, _ = our_iod.izzo(K, R0, R1, TOF, 0, True, True, 35, 1e-8)
    end, _ = ours.farnocchia(K, R0, v0, TOF)
    assert_close(end, R1, rtol=1e-6, label="izzo round trip")


def test_izzo_rejects_collinear():
    with pytest.raises(ValueError, match="collinear"):
        our_iod.izzo(K, R0, R0 * 2.0, TOF, 0, True, True, 35, 1e-8)


def test_vallado_iod_scalar():
    v0_o, v_o = our_iod.vallado(K, R0, R1, TOF, 0, True, True, 100, 1e-8)
    v0_u, v_u = their_iod.vallado(K, R0, R1, TOF, 0, True, True, 100, 1e-8)
    assert_close(v0_o, v0_u, label="vallado v0")
    assert_close(v_o, v_u, label="vallado v")


def test_vallado_iod_rejects_multirevolution():
    with pytest.raises(NotImplementedError, match="Multi-revolution"):
        our_iod.vallado(K, R0, R1, TOF, 1, True, True, 100, 1e-8)


def test_vallado_iod_rejects_180_degree_phase():
    with pytest.raises(RuntimeError, match="phase angle is 180"):
        our_iod.vallado(K, R0, -R0, TOF, 0, True, True, 100, 1e-8)


def _start_state():
    r, v = their_elements.coe2rv(K, 11000.0, 0.3, 0.9, 0.2, 1.1, 2.0)
    return np.asarray(r), np.asarray(v)


@pytest.mark.parametrize("tof", [1.0, 600.0, 5000.0, 20000.0])
def test_propagators_agree_with_upstream(tof):
    r0, v0 = _start_state()
    assert_close(
        ours.farnocchia(K, r0, v0, tof)[0], their_prop.farnocchia(K, r0, v0, tof)[0],
        label="farnocchia",
    )
    assert_close(
        ours.markley(K, r0, v0, tof)[0], their_prop.markley(K, r0, v0, tof)[0],
        label="markley",
    )
    assert_close(
        ours.gooding(K, r0, v0, tof)[0], their_prop.gooding(K, r0, v0, tof)[0],
        label="gooding",
    )


def test_propagators_agree_with_each_other():
    # Three independent universal/anomaly formulations of the same orbit.
    r0, v0 = _start_state()
    f = ours.farnocchia(K, r0, v0, 3600.0)[0]
    m = ours.markley(K, r0, v0, 3600.0)[0]
    g = ours.gooding(K, r0, v0, 3600.0)[0]
    assert_close(f, m, label="farnocchia vs markley")
    assert_close(f, g, label="farnocchia vs gooding")


def test_propagators_batch():
    r0, v0 = _start_state()
    tofs = np.linspace(1.0, 20000.0, 64)
    got = ours.farnocchia(K, r0, v0, tofs)[0]
    for i, t in enumerate(tofs):
        assert_close(got[i], their_prop.farnocchia(K, r0, v0, t)[0], label="batch")


def test_vallado_lagrange_coefficients():
    r0, v0 = _start_state()
    got = ours.vallado(K, r0, v0, 600.0, 100)
    ref = their_prop.vallado(K, r0, v0, 600.0, 100)
    for i, name in enumerate(["f", "g", "fdot", "gdot"]):
        assert_close(got[i], ref[i], label=f"vallado uni {name}")


def test_vallado_universal_matches_farnocchia():
    r0, v0 = _start_state()
    f, g, fdot, gdot = ours.vallado(K, r0, v0, 600.0, 100)
    r1, v1 = _start_state()
    end_r, end_v = ours.farnocchia(K, r0, v0, 600.0)
    # r = f r0 + g v0 and v = fdot r0 + gdot v0 define the same transfer.
    assert_close(f * r0 + g * v0, end_r, rtol=1e-7, label="lagrange r")
    assert_close(fdot * r0 + gdot * v0, end_v, rtol=1e-6, label="lagrange v")


def test_func_twobody():
    r0, v0 = _start_state()
    got = ours.func_twobody(0.0, np.concatenate([r0, v0]), K)
    ref = their_prop.func_twobody(0.0, np.concatenate([r0, v0]), K)
    assert_close(got, ref, label="func_twobody")
    # A circular orbit has centripetal acceleration mu / r^2.
    speed = np.sqrt(K / 7000.0)
    got = ours.func_twobody(
        0.0, np.array([7000.0, 0.0, 0.0, 0.0, speed, 0.0]), K
    )
    assert_close(got[3], -(K / 7000.0**2), label="centripetal")
    assert_close(got[4], 0.0, label="centripetal y")


@pytest.mark.parametrize("name", ["S_x", "dS_x_alt", "d2S_x_alt"])
def test_farnocchia_series(name):
    ecc = np.full(25, 0.999)
    x = np.full(25, 0.1)
    assert_close(
        getattr(ours, name)(ecc, x), us(getattr(their_farnocchia, name), ecc, x),
        label=name,
    )


def test_farnocchia_series_terms():
    # Upstream's dS_x_alt and d2S_x_alt are the same series with the k-th
    # coefficient multiplied by (2k+3) and (2k+3)(2k+2); check the sums term
    # by term against a plain Python transcription of that definition.
    ecc, x, atol = 0.999, 0.1, 1e-12
    for fn, factor in (
        (ours.S_x, lambda k: 1.0),
        (ours.dS_x_alt, lambda k: 2 * k + 3),
        (ours.d2S_x_alt, lambda k: (2 * k + 3) * (2 * k + 2)),
    ):
        total = 0.0
        k = 0.0
        while True:
            old = total
            total += (ecc - 1.0 / (2 * k + 3)) * factor(k) * x**k
            k += 1.0
            if abs(total - old) < atol:
                break
        assert abs(fn(ecc, x) - total) < 1e-12


def test_near_parabolic_kepler():
    ecc = np.full(25, 0.999)
    D = np.linspace(-3.0, 3.0, 25)
    assert_close(
        ours.D_to_M_near_parabolic(D, ecc),
        us(their_farnocchia.D_to_M_near_parabolic, D, ecc),
        label="D_to_M_near_parabolic",
    )
    assert_close(
        ours.M_to_D_near_parabolic(D, ecc),
        us(their_farnocchia.M_to_D_near_parabolic, D, ecc),
        label="M_to_D_near_parabolic",
    )


@pytest.mark.parametrize("ecc", [0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 2.0])
def test_delta_t_from_nu_round_trip(ecc):
    nu = np.linspace(-1.0, 1.0, 17)
    dt = ours.delta_t_from_nu(nu, ecc, K, 8000.0)
    back = ours.nu_from_delta_t(dt, ecc, K, 8000.0)
    assert_close(back, nu, rtol=1e-6, label=f"delta_t round trip ecc={ecc}")


def test_delta_t_from_nu_matches_upstream():
    ecc = np.linspace(0.05, 0.95, 25)
    nu = np.linspace(-1.5, 1.5, 25)
    assert_close(
        ours.delta_t_from_nu(nu, ecc, K, 8000.0),
        us(their_farnocchia.delta_t_from_nu, nu, ecc, K, 8000.0),
        label="delta_t_from_nu",
    )
    assert_close(
        ours.nu_from_delta_t(nu, ecc, K, 8000.0),
        us(their_farnocchia.nu_from_delta_t, nu, ecc, K, 8000.0),
        label="nu_from_delta_t",
    )


def test_farnocchia_coe_matches_upstream():
    p, ecc, inc, raan, argp, nu = their_elements.rv2coe(K, *_start_state())
    cols = [np.full(25, v) for v in (p, ecc, inc, raan, argp, nu)]
    tofs = np.linspace(1.0, 20000.0, 25)
    assert_close(
        ours.farnocchia_coe(K, *cols, tofs),
        us(their_farnocchia.farnocchia_coe, K, *cols, tofs),
        label="farnocchia_coe",
    )


def test_markley_and_gooding_coe_match_upstream():
    p, ecc, inc, raan, argp, nu = their_elements.rv2coe(K, *_start_state())
    cols = [np.full(25, v) for v in (p, ecc, inc, raan, argp, nu)]
    tofs = np.linspace(1.0, 20000.0, 25)
    assert_close(
        ours.markley_coe(K, *cols, tofs),
        us(their_prop.markley_coe, K, *cols, tofs),
        label="markley_coe",
    )
    assert_close(
        ours.gooding_coe(K, *cols, tofs),
        us(their_prop.gooding_coe, K, *cols, tofs),
        label="gooding_coe",
    )


def test_markley_coe_keeps_the_shared_sin_and_cos():
    """The `sin(E)`/`cos(E)` hoisting must not change a single operation.

    `markley_coe` builds `f0..f3` from one `sin(E)` and one `cos(E)` instead of
    re-evaluating each twice.  The arithmetic and its order are otherwise the
    upstream ones, so the result has to agree with the literal double-eval
    formulation to the last bit, not merely to the parity tolerance.
    """
    import math

    def markley_coe_double_eval(k, p, ecc, nu, tof):
        PI = math.pi
        TWO_PI = 2.0 * math.pi
        E = 2.0 * math.atan(
            math.sqrt((1.0 - ecc) / (1.0 + ecc)) * math.tan(nu / 2.0)
        )
        M0 = E - ecc * math.sin(E)
        a = p / (1.0 - ecc * ecc)
        M = M0 + math.sqrt(k / (a * a * a)) * tof
        M = (M + PI) % TWO_PI - PI
        alpha = (3 * PI**2 + 1.6 * (PI - abs(M)) / (1 + ecc)) / (PI**2 - 6)
        d = 3 * (1 - ecc) + alpha * ecc
        q = 2 * alpha * d * (1 - ecc) - M**2
        r = 3 * alpha * d * (d - 1 + ecc) * M + M**3
        w = (abs(r) + math.sqrt(q**3 + r**2)) ** (2.0 / 3.0)
        E = (2 * r * w / (w**2 + w * q + q**2) + M) / d
        f0 = E - ecc * math.sin(E) - M          # sin(E)
        f1 = 1 - ecc * math.cos(E)              # cos(E)
        f2 = ecc * math.sin(E)                  # sin(E) again
        f3 = ecc * math.cos(E)                  # cos(E) again
        f4 = -f2
        delta3 = -f0 / (f1 - 0.5 * f0 * f2 / f1)
        delta4 = -f0 / (f1 + 0.5 * delta3 * f2 + 1.0 / 6.0 * delta3**2 * f3)
        delta5 = -f0 / (
            f1
            + 0.5 * delta4 * f2
            + 1.0 / 6.0 * delta4**2 * f3
            + 1.0 / 24.0 * delta4**3 * f4
        )
        E += delta5
        return 2.0 * math.atan(
            math.sqrt((1.0 + ecc) / (1.0 - ecc)) * math.tan(E / 2.0)
        )

    rng = np.random.default_rng(31)
    n = 64
    p = np.full(n, 11000.0)
    ecc = rng.uniform(0.0, 0.8, n)
    nu = rng.uniform(-3.0, 3.0, n)
    tof = rng.uniform(1.0, 20000.0, n)
    got = ours.markley_coe(K, p, ecc, np.full(n, 1.0), np.full(n, 2.0),
                           np.full(n, 3.0), nu, tof)
    want = np.array([markley_coe_double_eval(K, p[j], ecc[j], nu[j], tof[j])
                     for j in range(n)])
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_gooding_rejects_non_elliptic():
    # Upstream raises NotImplementedError; the kernel returns NaN instead,
    # because an abi("C") entry point cannot raise.
    p, _, inc, raan, argp, nu = their_elements.rv2coe(K, *_start_state())
    for ecc in (1.0, 1.5, 2.0):
        assert np.isnan(
            ours.gooding_coe(K, p, ecc, inc, raan, argp, nu, 600.0, 150, 1e-8)
        )
