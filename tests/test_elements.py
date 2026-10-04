"""Parity for `poliastro.core.elements` and `poliastro._math.linalg`."""

import numpy as np
import pytest

from parity import assert_close, us

import mojo_poliastro._math.linalg as our_linalg
import mojo_poliastro.core.elements as ours
import poliastro._math.linalg as their_linalg
import poliastro.core.elements as theirs

K = 398600.4418
N = 25

P = np.full(N, 11000.0)
ECC = np.linspace(0.0, 0.9, N)
INC = np.linspace(0.05, 3.0, N)
RAAN = np.linspace(0.0, 6.0, N)
ARGP = np.linspace(0.0, 6.0, N)
NU = np.linspace(-3.0, 3.0, N)


def coe2rv_upstream():
    stacked = us(theirs.coe2rv, K, P, ECC, INC, RAAN, ARGP, NU)
    return stacked[:, 0, :], stacked[:, 1, :]


def test_circular_velocity():
    assert_close(
        ours.circular_velocity(K, P), us(theirs.circular_velocity, K, P),
        label="circular_velocity",
    )


def test_rv_pqw():
    assert_close(
        ours.rv_pqw(K, P, ECC, NU), us(theirs.rv_pqw, K, P, ECC, NU),
        label="rv_pqw",
    )


def test_coe_rotation_matrix():
    assert_close(
        ours.coe_rotation_matrix(INC, RAAN, ARGP),
        us(theirs.coe_rotation_matrix, INC, RAAN, ARGP),
        label="coe_rotation_matrix",
    )


def test_coe2rv():
    r_u, v_u = coe2rv_upstream()
    r_o, v_o = ours.coe2rv(K, P, ECC, INC, RAAN, ARGP, NU)
    assert r_o.shape == (N, 3) and v_o.shape == (N, 3)
    assert_close(r_o, r_u, label="coe2rv r")
    assert_close(v_o, v_u, label="coe2rv v")


def test_coe2rv_scalar_signature():
    # Upstream returns a pair of 3-vectors for a scalar call.
    r, v = ours.coe2rv(K, 11000.0, 0.3, 0.9, 0.2, 1.1, 2.0)
    assert r.shape == (3,) and v.shape == (3,)
    ru, vu = theirs.coe2rv(K, 11000.0, 0.3, 0.9, 0.2, 1.1, 2.0)
    assert_close(r, ru, label="coe2rv scalar r")
    assert_close(v, vu, label="coe2rv scalar v")


def test_coe2rv_many():
    r_u, v_u = coe2rv_upstream()
    r_o, v_o = ours.coe2rv_many(np.full(N, K), P, ECC, INC, RAAN, ARGP, NU)
    assert_close(r_o, r_u, label="coe2rv_many r")
    assert_close(v_o, v_u, label="coe2rv_many v")


def test_coe2rv_many_parallel_threshold():
    # The parallel branch only engages at 4096 rows; both must agree.
    n = 5000
    p = np.full(n, 11000.0)
    ecc = np.linspace(0.0, 0.9, n)
    inc = np.linspace(0.05, 3.0, n)
    raan = np.linspace(0.0, 6.0, n)
    argp = np.linspace(0.0, 6.0, n)
    nu = np.linspace(-3.0, 3.0, n)
    r_many, _ = ours.coe2rv_many(np.full(n, K), p, ecc, inc, raan, argp, nu)
    r_one, _ = ours.coe2rv(K, p, ecc, inc, raan, argp, nu)
    assert_close(r_many, r_one, label="parallel vs serial")


def test_eccentricity_vector():
    r_u, v_u = coe2rv_upstream()
    assert_close(
        ours.eccentricity_vector(K, r_u, v_u),
        us(theirs.eccentricity_vector, K, r_u, v_u),
        label="eccentricity_vector",
    )


def test_rv2coe():
    r_u, v_u = coe2rv_upstream()
    ref = us(theirs.rv2coe, K, r_u, v_u)
    got = ours.rv2coe(K, r_u, v_u)
    names = ["p", "ecc", "inc", "raan", "argp", "nu"]
    for i, name in enumerate(names):
        assert_close(got[i], ref[i], label=f"rv2coe {name}")


def test_rv2coe_round_trip():
    r_u, v_u = coe2rv_upstream()
    p, ecc, inc, raan, argp, nu = ours.rv2coe(K, r_u, v_u)
    r_o, v_o = ours.coe2rv(K, p, ecc, inc, raan, argp, nu)
    assert_close(r_o, r_u, label="rv2coe round trip r")
    assert_close(v_o, v_u, label="rv2coe round trip v")


def test_rv2coe_degenerate_branches():
    # A circular equatorial orbit takes the branch upstream special-cases.
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(K / 7000.0), 0.0])
    got = ours.rv2coe(K, r, v)
    ref = theirs.rv2coe(K, r, v)
    for i in range(6):
        assert_close(got[i], ref[i], label="degenerate rv2coe")


def test_coe2mee_mee2coe():
    mee_ref = us(theirs.coe2mee, P, ECC, INC, RAAN, ARGP, NU)
    mee_ours = ours.coe2mee(P, ECC, INC, RAAN, ARGP, NU)
    for i, name in enumerate("p f g h k L".split()):
        assert_close(mee_ours[i], mee_ref[i], label=f"coe2mee {name}")
    back_ref = us(theirs.mee2coe, *mee_ref)
    back_ours = ours.mee2coe(*mee_ref)
    for i, name in enumerate(["p", "ecc", "inc", "raan", "argp", "nu"]):
        assert_close(back_ours[i], back_ref[i], label=f"mee2coe {name}")


def test_coe2mee_scalar_returns_tuple():
    got = ours.coe2mee(11000.0, 0.3, 0.9, 0.2, 1.1, 2.0)
    assert isinstance(got, tuple) and len(got) == 6
    ref = theirs.coe2mee(11000.0, 0.3, 0.9, 0.2, 1.1, 2.0)
    for a, b in zip(got, ref):
        assert_close(a, b, label="coe2mee scalar")


def test_coe2mee_rejects_retrograde():
    with pytest.raises(ValueError, match="180 degrees"):
        ours.coe2mee(11000.0, 0.3, np.pi, 0.2, 1.1, 2.0)


def mee2rv_reference(p, f, g, h, k, L):
    """Upstream's `mee2rv`, transcribed.

    Upstream's own source cannot run: it writes `(r / s2)(...)`, which calls
    a float.  Both numba and CPython reject it, so upstream offers no
    executable oracle for this function.  The transcription is upstream's
    source with the missing operator restored.
    """
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


def test_mee2rv():
    mee = us(theirs.coe2mee, P, ECC, INC, RAAN, ARGP, NU)
    cols = [mee[i] for i in range(6)]
    ref = np.stack([mee2rv_reference(*[c[j] for c in cols]) for j in range(N)])
    r_o, v_o = ours.mee2rv(*cols)
    assert_close(r_o, ref[:, 0, :], label="mee2rv r")
    assert_close(v_o, ref[:, 1, :], label="mee2rv v")


def test_linalg_norm():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(N, 3))
    assert_close(our_linalg.norm(a), us(their_linalg.norm, a), label="norm")
    assert isinstance(our_linalg.norm(np.array([3.0, 4.0, 0.0])), float)
    assert_close(our_linalg.norm([3.0, 4.0, 0.0]), 5.0, label="norm scalar")
