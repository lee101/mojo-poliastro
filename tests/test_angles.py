"""Parity for `poliastro.core.angles`."""

import numpy as np
import pytest

from parity import assert_close, us

import mojo_poliastro.core.angles as ours
import poliastro.core.angles as theirs

ELLIPTIC = (np.linspace(-3.0, 3.0, 25), np.linspace(0.0, 0.9, 25))
HYPERBOLIC = (np.linspace(-1.5, 1.5, 25), np.linspace(1.05, 3.0, 25))


@pytest.mark.parametrize(
    "name,args",
    [
        ("nu_to_E", ELLIPTIC),
        ("E_to_nu", ELLIPTIC),
        ("E_to_M", ELLIPTIC),
        ("M_to_E", ELLIPTIC),
        ("fp_angle", ELLIPTIC),
        ("nu_to_F", HYPERBOLIC),
        ("F_to_nu", HYPERBOLIC),
        ("F_to_M", HYPERBOLIC),
        ("M_to_F", HYPERBOLIC),
    ],
)
def test_two_argument_conversions(name, args):
    a, b = args
    assert_close(
        getattr(ours, name)(a, b), us(getattr(theirs, name), a, b), label=name
    )


@pytest.mark.parametrize("name", ["D_to_nu", "nu_to_D", "M_to_D", "D_to_M"])
def test_one_argument_conversions(name):
    a = np.linspace(-3.0, 3.0, 25)
    assert_close(getattr(ours, name)(a), us(getattr(theirs, name), a), label=name)


def test_scalar_in_scalar_out():
    # upstream's calling convention: scalars in, scalars out.
    for name, args in [
        ("nu_to_E", (0.4, 0.3)),
        ("E_to_nu", (0.4, 0.3)),
        ("M_to_E", (0.4, 0.3)),
        ("fp_angle", (0.4, 0.3)),
        ("M_to_F", (0.4, 1.3)),
        ("D_to_nu", (0.4,)),
        ("M_to_D", (0.4,)),
    ]:
        got = getattr(ours, name)(*args)
        assert isinstance(got, float), f"{name} should return a float"
        assert_close(got, us(getattr(theirs, name), *args)[0], label=name)


def test_kepler_equation_round_trip():
    # M -> E -> M must return the mean anomaly it started from.
    ecc = np.linspace(0.0, 0.95, 25)
    M = np.linspace(-3.0, 3.0, 25)
    assert_close(ours.E_to_M(ours.M_to_E(M, ecc), ecc), M, label="E round trip")
    assert_close(ours.D_to_M(ours.M_to_D(M)), M, label="D round trip")


def test_newton_converges_across_the_elliptic_domain():
    ecs = np.linspace(0.0, 0.999, 241)
    got = np.array([ours.M_to_E(0.4, float(e)) for e in ecs])
    ref = np.array([theirs.M_to_E(0.4, float(e)) for e in ecs])
    assert not np.isnan(got).any() and not np.isnan(ref).any()
    assert_close(got, ref, label="elliptic domain")


def test_newton_gives_up_with_nan():
    # Upstream's Newton wrapper returns np.nan when it does not converge in
    # `maxiter`.  Outside the elliptic regime 1 - e cos(E) passes through
    # zero, and which iterate lands there first is decided in the last bit,
    # so this asserts the give-up contract on inputs where both agree.
    for ecc in (-3.0, -2.8, -2.575, -2.375):
        assert np.isnan(theirs.M_to_E(0.4, ecc))
        assert np.isnan(ours.M_to_E(0.4, ecc))


def test_broadcasting():
    # Upstream is numba-typed for scalars and cannot broadcast, so the
    # reference is an explicit double loop.
    M = np.linspace(-1.0, 1.0, 7)[:, None]
    ecc = np.linspace(0.1, 0.8, 5)[None, :]
    got = ours.M_to_E(M, ecc)
    assert got.shape == (7, 5)
    ref = np.array(
        [[theirs.M_to_E(float(M[i, 0]), float(ecc[0, j])) for j in range(5)]
         for i in range(7)]
    )
    assert_close(got, ref, label="broadcast")
