"""Parity for `poliastro._math.special` and `poliastro.core.util`."""

import numpy as np
import pytest

from parity import assert_close, us

import mojo_poliastro._math.special as our_special
import mojo_poliastro.core.util as ours
import poliastro._math.special as their_special
import poliastro.core.util as theirs


@pytest.mark.parametrize("name", ["stumpff_c2", "stumpff_c3"])
def test_stumpff(name):
    psi = np.linspace(-4.0, 4.0, 41)
    assert_close(
        getattr(our_special, name)(psi), us(getattr(their_special, name), psi),
        label=name,
    )


def test_stumpff_matches_closed_forms():
    # psi > 1 and psi < -1 have closed forms in the two branches.
    psi = np.linspace(1.5, 6.0, 21)
    c2 = (1 - np.cos(np.sqrt(psi))) / psi
    c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    assert_close(our_special.stumpff_c2(psi), c2, label="c2 closed form")
    assert_close(our_special.stumpff_c3(psi), c3, label="c3 closed form")

    psi = np.linspace(-6.0, -1.5, 21)
    c2 = (np.cosh(np.sqrt(-psi)) - 1) / -psi
    c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))
    assert_close(our_special.stumpff_c2(psi), c2, label="c2 closed form neg")
    assert_close(our_special.stumpff_c3(psi), c3, label="c3 closed form neg")


def test_stumpff_small_argument_series():
    # The |psi| <= 1 branch is the Taylor series; c2 -> 1/2, c3 -> 1/6.
    assert abs(our_special.stumpff_c2(0.0) - 0.5) < 1e-15
    assert abs(our_special.stumpff_c3(0.0) - 1.0 / 6.0) < 1e-15


def test_hyp2f1b():
    x = np.linspace(-0.9, 0.9, 37)
    assert_close(
        our_special.hyp2f1b(x), us(their_special.hyp2f1b, x), label="hyp2f1b"
    )
    assert our_special.hyp2f1b(1.0) == np.inf
    assert_close(our_special.hyp2f1b(0.0), 1.0, label="hyp2f1b at zero")


def test_rotation_matrix():
    angle = np.linspace(-3.0, 3.0, 25)
    for axis in (0, 1, 2):
        assert_close(
            ours.rotation_matrix(angle, axis), theirs.rotation_matrix(angle, axis),
            label=f"rotation_matrix axis {axis}",
        )
    m = ours.rotation_matrix(0.5, 2)
    assert m.shape == (3, 3)
    assert_close(m @ m.T, np.eye(3), label="rotation_matrix orthogonality")


def test_alinspace():
    for kwargs in ({}, {"stop": 6.0}, {"stop": -6.0}, {"endpoint": False},
                   {"stop": 6.0, "endpoint": False}, {"num": 3},
                   {"num": 0}, {"num": 1}, {"num": 0, "endpoint": False},
                   {"num": 1, "endpoint": False}):
        got = ours.alinspace(0.3, **kwargs)
        ref = theirs.alinspace(0.3, **kwargs)
        assert got.shape == ref.shape, f"alinspace{kwargs} shape"
        assert_close(got, ref, label=f"alinspace{kwargs}")


def test_alinspace_wraps_past_2pi():
    # stop <= start adds whole revolutions, upstream's job when sampling a
    # full orbit.
    assert_close(ours.alinspace(3.0, 1.0, 9), theirs.alinspace(3.0, 1.0, 9),
                 label="alinspace wrap")


def test_alinspace_degenerate_lengths_are_not_divided_by_zero():
    """`num` of 0 and 1 leave no interval to step across.

    The kernel divides by `count - 1` to space the samples and pins the last
    one to the stop value, so a length of 1 divides by zero and a length of 0
    writes to index -1.  Both have to come back exactly as numpy's linspace
    returns them, and neither may write outside the caller's buffer.
    """
    for num in (0, 1):
        for endpoint in (True, False):
            got = ours.alinspace(0.3, num=num, endpoint=endpoint)
            ref = theirs.alinspace(0.3, num=num, endpoint=endpoint)
            assert got.shape == ref.shape
            assert np.array_equal(got, ref), f"alinspace num={num} endpoint={endpoint}"
    assert np.isfinite(ours.alinspace(0.3, num=1)).all()


def test_spherical_to_cartesian():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(25, 3))
    v[:, 0] = np.abs(v[:, 0]) * 3.0
    assert_close(
        ours.spherical_to_cartesian(v), us(theirs.spherical_to_cartesian, v),
        label="spherical_to_cartesian",
    )


def test_planetocentric_to_AltAz():
    theta = np.linspace(0.0, 6.0, 25)
    phi = np.linspace(-1.5, 1.5, 25)
    assert_close(
        ours.planetocentric_to_AltAz(theta, phi),
        us(theirs.planetocentric_to_AltAz, theta, phi),
        label="planetocentric_to_AltAz",
    )
