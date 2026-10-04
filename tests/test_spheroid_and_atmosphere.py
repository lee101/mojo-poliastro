"""Parity for `poliastro.core.spheroid_location` and the atmosphere models."""

import numpy as np
import pytest

from parity import assert_close, us

import mojo_poliastro.core.earth_atmosphere.jacchia as our_jacchia
import mojo_poliastro.core.earth_atmosphere.util as our_atmos
import mojo_poliastro.core.spheroid_location as ours
import poliastro.core.earth_atmosphere.jacchia as their_jacchia
import poliastro.core.earth_atmosphere.util as their_atmos
import poliastro.core.spheroid_location as theirs

A_EQ = 6378.137
C_EQ = 6356.752314245


def test_cartesian_cords():
    got = ours.cartesian_cords(A_EQ, C_EQ, 0.5, 0.2, 100.0)
    ref = theirs.cartesian_cords(A_EQ, C_EQ, 0.5, 0.2, 100.0)
    assert_close(got, np.asarray(ref), label="cartesian_cords")


def test_normal_and_tangential_frame():
    cords = ours.cartesian_cords(A_EQ, C_EQ, 0.5, 0.2, 100.0)
    n = ours.N(A_EQ, A_EQ, C_EQ, cords)
    ref = np.asarray(theirs.N(A_EQ, A_EQ, C_EQ, np.asarray(cords)))
    assert_close(n, ref, label="N")
    u, v = ours.tangential_vecs(n)
    ru, rv = theirs.tangential_vecs(ref)
    assert_close(u, ru, label="tangential u")
    assert_close(v, rv, label="tangential v")
    # The three vectors are an orthonormal right-handed frame.
    assert_close(np.dot(n, u), 0.0, label="n.u")
    assert_close(np.dot(n, v), 0.0, label="n.v")
    assert_close(np.dot(u, v), 0.0, label="u.v")
    assert_close(np.linalg.norm(n), 1.0, label="|n|")


def test_flattening_and_curvature():
    assert_close(ours.f(A_EQ, C_EQ), theirs.f(A_EQ, C_EQ), label="f")
    for lat in (-1.2, -0.4, 0.0, 0.4, 1.2):
        assert_close(
            ours.radius_of_curvature(A_EQ, C_EQ, lat),
            theirs.radius_of_curvature(A_EQ, C_EQ, lat),
            label="radius_of_curvature",
        )


def test_distance_and_visibility():
    cords = ours.cartesian_cords(A_EQ, C_EQ, 0.5, 0.2, 100.0)
    n = ours.N(A_EQ, A_EQ, C_EQ, cords)
    ref_cords = np.asarray(theirs.cartesian_cords(A_EQ, C_EQ, 0.5, 0.2, 100.0))
    ref_n = np.asarray(theirs.N(A_EQ, A_EQ, C_EQ, ref_cords))
    assert_close(
        ours.distance(cords, 0.0, 0.0, 10000.0),
        theirs.distance(ref_cords, 0.0, 0.0, 10000.0),
        label="distance",
    )
    # Above the local horizon the point is visible, below it is not.
    for height in (-1000.0, 0.0, 1.0, 10000.0, 20000.0):
        ours_sees = bool(ours.is_visible(cords, 0.0, 0.0, height, n))
        theirs_sees = bool(theirs.is_visible(ref_cords, 0.0, 0.0, height, ref_n))
        assert ours_sees == theirs_sees, f"visibility disagreement at {height}"
    assert bool(ours.is_visible(cords, 0.0, 0.0, 20000.0, n)) is True
    assert bool(ours.is_visible(cords, 0.0, 0.0, 0.0, n)) is False


def test_cartesian_to_ellipsoidal_round_trip():
    lat, lon, h = 0.5, 0.2, 100.0
    cords = np.asarray(ours.cartesian_cords(A_EQ, C_EQ, lat, lon, h))
    got = ours.cartesian_to_ellipsoidal(A_EQ, C_EQ, *cords)
    ref = theirs.cartesian_to_ellipsoidal(A_EQ, C_EQ, *cords)
    for a, b, name in zip(got, ref, ["lat", "lon", "h"]):
        assert_close(a, b, label=f"cartesian_to_ellipsoidal {name}")
    # The inverse of cartesian_cords is its own companion.
    assert_close(got[0], lat, rtol=1e-6, label="lat round trip")
    assert_close(got[1], lon, rtol=1e-6, label="lon round trip")
    assert_close(got[2], h, rtol=1e-6, label="h round trip")



def test_geodetic_helpers_batch_a_batched_vector_against_scalar_scalars():
    """A 0-d scalar paired with an (n, 3) vector is one batch of n.

    Every buffer the kernel is handed is read `n` elements deep, so a scalar
    argument that stayed one element long would be read past its end and the
    later rows would come back as whatever followed it in the heap.
    """
    cords = np.tile(np.array([6000.0, 1000.0, 2000.0]), (5, 1))

    got = np.asarray(ours.N(A_EQ, A_EQ, C_EQ, cords))
    want = np.array(
        [np.asarray(theirs.N(A_EQ, A_EQ, C_EQ, c)) for c in cords]
    )
    assert got.shape == want.shape
    assert_close(got, want, label="batched N")

    x, y, z = cords[:, 0], cords[:, 1], cords[:, 2]
    got = np.asarray(ours.cartesian_to_ellipsoidal(A_EQ, C_EQ, x, y, z))
    want = np.array(
        [theirs.cartesian_to_ellipsoidal(A_EQ, C_EQ, *c) for c in cords]
    ).T
    assert_close(got, want, label="batched cartesian_to_ellipsoidal")

    one = np.array([6000.0, 1000.0, 2000.0])
    got = np.asarray(ours.distance(one, x, y, z))
    want = np.array([theirs.distance(one, *c) for c in cords])
    assert_close(got, want, label="batched distance")

    # A 1-D 3-vector argument is one point repeated across the batch, not a
    # single-element buffer read n times.
    n = ours.N(A_EQ, A_EQ, C_EQ, np.asarray(ours.cartesian_cords(A_EQ, C_EQ, 0.5, 0.2, 100.0)))
    assert n.shape == (3,)
    assert np.isfinite(n).all()


def test_geodetic_helpers_reject_ragged_batches():
    with pytest.raises(ValueError, match="cannot broadcast"):
        ours.N(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros((5, 3)))

R0 = 6356.766


def test_altitude_conversions():
    z = np.linspace(0.0, 1000.0, 25)
    assert_close(
        our_atmos.geometric_to_geopotential(z, R0),
        us(their_atmos.geometric_to_geopotential, z, R0),
        label="geometric_to_geopotential",
    )
    assert_close(
        our_atmos.geopotential_to_geometric(z, R0),
        us(their_atmos.geopotential_to_geometric, z, R0),
        label="geopotential_to_geometric",
    )
    assert our_atmos.z_to_h is our_atmos.geometric_to_geopotential
    assert our_atmos.h_to_z is our_atmos.geopotential_to_geometric


def test_altitude_conversions_are_inverses():
    z = np.linspace(0.0, 800.0, 25)
    back = our_atmos.geopotential_to_geometric(
        our_atmos.geometric_to_geopotential(z, R0), R0
    )
    assert_close(back, z, rtol=1e-10, label="z/h round trip")


def test_gravity():
    z = np.linspace(0.0, 1000.0, 25)
    assert_close(
        our_atmos.gravity(z, 9.80665, R0), us(their_atmos.gravity, z, 9.80665, R0),
        label="gravity",
    )
    assert_close(our_atmos.gravity(0.0, 9.80665, R0), 9.80665, label="g0")


def test_check_altitude():
    z = np.linspace(0.0, 1000.0, 25)
    for geometric in (True, False):
        got = our_atmos.check_altitude(z, R0, geometric)
        ref = us(their_atmos._check_altitude, z, R0, geometric)
        assert_close(got[0], ref[0], label="check_altitude z")
        assert_close(got[1], ref[1], label="check_altitude h")


PROFILE_NAMES = ("Z", "T", "CN2", "CO2", "CO", "CAr", "CHe", "CH", "CM", "WM")


def their_profile(alt, Texo):
    e5m = np.zeros(11)
    e6p = np.zeros(11)
    out = their_jacchia._altitude_profile(alt, Texo, 0.0, 0.0, e5m, e6p)
    return dict(zip(PROFILE_NAMES, [np.asarray(x, dtype=float) for x in out]))


@pytest.mark.parametrize("alt,Texo", [(90, 288.0), (200, 1000.0), (400, 1000.0),
                                      (1000, 800.0), (2500, 1000.0)])
def test_jacchia_altitude_profile(alt, Texo):
    got = our_jacchia.altitude_profile(alt, Texo)
    ref = their_profile(alt, Texo)
    assert len(got) == len(PROFILE_NAMES)
    for name in PROFILE_NAMES:
        assert got[name].shape == (alt + 1,)
        # The composition is built from two chained recurrences over up to
        # 2500 steps, each seeded by exp/log, so Mojo's ~1e-11 fast-math
        # difference compounds.  1e-7 is still three orders below the
        # physical accuracy of the model.
        assert_close(got[name], ref[name], rtol=1e-7, label=f"jacchia {name}")


def test_jacchia_rejects_out_of_range():
    for alt in (89, 2501):
        with pytest.raises(ValueError, match="90km - 2500km"):
            our_jacchia.altitude_profile(alt, 1000.0)


def test_jacchia_temperature_is_monotonic():
    # Above the thermopause the US Standard 1976 profile is monotone in the
    # 90-500 km band the model covers.
    prof = our_jacchia.altitude_profile(500, 1000.0)
    t = prof["T"][90:500]
    assert np.all(np.diff(t) > 0)


def test_jacchia_h_correction_never_reads_past_the_buffers():
    """`_H_correction` indexes WM[500] and CM[500], so its guard has to
    account for the `int(alt + 1)` the profile applies to its own length.

    Upstream tests `500 <= alt` against the already-incremented length, so at
    a requested altitude of 499 it calls `_H_correction` on a 500-element
    list and raises IndexError.  The port reads the same index and must not
    touch it, so 499 is the boundary it still answers for and 500 is the
    first altitude with H densities.
    """
    without_h = our_jacchia.altitude_profile(499, 1000.0)
    assert without_h["T"].shape == (500,)
    assert without_h["CH"][150] == 0.0

    with_h = our_jacchia.altitude_profile(500, 1000.0)
    assert with_h["T"].shape == (501,)
    assert with_h["CH"][150] != 0.0
    assert np.isfinite(with_h["T"]).all()
    assert np.isfinite(with_h["WM"]).all()
