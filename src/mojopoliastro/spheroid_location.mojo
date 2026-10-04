"""Port of `poliastro.core.spheroid_location`.

Geodetic helpers for a triaxial-free oblate spheroid: the point/normal/tangent
frame at a surface point, the radius of curvature, horizon visibility, and the
cartesian-to-geodetic inversion.  Functions appear in upstream's source order.

Note upstream's `cartesian_cords` passes `_lon` where the formula reads like a
latitude; the port keeps upstream's argument order and its arithmetic, quirk
included, so results match.
"""

from std.math import atan, atan2, cos, sin, sqrt

from mojopoliastro.linalg import V3, cross, dot, norm
from mojopoliastro.util import Mat3


def cartesian_cords(
    mut out: V3, _a: Float64, _c: Float64, _lon: Float64, _lat: Float64, _h: Float64
):
    var e2 = 1 - (_c / _a) ** 2
    var N = _a / sqrt(1 - e2 * sin(_lon) ** 2)
    out.x = (N + _h) * cos(_lon) * cos(_lat)
    out.y = (N + _h) * cos(_lon) * sin(_lat)
    out.z = ((1 - e2) * N + _h) * sin(_lon)


def f(_a: Float64, _c: Float64) -> Float64:
    return 1 - _c / _a


def N(mut out: V3, a: Float64, b: Float64, c: Float64, cartesian_cords: V3):
    var x = cartesian_cords.x
    var y = cartesian_cords.y
    var z = cartesian_cords.z
    out.x = 2 * x / a**2
    out.y = 2 * y / b**2
    out.z = 2 * z / c**2
    out = out / norm(out)


def tangential_vecs(mut u: V3, mut v: V3, N: V3):
    u = V3(1.0, 0.0, 0.0)
    u = u - N * dot(u, N)
    u = u / norm(u)
    v = cross(N, u)


def radius_of_curvature(_a: Float64, _c: Float64, _lat: Float64) -> Float64:
    var e2 = 1 - (_c / _a) ** 2
    var rc = _a * (1 - e2) / (1 - e2 * sin(_lat) ** 2) ** 1.5
    return rc


def distance(cartesian_cords: V3, u: V3) -> Float64:
    var d = norm(cartesian_cords - u)
    return d


def is_visible(cartesian_cords: V3, u: V3, N: V3) -> Bool:
    var d = -(dot(N, cartesian_cords))
    var p = dot(N, u) + d
    return p >= 0


def cartesian_to_ellipsoidal(
    mut out: SIMD[DType.float64, 8], _a: Float64, _c: Float64, x: Float64, y: Float64, z: Float64
):
    var e2 = 1 - (_c / _a) ** 2
    var e2_ = e2 / (1 - e2)
    var p = sqrt(x**2 + y**2)
    var th = atan(z * _a / (p * _c))
    var lon = atan2(y, x)
    var lat = atan((z + e2_ * _c * sin(th) ** 3) / (p - e2 * _a * cos(th) ** 3))
    var v = _a / sqrt(1 - e2 * sin(lat) ** 2)
    var h: Float64
    if lat == 0.0:
        h = x / cos(lat) - v
    else:
        h = z / sin(lat) - (1 - e2) * v
    out = SIMD[DType.float64, 8](lat, lon, h, 0.0, 0.0, 0.0, 0.0, 0.0)
