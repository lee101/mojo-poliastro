"""Port of `poliastro.core.elements`.

Conversions between the classical orbital elements (COE), the modified
equinoctial elements (MEE) and the position/velocity state vector.  Functions
appear in upstream's source order.

DIVERGENCE: Mojo structs are move-only in this dialect, so the functions
upstream returns as numpy arrays fill a caller-provided `Mat3` through a
leading `out` argument, and the ones that return several scalars return a
`SIMD` lane vector that the caller unpacks in upstream's order.  The lane
width is 8 rather than 6 because Mojo only has power-of-two SIMD widths, and
the trailing lanes must be written explicitly: a short lane list compiles but
crashes at run time.
"""

from std.math import abs, acos, atan, atan2, cos, log, pi, sin, sqrt, tan

from mojopoliastro.angles import E_to_nu, F_to_nu
from mojopoliastro.linalg import V3, cross, dot, norm
from mojopoliastro.util import Mat3, matmul, rotation_matrix

comptime TWO_PI: Float64 = 6.283185307179586


def eccentricity_vector(mut out: V3, k: Float64, r: V3, v: V3):
    r"""Eccentricity vector.

    .. math::

       \vec{e} = \frac{1}{\mu} \left(
           \left( v^{2} - \frac{\mu}{r}\right ) \vec{r}  - (\vec{r} \cdot \vec{v})\vec{v}
       \right)
    """
    out = (r * (dot(v, v) - k / norm(r)) - v * dot(r, v)) / k


def circular_velocity(k: Float64, a: Float64) -> Float64:
    r"""Circular velocity for a given body.

    .. math::

       v = \sqrt{\frac{\mu}{a}}
    """
    return sqrt(k / a)


def rv_pqw(mut out: Mat3, k: Float64, p: Float64, ecc: Float64, nu: Float64):
    r"""Returns r and v vectors in perifocal frame.

    Row 0 is the position, row 1 the velocity, row 2 is unused.

    .. math::

        \vec{r} = \frac{p}{1 + e\cos(\theta)}
        \begin{bmatrix}\cos(\theta)\\ \sin(\theta)\\ 0\end{bmatrix}

        \vec{v} = \sqrt{\frac{\mu}{p}}
        \begin{bmatrix}-\sin(\theta)\\ e+\cos(\theta)\\ 0\end{bmatrix}
    """
    var c_nu = cos(nu)
    var s_nu = sin(nu)
    var pqw = Mat3()
    pqw.set(0, 0, c_nu)
    pqw.set(0, 1, s_nu)
    pqw.set(0, 2, 0.0)
    pqw.set(1, 0, -s_nu)
    pqw.set(1, 1, ecc + c_nu)
    pqw.set(1, 2, 0.0)
    # upstream multiplies by the 2x1 column [p / (1 + e cos nu), sqrt(k / p)],
    # which numpy broadcasts across the three columns of the 2x3.  Both
    # factors are the same in all three columns, so they are computed once.
    var rp = p / (1 + ecc * c_nu)
    var vp = sqrt(k / p)
    for j in range(3):
        out.set(0, j, pqw.get(0, j) * rp)
        out.set(1, j, pqw.get(1, j) * vp)


def coe_rotation_matrix(mut out: Mat3, inc: Float64, raan: Float64, argp: Float64):
    """Create a rotation matrix for coe transformation"""
    var r = Mat3()
    rotation_matrix(r, raan, 2)
    var r_inc = Mat3()
    rotation_matrix(r_inc, inc, 0)
    var r_tmp = Mat3()
    matmul(r_tmp, r, r_inc)
    var r_argp = Mat3()
    rotation_matrix(r_argp, argp, 2)
    matmul(out, r_tmp, r_argp)


def coe2rv(
    mut out: Mat3,
    k: Float64,
    p: Float64,
    ecc: Float64,
    inc: Float64,
    raan: Float64,
    argp: Float64,
    nu: Float64,
):
    r"""Converts from classical orbital to state vectors.

    Row 0 of `out` is `r_ijk`, row 1 is `v_ijk`.
    """
    var pqw = Mat3()
    rv_pqw(pqw, k, p, ecc, nu)
    var rm = Mat3()
    coe_rotation_matrix(rm, inc, raan, argp)

    # pqw @ rm.T
    for i in range(2):
        for j in range(3):
            var acc = 0.0
            for t in range(3):
                acc += pqw.get(i, t) * rm.get(j, t)
            out.set(i, j, acc)


def coe2mee(
    mut out: SIMD[DType.float64, 8],
    p: Float64,
    ecc: Float64,
    inc: Float64,
    raan: Float64,
    argp: Float64,
    nu: Float64,
) -> Int:
    r"""Converts from classical orbital elements to modified equinoctial elements.

    .. math::
        \begin{align}
        p &= a(1-e^2) \\
        f &= e\cos(\omega + \Omega) \\
        g &= e\sin(\omega + \Omega) \\
        h &= \tan(\frac{i}{2})\cos(\Omega) \\
        k &= \tan(\frac{i}{2})\sin(\Omega) \\
        L &= \Omega + \omega + \theta \\
        \end{align}

    Returns 0 on success, -1 for the 180 degree inclination that upstream
    rejects with a ValueError.
    """
    if inc == pi:
        return -1
    var lonper = raan + argp
    var f = ecc * cos(lonper)
    var g = ecc * sin(lonper)
    var h = tan(inc / 2) * cos(raan)
    var k = tan(inc / 2) * sin(raan)
    var L = lonper + nu
    out = SIMD[DType.float64, 8](p, f, g, h, k, L, 0.0, 0.0)
    return 0


def rv2coe(
    mut out: SIMD[DType.float64, 8], k: Float64, r: V3, v: V3, tol: Float64 = 1e-8
):
    r"""Converts from vectors to classical orbital elements."""
    var h = cross(r, v)
    var n = cross(V3(0.0, 0.0, 1.0), h)
    var e = V3(0.0, 0.0, 0.0)
    eccentricity_vector(e, k, r, v)
    var ecc = norm(e)
    var p = dot(h, h) / k
    # `norm(h)` is used by `inc` and again by `nu`/`argp` in the general
    # branch below; it is the same value, so it is computed once.
    var inc = acos(h.z / norm(h))
    var circular = ecc < tol
    var equatorial = abs(inc) < tol
    var raan: Float64
    var argp: Float64
    var nu: Float64
    if equatorial and (not circular):
        raan = 0.0
        argp = atan2(e.y, e.x) % TWO_PI
        nu = atan2(dot(h, cross(e, r)) / norm(h), dot(r, e))
    elif (not equatorial) and circular:
        raan = atan2(n.y, n.x) % TWO_PI
        argp = 0.0
        nu = atan2(dot(r, cross(h, n)) / norm(h), dot(r, n))
    elif equatorial and circular:
        raan = 0.0
        argp = 0.0
        nu = atan2(r.y, r.x) % TWO_PI
    else:
        var a = p / (1 - ecc**2)
        var ka = k * a
        if a > 0.0:
            var e_se = dot(r, v) / sqrt(ka)
            var e_ce = norm(r) * dot(v, v) / k - 1
            nu = E_to_nu(atan2(e_se, e_ce), ecc)
        else:
            var e_sh = dot(r, v) / sqrt(-ka)
            var e_ch = norm(r) * norm(v) ** 2 / k - 1
            nu = F_to_nu(log((e_ch + e_sh) / (e_ch - e_sh)) / 2, ecc)
        raan = atan2(n.y, n.x) % TWO_PI
        var px = dot(r, n)
        var py = dot(r, cross(h, n)) / norm(h)
        argp = (atan2(py, px) - nu) % TWO_PI
    nu = (nu + pi) % TWO_PI - pi
    out = SIMD[DType.float64, 8](p, ecc, inc, raan, argp, nu, 0.0, 0.0)


def mee2coe(
    mut out: SIMD[DType.float64, 8],
    p: Float64,
    f: Float64,
    g: Float64,
    h: Float64,
    k: Float64,
    L: Float64,
):
    r"""Converts from modified equinoctial to classical orbital elements.

    .. math::

        \begin{align}
            p &= a(1 - e^{2})\\
            e &= \sqrt{f^{2} + g^{2}}\\
            i &= 2\arctan{(\sqrt{h^{2} + k^{2}})}\\
            raan &= atan2(k, h) \pmod{2\pi}\\
            argp &= (atan2(g, f) - raan) \pmod{2\pi}\\
            nu &= (L - atan2(g, f)) \pmod{2\pi}\\
        \end{align}
    """
    var ecc = sqrt(f**2 + g**2)
    var inc = 2 * atan(sqrt(h**2 + k**2))
    var lonper = atan2(g, f)
    var raan = atan2(k, h) % TWO_PI
    var argp = (lonper - raan) % TWO_PI
    var nu = (L - lonper) % TWO_PI
    out = SIMD[DType.float64, 8](p, ecc, inc, raan, argp, nu, 0.0, 0.0)


def mee2rv(
    mut out: Mat3,
    p: Float64,
    f: Float64,
    g: Float64,
    h: Float64,
    k: Float64,
    L: Float64,
):
    """Position and velocity vector from modified equinoctial elements.

    Equations 3a and 3b of
    https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf
    """
    # upstream evaluates `cos(L)` five times, `sin(L)` three, and
    # `sqrt(k / p)` twice per call; every one of them is the same value.
    var c_L = cos(L)
    var s_L = sin(L)
    var w = 1 + f * c_L + g * s_L
    var r = p / w
    var s2 = 1 + h**2 + k**2
    var alpha2 = h**2 - k**2
    var alpha4 = alpha2 * alpha2
    var hk2 = 2.0 * h * k
    var vk = sqrt(k / p)

    var rx = (r / s2) * (c_L + alpha4 * c_L + hk2 * s_L)
    var ry = (r / s2) * (s_L - alpha4 * s_L + hk2 * c_L)
    var rz = (2 * r / s2) * (h * s_L - k * c_L)

    var vx = (
        (-1 / s2)
        * vk
        * (
            s_L + alpha2 * s_L - hk2 * c_L + g - 2.0 * f * h * k + alpha2 * g
        )
    )
    var vy = (
        (-1 / s2)
        * vk
        * (
            -c_L + alpha2 * c_L + hk2 * s_L - f + 2.0 * g * h * k + alpha2 * f
        )
    )
    var vz = (2 / s2) * vk * (h * c_L + k * s_L + f * h + g * k)

    out.set(0, 0, rx)
    out.set(0, 1, ry)
    out.set(0, 2, rz)
    out.set(1, 0, vx)
    out.set(1, 1, vy)
    out.set(1, 2, vz)
