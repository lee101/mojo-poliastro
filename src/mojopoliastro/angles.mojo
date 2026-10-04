"""Port of `poliastro.core.angles`.

Conversions between true, eccentric, hyperbolic, parabolic and mean anomalies,
and the Newton iterations that invert Kepler's equation in both the elliptic
and the hyperbolic regime.  Functions appear in upstream's source order.
"""

from std.math import asinh, atan, atan2, atanh, cos, cosh, nan, sin, sinh, sqrt, tan, tanh

comptime PI: Float64 = 3.141592653589793
comptime TWO_PI: Float64 = 6.283185307179586


def _kepler_equation(E: Float64, M: Float64, ecc: Float64) -> Float64:
    return E_to_M(E, ecc) - M


def _kepler_equation_prime(E: Float64, M: Float64, ecc: Float64) -> Float64:
    return 1 - ecc * cos(E)


def _kepler_equation_hyper(F: Float64, M: Float64, ecc: Float64) -> Float64:
    return F_to_M(F, ecc) - M


def _kepler_equation_prime_hyper(F: Float64, M: Float64, ecc: Float64) -> Float64:
    return ecc * cosh(F) - 1


# DIVERGENCE: upstream builds these two with `newton_factory(func, fprime)`,
# which returns a closure.  Mojo cannot pass a `def` as a value, so the
# factory's two instantiations are written out with the factory's loop body
# unchanged.  Argument order follows the call sites: x0, then `args`, then
# the factory's own `tol` / `maxiter` defaults.
def _newton_elliptic(
    x0: Float64, M: Float64, ecc: Float64, tol: Float64 = 1.48e-08, maxiter: Int = 50
) -> Float64:
    var p0 = x0
    for _ in range(maxiter):
        var fval = _kepler_equation(p0, M, ecc)
        var fder = _kepler_equation_prime(p0, M, ecc)
        var newton_step = fval / fder
        var p = p0 - newton_step
        if abs(p - p0) < tol:
            return p
        p0 = p

    return nan[DType.float64]()


def _newton_hyperbolic(
    x0: Float64, M: Float64, ecc: Float64, tol: Float64 = 1.48e-08, maxiter: Int = 50
) -> Float64:
    var p0 = x0
    for _ in range(maxiter):
        var fval = _kepler_equation_hyper(p0, M, ecc)
        var fder = _kepler_equation_prime_hyper(p0, M, ecc)
        var newton_step = fval / fder
        var p = p0 - newton_step
        if abs(p - p0) < tol:
            return p
        p0 = p

    return nan[DType.float64]()


def D_to_nu(D: Float64) -> Float64:
    r"""True anomaly from parabolic anomaly.

    .. math::

        \nu = 2 \arctan{D}
    """
    return 2.0 * atan(D)


def nu_to_D(nu: Float64) -> Float64:
    r"""Parabolic anomaly from true anomaly.

    .. math::

        B = \tan{\frac{\nu}{2}}
    """
    return tan(nu / 2.0)


def nu_to_E(nu: Float64, ecc: Float64) -> Float64:
    r"""Eccentric anomaly from true anomaly.

    .. math::

        E = 2 \arctan \left ( \sqrt{\frac{1 - e}{1 + e}} \tan{\frac{\nu}{2}} \right)
        \in (-\pi, \pi]
    """
    var E = 2 * atan(sqrt((1 - ecc) / (1 + ecc)) * tan(nu / 2))
    return E


def nu_to_F(nu: Float64, ecc: Float64) -> Float64:
    r"""Hyperbolic anomaly from true anomaly.

    .. math::

        F = 2 \operatorname{arctanh} \left( \sqrt{\frac{e-1}{e+1}} \tan{\frac{\nu}{2}} \right)
    """
    var F = 2 * atanh(sqrt((ecc - 1) / (ecc + 1)) * tan(nu / 2))
    return F


def E_to_nu(E: Float64, ecc: Float64) -> Float64:
    r"""True anomaly from eccentric anomaly.

    .. math::

        \nu = 2 \arctan \left( \sqrt{\frac{1 + e}{1 - e}} \tan{\frac{E}{2}} \right)
        \in (-\pi, \pi]
    """
    var nu = 2 * atan(sqrt((1 + ecc) / (1 - ecc)) * tan(E / 2))
    return nu


def F_to_nu(F: Float64, ecc: Float64) -> Float64:
    r"""True anomaly from hyperbolic anomaly.

    .. math::

        \nu = 2 \arctan \left( \sqrt{\frac{e + 1}{e - 1}} \tanh{\frac{F}{2}} \right)
    """
    var nu = 2 * atan(sqrt((ecc + 1) / (ecc - 1)) * tanh(F / 2))
    return nu


def M_to_E(M: Float64, ecc: Float64) -> Float64:
    r"""Eccentric anomaly from mean anomaly."""
    var E0: Float64
    if -PI < M < 0 or PI < M:
        E0 = M - ecc
    else:
        E0 = M + ecc
    var E = _newton_elliptic(E0, M, ecc)
    return E


def M_to_F(M: Float64, ecc: Float64) -> Float64:
    r"""Hyperbolic anomaly from mean anomaly."""
    var F0 = asinh(M / ecc)
    var F = _newton_hyperbolic(F0, M, ecc, 1.48e-08, 100)
    return F


def M_to_D(M: Float64) -> Float64:
    r"""Parabolic anomaly from mean anomaly."""
    var B = 3.0 * M / 2.0
    # DIVERGENCE: `** 0.5` is written `sqrt`.  Mojo's `pow` is a fast
    # approximation and disagrees with glibc's by ~1e-9 relative, which this
    # catastrophically cancelling expression amplifies; `sqrt` is the
    # bit-exact match for CPython's `x ** 0.5`.
    var A = (B + sqrt(1.0 + B**2)) ** (2.0 / 3.0)
    var D = 2 * A * B / (1 + A + A**2)
    return D


def E_to_M(E: Float64, ecc: Float64) -> Float64:
    r"""Mean anomaly from eccentric anomaly.

    .. math::

        M = E - e \sin{E}
    """
    var M = E - ecc * sin(E)
    return M


def F_to_M(F: Float64, ecc: Float64) -> Float64:
    r"""Mean anomaly from hyperbolic anomaly.

    .. math::

        M = e \sinh{F} - F
    """
    var M = ecc * sinh(F) - F
    return M


def D_to_M(D: Float64) -> Float64:
    r"""Mean anomaly from parabolic anomaly.

    .. math::

        M = \frac{1}{3} D^3 + D
    """
    var M = D + D**3 / 3
    return M


def fp_angle(nu: Float64, ecc: Float64) -> Float64:
    r"""Flight path angle.

    .. math::

        \phi = \arctan{\frac{e \sin{\nu}}{1 + e \cos{\nu}}}
    """
    return atan2(ecc * sin(nu), 1 + ecc * cos(nu))
