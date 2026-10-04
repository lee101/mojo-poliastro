"""Port of `poliastro.core.angles` -- see `src/mojopoliastro/angles.mojo`.

Names, argument order and defaults are upstream's.  Upstream's functions are
scalar; passing arrays here vectorises them over the broadcast shape, which is
the same thing numba would do for a jitted caller.
"""

from .._dispatch import map1, map2

__all__ = [
    "D_to_nu", "nu_to_D", "nu_to_E", "nu_to_F", "E_to_nu", "F_to_nu",
    "M_to_E", "M_to_F", "M_to_D", "E_to_M", "F_to_M", "D_to_M", "fp_angle",
]


def D_to_nu(D):
    r"""True anomaly from parabolic anomaly.

    .. math::

        \nu = 2 \arctan{D}
    """
    return map1("mp_D_to_nu", D)


def nu_to_D(nu):
    r"""Parabolic anomaly from true anomaly.

    .. math::

        B = \tan{\frac{\nu}{2}}
    """
    return map1("mp_nu_to_D", nu)


def nu_to_E(nu, ecc):
    r"""Eccentric anomaly from true anomaly.

    .. math::

        E = 2 \arctan \left ( \sqrt{\frac{1 - e}{1 + e}} \tan{\frac{\nu}{2}} \right)
        \in (-\pi, \pi]
    """
    return map2("mp_nu_to_E", nu, ecc)


def nu_to_F(nu, ecc):
    r"""Hyperbolic anomaly from true anomaly.

    .. math::

        F = 2 \operatorname{arctanh} \left( \sqrt{\frac{e-1}{e+1}} \tan{\frac{\nu}{2}} \right)
    """
    return map2("mp_nu_to_F", nu, ecc)


def E_to_nu(E, ecc):
    r"""True anomaly from eccentric anomaly.

    .. math::

        \nu = 2 \arctan \left( \sqrt{\frac{1 + e}{1 - e}} \tan{\frac{E}{2}} \right)
        \in (-\pi, \pi]
    """
    return map2("mp_E_to_nu", E, ecc)


def F_to_nu(F, ecc):
    r"""True anomaly from hyperbolic anomaly.

    .. math::

        \nu = 2 \arctan \left( \sqrt{\frac{e + 1}{e - 1}} \tanh{\frac{F}{2}} \right)
    """
    return map2("mp_F_to_nu", F, ecc)


def M_to_E(M, ecc):
    r"""Eccentric anomaly from mean anomaly."""
    return map2("mp_M_to_E", M, ecc)


def M_to_F(M, ecc):
    r"""Hyperbolic anomaly from mean anomaly."""
    return map2("mp_M_to_F", M, ecc)


def M_to_D(M):
    r"""Parabolic anomaly from mean anomaly."""
    return map1("mp_M_to_D", M)


def E_to_M(E, ecc):
    r"""Mean anomaly from eccentric anomaly.

    .. math::

        M = E - e \sin{E}
    """
    return map2("mp_E_to_M", E, ecc)


def F_to_M(F, ecc):
    r"""Mean anomaly from hyperbolic anomaly.

    .. math::

        M = e \sinh{F} - F
    """
    return map2("mp_F_to_M", F, ecc)


def D_to_M(D):
    r"""Mean anomaly from parabolic anomaly.

    .. math::

        M = \frac{1}{3} D^3 + D
    """
    return map1("mp_D_to_M", D)


def fp_angle(nu, ecc):
    r"""Flight path angle.

    .. math::

        \phi = \arctan{\frac{e \sin{\nu}}{1 + e \cos{\nu}}}
    """
    return map2("mp_fp_angle", nu, ecc)
