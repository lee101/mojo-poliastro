"""Port of `poliastro._math.special` -- see `src/mojopoliastro/special.mojo`."""

from .._dispatch import map1

__all__ = ["hyp2f1b", "stumpff_c2", "stumpff_c3"]


def hyp2f1b(x):
    r"""The regularised hypergeometric function 2F1(3/2, 1; 5/2; x)."""
    return map1("mp_hyp2f1b", x)


def stumpff_c2(psi):
    r"""Second Stumpff function :math:`C_2(\psi)`."""
    return map1("mp_stumpff_c2", psi)


def stumpff_c3(psi):
    r"""Third Stumpff function :math:`C_3(\psi)`."""
    return map1("mp_stumpff_c3", psi)
