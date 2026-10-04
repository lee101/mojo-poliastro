"""Port of `poliastro._math.special`.

Hypergeometric series and the Stumpff functions `C2(psi)` / `C3(psi)`, used by
Izzo's Lambert solver and by the universal-variable propagators.

DIVERGENCE: Mojo has no implicit int/float promotion, so integer series
counters are carried as `Float64` (`ii = 0` becomes `ii = 0.0`) and integer
literals inside float expressions are written as floats (`3` becomes `3.0`).
The arithmetic and the values are unchanged.
"""

from std.math import cos, cosh, inf, sin, sinh, sqrt


def hyp2f1b(x: Float64) -> Float64:
    if x >= 1.0:
        return inf[DType.float64]()
    else:
        var res = 1.0
        # upstream's `term * (3+ii)*(1+ii) / (2.5+ii) * x / (ii+1)` has
        # `(1+ii)` in both numerator and denominator; the recurrence is
        # `term * x * (3+ii) / (2.5+ii)`, one division instead of two.
        var term = 1.0
        var ii = 0.0
        while True:
            term = term * x * (ii + 3.0) / (ii + 2.5)
            var res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1.0


@always_inline
def _c2_series(psi: Float64) -> Float64:
    r"""`C_2(psi)` for `abs(psi) <= 1` as `sum_k (-psi)^k / (2k+2)!`.

    Upstream walks the series with `gamma(2k+3)` per term, which is a
    transcendental evaluation of an integer factorial.  The same series
    follows from the term ratio `(-psi) / ((2k+4)(2k+3))`, so the whole
    branch is two multiplies and one divide per term.
    """
    var res = 0.5
    var delta = -psi * (1.0 / 24.0)
    var d = 4.0
    while res + delta != res:
        res = res + delta
        d += 2.0
        delta = delta * (-psi) / (d * (d - 1.0))
    return res


@always_inline
def _c3_series(psi: Float64) -> Float64:
    r"""`C_3(psi)` for `abs(psi) <= 1` as `sum_k (-psi)^k / (2k+3)!`."""
    var res = 1.0 / 6.0
    var delta = -psi * (1.0 / 120.0)
    var d = 5.0
    while res + delta != res:
        res = res + delta
        d += 2.0
        delta = delta * (-psi) / (d * (d - 1.0))
    return res



def stumpff_c2(psi: Float64) -> Float64:
    var eps = 1.0
    if psi > eps:
        return (1 - cos(sqrt(psi))) / psi
    elif psi < -eps:
        return (cosh(sqrt(-psi)) - 1) / -psi
    return _c2_series(psi)


def stumpff_c3(psi: Float64) -> Float64:
    var eps = 1.0
    if psi > eps:
        return (sqrt(psi) - sin(sqrt(psi))) / (psi * sqrt(psi))
    elif psi < -eps:
        return (sinh(sqrt(-psi)) - sqrt(-psi)) / (-psi * sqrt(-psi))
    return _c3_series(psi)
