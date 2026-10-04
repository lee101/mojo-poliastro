"""Port of `poliastro.core.iod`.

Lambert's problem, solved two ways: `vallado` bisects on the universal
variable psi, `izzo` runs Izzo's Householder iteration on the normalised
variable x.  Functions appear in upstream's source order.

DIVERGENCE: upstream signals failure with `raise` (NotImplementedError for
multi-revolution Vallado, RuntimeError for a 180 degree phase angle, a
non-convergent iteration, and ValueError for Izzo's collinear / infeasible
cases).  The C ABI cannot raise, so each entry point here returns a status
`Int`: 0 on success and a negative code otherwise, and the Python layer turns
the code back into upstream's exception.  See `python/mojo_poliastro/core/iod.py`.
"""

from std.math import abs, acos, asinh, exp, floor, inf, log, nan, pi, sqrt

from mojopoliastro.linalg import V3, cross, dot, norm
from mojopoliastro.util import Mat3
from mojopoliastro.special import hyp2f1b
from mojopoliastro.special import stumpff_c2 as c2
from mojopoliastro.special import stumpff_c3 as c3

# status codes shared with the Python layer
comptime OK: Int = 0
comptime ERR_MULTIREV: Int = -1
comptime ERR_PHASE_180: Int = -2
comptime ERR_MAX_ITER: Int = -3
comptime ERR_COLLINEAR: Int = -4
comptime ERR_INFEASIBLE: Int = -5


def vallado(
    k: Float64,
    r0: V3,
    r: V3,
    tof: Float64,
    M: Int,
    prograde: Bool,
    lowpath: Bool,
    numiter: Int,
    rtol: Float64,
    mut out: Mat3,
) -> Int:
    r"""Solves the Lambert's problem.

    The algorithm returns the initial velocity vector and the final one.

    .. math::

        \vec{v_{o}} &= \frac{1}{g}(\vec{r} - f\vec{r_{0}}) \\
        \vec{v} &= \frac{1}{g}(\dot{g}\vec{r} - \vec{r_{0}})
    """
    if M > 0:
        return ERR_MULTIREV
    var t_m = 1.0
    if not prograde:
        t_m = -1.0
    var norm_r0 = norm(r0)
    var norm_r = norm(r)
    var norm_r0_times_norm_r = norm_r0 * norm_r
    var norm_r0_plus_norm_r = norm_r0 + norm_r
    var cos_dnu = dot(r0, r) / norm_r0_times_norm_r
    var A = t_m * sqrt(norm_r * norm_r0 * (1 + cos_dnu))
    if A == 0.0:
        return ERR_PHASE_180
    var psi = 0.0
    var psi_low = -4 * pi**2
    var psi_up = 4 * pi**2
    var count = 0
    var y = 0.0
    var converged = False
    var sqrt_k = sqrt(k)
    # `c2(psi)` / `c3(psi)` are the most expensive thing in this loop and
    # upstream re-evaluates them up to five times for the same `psi` inside
    # one bisection step.  Hoist them: recompute only when `psi` moves.
    var c2v = c2(psi)
    var c3v = c3(psi)
    while count < numiter:
        y = norm_r0_plus_norm_r + A * (psi * c3v - 1) / sqrt(c2v)
        if A > 0.0:
            while y < 0.0:
                psi_low = psi
                psi = 0.8 * (1.0 / c3v) * (
                    1.0 - norm_r0_times_norm_r * sqrt(c2v) / A
                )
                c2v = c2(psi)
                c3v = c3(psi)
                y = norm_r0_plus_norm_r + A * (psi * c3v - 1) / sqrt(c2v)
        var xi = sqrt(y / c2v)
        var tof_new = (xi**3 * c3v + A * sqrt(y)) / sqrt_k
        if abs((tof_new - tof) / tof) < rtol:
            converged = True
            break
        count += 1
        var condition = tof_new <= tof
        if condition:
            psi_low = psi_low + (psi - psi_low)
        if not condition:
            psi_up = psi_up + (psi - psi_up)
        psi = (psi_up + psi_low) / 2
        c2v = c2(psi)
        c3v = c3(psi)
    if not converged:
        return ERR_MAX_ITER
    var f = 1 - y / norm_r0
    var g = A * sqrt(y / k)
    var gdot = 1 - y / norm_r
    var v0 = (r - r0 * f) / g
    var v = (r * gdot - r0) / g
    out.set(0, 0, v0.x)
    out.set(0, 1, v0.y)
    out.set(0, 2, v0.z)
    out.set(1, 0, v.x)
    out.set(1, 1, v.y)
    out.set(1, 2, v.z)
    return OK


def _reconstruct(
    x: Float64,
    y: Float64,
    r1: Float64,
    r2: Float64,
    ll: Float64,
    gamma: Float64,
    rho: Float64,
    sigma: Float64,
) -> SIMD[DType.float64, 4]:
    var V_r1 = gamma * (ll * y - x - rho * (ll * y + x)) / r1
    var V_r2 = -gamma * (ll * y - x + rho * (ll * y + x)) / r2
    var V_t1 = gamma * sigma * (y + ll * x) / r1
    var V_t2 = gamma * sigma * (y + ll * x) / r2
    return SIMD[DType.float64, 4](V_r1, V_r2, V_t1, V_t2)


def _find_xy(
    ll: Float64, T: Float64, M: Int, numiter: Int, lowpath: Bool, rtol: Float64
) -> Tuple[Float64, Float64, Int]:
    var M_max = Int(floor(T / pi))
    var T_00 = acos(ll) + ll * sqrt(1 - ll**2)
    if T < T_00 + Float64(M_max) * pi and M_max > 0:
        var T_min = _compute_T_min(ll, M_max, numiter, rtol)[1]
        if T < T_min:
            M_max -= 1
    if M > M_max:
        return Tuple(0.0, 0.0, ERR_INFEASIBLE)
    var x_0 = _initial_guess(T, ll, M, lowpath)
    var x = _householder(x_0, T, ll, M, rtol, numiter)
    var y = _compute_y(x, ll)
    return Tuple(x, y, OK)


def _compute_y(x: Float64, ll: Float64) -> Float64:
    return sqrt(1 - ll**2 * (1 - x**2))


def _compute_psi(x: Float64, y: Float64, ll: Float64) -> Float64:
    if -1.0 <= x < 1.0:
        return acos(x * y + ll * (1 - x**2))
    elif x > 1:
        return asinh((y - x * ll) * sqrt(x**2 - 1))
    else:
        return 0.0


def _tof_equation(x: Float64, T0: Float64, ll: Float64, M: Int) -> Float64:
    return _tof_equation_y(x, _compute_y(x, ll), T0, ll, M)


def _tof_equation_y(
    x: Float64, y: Float64, T0: Float64, ll: Float64, M: Int
) -> Float64:
    var T_: Float64
    if M == 0 and sqrt(0.6) < x < sqrt(1.4):
        var eta = y - ll * x
        var S_1 = (1 - ll - x * eta) * 0.5
        var Q = 4.0 / 3.0 * hyp2f1b(S_1)
        T_ = (eta**3 * Q + 4 * ll * eta) * 0.5
    else:
        var psi = _compute_psi(x, y, ll)
        var num = (psi + Float64(M) * pi) / sqrt(abs(1 - x**2)) - x + ll * y
        T_ = num / (1 - x**2)
    return T_ - T0


def _tof_equation_p(x: Float64, y: Float64, T: Float64, ll: Float64) -> Float64:
    return (3 * T * x - 2 + 2 * ll**3 * x / y) / (1 - x**2)


def _tof_equation_p2(
    x: Float64, y: Float64, T: Float64, dT: Float64, ll: Float64
) -> Float64:
    return (3 * T + 5 * x * dT + 2 * (1 - ll**2) * ll**3 / y**3) / (1 - x**2)


def _tof_equation_p3(
    x: Float64, y: Float64, `_`: Float64, dT: Float64, ddT: Float64, ll: Float64
) -> Float64:
    return (7 * x * ddT + 8 * dT - 6 * (1 - ll**2) * ll**5 * x / y**5) / (1 - x**2)


def _compute_T_min(ll: Float64, M: Int, numiter: Int, rtol: Float64) -> SIMD[DType.float64, 2]:
    var x_T_min: Float64
    var T_min: Float64
    if ll == 1:
        x_T_min = 0.0
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
    elif M == 0:
        x_T_min = inf[DType.float64]()
        T_min = 0.0
    else:
        var x_i = 0.1
        var T_i = _tof_equation(x_i, 0.0, ll, M)
        x_T_min = _halley(x_i, T_i, ll, rtol, numiter)
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
    return SIMD[DType.float64, 2](x_T_min, T_min)


def _initial_guess(
    T: Float64, ll: Float64, M: Int, lowpath: Bool
) -> Float64:
    var x_0: Float64
    if M == 0:
        var T_0 = acos(ll) + ll * sqrt(1 - ll**2) + Float64(M) * pi
        var T_1 = 2 * (1 - ll**3) / 3
        if T >= T_0:
            x_0 = (T_0 / T) ** (2.0 / 3.0) - 1
        elif T < T_1:
            x_0 = 5.0 / 2.0 * T_1 / T * (T_1 - T) / (1 - ll**5) + 1
        else:
            x_0 = exp(log(2.0) * log(T / T_0) / log(T_1 / T_0)) - 1
        return x_0
    else:
        var x_0l = (
            ((Float64(M) * pi + pi) / (8 * T)) ** (2.0 / 3.0) - 1
        ) / (((Float64(M) * pi + pi) / (8 * T)) ** (2.0 / 3.0) + 1)
        var x_0r = (
            (8 * T / (Float64(M) * pi)) ** (2.0 / 3.0) - 1
        ) / ((8 * T / (Float64(M) * pi)) ** (2.0 / 3.0) + 1)
        if lowpath:
            x_0 = max(x_0l, x_0r)
        else:
            x_0 = min(x_0l, x_0r)
        return x_0


def _halley(
    mut p0: Float64, T0: Float64, ll: Float64, tol: Float64, maxiter: Int
) -> Float64:
    for _ in range(maxiter):
        var y = _compute_y(p0, ll)
        var fder = _tof_equation_p(p0, y, T0, ll)
        var fder2 = _tof_equation_p2(p0, y, T0, fder, ll)
        if fder2 == 0:
            return nan[DType.float64]()
        var fder3 = _tof_equation_p3(p0, y, T0, fder, fder2, ll)
        var p = p0 - 2 * fder * fder2 / (2 * fder2**2 - fder * fder3)
        if abs(p - p0) < tol:
            return p
        p0 = p
    return nan[DType.float64]()


def _householder(
    mut p0: Float64, T0: Float64, ll: Float64, M: Int, tol: Float64, maxiter: Int
) -> Float64:
    for _ in range(maxiter):
        var y = _compute_y(p0, ll)
        var fval = _tof_equation_y(p0, y, T0, ll, M)
        var T = fval + T0
        var fder = _tof_equation_p(p0, y, T, ll)
        var fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        var fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)
        var p = p0 - fval * (
            (fder**2 - fval * fder2 / 2)
            / (fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6)
        )
        if abs(p - p0) < tol:
            return p
        p0 = p
    return nan[DType.float64]()


def izzo(
    k: Float64,
    r1: V3,
    r2: V3,
    tof: Float64,
    M: Int,
    prograde: Bool,
    lowpath: Bool,
    numiter: Int,
    rtol: Float64,
    mut out: Mat3,
) -> Int:
    """Solves the Lambert problem using the Izzo algorithm."""
    var c = r2 - r1
    var c_norm = norm(c)
    var r1_norm = norm(r1)
    var r2_norm = norm(r2)
    if c_norm == 0.0:
        return ERR_COLLINEAR
    var s = (r1_norm + r2_norm + c_norm) * 0.5
    var i_r1 = r1 / r1_norm
    var i_r2 = r2 / r2_norm
    var i_h = cross(i_r1, i_r2)
    var i_hn = norm(i_h)
    if i_hn == 0.0:
        return ERR_COLLINEAR
    i_h = i_h / i_hn
    var ll = sqrt(1 - min(1.0, c_norm / s))
    var i_t1: V3
    var i_t2: V3
    if i_h.z < 0:
        ll = -ll
        i_t1 = cross(i_r1, i_h)
        i_t2 = cross(i_r2, i_h)
    else:
        i_t1 = cross(i_h, i_r1)
        i_t2 = cross(i_h, i_r2)
    if not prograde:
        ll = -ll
        i_t1 = -i_t1
        i_t2 = -i_t2
    var T = sqrt(2 * k / s**3) * tof
    var xy = _find_xy(ll, T, M, numiter, lowpath, rtol)
    if xy[2] != OK:
        return xy[2]
    var x = xy[0]
    var y = xy[1]
    var gamma = sqrt(k * s / 2)
    var rho = (r1_norm - r2_norm) / c_norm
    var sigma = sqrt(1 - rho**2)
    var rec = _reconstruct(x, y, r1_norm, r2_norm, ll, gamma, rho, sigma)
    var V_r1 = rec[0]
    var V_r2 = rec[1]
    var V_t1 = rec[2]
    var V_t2 = rec[3]
    var v1 = i_r1 * V_r1 + i_t1 * V_t1
    var v2 = i_r2 * V_r2 + i_t2 * V_t2
    out.set(0, 0, v1.x)
    out.set(0, 1, v1.y)
    out.set(0, 2, v1.z)
    out.set(1, 0, v2.x)
    out.set(1, 1, v2.y)
    out.set(1, 2, v2.z)
    return OK
