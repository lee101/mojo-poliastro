"""Port of `poliastro.core.propagation`.

Modules appear in upstream's order: `base`, `farnocchia`, `vallado`,
`markley`, `gooding`.  `cowell` is not ported: it is a thin wrapper around
`scipy.integrate.solve_ivp` with `DOP853`, and there is no faithful Mojo
counterpart to hand off to here.  See the README's coverage section.
"""

from std.math import abs, acos, acosh, cos, cosh, log, nan, pi, sin, sqrt

from mojopoliastro.angles import (
    D_to_M,
    D_to_nu,
    E_to_M,
    E_to_nu,
    F_to_M,
    F_to_nu,
    M_to_D,
    M_to_E,
    M_to_F,
    nu_to_D,
    nu_to_E,
    nu_to_F,
)
from mojopoliastro.elements import coe2rv, rv2coe
from mojopoliastro.linalg import V3, dot, norm
from mojopoliastro.special import stumpff_c2 as c2
from mojopoliastro.special import stumpff_c3 as c3
from mojopoliastro.util import Mat3

comptime PI: Float64 = 3.141592653589793
comptime TWO_PI: Float64 = 6.283185307179586


# ------------------------------------------------------------------ base
def func_twobody(
    k: Float64, x: Float64, y: Float64, z: Float64, vx: Float64, vy: Float64, vz: Float64
) -> SIMD[DType.float64, 8]:
    r"""Right-hand side of the two-body problem.

    .. math::

        \ddot{\vec{r}} = -\frac{\mu}{r^3}\vec{r}
    """
    var r3 = (x**2 + y**2 + z**2) ** 1.5
    return SIMD[DType.float64, 8](
        vx,
        vy,
        vz,
        -k * x / r3,
        -k * y / r3,
        -k * z / r3,
        0.0,
        0.0,
    )


# ------------------------------------------------------------ farnocchia
def _kepler_equation_near_parabolic(
    D: Float64, M: Float64, ecc: Float64
) -> Float64:
    return D_to_M_near_parabolic(D, ecc) - M


def _kepler_equation_prime_near_parabolic(
    D: Float64, M: Float64, ecc: Float64
) -> Float64:
    var x = (ecc - 1.0) / (ecc + 1.0) * D**2
    var S = dS_x_alt(ecc, x)
    return sqrt(2.0 / (1.0 + ecc)) + sqrt(2.0 / (1.0 + ecc) ** 3) * D**2 * S


def S_x(ecc: Float64, x: Float64, atol: Float64 = 1e-12) -> Float64:
    var S = 0.0
    var k = 0.0
    while True:
        var S_old = S
        S += (ecc - 1 / (2 * k + 3)) * x**k
        k += 1.0
        if abs(S - S_old) < atol:
            return S


def dS_x_alt(ecc: Float64, x: Float64, atol: Float64 = 1e-12) -> Float64:
    var S = 0.0
    var k = 0.0
    while True:
        var S_old = S
        S += (ecc - 1 / (2 * k + 3)) * (2 * k + 3) * x**k
        k += 1.0
        if abs(S - S_old) < atol:
            return S


def d2S_x_alt(ecc: Float64, x: Float64, atol: Float64 = 1e-12) -> Float64:
    var S = 0.0
    var k = 0.0
    while True:
        var S_old = S
        S += (ecc - 1 / (2 * k + 3)) * (2 * k + 3) * (2 * k + 2) * x**k
        k += 1.0
        if abs(S - S_old) < atol:
            return S


def D_to_M_near_parabolic(D: Float64, ecc: Float64) -> Float64:
    var x = (ecc - 1.0) / (ecc + 1.0) * D**2
    var S = S_x(ecc, x)
    return sqrt(2.0 / (1.0 + ecc)) * D + sqrt(2.0 / (1.0 + ecc) ** 3) * D**3 * S


def M_to_D_near_parabolic(
    M: Float64, ecc: Float64, tol: Float64 = 1.48e-08, maxiter: Int = 50
) -> Float64:
    var D0 = M_to_D(M)
    for _ in range(maxiter):
        var fval = _kepler_equation_near_parabolic(D0, M, ecc)
        var fder = _kepler_equation_prime_near_parabolic(D0, M, ecc)
        var newton_step = fval / fder
        var D = D0 - newton_step
        if abs(D - D0) < tol:
            return D
        D0 = D
    return nan[DType.float64]()


def delta_t_from_nu(
    nu: Float64,
    ecc: Float64,
    k: Float64 = 1.0,
    q: Float64 = 1.0,
    delta: Float64 = 0.01,
) -> Float64:
    var M: Float64
    var n: Float64
    if ecc < 1 - delta:
        var E = nu_to_E(nu, ecc)
        M = E_to_M(E, ecc)
        n = sqrt(k * (1 - ecc) ** 3 / q**3)
    elif 1 - delta <= ecc < 1:
        var E = nu_to_E(nu, ecc)
        if delta <= 1 - ecc * cos(E):
            M = E_to_M(E, ecc)
            n = sqrt(k * (1 - ecc) ** 3 / q**3)
        else:
            var D = nu_to_D(nu)
            M = D_to_M_near_parabolic(D, ecc)
            n = sqrt(k / (2 * q**3))
    elif ecc == 1:
        var D = nu_to_D(nu)
        M = D_to_M(D)
        n = sqrt(k / (2 * q**3))
    elif 1 + ecc * cos(nu) < 0:
        return nan[DType.float64]()
    elif 1.0 < ecc <= 1 + delta:
        var F = nu_to_F(nu, ecc)
        if delta <= ecc * cosh(F) - 1:
            M = F_to_M(F, ecc)
            n = sqrt(k * (ecc - 1) ** 3 / q**3)
        else:
            var D = nu_to_D(nu)
            M = D_to_M_near_parabolic(D, ecc)
            n = sqrt(k / (2 * q**3))
    elif 1 + delta < ecc:
        var F = nu_to_F(nu, ecc)
        M = F_to_M(F, ecc)
        n = sqrt(k * (ecc - 1) ** 3 / q**3)
    else:
        return nan[DType.float64]()
    return M / n


def nu_from_delta_t(
    delta_t: Float64,
    ecc: Float64,
    k: Float64 = 1.0,
    q: Float64 = 1.0,
    delta: Float64 = 0.01,
) -> Float64:
    if ecc < 1 - delta:
        var n = sqrt(k * (1 - ecc) ** 3 / q**3)
        var M = n * delta_t
        var E = M_to_E((M + PI) % TWO_PI - PI, ecc)
        var nu = E_to_nu(E, ecc)
        return nu
    elif 1 - delta <= ecc < 1:
        var E_delta = acos((1 - delta) / ecc)
        var n = sqrt(k * (1 - ecc) ** 3 / q**3)
        var M = n * delta_t
        if E_to_M(E_delta, ecc) <= abs(M):
            var E = M_to_E((M + PI) % TWO_PI - PI, ecc)
            return E_to_nu(E, ecc)
        else:
            n = sqrt(k / (2 * q**3))
            M = n * delta_t
            var D = M_to_D_near_parabolic(M, ecc)
            return D_to_nu(D)
    elif ecc == 1:
        var n = sqrt(k / (2 * q**3))
        var M = n * delta_t
        var D = M_to_D(M)
        return D_to_nu(D)
    elif 1.0 < ecc <= 1 + delta:
        var F_delta = acosh((1 + delta) / ecc)
        var n = sqrt(k * (ecc - 1) ** 3 / q**3)
        var M = n * delta_t
        if F_to_M(F_delta, ecc) <= abs(M):
            var F = M_to_F(M, ecc)
            return F_to_nu(F, ecc)
        else:
            n = sqrt(k / (2 * q**3))
            M = n * delta_t
            var D = M_to_D_near_parabolic(M, ecc)
            return D_to_nu(D)
    else:
        var n = sqrt(k * (ecc - 1) ** 3 / q**3)
        var M = n * delta_t
        var F = M_to_F(M, ecc)
        return F_to_nu(F, ecc)


def farnocchia_coe(
    k: Float64,
    p: Float64,
    ecc: Float64,
    inc: Float64,
    raan: Float64,
    argp: Float64,
    nu: Float64,
    tof: Float64,
) -> Float64:
    var q = p / (1 + ecc)
    var delta_t0 = delta_t_from_nu(nu, ecc, k, q)
    var delta_t = delta_t0 + tof
    return nu_from_delta_t(delta_t, ecc, k, q)


def farnocchia_rv(
    mut out: Mat3, k: Float64, r0: V3, v0: V3, tof: Float64
):
    var coe = SIMD[DType.float64, 8]()
    rv2coe(coe, k, r0, v0)
    var p = coe[0]
    var ecc = coe[1]
    var inc = coe[2]
    var raan = coe[3]
    var argp = coe[4]
    var nu0 = coe[5]
    var nu = farnocchia_coe(k, p, ecc, inc, raan, argp, nu0, tof)
    coe2rv(out, k, p, ecc, inc, raan, argp, nu)


# -------------------------------------------------------------- vallado
def vallado(
    k: Float64,
    r0: V3,
    v0: V3,
    tof: Float64,
    numiter: Int,
    mut out: SIMD[DType.float64, 8],
) -> Int:
    r"""Universal-variable Lagrange coefficients, Vallado's iteration.

    Returns 0 on success, -3 if `numiter` is exhausted (upstream raises
    RuntimeError there).
    """
    var dot_r0v0 = dot(r0, v0)
    var norm_r0 = norm(r0)
    var sqrt_mu = k**0.5
    var alpha = -(dot(v0, v0)) / k + 2 / norm_r0
    var xi_new: Float64
    if alpha > 0:
        xi_new = sqrt_mu * tof * alpha
    elif alpha < 0:
        # DIVERGENCE: numpy's np.sign(0.0) is 0.0; Mojo has no sign(), so the
        # three cases are spelled out.
        var s = 0.0
        if tof > 0:
            s = 1.0
        elif tof < 0:
            s = -1.0
        xi_new = (
            s
            * sqrt(-1 / alpha)
            * log(
                -2 * k * alpha * tof
                / (dot_r0v0 + s * sqrt(-k / alpha) * (1 - norm_r0 * alpha))
            )
        )
    else:
        xi_new = sqrt_mu * tof / norm_r0
    var count = 0
    var converged = False
    var c2_psi = 0.0
    var c3_psi = 0.0
    var xi = 0.0
    var psi = 0.0
    var norm_r = 0.0
    while count < numiter:
        xi = xi_new
        psi = xi * xi * alpha
        c2_psi = c2(psi)
        c3_psi = c3(psi)
        norm_r = (
            xi * xi * c2_psi
            + dot_r0v0 / sqrt_mu * xi * (1 - psi * c3_psi)
            + norm_r0 * (1 - psi * c2_psi)
        )
        xi_new = xi + (
            sqrt_mu * tof
            - xi * xi * xi * c3_psi
            - dot_r0v0 / sqrt_mu * xi * xi * c2_psi
            - norm_r0 * xi * (1 - psi * c3_psi)
        ) / norm_r
        if abs(xi_new - xi) < 1e-07:
            converged = True
            break
        else:
            count += 1
    if not converged:
        return -3
    var f = 1 - xi**2 / norm_r0 * c2_psi
    var g = tof - xi**3 / sqrt_mu * c3_psi
    var gdot = 1 - xi**2 / norm_r * c2_psi
    var fdot = sqrt_mu / (norm_r * norm_r0) * xi * (psi * c3_psi - 1)
    out = SIMD[DType.float64, 8](f, g, fdot, gdot, 0.0, 0.0, 0.0, 0.0)
    return 0


# -------------------------------------------------------------- markley
def markley_coe(
    k: Float64,
    p: Float64,
    ecc: Float64,
    inc: Float64,
    raan: Float64,
    argp: Float64,
    nu: Float64,
    tof: Float64,
) -> Float64:
    var M0 = E_to_M(nu_to_E(nu, ecc), ecc)
    var a = p / (1 - ecc**2)
    var n = sqrt(k / a**3)
    var M = M0 + n * tof
    M = (M + PI) % TWO_PI - PI
    var alpha = (3 * PI**2 + 1.6 * (PI - abs(M)) / (1 + ecc)) / (PI**2 - 6)
    var d = 3 * (1 - ecc) + alpha * ecc
    var q = 2 * alpha * d * (1 - ecc) - M**2
    var r = 3 * alpha * d * (d - 1 + ecc) * M + M**3
    var w = (abs(r) + sqrt(q**3 + r**2)) ** (2.0 / 3.0)
    var E = (2 * r * w / (w**2 + w * q + q**2) + M) / d
    # `f0`/`f1` are `_kepler_equation`/`_kepler_equation_prime`, which are
    # `E - ecc sin E - M` and `1 - ecc cos E`; `f2`/`f3` are the same `sin E`
    # and `cos E` again.  Evaluate each transcendental once.  The
    # operations, and their order, are unchanged.
    var sE = sin(E)
    var cE = cos(E)
    var f0 = E - ecc * sE - M
    var f1 = 1 - ecc * cE
    var f2 = ecc * sE
    var f3 = ecc * cE
    var f4 = -f2
    var delta3 = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    var delta4 = -f0 / (f1 + 0.5 * delta3 * f2 + 1.0 / 6.0 * delta3**2 * f3)
    var delta5 = -f0 / (
        f1
        + 0.5 * delta4 * f2
        + 1.0 / 6.0 * delta4**2 * f3
        + 1.0 / 24.0 * delta4**3 * f4
    )
    E += delta5
    return E_to_nu(E, ecc)


def markley(mut out: Mat3, k: Float64, r0: V3, v0: V3, tof: Float64):
    var coe = SIMD[DType.float64, 8]()
    rv2coe(coe, k, r0, v0)
    var nu = markley_coe(
        k, coe[0], coe[1], coe[2], coe[3], coe[4], coe[5], tof
    )
    coe2rv(out, k, coe[0], coe[1], coe[2], coe[3], coe[4], nu)


# -------------------------------------------------------------- gooding
def gooding_coe(
    k: Float64,
    p: Float64,
    ecc: Float64,
    inc: Float64,
    raan: Float64,
    argp: Float64,
    nu: Float64,
    tof: Float64,
    numiter: Int = 150,
    rtol: Float64 = 1e-08,
) -> Float64:
    if ecc >= 1.0:
        return nan[DType.float64]()
    var M0 = E_to_M(nu_to_E(nu, ecc), ecc)
    var semi_axis_a = p / (1 - ecc**2)
    var n = sqrt(k / abs(semi_axis_a) ** 3)
    var M = M0 + n * tof
    # upstream reuses `n` as the iteration counter here; Mojo cannot change a
    # variable's type, so the counter gets its own name.
    var c = ecc * cos(M)
    var s = ecc * sin(M)
    var psi = s / sqrt(1 - 2 * c + ecc**2)
    var f = 1.0
    var it = 0
    while f**2 >= rtol and it <= numiter:
        var xi = cos(psi)
        var eta = sin(psi)
        var fd = 1 - c * xi + s * eta
        var fdd = c * eta + s * xi
        f = psi - fdd
        psi = psi - f * fd / (fd**2 - 0.5 * f * fdd)
        it += 1
    var E = M + psi
    return E_to_nu(E, ecc)


def gooding(
    mut out: Mat3,
    k: Float64,
    r0: V3,
    v0: V3,
    tof: Float64,
    numiter: Int = 150,
    rtol: Float64 = 1e-08,
):
    var coe = SIMD[DType.float64, 8]()
    rv2coe(coe, k, r0, v0)
    var nu = gooding_coe(
        k, coe[0], coe[1], coe[2], coe[3], coe[4], coe[5], tof, numiter, rtol
    )
    coe2rv(out, k, coe[0], coe[1], coe[2], coe[3], coe[4], nu)
