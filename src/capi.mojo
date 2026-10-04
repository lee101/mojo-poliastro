"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric.  Callers own all
memory, including scratch — nothing here allocates, so nothing here can leak.

Each export is the batched form of the same-named kernel in
`mojopoliastro.*`.  Upstream is a scalar `@njit` function per element; the
loop below is the one place the C ABI forces a divergence, and it is kept here
rather than pushed into the kernels so the kernels keep reading like the
upstream source they came from.

Every export ends with an `Int` worker count.  `parallelize` builds a pool
sized from the machine for every launch, so in a container the question is not
"how many workers for this batch" but "is there room in the cgroup for a pool
at all", and that is a property of the machine at that moment rather than a
constant the library can bake in.  The count crosses as an argument because
Mojo has no mutable globals in this dialect.  A count of 1 means *do not
launch*: the export runs the same `work` item through its own serial loop, one
element at a time, which is the same code the pool would have run.  The Python
layer decides, in `_lib.worker_count`.  The two exports that never parallelize
ignore the count entirely.
"""

from max.algorithm import parallelize
from std.math import atan, atan2, atanh, cos, cosh, inf, sin, sinh, sqrt, tan, tanh
from std.sys import simd_width_of

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
    fp_angle,
    nu_to_D,
    nu_to_E,
    nu_to_F,
)
from mojopoliastro.earth_atmosphere import (
    _altitude_profile,
    _check_altitude,
    geometric_to_geopotential,
    geopotential_to_geometric,
    gravity,
)
from mojopoliastro.elements import (
    circular_velocity,
    coe2mee,
    coe2rv,
    coe_rotation_matrix,
    eccentricity_vector,
    mee2coe,
    mee2rv,
    rv2coe,
    rv_pqw,
)
from mojopoliastro.iod import izzo, vallado
from mojopoliastro.linalg import V3, norm
from mojopoliastro.propagation import (
    D_to_M_near_parabolic,
    M_to_D_near_parabolic,
    S_x,
    d2S_x_alt,
    dS_x_alt,
    delta_t_from_nu,
    farnocchia_coe,
    farnocchia_rv,
    func_twobody,
    gooding,
    gooding_coe,
    markley,
    markley_coe,
    nu_from_delta_t,
    vallado as vallado_uni,
)
from mojopoliastro.special import hyp2f1b, stumpff_c2, stumpff_c3
from mojopoliastro.spheroid_location import (
    N as normal_at,
    cartesian_cords,
    cartesian_to_ellipsoidal,
    f as flattening,
    is_visible,
    radius_of_curvature,
    tangential_vecs,
)
from mojopoliastro.spheroid_location import distance as sph_distance
from mojopoliastro.util import (
    Mat3,
    alinspace,
    planetocentric_to_AltAz,
    rotation_matrix,
    spherical_to_cartesian,
)

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]

# Batched kernels below this stay on one thread; launching workers costs more
# than the arithmetic for small batches.  Every export here is a map over
# independent elements with disjoint output slots, so above the threshold the
# batch divides cleanly across cores.
# A parallel launch costs about 5 ms on this machine.  A solver that spends
# microseconds per element pays that back after a few thousand rows, so the
# iterative kernels and the Mat3 conversions use the low figure.  The
# vectorised elementwise maps run ~50 ns an element serially, which puts
# their own break-even near 10^5 rows, and they use the high one.  Measured
# on a 36-thread Xeon E5-2697 v4; see the README.
comptime PARALLEL_THRESHOLD: Int = 4096
comptime PARALLEL_THRESHOLD_CHEAP: Int = 131072


# ---------------------------------------------------------------- SIMD
comptime W: Int = simd_width_of[DType.float64]()
comptime Fv = SIMD[DType.float64, W]
comptime Bv = SIMD[DType.bool, W]


@always_inline
def vload(p_: Ptr, i: Int) -> Fv:
    return p_.unsafe_load[width=W](i)


@always_inline
def vstore(p_: Ptr, i: Int, v: Fv):
    p_.unsafe_store(i, v)


@always_inline
def nlanes(n: Int) -> Int:
    return (n + W - 1) // W


@always_inline
def mselect(c: Bv, a: Fv, b: Fv) -> Fv:
    """Lane-wise pick: `a` where `c` is true, `b` where it is false.

    There is no `select` on `SIMD` in this dialect and no bitcast to build one
    out of, so the mask becomes a 0/1 vector and the blend is arithmetic.
    That means neither argument may be an `inf` or a `NaN`: `inf - res` is
    `inf` and `inf * 0` is `NaN`, so a value the mask discards would still
    poison the lanes it selects out.  Callers whose discarded branch can go
    non-finite have to feed that branch a finite argument instead.
    """
    return b + (a - b) * Fv(c)


@always_inline
def any_true(c: Bv) -> Bool:
    return SIMD[DType.int64, W](c).reduce_add() > 0


@always_inline
def changed(a: Fv, b: Fv) -> Bv:
    return ~SIMD.eq(a, b)


@always_inline
def above_one(x: Fv) -> Bv:
    return ~SIMD.lt(x, 1.0)


@always_inline
def hyp2f1b_v(x: Fv) -> Fv:
    """`hyp2f1b` a vector at a time.

    The series is the one the scalar kernel sums, term for term, but every
    lane runs until *no* lane is still changing -- so each lane stops adding
    exactly when it would have stopped on its own, and lands on the same
    value.
    """
    var big = above_one(x)
    # The lanes that take the `x >= 1` branch get a series input of zero.
    # The series has to be evaluated for the whole vector, and an inf or a
    # NaN in a lane that is thrown away would survive the blend below.
    var x_s = mselect(big, Fv(0.0), x)
    var res = Fv(1.0)
    var term = Fv(1.0)
    var ii = Fv(0.0)
    var active = ~big
    while any_true(active):
        term = term * x_s * (ii + 3.0) / (ii + 2.5)
        var res_old = res
        res += term
        active = changed(res_old, res) & ~big
        ii += 1.0
    # `inf` cannot go through an arithmetic blend -- `inf - res` is inf and
    # `inf * 0` is NaN, so the mask would poison the very lanes it selects
    # out.  A lane with `x >= 1` ran the series at `x = 0` and so holds
    # exactly 1.0, which makes dividing by a masked zero an exact `+inf` and
    # leaves every other lane untouched.
    return res / mselect(big, Fv(0.0), Fv(1.0))


@always_inline
def _c2_series_v(psi: Fv, small: Bv) -> Fv:
    var res = Fv(0.5)
    var delta = psi * (-1.0 / 24.0)
    var d = Fv(4.0)
    var active = small
    while any_true(active):
        # The scalar kernel adds the term it already holds and only then
        # forms the next one; adding after the update would sum the series
        # one term out of order.
        var res_old = res
        res += delta
        active = small & changed(res_old, res)
        d += 2.0
        delta = delta * (-psi) / (d * (d - 1.0))
    return res


@always_inline
def _c3_series_v(psi: Fv, small: Bv) -> Fv:
    var res = Fv(1.0 / 6.0)
    var delta = psi * (-1.0 / 120.0)
    var d = Fv(5.0)
    var active = small
    while any_true(active):
        # The scalar kernel adds the term it already holds and only then
        # forms the next one; adding after the update would sum the series
        # one term out of order.
        var res_old = res
        res += delta
        active = small & changed(res_old, res)
        d += 2.0
        delta = delta * (-psi) / (d * (d - 1.0))
    return res


@always_inline
def stumpff_c2_v(psi: Fv) -> Fv:
    var pos = SIMD.gt(psi, 1.0)
    var neg = SIMD.lt(psi, -1.0)
    var small = ~(pos | neg)
    # All three branches are evaluated for the whole vector, so each is fed
    # an argument that keeps it finite: `psi_hi >= 1` never divides by zero
    # and `psi_sm` keeps the series inside its convergent range.  Without
    # this a lane whose branch is discarded still poisons the blend.
    var psi_hi = mselect(pos, psi, Fv(2.0))
    var psi_lo = mselect(neg, psi, Fv(-2.0))
    var psi_sm = mselect(small, psi, Fv(0.0))
    var hi = (1.0 - cos(sqrt(psi_hi))) / psi_hi
    var lo = (cosh(sqrt(-psi_lo)) - 1.0) / (-psi_lo)
    return mselect(pos, hi, mselect(neg, lo, _c2_series_v(psi_sm, small)))


@always_inline
def stumpff_c3_v(psi: Fv) -> Fv:
    var pos = SIMD.gt(psi, 1.0)
    var neg = SIMD.lt(psi, -1.0)
    var small = ~(pos | neg)
    var psi_hi = mselect(pos, psi, Fv(2.0))
    var psi_lo = mselect(neg, psi, Fv(-2.0))
    var psi_sm = mselect(small, psi, Fv(0.0))
    var sp = sqrt(psi_hi)
    var sm = sqrt(-psi_lo)
    var hi = (sp - sin(sp)) / (psi_hi * sp)
    var lo = (sinh(sm) - sm) / ((-psi_lo) * sm)
    return mselect(pos, hi, mselect(neg, lo, _c3_series_v(psi_sm, small)))


def E_to_M_v(E: Fv, ecc: Fv) -> Fv:
    return E - ecc * sin(E)


@always_inline
def F_to_M_v(F: Fv, ecc: Fv) -> Fv:
    return ecc * sinh(F) - F


@always_inline
def nu_to_E_v(nu: Fv, ecc: Fv) -> Fv:
    return 2.0 * atan(sqrt((1.0 - ecc) / (1.0 + ecc)) * tan(nu * 0.5))


@always_inline
def E_to_nu_v(E: Fv, ecc: Fv) -> Fv:
    return 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(E * 0.5))


@always_inline
def nu_to_F_v(nu: Fv, ecc: Fv) -> Fv:
    return 2.0 * atanh(sqrt((ecc - 1.0) / (ecc + 1.0)) * tan(nu * 0.5))


@always_inline
def F_to_nu_v(F: Fv, ecc: Fv) -> Fv:
    return 2.0 * atan(sqrt((ecc + 1.0) / (ecc - 1.0)) * tanh(F * 0.5))


@always_inline
def fp_angle_v(nu: Fv, ecc: Fv) -> Fv:
    return atan2(ecc * sin(nu), 1.0 + ecc * cos(nu))


@always_inline
def D_to_nu_v(D: Fv) -> Fv:
    return 2.0 * atan(D)


@always_inline
def nu_to_D_v(nu: Fv) -> Fv:
    return tan(nu * 0.5)


@always_inline
def D_to_M_v(D: Fv) -> Fv:
    return D + D * D * D / 3.0


@always_inline
def M_to_D_v(M: Fv) -> Fv:
    var B = 3.0 * M * 0.5
    var A = (B + sqrt(1.0 + B * B)) ** (2.0 / 3.0)
    return 2.0 * A * B / (1.0 + A + A * A)

def reduce_status(sp: Ptr, n: Int) -> Int:
    """The status the serial loop used to accumulate in a local.

    A parallel worker cannot report back through a captured variable, so each
    element parks its code in a slot the caller owns and the last non-zero one
    wins -- the order the serial loop produced.
    """
    var status = 0
    for i in range(n):
        var rc = Int(ld(sp, i))
        if rc != 0:
            status = rc
    return status


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@always_inline
def ld(p_: Ptr, i: Int) -> Float64:
    return p_.unsafe_load(i)


@always_inline
def st(p_: Ptr, i: Int, v: Float64):
    p_.unsafe_store(i, v)


# ---------------------------------------------------------------- linalg
@export("mp_norm")
def mp_norm(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, norm(V3(ld(s, 3 * i), ld(s, 3 * i + 1), ld(s, 3 * i + 2))))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


# --------------------------------------------------------------- special
@export("mp_hyp2f1b")
def mp_hyp2f1b(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, hyp2f1b_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, hyp2f1b(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_stumpff_c2")
def mp_stumpff_c2(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, stumpff_c2_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, stumpff_c2(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_stumpff_c3")
def mp_stumpff_c3(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, stumpff_c3_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, stumpff_c3(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


# ---------------------------------------------------------------- angles
@export("mp_D_to_nu")
def mp_D_to_nu(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, D_to_nu_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, D_to_nu(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_nu_to_D")
def mp_nu_to_D(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, nu_to_D_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, nu_to_D(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_nu_to_E")
def mp_nu_to_E(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, nu_to_E_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, nu_to_E(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_nu_to_F")
def mp_nu_to_F(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, nu_to_F_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, nu_to_F(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_E_to_nu")
def mp_E_to_nu(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, E_to_nu_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, E_to_nu(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_F_to_nu")
def mp_F_to_nu(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, F_to_nu_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, F_to_nu(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_M_to_E")
def mp_M_to_E(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, M_to_E(ld(s, i), ld(e, i)))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_M_to_F")
def mp_M_to_F(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, M_to_F(ld(s, i), ld(e, i)))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_M_to_D")
def mp_M_to_D(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, M_to_D_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, M_to_D(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_E_to_M")
def mp_E_to_M(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, E_to_M_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, E_to_M(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_F_to_M")
def mp_F_to_M(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, F_to_M_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, F_to_M(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_D_to_M")
def mp_D_to_M(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, D_to_M_v(vload(s, i)))
        else:
            for j in range(i, n):
                st(d, j, D_to_M(ld(s, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


@export("mp_fp_angle")
def mp_fp_angle(src: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var e = p(ecc)
    var d = p(dst)
    # One work item owns W consecutive elements, so the vector body never has
    # to know about its neighbour; only the last item can run short, and it
    # finishes with the scalar kernel, so the tail is exact.
    def work(v: Int) {imm}:
        var i = v * W
        if i + W <= n:
            vstore(d, i, fp_angle_v(vload(s, i), vload(e, i)))
        else:
            for j in range(i, n):
                st(d, j, fp_angle(ld(s, j), ld(e, j)))

    if n >= PARALLEL_THRESHOLD_CHEAP and workers > 1:
        parallelize(work, nlanes(n), workers)
    else:
        for v in range(nlanes(n)):
            work(v)


# ------------------------------------------------------------------ util
@export("mp_rotation_matrix")
def mp_rotation_matrix(src: Int, axis: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        rotation_matrix(m, ld(s, i), axis)
        for j in range(9):
            st(d, 9 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_alinspace")
def mp_alinspace(start: Float64, stop: Float64, num: Int, endpoint: Int, dst: Int, workers: Int) abi("C"):
    var out = p(dst)
    alinspace(start, stop, out, num, endpoint != 0)


@export("mp_spherical_to_cartesian")
def mp_spherical_to_cartesian(src: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(src)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = V3(0.0, 0.0, 0.0)
        spherical_to_cartesian(o, V3(ld(s, 3 * i), ld(s, 3 * i + 1), ld(s, 3 * i + 2)))
        st(d, 3 * i, o.x)
        st(d, 3 * i + 1, o.y)
        st(d, 3 * i + 2, o.z)

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_planetocentric_to_AltAz")
def mp_planetocentric_to_AltAz(theta: Int, phi: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var t = p(theta)
    var f_ = p(phi)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        planetocentric_to_AltAz(m, ld(t, i), ld(f_, i))
        for j in range(9):
            st(d, 9 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


# -------------------------------------------------------------- elements
@export("mp_eccentricity_vector")
def mp_eccentricity_vector(k: Float64, r: Int, v: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var rs = p(r)
    var vs = p(v)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = V3(0.0, 0.0, 0.0)
        eccentricity_vector(
            o, k, V3(ld(rs, 3 * i), ld(rs, 3 * i + 1), ld(rs, 3 * i + 2)),
            V3(ld(vs, 3 * i), ld(vs, 3 * i + 1), ld(vs, 3 * i + 2))
        )
        st(d, 3 * i, o.x)
        st(d, 3 * i + 1, o.y)
        st(d, 3 * i + 2, o.z)

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_circular_velocity")
def mp_circular_velocity(k: Float64, a: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(a)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, circular_velocity(k, ld(s, i)))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_rv_pqw")
def mp_rv_pqw(k: Float64, pp: Int, ecc: Int, nu: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var ps = p(pp)
    var e = p(ecc)
    var ns = p(nu)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        rv_pqw(m, k, ld(ps, i), ld(e, i), ld(ns, i))
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_coe_rotation_matrix")
def mp_coe_rotation_matrix(inc: Int, raan: Int, argp: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        coe_rotation_matrix(m, ld(is_, i), ld(rs, i), ld(as_, i))
        for j in range(9):
            st(d, 9 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_coe2rv")
def mp_coe2rv(
    k: Float64,
    pp: Int,
    ecc: Int,
    inc: Int,
    raan: Int,
    argp: Int,
    nu: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var ps = p(pp)
    var e = p(ecc)
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var ns = p(nu)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        coe2rv(m, k, ld(ps, i), ld(e, i), ld(is_, i), ld(rs, i), ld(as_, i), ld(ns, i))
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_coe2rv_many")
def mp_coe2rv_many(
    ks: Int,
    pp: Int,
    ecc: Int,
    inc: Int,
    raan: Int,
    argp: Int,
    nu: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var kv = p(ks)
    var ps = p(pp)
    var e = p(ecc)
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var ns = p(nu)
    var d = p(dst)
    # upstream decorates this one with `parallel=True`; it now takes the same
    # dispatch as every other batched export here.
    def work(i: Int) {imm}:
        var m = Mat3()
        coe2rv(m, ld(kv, i), ld(ps, i), ld(e, i), ld(is_, i), ld(rs, i), ld(as_, i), ld(ns, i))
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_coe2mee")
def mp_coe2mee(
    pp: Int,
    ecc: Int,
    inc: Int,
    raan: Int,
    argp: Int,
    nu: Int,
    dst: Int,
    sts: Int,
    n: Int,
    workers: Int,
) abi("C") -> Int:
    var ps = p(pp)
    var e = p(ecc)
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var ns = p(nu)
    var d = p(dst)
    var sp = p(sts)
    def work(i: Int) {imm}:
        var o = SIMD[DType.float64, 8]()
        var rc = coe2mee(o, ld(ps, i), ld(e, i), ld(is_, i), ld(rs, i), ld(as_, i), ld(ns, i))
        for j in range(6):
            st(d, 6 * i + j, o[j])
        st(sp, i, Float64(rc))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)

    return reduce_status(sp, n)


@export("mp_rv2coe")
def mp_rv2coe(k: Float64, r: Int, v: Int, tol: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var rs = p(r)
    var vs = p(v)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = SIMD[DType.float64, 8]()
        rv2coe(
            o, k, V3(ld(rs, 3 * i), ld(rs, 3 * i + 1), ld(rs, 3 * i + 2)),
            V3(ld(vs, 3 * i), ld(vs, 3 * i + 1), ld(vs, 3 * i + 2)), tol
        )
        for j in range(6):
            st(d, 6 * i + j, o[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_mee2coe")
def mp_mee2coe(
    pp: Int,
    f: Int,
    g: Int,
    h: Int,
    kk: Int,
    L: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var ps = p(pp)
    var fs = p(f)
    var gs = p(g)
    var hs = p(h)
    var ks = p(kk)
    var Ls = p(L)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = SIMD[DType.float64, 8]()
        mee2coe(o, ld(ps, i), ld(fs, i), ld(gs, i), ld(hs, i), ld(ks, i), ld(Ls, i))
        for j in range(6):
            st(d, 6 * i + j, o[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_mee2rv")
def mp_mee2rv(pp: Int, f: Int, g: Int, h: Int, kk: Int, L: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var ps = p(pp)
    var fs = p(f)
    var gs = p(g)
    var hs = p(h)
    var ks = p(kk)
    var Ls = p(L)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        mee2rv(m, ld(ps, i), ld(fs, i), ld(gs, i), ld(hs, i), ld(ks, i), ld(Ls, i))
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


# ------------------------------------------------------------------- iod
@export("mp_vallado_iod")
def mp_vallado_iod(
    k: Float64,
    r0: Int,
    r1: Int,
    tof: Int,
    M: Int,
    prograde: Int,
    lowpath: Int,
    numiter: Int,
    rtol: Float64,
    dst: Int,
    sts: Int,
    n: Int,
    workers: Int,
) abi("C") -> Int:
    var a = p(r0)
    var b = p(r1)
    var ts = p(tof)
    var d = p(dst)
    var sp = p(sts)
    def work(i: Int) {imm}:
        var m = Mat3()
        var rc = vallado(
            k, V3(ld(a, 3 * i), ld(a, 3 * i + 1), ld(a, 3 * i + 2)),
            V3(ld(b, 3 * i), ld(b, 3 * i + 1), ld(b, 3 * i + 2)), ld(ts, i), M,
            prograde != 0, lowpath != 0, numiter, rtol, m
        )
        for j in range(6):
            st(d, 6 * i + j, m.r[j])
        st(sp, i, Float64(rc))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)

    return reduce_status(sp, n)


@export("mp_izzo")
def mp_izzo(
    k: Float64,
    r0: Int,
    r1: Int,
    tof: Int,
    M: Int,
    prograde: Int,
    lowpath: Int,
    numiter: Int,
    rtol: Float64,
    dst: Int,
    sts: Int,
    n: Int,
    workers: Int,
) abi("C") -> Int:
    var a = p(r0)
    var b = p(r1)
    var ts = p(tof)
    var d = p(dst)
    var sp = p(sts)
    def work(i: Int) {imm}:
        var m = Mat3()
        var rc = izzo(
            k, V3(ld(a, 3 * i), ld(a, 3 * i + 1), ld(a, 3 * i + 2)),
            V3(ld(b, 3 * i), ld(b, 3 * i + 1), ld(b, 3 * i + 2)), ld(ts, i), M,
            prograde != 0, lowpath != 0, numiter, rtol, m
        )
        for j in range(6):
            st(d, 6 * i + j, m.r[j])
        st(sp, i, Float64(rc))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)

    return reduce_status(sp, n)


# ----------------------------------------------------------- propagation
@export("mp_func_twobody")
def mp_func_twobody(k: Float64, u: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(u)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = func_twobody(
            k, ld(s, 6 * i), ld(s, 6 * i + 1), ld(s, 6 * i + 2),
            ld(s, 6 * i + 3), ld(s, 6 * i + 4), ld(s, 6 * i + 5)
        )
        for j in range(6):
            st(d, 6 * i + j, o[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_S_x")
def mp_S_x(ecc: Int, x: Int, atol: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var e = p(ecc)
    var xs = p(x)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, S_x(ld(e, i), ld(xs, i), atol))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_dS_x_alt")
def mp_dS_x_alt(ecc: Int, x: Int, atol: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var e = p(ecc)
    var xs = p(x)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, dS_x_alt(ld(e, i), ld(xs, i), atol))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_d2S_x_alt")
def mp_d2S_x_alt(ecc: Int, x: Int, atol: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var e = p(ecc)
    var xs = p(x)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, d2S_x_alt(ld(e, i), ld(xs, i), atol))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_D_to_M_near_parabolic")
def mp_D_to_M_near_parabolic(D: Int, ecc: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(D)
    var e = p(ecc)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, D_to_M_near_parabolic(ld(s, i), ld(e, i)))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_M_to_D_near_parabolic")
def mp_M_to_D_near_parabolic(
    M: Int,
    ecc: Int,
    tol: Float64,
    maxiter: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var s = p(M)
    var e = p(ecc)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, M_to_D_near_parabolic(ld(s, i), ld(e, i), tol, maxiter))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_delta_t_from_nu")
def mp_delta_t_from_nu(
    nu: Int,
    ecc: Int,
    k: Float64,
    q: Float64,
    delta: Float64,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var s = p(nu)
    var e = p(ecc)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, delta_t_from_nu(ld(s, i), ld(e, i), k, q, delta))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_nu_from_delta_t")
def mp_nu_from_delta_t(
    delta_t: Int,
    ecc: Int,
    k: Float64,
    q: Float64,
    delta: Float64,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var s = p(delta_t)
    var e = p(ecc)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, nu_from_delta_t(ld(s, i), ld(e, i), k, q, delta))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_farnocchia_coe")
def mp_farnocchia_coe(
    k: Float64,
    pp: Int,
    ecc: Int,
    inc: Int,
    raan: Int,
    argp: Int,
    nu: Int,
    tof: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var ps = p(pp)
    var e = p(ecc)
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var ns = p(nu)
    var ts = p(tof)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(
            d, i,
            farnocchia_coe(
                k, ld(ps, i), ld(e, i), ld(is_, i), ld(rs, i), ld(as_, i), ld(ns, i),
                ld(ts, i)
            )
        )

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_farnocchia_rv")
def mp_farnocchia_rv(k: Float64, r0: Int, v0: Int, tof: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var a = p(r0)
    var b = p(v0)
    var ts = p(tof)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        farnocchia_rv(
            m, k, V3(ld(a, 3 * i), ld(a, 3 * i + 1), ld(a, 3 * i + 2)),
            V3(ld(b, 3 * i), ld(b, 3 * i + 1), ld(b, 3 * i + 2)), ld(ts, i)
        )
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_vallado_uni")
def mp_vallado_uni(
    k: Float64,
    r0: Int,
    v0: Int,
    tof: Int,
    numiter: Int,
    dst: Int,
    sts: Int,
    n: Int,
    workers: Int,
) abi("C") -> Int:
    var a = p(r0)
    var b = p(v0)
    var ts = p(tof)
    var d = p(dst)
    var sp = p(sts)
    def work(i: Int) {imm}:
        var o = SIMD[DType.float64, 8]()
        var rc = vallado_uni(
            k, V3(ld(a, 3 * i), ld(a, 3 * i + 1), ld(a, 3 * i + 2)),
            V3(ld(b, 3 * i), ld(b, 3 * i + 1), ld(b, 3 * i + 2)), ld(ts, i), numiter, o
        )
        for j in range(4):
            st(d, 4 * i + j, o[j])
        st(sp, i, Float64(rc))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)

    return reduce_status(sp, n)


@export("mp_markley_coe")
def mp_markley_coe(
    k: Float64,
    pp: Int,
    ecc: Int,
    inc: Int,
    raan: Int,
    argp: Int,
    nu: Int,
    tof: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var ps = p(pp)
    var e = p(ecc)
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var ns = p(nu)
    var ts = p(tof)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(
            d, i,
            markley_coe(
                k, ld(ps, i), ld(e, i), ld(is_, i), ld(rs, i), ld(as_, i), ld(ns, i),
                ld(ts, i)
            )
        )

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_markley")
def mp_markley(k: Float64, r0: Int, v0: Int, tof: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var a = p(r0)
    var b = p(v0)
    var ts = p(tof)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        markley(
            m, k, V3(ld(a, 3 * i), ld(a, 3 * i + 1), ld(a, 3 * i + 2)),
            V3(ld(b, 3 * i), ld(b, 3 * i + 1), ld(b, 3 * i + 2)), ld(ts, i)
        )
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_gooding_coe")
def mp_gooding_coe(
    k: Float64,
    pp: Int,
    ecc: Int,
    inc: Int,
    raan: Int,
    argp: Int,
    nu: Int,
    tof: Int,
    numiter: Int,
    rtol: Float64,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var ps = p(pp)
    var e = p(ecc)
    var is_ = p(inc)
    var rs = p(raan)
    var as_ = p(argp)
    var ns = p(nu)
    var ts = p(tof)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(
            d, i,
            gooding_coe(
                k, ld(ps, i), ld(e, i), ld(is_, i), ld(rs, i), ld(as_, i), ld(ns, i),
                ld(ts, i), numiter, rtol
            )
        )

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_gooding")
def mp_gooding(
    k: Float64,
    r0: Int,
    v0: Int,
    tof: Int,
    numiter: Int,
    rtol: Float64,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var a = p(r0)
    var b = p(v0)
    var ts = p(tof)
    var d = p(dst)
    def work(i: Int) {imm}:
        var m = Mat3()
        gooding(
            m, k, V3(ld(a, 3 * i), ld(a, 3 * i + 1), ld(a, 3 * i + 2)),
            V3(ld(b, 3 * i), ld(b, 3 * i + 1), ld(b, 3 * i + 2)), ld(ts, i), numiter, rtol
        )
        for j in range(6):
            st(d, 6 * i + j, m.r[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


# ----------------------------------------------------- spheroid_location
@export("mp_cartesian_cords")
def mp_cartesian_cords(
    a: Int,
    c: Int,
    lon: Int,
    lat: Int,
    h: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var as_ = p(a)
    var cs = p(c)
    var ls = p(lon)
    var bs = p(lat)
    var hs = p(h)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = V3(0.0, 0.0, 0.0)
        cartesian_cords(o, ld(as_, i), ld(cs, i), ld(ls, i), ld(bs, i), ld(hs, i))
        st(d, 3 * i, o.x)
        st(d, 3 * i + 1, o.y)
        st(d, 3 * i + 2, o.z)

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_f")
def mp_f(a: Int, c: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var as_ = p(a)
    var cs = p(c)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, flattening(ld(as_, i), ld(cs, i)))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_N")
def mp_N(a: Int, b: Int, c: Int, cords: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var as_ = p(a)
    var bs = p(b)
    var cs = p(c)
    var ps = p(cords)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = V3(0.0, 0.0, 0.0)
        normal_at(
            o, ld(as_, i), ld(bs, i), ld(cs, i),
            V3(ld(ps, 3 * i), ld(ps, 3 * i + 1), ld(ps, 3 * i + 2))
        )
        st(d, 3 * i, o.x)
        st(d, 3 * i + 1, o.y)
        st(d, 3 * i + 2, o.z)

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_tangential_vecs")
def mp_tangential_vecs(N: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var ns = p(N)
    var d = p(dst)
    def work(i: Int) {imm}:
        var u = V3(0.0, 0.0, 0.0)
        var v = V3(0.0, 0.0, 0.0)
        tangential_vecs(
            u, v, V3(ld(ns, 3 * i), ld(ns, 3 * i + 1), ld(ns, 3 * i + 2))
        )
        st(d, 6 * i, u.x)
        st(d, 6 * i + 1, u.y)
        st(d, 6 * i + 2, u.z)
        st(d, 6 * i + 3, v.x)
        st(d, 6 * i + 4, v.y)
        st(d, 6 * i + 5, v.z)

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_radius_of_curvature")
def mp_radius_of_curvature(a: Int, c: Int, lat: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var as_ = p(a)
    var cs = p(c)
    var ls = p(lat)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, radius_of_curvature(ld(as_, i), ld(cs, i), ld(ls, i)))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_distance")
def mp_distance(cords: Int, u: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var ps = p(cords)
    var us = p(u)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(
            d, i,
            sph_distance(
                V3(ld(ps, 3 * i), ld(ps, 3 * i + 1), ld(ps, 3 * i + 2)),
                V3(ld(us, 3 * i), ld(us, 3 * i + 1), ld(us, 3 * i + 2))
            )
        )

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_is_visible")
def mp_is_visible(cords: Int, u: Int, N: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var ps = p(cords)
    var us = p(u)
    var ns = p(N)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(
            d, i,
            1.0
            if is_visible(
                V3(ld(ps, 3 * i), ld(ps, 3 * i + 1), ld(ps, 3 * i + 2)),
                V3(ld(us, 3 * i), ld(us, 3 * i + 1), ld(us, 3 * i + 2)),
                V3(ld(ns, 3 * i), ld(ns, 3 * i + 1), ld(ns, 3 * i + 2))
            )
            else 0.0
        )

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_cartesian_to_ellipsoidal")
def mp_cartesian_to_ellipsoidal(
    a: Int,
    c: Int,
    x: Int,
    y: Int,
    z: Int,
    dst: Int,
    n: Int,
    workers: Int,
) abi("C"):
    var as_ = p(a)
    var cs = p(c)
    var xs = p(x)
    var ys = p(y)
    var zs = p(z)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = SIMD[DType.float64, 8]()
        cartesian_to_ellipsoidal(
            o, ld(as_, i), ld(cs, i), ld(xs, i), ld(ys, i), ld(zs, i)
        )
        for j in range(3):
            st(d, 3 * i + j, o[j])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


# ------------------------------------------------------ earth_atmosphere
@export("mp_geometric_to_geopotential")
def mp_geometric_to_geopotential(z: Int, r0: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(z)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, geometric_to_geopotential(ld(s, i), r0))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_geopotential_to_geometric")
def mp_geopotential_to_geometric(h: Int, r0: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(h)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, geopotential_to_geometric(ld(s, i), r0))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_gravity")
def mp_gravity(z: Int, g0: Float64, r0: Float64, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(z)
    var d = p(dst)
    def work(i: Int) {imm}:
        st(d, i, gravity(ld(s, i), g0, r0))

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_check_altitude")
def mp_check_altitude(alt: Int, r0: Float64, geometric: Int, dst: Int, n: Int, workers: Int) abi("C"):
    var s = p(alt)
    var d = p(dst)
    def work(i: Int) {imm}:
        var o = SIMD[DType.float64, 8]()
        _check_altitude(o, ld(s, i), r0, geometric != 0)
        st(d, 2 * i, o[0])
        st(d, 2 * i + 1, o[1])

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize(work, n, workers)
    else:
        for i in range(n):
            work(i)


@export("mp_jacchia_altitude_profile")
def mp_jacchia_altitude_profile(
    alt: Float64,
    Texo: Float64,
    Z: Int,
    T: Int,
    CN2: Int,
    CO2: Int,
    CO: Int,
    CAr: Int,
    CHe: Int,
    CH: Int,
    CM: Int,
    WM: Int,
    E5M: Int,
    E6P: Int,
    workers: Int,
) abi("C") -> Int:
    return _altitude_profile(
        alt, Texo, p(Z), p(T), p(CN2), p(CO2), p(CO), p(CAr), p(CHe), p(CH), p(CM),
        p(WM), p(E5M), p(E6P)
    )
