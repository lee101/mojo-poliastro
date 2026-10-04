"""Port of `poliastro.core.earth_atmosphere`.

`util` first (geometric/geopotential altitude conversions and the variable
g), then `jacchia` (the Jacchia 1977 US Standard Atmosphere 1976 model, which
is a long chain of altitude-indexed recurrences).  Modules and functions
appear in upstream's source order.

DIVERGENCE: upstream's `_altitude_profile` allocates ten Python lists of
length `alt` and indexes them as `Z[iz]`.  The C ABI cannot allocate on the
caller's behalf, so the profile is written into ten caller-provided `Float64`
buffers, and `Z[iz]` becomes `Z.unsafe_load(iz)` to say so explicitly.  The
altitude range check that upstream raises on is returned as a status code.
"""

from std.math import atan, exp, log, sqrt, tanh

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]

comptime pi2: Float64 = 1.5707963267948966
comptime wm0: Float64 = 28.96
comptime wmN2: Float64 = 28.0134
comptime wmO2: Float64 = 31.9988
comptime wmO: Float64 = 15.9994
comptime wmAr: Float64 = 39.948
comptime wmHe: Float64 = 4.0026
comptime wmH: Float64 = 1.0079
comptime qN2: Float64 = 0.7811
comptime qO2: Float64 = 0.20955
comptime qAr: Float64 = 0.009343
comptime qHe: Float64 = 5.242e-06
comptime R0: Float64 = 6356.766
comptime R: Float64 = 8314.32


# ------------------------------------------------------------------ util
def geometric_to_geopotential(z: Float64, r0: Float64) -> Float64:
    var h = r0 * z / (r0 + z)
    return h


def z_to_h(z: Float64, r0: Float64) -> Float64:
    return geometric_to_geopotential(z, r0)


def geopotential_to_geometric(h: Float64, r0: Float64) -> Float64:
    var z = r0 * h / (r0 - h)
    return z


def h_to_z(h: Float64, r0: Float64) -> Float64:
    return geopotential_to_geometric(h, r0)


def gravity(z: Float64, g0: Float64, r0: Float64) -> Float64:
    var g = g0 * (r0 / (r0 + z)) ** 2
    return g


def _get_index(x: Float64, x_levels: Ptr, n: Int) -> Int:
    for i in range(n):
        var value = x_levels.unsafe_load(i)
        if i < n and value < x:
            continue
        elif x == value:
            return i
        else:
            return i - 1
    return n - 1


def _check_altitude(
    mut out: SIMD[DType.float64, 8], alt: Float64, r0: Float64, geometric: Bool
):
    var z: Float64
    var h: Float64
    if geometric:
        z = alt
        h = z_to_h(z, r0)
    else:
        h = alt
        z = h_to_z(h, r0)
    out = SIMD[DType.float64, 8](z, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------- jacchia
def _O_and_O2_correction(
    alt: Int,
    Texo: Float64,
    Z: Ptr,
    CN2: Ptr,
    CO2: Ptr,
    CO: Ptr,
    CAr: Ptr,
    CHe: Ptr,
    CH: Ptr,
    CM: Ptr,
    WM: Ptr,
):
    for iz in range(90, alt):
        CO2.unsafe_store(iz, CO2.unsafe_load(iz) * 10.0 ** (-0.07 * (1.0 + tanh(0.18 * (Z.unsafe_load(iz) - 111.0)))))
        CO.unsafe_store(iz, CO.unsafe_load(iz) * 10.0 ** (-0.24 * exp(-0.009 * (Z.unsafe_load(iz) - 97.7) ** 2)))
        CM.unsafe_store(iz, CN2.unsafe_load(iz) + CO2.unsafe_load(iz) + CO.unsafe_load(iz) + CAr.unsafe_load(iz) + CHe.unsafe_load(iz) + CH.unsafe_load(iz))
        WM.unsafe_store(iz, (
            wmN2 * CN2.unsafe_load(iz)
            + wmO2 * CO2.unsafe_load(iz)
            + wmO * CO.unsafe_load(iz)
            + wmAr * CAr.unsafe_load(iz)
            + wmHe * CHe.unsafe_load(iz)
            + wmH * CH.unsafe_load(iz)
        ) / CM.unsafe_load(iz))


def _H_correction(
    alt: Int,
    Texo: Float64,
    Z: Ptr,
    CN2: Ptr,
    CO2: Ptr,
    CO: Ptr,
    CAr: Ptr,
    CHe: Ptr,
    CH: Ptr,
    CM: Ptr,
    WM: Ptr,
    T: Ptr,
):
    var phid00 = 10.0 ** (6.9 + 28.9 * Texo ** (-0.25)) / 2e20
    phid00 = phid00 * 524.0
    var H_500 = 10.0 ** (-0.06 + 28.9 * Texo ** (-0.25))
    for iz in range(150, alt):
        var phid0 = phid00 / sqrt(T.unsafe_load(iz))
        WM.unsafe_store(iz, wmH * 0.5897446 * (1.0 + Z.unsafe_load(iz) / R0) ** (-2) / T.unsafe_load(iz) + phid0)
        CM.unsafe_store(iz, CM.unsafe_load(iz) * phid0)
    var y = WM.unsafe_load(150)
    WM.unsafe_store(150, 0.0)
    for iz in range(151, alt):
        var x = WM.unsafe_load(iz - 1) + (y + WM.unsafe_load(iz))
        y = WM.unsafe_load(iz)
        WM.unsafe_store(iz, x)
    for iz in range(150, alt):
        WM.unsafe_store(iz, exp(WM.unsafe_load(iz)) * (T.unsafe_load(iz) / T.unsafe_load(150)) ** 0.75)
        CM.unsafe_store(iz, WM.unsafe_load(iz) * CM.unsafe_load(iz))
    y = CM.unsafe_load(150)
    CM.unsafe_store(150, 0.0)
    for iz in range(151, alt):
        var x = CM.unsafe_load(iz - 1) + 0.5 * (y + CM.unsafe_load(iz))
        y = CM.unsafe_load(iz)
        CM.unsafe_store(iz, x)
    for iz in range(150, alt):
        CH.unsafe_store(iz, WM.unsafe_load(500) / WM.unsafe_load(iz) * (H_500 - (CM.unsafe_load(iz) - CM.unsafe_load(500))))
    for iz in range(150, alt):
        CM.unsafe_store(iz, CN2.unsafe_load(iz) + CO2.unsafe_load(iz) + CO.unsafe_load(iz) + CAr.unsafe_load(iz) + CHe.unsafe_load(iz) + CH.unsafe_load(iz))
        WM.unsafe_store(iz, (
            wmN2 * CN2.unsafe_load(iz)
            + wmO2 * CO2.unsafe_load(iz)
            + wmO * CO.unsafe_load(iz)
            + wmAr * CAr.unsafe_load(iz)
            + wmHe * CHe.unsafe_load(iz)
            + wmH * CH.unsafe_load(iz)
        ) / CM.unsafe_load(iz))


def _altitude_profile(
    alt_in: Float64,
    Texo_in: Float64,
    Z: Ptr,
    T: Ptr,
    CN2: Ptr,
    CO2: Ptr,
    CO: Ptr,
    CAr: Ptr,
    CHe: Ptr,
    CH: Ptr,
    CM: Ptr,
    WM: Ptr,
    E5M: Ptr,
    E6P: Ptr,
) -> Int:
    # -1 is upstream's "Jacchia77 has been implemented in range 90km - 2500km."
    if alt_in < 90.0 or 2500.0 < alt_in:
        return -1
    var alt = Int(alt_in) + 1
    # upstream does `Texo = int(Texo)`; the truncated value is what every
    # later branch uses, so it is carried as a Float64 to keep the arithmetic.
    var Texo = Float64(Int(Texo_in))
    var x = 0.0
    var y = 0.0
    for iz in range(90, alt):
        Z.unsafe_store(iz, Float64(iz))
        CH.unsafe_store(iz, 0.0)
        if iz <= 90:
            T.unsafe_store(iz, 188.0)
        elif Texo < 188.1:
            T.unsafe_store(iz, 188.0)
        else:
            x = 0.0045 * (Texo - 188.0)
            var Tx = 188 + 110.5 * log(x + sqrt(x * x + 1))
            var Gx = pi2 * 1.9 * (Tx - 188.0) / (125.0 - 90.0)
            if iz <= 125:
                T.unsafe_store(iz, Tx + (Tx - 188.0) / pi2 * atan(
                    Gx / (Tx - 188.0) * (Z.unsafe_load(iz) - 125.0) * (
                        1.0 + 1.7 * ((Z.unsafe_load(iz) - 125.0) / (Z.unsafe_load(iz) - 90.0)) ** 2
                    )
                ))
            else:
                T.unsafe_store(iz, Tx + (Texo - Tx) / pi2 * atan(
                    Gx / (Texo - Tx) * (Z.unsafe_load(iz) - 125.0) * (
                        1.0 + 5.5e-05 * (Z.unsafe_load(iz) - 125.0) ** 2
                    )
                ))
        if iz <= 100:
            x = Float64(iz - 90)
            E5M.unsafe_store(iz - 90, 28.89122 + x * (
                -0.0283071 + x * (
                    -0.00659924 + x * (
                        -0.000339574 + x * (+6.19256e-05 + x * -1.84796e-06)
                    )
                )
            ))
            if iz <= 90:
                E6P.unsafe_store(0, 71450000000000.0 * T.unsafe_load(90))
            else:
                var G0 = (1 + Z.unsafe_load(iz - 1) / R0) ** (-2)
                var G1 = (1 + Z.unsafe_load(iz) / R0) ** (-2)
                E6P.unsafe_store(iz - 90, E6P.unsafe_load(iz - 91) * exp(
                    -0.5897446 * (
                        G1 * E5M.unsafe_load(iz - 90) / T.unsafe_load(iz)
                        + G0 * E5M.unsafe_load(iz - 91) / T.unsafe_load(iz - 1)
                    )
                ))
            x = E5M.unsafe_load(iz - 90) / wm0
            y = E6P.unsafe_load(iz - 90) / T.unsafe_load(iz)
            CN2.unsafe_store(iz, qN2 * y * x)
            CO.unsafe_store(iz, 2.0 * (1.0 - x) * y)
            CO2.unsafe_store(iz, (x * (1.0 + qO2) - 1.0) * y)
            CAr.unsafe_store(iz, qAr * y * x)
            CHe.unsafe_store(iz, qHe * y * x)
            CH.unsafe_store(iz, 0.0)
        else:
            var G0 = (1 + Z.unsafe_load(iz - 1) / R0) ** (-2)
            var G1 = (1 + Z.unsafe_load(iz) / R0) ** (-2)
            x = 0.5897446 * (G1 / T.unsafe_load(iz) + G0 / T.unsafe_load(iz - 1))
            y = T.unsafe_load(iz - 1) / T.unsafe_load(iz)
            CN2.unsafe_store(iz, CN2.unsafe_load(iz - 1) * y * exp(-wmN2 * x))
            CO2.unsafe_store(iz, CO2.unsafe_load(iz - 1) * y * exp(-wmO2 * x))
            CO.unsafe_store(iz, CO.unsafe_load(iz - 1) * y * exp(-wmO * x))
            CAr.unsafe_store(iz, CAr.unsafe_load(iz - 1) * y * exp(-wmAr * x))
            CHe.unsafe_store(iz, CHe.unsafe_load(iz - 1) * y**0.62 * exp(-wmHe * x))
            CH.unsafe_store(iz, 0.0)
    _O_and_O2_correction(alt, Texo, Z, CN2, CO2, CO, CAr, CHe, CH, CM, WM)
    # `_H_correction` reads index 500 (WM[500], CM[500]), so the ten profile
    # buffers must have a 501st slot.  `alt` here is the caller's `alt + 1`,
    # and the buffers are `alt` long, so the guard is `alt >= 501`, which is
    # the model's own "H densities above 500 km" boundary.  Upstream's `if
    # 500 <= alt` is off by one against its own `int(alt + 1)` and raises
    # IndexError for alt == 499; see the README's divergence list.
    if 501 <= alt:
        _H_correction(alt, Texo, Z, CN2, CO2, CO, CAr, CHe, CH, CM, WM, T)
    return 0
