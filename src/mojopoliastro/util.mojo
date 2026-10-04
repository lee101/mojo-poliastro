"""Port of `poliastro.core.util`.

Small numeric helpers shared by the element conversions: elementary rotation
matrices, an angular `linspace`, and the spherical/planetocentric frame
transforms.  Functions appear in upstream's source order.

DIVERGENCE: Mojo structs are move-only in this dialect, so the functions that
upstream returns as numpy arrays fill a caller-provided `Mat3` through a
leading `out` argument instead of returning one.
"""

from std.math import cos, floor, isnan, sin
from std.utils import StaticTuple

from mojopoliastro.linalg import V3

comptime PI: Float64 = 3.141592653589793
comptime TWO_PI: Float64 = 6.283185307179586

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]


struct Mat3:
    var r: StaticTuple[Float64, 9]

    def __init__(out self):
        self.r = StaticTuple[Float64, 9](
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    @always_inline
    def get(self, i: Int, j: Int) -> Float64:
        return self.r[3 * i + j]

    @always_inline
    def set(mut self, i: Int, j: Int, v: Float64):
        self.r[3 * i + j] = v


def matmul(mut out: Mat3, a: Mat3, b: Mat3):
    """Stands in for numpy's `a @ b` on two 3x3 matrices.

    The row of `a` is hoisted out of the column loop: three loads instead of
    nine, which matters because this runs once per orbit in `coe2rv`.
    """
    for i in range(3):
        var a0 = a.get(i, 0)
        var a1 = a.get(i, 1)
        var a2 = a.get(i, 2)
        for j in range(3):
            out.set(i, j, a0 * b.get(0, j) + a1 * b.get(1, j) + a2 * b.get(2, j))


def transpose(mut out: Mat3, a: Mat3):
    for i in range(3):
        for j in range(3):
            out.set(i, j, a.get(j, i))


def rotation_matrix(mut out: Mat3, angle: Float64, axis: Int):
    var c = cos(angle)
    var s = sin(angle)

    var a1 = (axis + 1) % 3
    var a2 = (axis + 2) % 3
    out.set(axis, axis, 1.0)
    out.set(a1, a1, c)
    out.set(a1, a2, -s)
    out.set(a2, a1, s)
    out.set(a2, a2, c)


# DIVERGENCE: upstream's signature is `alinspace(start, stop=None, num=50,
# endpoint=True)` and it returns an array.  Mojo default arguments cannot
# reference another argument, so `stop` arrives as a NaN sentinel meaning
# "not given" (upstream's `None`), and the samples are written into a
# caller-provided buffer instead of being returned.
def alinspace(start: Float64, stop: Float64, mut dst: Ptr, num: Int, endpoint: Bool):
    var stop_: Float64
    if isnan(stop):
        stop_ = start + TWO_PI
    elif stop <= start:
        stop_ = stop + (floor((start - stop) / 2 / PI) + 1) * TWO_PI
    else:
        stop_ = stop

    var count = num
    if not endpoint:
        count = num + 1

    # numpy's linspace pins the final sample to the stop value.  There is no
    # interval to divide by for a single sample, and no sample to pin for an
    # empty one.
    if count <= 1:
        if count == 1:
            dst.unsafe_store(0, start)
        return
    var i = 0
    while i < count:
        dst.unsafe_store(i, start + (stop_ - start) * Float64(i) / Float64(count - 1))
        i += 1
    dst.unsafe_store(count - 1, stop_)


def spherical_to_cartesian(mut out: V3, v: V3):
    r"""Compute cartesian coordinates from spherical coordinates (norm, colat, long).

    .. math::

       v = norm \cdot \begin{bmatrix}
       \sin(colat)\cos(long)\\
       \sin(colat)\sin(long)\\
       \cos(colat)\\
       \end{bmatrix}
    """
    var norm_vecs = v.x
    var vsin = sin(v.y)
    var vsin1 = sin(v.z)
    var vcos = cos(v.y)
    var vcos1 = cos(v.z)
    var x = vsin * vcos1
    var y = vsin * vsin1
    var z = vcos
    out.x = norm_vecs * x
    out.y = norm_vecs * y
    out.z = norm_vecs * z


def planetocentric_to_AltAz(mut out: Mat3, theta: Float64, phi: Float64):
    r"""Transformation matrix from planetocentric to Altitude-Azimuth.

    .. math::
       t\_matrix = \begin{bmatrix}
       -\sin(theta) & \cos(theta) & 0\\
       -\sin(phi)\cdot\cos(theta) & -\sin(phi)\cdot\sin(theta) & \cos(phi)\\
       \cos(phi)\cdot\cos(theta) & \cos(phi)\cdot\sin(theta) & \sin(phi)
       \end{bmatrix}
    """
    out.set(0, 0, -sin(theta))
    out.set(0, 1, cos(theta))
    out.set(0, 2, 0.0)
    out.set(1, 0, -sin(phi) * cos(theta))
    out.set(1, 1, -sin(phi) * sin(theta))
    out.set(1, 2, cos(phi))
    out.set(2, 0, cos(phi) * cos(theta))
    out.set(2, 1, cos(phi) * sin(theta))
    out.set(2, 2, sin(phi))
