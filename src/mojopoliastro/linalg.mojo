"""Port of `poliastro._math.linalg`.

Upstream this file holds a single numba function::

    @jit
    def norm(arr):
        return np.sqrt(arr @ arr)

Upstream's `arr` is a numpy 3-vector and `@` is `np.dot`.  Mojo has no array
type at this level, so a 3-vector is a `V3` and `@` becomes `dot`.  `cross` is
lifted in from the `numpy` import upstream uses, for the same reason.  The
arithmetic operators on `V3` exist so the ported expressions read like
upstream writes them.  Mojo resolves a binary operator on its left operand,
so upstream's `scalar * vec` is written `vec * scalar`; the products are
identical.
"""

from std.math import sqrt


struct V3:
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z

    @always_inline
    def __add__(a: V3, b: V3) -> V3:
        return V3(a.x + b.x, a.y + b.y, a.z + b.z)

    @always_inline
    def __sub__(a: V3, b: V3) -> V3:
        return V3(a.x - b.x, a.y - b.y, a.z - b.z)

    @always_inline
    def __neg__(a: V3) -> V3:
        return V3(-a.x, -a.y, -a.z)

    @always_inline
    def __mul__(a: V3, s: Float64) -> V3:
        return V3(a.x * s, a.y * s, a.z * s)

    @always_inline
    def __truediv__(a: V3, s: Float64) -> V3:
        return V3(a.x / s, a.y / s, a.z / s)


def dot(a: V3, b: V3) -> Float64:
    """Stands in for numpy's `a @ b`."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: V3, b: V3) -> V3:
    """Stands in for `numpy.cross` (upstream restricts it to 3-vectors too)."""
    return V3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def norm(arr: V3) -> Float64:
    return sqrt(dot(arr, arr))
