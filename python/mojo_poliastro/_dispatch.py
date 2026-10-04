"""Broadcasting glue between numpy callers and the batched C ABI.

Upstream's kernels are scalar `@njit` functions: scalars in, scalars out.  The
Mojo build batches them, so this module restores upstream's calling
convention -- every array argument is broadcast against the others, and a call
made entirely with scalars returns a Python float.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, fn


class Batch:
    """Broadcast views of the arguments of one call, plus the output buffer."""

    __slots__ = ("shape", "scalar", "n", "inp", "out")

    def __init__(self, *args, tail: tuple[int, ...] = (), fill: bool = True):
        arrs = [np.asarray(a, dtype=np.float64) for a in args]
        self.scalar = all(a.ndim == 0 for a in arrs)
        if self.scalar:
            self.shape = ()
            self.n = 1
            self.inp = [f64([a.item()]) for a in arrs]
        else:
            self.shape = np.broadcast_shapes(*[a.shape for a in arrs])
            self.n = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
            self.inp = [
                f64(np.broadcast_to(a, self.shape).reshape(-1)) for a in arrs
            ]
        self.out = np.empty((self.n,) + tail, dtype=np.float64) if fill else None

    def __call__(self, symbol: str, *args) -> "Batch":
        fn(symbol)(*args, addr(self.out), self.n)
        return self

    def result(self):
        if self.scalar:
            return float(self.out[0])
        return self.out.reshape(self.shape + self.out.shape[1:])

    def rows(self, out):
        """`out` with the batch axis dropped for a scalar call, kept and
        reshaped otherwise.  Use this wherever the result is indexed
        further (a pair of state vectors, a six-tuple of elements)."""
        if self.scalar:
            return out[0]
        return out.reshape(self.shape + out.shape[1:])


class MixedBatch:
    """Broadcast a mix of scalar-valued and ``(..., 3)`` vector arguments.

    Upstream's geodetic and perifocal helpers take 3-vectors, not sequences
    of them, so a 1-D length-3 array is one argument rather than three.

    Every argument is broadcast to the full batch before it crosses the ABI:
    a 0-d scalar becomes `n` copies and a 1-D 3-vector becomes `n` copies of
    itself.  A kernel reads `n` elements from every buffer it is handed, so
    a buffer left at its own shorter length would be read past its end.
    """

    __slots__ = ("scalar", "n", "s", "v", "out")

    def __init__(self, scalars=(), vectors=(), tail: tuple[int, ...] = ()):
        self.s = [np.asarray(x, dtype=np.float64) for x in scalars]
        self.v = [np.asarray(x, dtype=np.float64) for x in vectors]
        lens = [a.shape[0] for a in self.s if a.ndim == 1]
        lens += [a.shape[0] for a in self.v if a.ndim >= 2]
        self.scalar = not lens
        if lens:
            self.n = max(lens)
            if any(length not in (1, self.n) for length in lens):
                raise ValueError(
                    f"cannot broadcast arguments with leading lengths {lens}"
                )
        else:
            self.n = 1
        self.s = [
            np.ascontiguousarray(np.broadcast_to(a, (self.n,)))
            for a in self.s
        ]
        self.v = [
            np.ascontiguousarray(np.broadcast_to(a, (self.n, 3)).reshape(-1))
            for a in self.v
        ]
        self.out = np.empty((self.n,) + tail, dtype=np.float64)


def map1(symbol: str, *args, tail: tuple[int, ...] = ()):
    """One array argument, one array result of `tail` width."""
    b = Batch(*args, tail=tail)
    fn(symbol)(*(addr(a) for a in b.inp), addr(b.out), b.n)
    return b.result()


def map2(symbol: str, a, b, tail: tuple[int, ...] = ()):
    """Two array arguments, one array result of `tail` width."""
    bb = Batch(a, b, tail=tail)
    fn(symbol)(addr(bb.inp[0]), addr(bb.inp[1]), addr(bb.out), bb.n)
    return bb.result()


