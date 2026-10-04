"""A Mojo port of poliastro's numeric core.

The shared library is loaded eagerly here.  Upstream poliastro JIT-compiles
through numba, and numba's LLVM runtime conflicts with the Mojo runtime if it
is initialised first; loading this library at import time fixes the order.
`tests/conftest.py` imports this package before any upstream import so the
ordering holds for the whole session.
"""

from ._lib import build, lib

__version__ = "0.1.0"

__all__ = ["build", "lib", "__version__"]

lib()
