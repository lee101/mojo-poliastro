"""Session setup for the parity suite.

The tests exercise the compiled `dist/libmojo-poliastro.so`, not a Python
fallback, so the load is made explicit here rather than left implicit:

* `mojo_poliastro` is imported first, which builds the library if it is
  missing or stale and loads it.  That has to happen before any upstream
  poliastro import -- numba's JIT runtime and the Mojo runtime cannot both be
  initialised into one process in the other order.
* the resolved path is asserted to be a real shared object, so a session
  that somehow ran without one fails loudly at collection instead of passing
  quietly.
* `MOJO_POLIASTRO_REQUIRE_BUILT=1` (set below) forbids rebuilding during the
  test session.  Without it a deleted `dist/` is silently recompiled, which
  would let the suite pass on a freshly built library rather than the one the
  gate built, and would hide a broken install.
* `parity_report` is registered as a plugin so the session ends by printing the
  worst error each parity assertion reached.  It is not imported at module
  level: doing so both puts pytest's assert-rewrite warning in the output and
  leaves `parity.assert_close` unable to find it, since it is not yet in
  `sys.modules` when the tests import `parity`.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "python"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

# Fail loudly rather than rebuilding mid-session: the library under test must
# be the one `pixi run build` produced.
os.environ.setdefault("MOJO_POLIASTRO_REQUIRE_BUILT", "1")

import mojo_poliastro  # noqa: E402,F401

_LIB = mojo_poliastro._lib.LIB

if not os.path.isfile(_LIB):
    raise RuntimeError(
        f"the compiled library is missing at {_LIB}; run `pixi run build` "
        "before `pixi run test`. The suite deliberately refuses to fall back "
        "to a pure-Python path."
    )

pytest_plugins = ["parity_report"]
