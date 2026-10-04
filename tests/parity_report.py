"""Record and print the worst parity error each assertion label reached.

The suite asserts against `parity.RTOL` but nothing records how close the run
actually came, so a change that quietly moved a kernel from 1e-16 to 8e-10
would still be green. This prints the worst per label at the end of the
session, so the README's tolerance claim can be read against a number.

Imported by `conftest.py` as `parity_report`; `parity.assert_close` looks it
up in `sys.modules` and is a no-op when it is absent.
"""

import sys

WORST: dict[str, tuple[float, float]] = {}


def record(label: str, error: float, rtol: float) -> None:
    key = label.split(":")[0] or "(unlabelled)"
    prev = WORST.get(key, (0.0, float(rtol)))
    WORST[key] = (max(prev[0], float(error)), max(prev[1], float(rtol)))


def pytest_sessionfinish(session, exitstatus):
    if not WORST:
        return
    order = sorted(WORST.items(), key=lambda kv: -kv[1][0])
    print(f"\nworst parity error per label ({len(WORST)} labels, "
          f"pytest exit {exitstatus})", file=sys.stderr)
    for label, (error, rtol) in order[:12]:
        print(f"  {label:<50} {error:10.3e}  tol {rtol:.1e}", file=sys.stderr)
    top_label, (top_error, top_rtol) = order[0]
    print(f"  worst: {top_label} at {top_error:.3e} "
          f"(tolerance {top_rtol:.1e})", file=sys.stderr)
