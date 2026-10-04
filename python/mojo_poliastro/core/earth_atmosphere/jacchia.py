"""Port of `poliastro.core.earth_atmosphere.jacchia`.

Upstream's `_altitude_profile` allocates ten Python lists; the C ABI cannot
allocate on the caller's behalf, so this module allocates the ten buffers and
hands their addresses over.  The returned arrays are exactly the ten upstream
returns, each of length ``int(alt) + 1``.
"""

import numpy as np

from ..._lib import addr, fn

__all__ = ["altitude_profile"]

_NAMES = ("Z", "T", "CN2", "CO2", "CO", "CAr", "CHe", "CH", "CM", "WM")


def altitude_profile(alt, Texo):
    """Jacchia 1977 atmosphere profile at `alt` km for exospheric `Texo` K."""
    alt = int(alt)
    Texo = int(Texo)
    if not 90 <= alt <= 2500:
        raise ValueError("Jacchia77 has been implemented in range 90km - 2500km.")
    n = alt + 1
    bufs = [np.zeros(n, dtype=np.float64) for _ in range(10)]
    e5m = np.zeros(11, dtype=np.float64)
    e6p = np.zeros(11, dtype=np.float64)
    status = fn("mp_jacchia_altitude_profile")(
        float(alt), float(Texo), *(addr(b) for b in bufs), addr(e5m), addr(e6p)
    )
    if status != 0:
        raise ValueError("Jacchia77 has been implemented in range 90km - 2500km.")
    return {name: buf for name, buf in zip(_NAMES, bufs)}
