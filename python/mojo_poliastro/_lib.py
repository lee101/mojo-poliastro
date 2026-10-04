"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")


def _default_lib() -> str:
    """The build output: the repo's `dist/` in a checkout.

    A non-editable install lands in site-packages, where `ROOT` no longer
    names a checkout; there the sources are absent too and a library must be
    supplied with `MOJO_POLIASTRO_LIB`.
    """
    return os.path.join(ROOT, "dist", "libmojo-poliastro.so")


LIB = os.environ.get("MOJO_POLIASTRO_LIB") or _default_lib()

I = ctypes.c_int64
F = ctypes.c_double

# Ceiling on the worker pool.  A batch is a map over independent elements, so
# past this width the extra workers stop paying for themselves.
_MAX_WORKERS = 32

# A `parallelize` launch builds a pool sized from the machine rather than from
# the worker count, so a cgroup needs room for roughly one thread per core
# plus slack before a launch is worth attempting at all.
_POOL_SLACK = 2

# name -> (argtypes, restype).  Every export takes the `parallelize` worker
# count as its last argument; see `src/capi.mojo`.
_SIGNATURES = {
    "mp_norm": ([I, I, I], None),
    "mp_hyp2f1b": ([I, I, I], None),
    "mp_stumpff_c2": ([I, I, I], None),
    "mp_stumpff_c3": ([I, I, I], None),
    "mp_D_to_nu": ([I, I, I], None),
    "mp_nu_to_D": ([I, I, I], None),
    "mp_nu_to_E": ([I, I, I, I], None),
    "mp_nu_to_F": ([I, I, I, I], None),
    "mp_E_to_nu": ([I, I, I, I], None),
    "mp_F_to_nu": ([I, I, I, I], None),
    "mp_M_to_E": ([I, I, I, I], None),
    "mp_M_to_F": ([I, I, I, I], None),
    "mp_M_to_D": ([I, I, I], None),
    "mp_E_to_M": ([I, I, I, I], None),
    "mp_F_to_M": ([I, I, I, I], None),
    "mp_D_to_M": ([I, I, I], None),
    "mp_fp_angle": ([I, I, I, I], None),
    "mp_rotation_matrix": ([I, I, I, I], None),
    "mp_alinspace": ([F, F, I, I, I], None),
    "mp_spherical_to_cartesian": ([I, I, I], None),
    "mp_planetocentric_to_AltAz": ([I, I, I, I], None),
    "mp_eccentricity_vector": ([F, I, I, I, I], None),
    "mp_circular_velocity": ([F, I, I, I], None),
    "mp_rv_pqw": ([F, I, I, I, I, I], None),
    "mp_coe_rotation_matrix": ([I, I, I, I, I], None),
    "mp_coe2rv": ([F, I, I, I, I, I, I, I, I], None),
    "mp_coe2rv_many": ([I] * 9, None),
    "mp_coe2mee": ([I, I, I, I, I, I, I, I, I], I),
    "mp_rv2coe": ([F, I, I, F, I, I], None),
    "mp_mee2coe": ([I] * 8, None),
    "mp_mee2rv": ([I] * 8, None),
    "mp_vallado_iod": ([F, I, I, I, I, I, I, I, F, I, I, I], I),
    "mp_izzo": ([F, I, I, I, I, I, I, I, F, I, I, I], I),
    "mp_func_twobody": ([F, I, I, I], None),
    "mp_S_x": ([I, I, F, I, I], None),
    "mp_dS_x_alt": ([I, I, F, I, I], None),
    "mp_d2S_x_alt": ([I, I, F, I, I], None),
    "mp_D_to_M_near_parabolic": ([I, I, I, I], None),
    "mp_M_to_D_near_parabolic": ([I, I, F, I, I, I], None),
    "mp_delta_t_from_nu": ([I, I, F, F, F, I, I], None),
    "mp_nu_from_delta_t": ([I, I, F, F, F, I, I], None),
    "mp_farnocchia_coe": ([F] + [I] * 9, None),
    "mp_farnocchia_rv": ([F, I, I, I, I, I], None),
    "mp_vallado_uni": ([F, I, I, I, I, I, I, I], I),
    "mp_markley_coe": ([F] + [I] * 9, None),
    "mp_markley": ([F, I, I, I, I, I], None),
    "mp_gooding_coe": ([F] + [I] * 8 + [F] + [I, I], None),
    "mp_gooding": ([F, I, I, I, I, F, I, I], None),
    "mp_cartesian_cords": ([I] * 7, None),
    "mp_f": ([I, I, I, I], None),
    "mp_N": ([I] * 6, None),
    "mp_tangential_vecs": ([I, I, I], None),
    "mp_radius_of_curvature": ([I, I, I, I, I], None),
    "mp_distance": ([I, I, I, I], None),
    "mp_is_visible": ([I, I, I, I, I], None),
    "mp_cartesian_to_ellipsoidal": ([I] * 7, None),
    "mp_geometric_to_geopotential": ([I, F, I, I], None),
    "mp_geopotential_to_geometric": ([I, F, I, I], None),
    "mp_gravity": ([I, F, F, I, I], None),
    "mp_check_altitude": ([I, F, I, I, I], None),
    "mp_jacchia_altitude_profile": (
        [F, F] + [I] * 12,
        I,
    ),
}

# One trailing `Int` on every export: the worker count.
_SIGNATURES = {
    name: (argtypes + [I], restype)
    for name, (argtypes, restype) in _SIGNATURES.items()
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_POLIASTRO_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_POLIASTRO_MOJO=/path/to/mojo")



def _build_env(cmd: list[str]) -> dict:
    """The environment the compiler needs to find `std` and `max`.

    `pixi run` sets `MODULAR_HOME` itself; a bare `mojo` on `PATH` does not,
    and every import then fails with `unable to locate module 'std'`, which
    reads like a missing package but is not.  Derive it from the binary's
    own location when it is not already set.
    """
    env = dict(os.environ)
    if env.get("MODULAR_HOME"):
        return env
    binary = shutil.which(cmd[0]) or cmd[0]
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(binary)))
    candidate = os.path.join(prefix, "share", "max")
    if os.path.isdir(candidate):
        env["MODULAR_HOME"] = candidate
    return env


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into the shared library if it is stale.

    Set `MOJO_POLIASTRO_REQUIRE_BUILT=1` to refuse to compile and raise
    instead.  The test suite sets it so a deleted or stale `dist/` fails the
    run outright rather than being silently rebuilt: a suite that rebuilds
    what it is meant to be testing proves nothing about the shipped library.
    """
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no library at {LIB}; this looks "
            "like a copied install - point MOJO_POLIASTRO_LIB at a built "
            "libmojo-poliastro.so"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    if os.environ.get("MOJO_POLIASTRO_REQUIRE_BUILT"):
        raise BuildError(
            f"{LIB} is missing or older than its sources, and "
            "MOJO_POLIASTRO_REQUIRE_BUILT is set, so it will not be rebuilt. "
            "Run `pixi run build` first."
        )
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        "-j", os.environ.get("MOJO_BUILD_JOBS", "4"),
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, env=_build_env(cmd)
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def _cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _pids_budget() -> int | None:
    """How many more threads this cgroup may still create, or None if unlimited.

    A container's pids limit is shared by every process inside it, so the size
    of the pool a launch may build is not a constant of this library.  The
    failure is a hard `pthread_create failed` abort inside LLVM, not an
    exception the caller can catch, so it has to be avoided before the launch.
    """
    try:
        with open("/proc/self/cgroup") as fh:
            path = fh.read().strip().rsplit(":", 1)[-1]
        base = "/sys/fs/cgroup" + path.rstrip("/")
        with open(base + "/pids.max") as fh:
            limit = fh.read().strip()
        if limit == "max":
            return None
        with open(base + "/pids.current") as fh:
            used = int(fh.read().strip())
    except (OSError, ValueError, IndexError):
        return None
    # Leave room for this process, its threads, and anything else that lands
    # in the cgroup while the call runs.
    return int(limit) - used - 24


def _compute_workers() -> int:
    """1 to run every batch serially, more to launch a pool."""
    override = os.environ.get("MOJO_POLIASTRO_WORKERS")
    if override:
        return max(1, int(override))
    cpus = _cpus()
    budget = _pids_budget()
    if budget is not None and budget < _POOL_SLACK * cpus:
        # `parallelize` sizes its pool from the machine rather than from the
        # worker count, so a launch needs room for roughly one thread per core
        # plus slack.  Without it, stay serial: the same `work` item runs, one
        # element at a time, and the numbers are identical.
        return 1
    return min(_MAX_WORKERS, cpus)


_workers: tuple[float, int] | None = None


def worker_count() -> int:
    """Worker threads a batched export should launch; 1 means stay serial.

    Re-read at most once a second: the cgroup budget moves while a long run is
    in progress, and the cost of two small file reads is not worth paying on
    every scalar call.
    """
    global _workers
    now = time.monotonic()
    if _workers is None or now - _workers[0] > 1.0:
        _workers = (now, _compute_workers())
    return _workers[1]


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            sym = getattr(_lib, name)
            sym.argtypes = argtypes
            sym.restype = restype
    return _lib


def fn(name: str):
    """A symbol carrying its configured `argtypes` / `restype`.

    This must be `getattr(lib(), name)` and never `lib()[name]`: ctypes
    defines `CDLL.__getitem__`, which builds a fresh function pointer on
    every access and so ignores the `argtypes` set above.  An unconfigured
    pointer defaults to C `int` arguments, which truncates a 64-bit buffer
    address and segfaults on the call.

    The returned callable appends the `parallelize` worker count, which every
    export takes last; callers never pass it themselves.
    """
    sym = getattr(lib(), name)

    def call(*args):
        return sym(*args, worker_count())

    return call


def f64(a) -> np.ndarray:
    """A C-contiguous float64 array of `a`, copying only when it has to.

    Callers own the result and keep it alive across the C call, so the
    address taken from it stays valid for the whole call.
    """
    return np.ascontiguousarray(a, dtype=np.float64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojo_poliastro._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
