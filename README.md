# mojo-poliastro

A Mojo port of [poliastro](https://github.com/poliastro/poliastro)'s numeric
core, with a Python API that mirrors upstream's names, argument order and
defaults. Where it covers a function, it is a drop-in replacement for it.

Upstream `poliastro.core` is `@njit`-compiled but *scalar*: most of it cannot
accept an array at all. This port keeps the same scalar entry points and adds a
batched call underneath, so one `izzo(...)` over 200,000 transfers is one
ctypes call instead of 200,000 Python calls.

The headline number is that batched Lambert solve: 17.90 ms against upstream's
855.05 ms for the same 200,000 transfers. The ratio is almost entirely the cost
of Python-to-numba dispatch that the port removes, not better arithmetic, and
[Benchmarks](#benchmarks) says so at length. Absolute times are the sound
comparison.

## Install

From a checkout, with the pinned toolchain:

```bash
pixi install
pixi run build      # mojo build --emit shared-lib -> dist/libmojo-poliastro.so
pixi run test
pixi run bench
```

Or as an ordinary Python package, which builds the shared library on first
import if it is missing or stale:

```bash
pip install -e .
```

The build needs a compiler: `mojo` and `max` at the versions pinned in
`pixi.toml`. If `mojo` is not on `PATH`, point `MOJO_POLIASTRO_MOJO` at it; if
the compiler cannot find its own standard library, set `MODULAR_HOME` to the
`share/max` next to that binary (`pixi run` sets both for you). To use a
prebuilt library instead of compiling — which is what a non-editable install
in site-packages needs, where the `src/` tree is not present — set
`MOJO_POLIASTRO_LIB` to its path.

`MOJO_POLIASTRO_WORKERS` pins the thread pool; `1` runs every batch serially.

## Usage

```python
import numpy as np
import mojo_poliastro.core.iod as iod
import mojo_poliastro.core.elements as elements

GM_earth = 398600.4418

# One transfer, exactly upstream's call:
v0, v = iod.izzo(GM_earth, [7000.0, 0.0, 0.0], [0.0, 9000.0, 0.0], 3600.0,
                 0, True, True, 35, 1e-8)
# v0 -> array([4.44819619, 6.16205154, 0.        ])
# v  -> array([-4.79270676, -3.0788514 , 0.        ])

# The same call over a batch of 200,000 transfers:
r0 = np.random.default_rng(0).normal(size=(200_000, 3)) * 9000.0
r1 = np.random.default_rng(1).normal(size=(200_000, 3)) * 9000.0
tof = np.random.default_rng(2).uniform(600.0, 20_000.0, 200_000)
v0, v = iod.izzo(GM_earth, r0, r1, tof, 0, True, True, 35, 1e-8)
v0.shape   # (200000, 3)

# Element conversions vectorise the same way.
p, ecc, inc, raan, argp, nu = elements.coe2mee(
    11000.0, 0.3, 0.9, 0.2, 1.1, 2.0
)   # -> a 6-tuple of floats, as upstream returns
```

Scalars in give scalars out, exactly as upstream; arrays in give arrays out.
Arguments broadcast against each other, so a `(n, 1)` and an `(m,)` argument
produce an `(n, m)` batch — upstream's numba-typed scalars cannot do that, so
it is an addition rather than a change.

## What is covered

One Mojo kernel per upstream function, in upstream's file, in upstream's
source order. `src/capi.mojo` is the C-ABI layer; the kernels
live in `src/mojopoliastro/`, one file per
upstream module, and each reads
top-to-bottom like the upstream file it came from.

| upstream module | covered |
| --- | --- |
| `poliastro._math.linalg` | `norm` |
| `poliastro._math.special` | `hyp2f1b`, `stumpff_c2`, `stumpff_c3` |
| `poliastro.core.angles` | all 13 conversions, plus the Newton solves behind `M_to_E` / `M_to_F` |
| `poliastro.core.util` | `rotation_matrix`, `alinspace`, `spherical_to_cartesian`, `planetocentric_to_AltAz` |
| `poliastro.core.elements` | all 10: `eccentricity_vector`, `circular_velocity`, `rv_pqw`, `coe_rotation_matrix`, `coe2rv`, `coe2rv_many`, `coe2mee`, `rv2coe`, `mee2coe`, `mee2rv` |
| `poliastro.core.iod` | `vallado`, `izzo` and all 13 of Izzo's private helpers |
| `poliastro.core.propagation.base` | `func_twobody` |
| `poliastro.core.propagation.farnocchia` | all 11 |
| `poliastro.core.propagation.vallado` | `vallado` (universal-variable Lagrange coefficients) |
| `poliastro.core.propagation.markley` | `markley_coe`, `markley` |
| `poliastro.core.propagation.gooding` | `gooding_coe`, `gooding` |
| `poliastro.core.spheroid_location` | all 8 |
| `poliastro.core.earth_atmosphere.util` | `geometric_to_geopotential` (= `z_to_h`), `geopotential_to_geometric` (= `h_to_z`), `gravity`, `_check_altitude` (= `check_altitude`) |
| `poliastro.core.earth_atmosphere.jacchia` | `_altitude_profile` and its two helpers |

61 exported entry points. The kernels cover all 84 `def`s in those 14 upstream
modules, of which 83 are reachable from Python and tested. The exception is
`_get_index`, whose only callers are in the astropy `earth.atmosphere` layer
below, which is not part of this port; it is ported in Mojo but not exported.

`tests/test_coverage_claims.py` enforces that table rather than describing it:
each listed module must still exist, still define its kernels, and still be
imported by the ABI layer; each listed function must have a test that calls it
by name; the propagators documented as *absent* must be absent. An export
added later without a test fails the suite.

## What is not covered, and why

- **`cowell`** is a thin wrapper around `scipy.integrate.solve_ivp` with
  `DOP853`; there is nothing to port, only a scipy dependency to keep.
- **`danby`, `mikkola`, `pimienta`, `recseries`** are not ported. They are
  further universal-variable refinements of the same problem `vallado` and
  `farnocchia` already cover here, and the remaining gap does not earn a
  second propagation stack.
- **The `poliastro.twobody.*` astropy layer** (`Orbit`, `twobody.elements`,
  `twobody.angles`, `twobody.sampling`, `mean_elements`, `threebody`, `earth.*`)
  is not ported. Those are unit-aware wrappers, not kernels; upstream itself
  is a thin `Quantity` shell over the `poliastro.core` functions that are
  here. They also do not import against a modern astropy, which is why the
  parity oracle is `poliastro.core` and not `poliastro.twobody`.
- **`plotting`, `czml`, `ephem`, `sensors`, `io`, `bodies`, `constants`** are
  I/O and data plumbing with no compute to accelerate.
- **No GPU path.** Measured on this machine, and the numbers are below: a
  `DeviceContext()` costs about 240 ms to construct and cannot be cached
  across an `abi("C")` export, which is more than thirteen times the entire
  batched CPU kernel it would replace.

## Divergences from upstream

Each of these is a place where the two cannot be made identical. All are
deliberate; none is silent.

**Language-level, values unchanged.** Mojo has no implicit int/float
promotion, so integer series counters are carried as `Float64` (`ii = 0`
becomes `ii = 0.0`) and integer literals inside float expressions are written
as floats (`3` becomes `3.0`). `scalar * vec` is written `vec * scalar`
because Mojo resolves a binary operator on its left operand. numpy's `arr @
arr` and `numpy.cross` become `dot(a, b)` and `cross(a, b)` over a `V3`
struct. All products are identical.

**Array returns.** Mojo structs are move-only in this dialect and
`return <local struct>` does not parse, so the functions upstream returns as
numpy arrays fill a caller-provided `Mat3` through a leading `out` argument.
The Python layer allocates and hands that over, so the *API* is unchanged:
`coe2rv` still returns `(r_ijk, v_ijk)`.

**`newton_factory`.** Upstream builds its two Newton solvers by returning a
closure from `newton_factory(func, fprime)`. Mojo cannot pass a `def` as a
value, so `_newton_elliptic` and `_newton_hyperbolic` are written out with the
factory's loop body unchanged. Argument order follows the call sites.

**`alinspace`.** Upstream's signature is `alinspace(start, stop=None, num=50,
endpoint=True)`. Mojo default arguments cannot reference another argument, so
`stop` arrives as a NaN sentinel meaning "not given", and the samples are
written into a caller-provided buffer. The Python wrapper restores upstream's
signature and return value.

**Exceptions.** An `abi("C")` entry point cannot raise, so the kernels return
a status `Int` and the Python layer turns it back into upstream's exception
with upstream's message: `izzo` and `core.propagation.vallado` raise
`NotImplementedError` / `RuntimeError` / `ValueError` exactly as upstream does.
Three cases return NaN instead, and the README says so where they appear:
`coe2mee` raises `ValueError` for a 180 degree inclination (as upstream does);
`gooding_coe` returns NaN for `ecc >= 1` where upstream raises
`NotImplementedError`; `_newton_elliptic` / `_newton_hyperbolic` return NaN on
non-convergence, which is upstream's own behaviour.

**`x ** 0.5`.** Mojo's `pow` is a fast approximation and disagrees with
glibc's by around 1e-9 relative, which the catastrophically cancelling
`B + sqrt(1 + B**2)` in `M_to_D` amplifies. `M_to_D` therefore writes `sqrt`
where upstream writes `** 0.5`, which is the bit-exact match for CPython's
`x ** 0.5`. Mojo's `sin`, `cos`, `tan`, `atan` and `atan2` are bit-identical
to glibc; `pow`, `exp` and `log` are not, which is why the parity tolerance is
1e-9 rather than 1e-15. Three comparisons document their own looser tolerance
and say why: the Jacchia composition recursion (two chained recurrences over
up to 2500 altitude steps, each seeded by `exp`), the farnocchia `S_x` series
(compared against a re-derivation, not upstream's own truncated sum), and the
Lagrange coefficients. The suite prints the worst error each of its 347
assertions actually reached: 4.4e-09 against its own 1e-07 for `S_x`, and
1.7e-10 against the 1e-9 default, which is `M_to_D` — the `pow` divergence is
the whole of the 1e-9 budget.

**Newton give-up outside the elliptic domain.** For `|ecc| > 1`,
`M_to_E` solves a problem that has no solution, and which iterate first lands
on the vanishing derivative `1 - e cos E` is decided in the last bit. The two
implementations therefore give up on slightly different subsets of that
non-physical domain. Inside `0 <= ecc < 1` they agree exactly; the test suite
asserts the give-up contract only where both agree, and says why.

**`mee2rv`.** Upstream's `mee2rv` cannot be executed by anyone: it writes
`(r / s2)(np.cos(L) + ...)`, which calls a float. Both numba and CPython
reject it. This port implements the intended product, and the test suite
compares against a NumPy transcription of upstream's source with the missing
operator restored. The benchmark row for it is excluded from the total for the
same reason.

**`cartesian_cords`.** Upstream passes `_lon` where its formula reads like a
latitude. The port keeps upstream's argument order and its arithmetic, quirk
included, so the two agree.

**`np.sign` in `core.propagation.vallado`.** Mojo has no `sign()`, so the
three cases are spelled out.

**The Stumpff series no longer calls `gamma`.** Upstream walks
`C_2(psi) = sum_k (-psi)^k / (2k+2)!` and evaluates each denominator as
`gamma(2k+3)`, which is a transcendental evaluation of an integer factorial
and the single largest cost in `stumpff_c2`. The same series follows from the
term ratio `(-psi) / ((2k+4)(2k+3))`, so each term is now two multiplies and
a divide. The terms are the same to a few ulp and the sum still stops on the
same `res + delta == res` test, so the results agree to well inside the 1e-9
parity tolerance. In the table above, `stumpff_c2` over 1M elements takes
12.46 ms on this build.

**`hyp2f1b` drops a cancelling factor.** Upstream's term is
`term * (3+ii)*(1+ii) / (2.5+ii) * x / (ii+1)`, in which `(1+ii)` appears in
both the numerator and the denominator. The port evaluates the algebraically
identical `term * x * (3+ii) / (2.5+ii)`, which is one division per term
instead of two. The rounding differs by a few ulp per term and the series
still terminates on the same equality test, so the result agrees to about
1e-15 relative — far inside tolerance, and far below the 1e-8 that `izzo`'s
Householder iteration asks of `hyp2f1b`.

**Repeated subexpressions are hoisted.** `rv_pqw` evaluated `cos(nu)` four
times and `sqrt(k / p)` three times per orbit; `mee2rv` evaluated `cos(L)`
five times, `sin(L)` three and `sqrt(k / p)` twice; `markley_coe` built its
`f0..f3` Kepler terms from a `sin(E)` and a `cos(E)` it re-evaluated twice
each; and Vallado's bisection recomputed `c2(psi)` and `c3(psi)` up to five
times for the same `psi` inside a single iteration. Each is now computed
once. These are the same floating point operations in the same order, so the
values are unchanged — only the count moved. `matmul` also hoists the row of
`a` out of its column loop, which is three loads instead of nine.

**The status buffer.** The four exports that report a failure per element
(`mp_coe2mee`, `mp_vallado_iod`, `mp_izzo`, `mp_vallado_uni`) cannot keep the
status in a local any more, because a `parallelize` worker has no way to
report back through a captured variable. Each element parks its code in a
slot the caller owns, and the export reduces the slots afterwards, taking the
last non-zero one — the same answer, in the same order, that the serial loop
produced. This adds one `int64` per element to the C ABI of those four
functions; the Python wrappers allocate it. The Python API is unchanged, and
an element that fails deep inside a parallel batch still raises.

**Jacchia77 at 499 km.** Upstream allocates ten lists of length
`int(alt + 1)` and then guards its `_H_correction` call with `500 <= alt`,
which by then is the *incremented* length. At a requested altitude of 499 km
that guard passes while the lists are 500 long, and `_H_correction` indexes
`WM[500]` and `CM[500]`: upstream raises `IndexError`. The port tests
`501 <= alt` instead, so 499 km returns a profile with no H densities — which
is what the model's own documentation says should happen below 500 km — and
500 km is the first altitude that gets them. Every altitude upstream handles
without raising gives the same numbers here.

**`alinspace` with `num` of 0 or 1.** Upstream defers to `numpy.linspace`,
which returns `[]` and `[start]` for those lengths. The kernel spaces samples
by dividing by `count - 1` and pins the last one to the stop value, so it
special-cases both lengths instead of dividing by zero or writing to index
`-1`. The values returned match numpy's exactly; `tests/test_special_and_util.py`
pins it.

## Parity testing

`pixi run test` runs 392 tests against real upstream `poliastro` 0.17.0,
installed in the environment and imported from the same interpreter. They
assert numerical parity, not merely that the call returns, and the run ends by
printing the worst error each of the 347 parity assertions reached — a
tolerance that passes with no headroom left is a tolerance that will not catch
the next change.

Two upstream constraints shape the tests. First, most of `poliastro.core` is
numba-typed for scalars, so upstream has to be looped to produce a reference
for a vectorised call; `tests/parity.py` does that. Second, a handful of
upstream functions cannot be executed at all (`mee2rv`) or not in nopython
mode (`rv_pqw` with an array argument, `rotation_matrix` with a scalar angle),
so those are compared against a transcription of upstream's source.

Beyond upstream, the tests assert properties that hold independently of it:
Kepler round trips (`M -> E -> M`, `M -> D -> M`), `nu_from_delta_t`
inverting `delta_t_from_nu` across the elliptic, parabolic and hyperbolic
regimes, Izzo's initial velocity propagating back to the target position,
the three propagators agreeing with each other, the perifocal frame being
orthonormal, and the Lagrange coefficients reproducing `farnocchia`'s
endpoint.

`tests/test_kernel_symbols.py` calls all 61 `@export`ed kernels *by name*,
against the raw C ABI, and checks each one's real output. The parity tests
above reach the kernels only through the Python API, which leaves a gap: an
export that is wired up and numerically wrong still ships green, because
nothing calls its name and nothing observes what it writes. Every symbol is
covered here, and `test_every_export_is_covered` fails if one is added later
without a case. The check is mechanical:

```bash
for s in $(grep -hoE '@export\("[A-Za-z_0-9]+"\)' src/*.mojo \
          | sed 's/@export("//;s/")//'); do
    grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"
done
```

It prints nothing. Those calls pass `workers=1`, which selects the serial
path, so each result is the arithmetic alone and does not depend on how many
threads the host offered.

`tests/test_coverage_claims.py` holds the README's coverage table to the
source: every export is present in the compiled `.so`, in `_SIGNATURES`, and
named by a test; every `_SIGNATURES` row matches the `@export` declaration it
describes, argument by argument; and every `fn("mp_x")(...)` call in the
package passes exactly the declared number of arguments. The signature check
matters because `_SIGNATURES` is transcribed by hand — an `Int` where the
declaration says `Float64` shifts every later register, and an address passed
through an unconfigured pointer is truncated to 32 bits by the C default, so
the failure is a segfault rather than a wrong answer.

`tests/test_batching.py` covers the batched dispatch layer, which the
per-function parity tests only ever call with a handful of elements:

- the SIMD body and its scalar tail, at every tail length and checked against
  the scalar kernel one element at a time, with `psi = 0`, `psi = +-1` and
  `x = 1` on the branch boundaries;
- both sides of both dispatch thresholds (4096 and 131072 rows, and either
  side of each), which must return the same numbers;
- the status buffer, including an element that fails inside a 6000-row
  parallel batch and must still raise.

## Benchmarks

`pixi run bench`, run under a machine-wide `flock`. Upstream is timed as a
Python loop over the same batch, because that is what a caller of
scalar-only upstream actually pays.

The environment pins `OPENBLAS_NUM_THREADS=8`. No kernel benchmarked here is
wide enough for BLAS threading to matter — upstream's hot path is a scalar
`@njit` function and ours is scalar or 3x3 — and OpenBLAS otherwise claims a
thread per core at import, which is enough on a busy host to starve the
compiler and the test runner of process ids. The row that touches BLAS at all
is `coe2rv`, and there the same 3x3 product is timed on both sides.

Measured on: Intel Xeon E5-2697 v4 @ 2.30 GHz, 36 physical cores (72 hardware
threads, `sched_getaffinity` 72), Linux 6.8.0-142-generic (glibc 2.39), Python
3.12.14, NumPy 1.26.4, Mojo 1.2.0.dev2026092905. The table below is one run of
`pixi run bench` on that machine; each row is the best of three timings.

| kernel | mojo-poliastro | upstream poliastro | speedup | note |
| --- | ---: | ---: | ---: | --- |
| nu_to_E (1M anomalies) | 12.14 ms | 1392.92 ms | 114.78x | upstream: Python loop |
| M_to_E Newton solve (1M) | 25.83 ms | 1429.59 ms | 55.35x | upstream: Python loop |
| E_to_M (1M) | 9.53 ms | 923.84 ms | 96.97x | upstream: Python loop |
| stumpff_c2 (1M) | 12.46 ms | 1193.61 ms | 95.79x | upstream: Python loop |
| hyp2f1b (200k) | 14.53 ms | 325.61 ms | 22.40x | upstream: Python loop |
| coe2rv (1M orbits) | 54.42 ms | 10706.08 ms | 196.75x | upstream: Python loop |
| coe2rv_many (200k orbits) | 17.60 ms | 928.29 ms | 52.73x | ours: parallel, upstream: loop |
| rv2coe (1M) | 23.61 ms | 8921.16 ms | 377.84x | upstream: Python loop |
| coe2mee (1M) | 21.14 ms | 3126.01 ms | 147.88x | upstream: Python loop |
| mee2rv (500k) | 10.14 ms | 13513.83 ms | 1333.23x | reference is upstream transcribed: upstream's own source is unrunnable |
| Izzo Lambert (200k transfers) | 17.90 ms | 855.05 ms | 47.76x | upstream: Python loop |
| Vallado Lambert (200k transfers) | 39.44 ms | 1369.77 ms | 34.73x | upstream: Python loop |
| farnocchia propagate (500k) | 49.99 ms | 8999.32 ms | 180.02x | upstream: Python loop |
| markley propagate (500k) | 42.29 ms | 6054.48 ms | 143.15x | upstream: Python loop |
| gooding propagate (500k) | 51.69 ms | 6107.88 ms | 118.17x | upstream: Python loop |
| delta_t_from_nu (1M) | 13.87 ms | 2632.50 ms | 189.75x | upstream: Python loop, delta passed explicitly |
| delta_t_from_nu (1M, upstream default arg) | 14.51 ms | 47216.94 ms | 3255.10x | numba default-arg dispatch dominates, not the kernel |
| **total (15 counted rows)** | **406.45 ms** | **54966.12 ms** | **135.23x** | |

**This box was busy.** `bench/bench.py` prints the load average next to the
table so a reader can see it; the run above started at a 1-minute load average
of 74-102 against 72 hardware threads across the runs, with other jobs on it.
Both columns are wall clock, so both are inflated, and the upstream column —
which is almost pure interpreter dispatch — inflates proportionally more than
the mojo column, which flatters the ratio. Eight runs of this same build across
the session moved the mojo total between roughly 405 ms and 831 ms while the
upstream total moved between roughly 53 s and 90 s. Read the ratios and the
ordering, not the absolute milliseconds, and treat a few tens of percent on
any single row as noise: `coe2rv`, `rv2coe`, `M_to_E` and `markley` land on
however many cores are free at that instant — `rv2coe` ranged over 23 ms to
58 ms and `markley` over 42 ms to 104 ms across runs of identical code. Re-run
`pixi run bench` on an idle machine for a clean-machine figure.

The upstream column is dominated by dispatch, not arithmetic, and that
matters for how you read the ratios. Upstream's kernels are scalar `@njit`
functions, so a batch has to be driven by a Python loop that pays about
45 us of Python-to-numba dispatch per element; the arithmetic behind each
call is a few nanoseconds. Almost all of the factor above is the cost of
that dispatch, not this port being better at the maths. A caller who wrote
their own `@njit` wrapper around upstream's kernels would amortise the
dispatch away and land far closer to the mojo column. Compare the absolute
times, not the ratios.

Read it honestly:

- The win is against a **Python loop**, not against a vectorised upstream.
  That is the real comparison — upstream cannot take the array — but it is a
  comparison against a scalar API, not against tuned BLAS. Nothing here
  should be read as "Mojo beats LAPACK".
- Most of the win is **parallelism and removing repeated work**, not vector
  width. The three changes are separable and were measured separately: the
  algebraic fixes in the kernels, the SIMD bodies, and the parallel dispatch.
  The parallel dispatch is the largest single contributor and the SIMD bodies
  are the smallest, which is the opposite of the usual expectation.
- The machine has 36 physical cores and `sched_getaffinity` reports 72, so the
  32-worker launch in the export layer is a deliberate choice, not an accident
  of `nproc` (which prints 8 here, under the cgroup's CPU quota). The batch is
  not perfectly balanced, so oversubscribing measured faster than matching the
  core count.
- The 3255x on the last row is numba's cost for a jitted function called with
  an omitted default argument, not the kernel. The same kernel with `delta`
  passed explicitly, one row above, is 189.75x. The row is shown to make that
  visible and is excluded from the total.
- `hyp2f1b` and `Vallado` are no longer the parity rows they were. Both were
  slow for a reason that had nothing to do with width: `hyp2f1b` was summing
  a series whose `(1 + ii)` factor cancelled, and Vallado's bisection was
  recomputing `c2(psi)` and `c3(psi)` up to five times for the same `psi`.
  Both are fixed below.

## What the GPU is worth here

`max.gpu.host` is present on this toolchain, so this is not a case of the API
being unavailable. It is a case of the GPU losing, and the reason is measured
on this machine (NVIDIA RTX 5090, driver 595.84) rather than assumed. A probe
was built and run against the driver; what it measured:

- **A `DeviceContext()` costs 239-245 ms to construct** (four runs: 237, 245,
  240, 239 ms), and it is the *first* one in a process that is expensive — a
  second construction in the same process measures 0.22-0.36 ms. Allocating a
  200k-element `Float64` device buffer costs 0.19-0.29 ms, so driver
  initialisation is the whole cost. Every entry point here is reached through
  an `abi("C")` `@export`, which may not be `raises` and which cannot hold a
  cached context: this dialect has no module-level mutable globals (the
  compiler rejects them outright), so a context could only be built once per
  process and would have to be rebuilt for each process that loads the
  library. The whole CPU Izzo kernel over 200k transfers — the most
  compute-dense per-element solve here, a Householder iteration with a
  data-dependent trip count — takes 17.90 ms in the table above, so that
  one-time cost is thirteen times the whole batch.
- **The transfers are cheap; the context is the whole argument.** A
  device-to-host copy of 200k `Float64` measures 0.055-0.097 ms and a
  device-to-device copy 0.022-0.041 ms. Whatever the device kernel itself
  costs, one round trip per `abi("C")` call is not what rules it out — the
  per-process context construction is.
- **The elementwise maps are memory-shaped, not compute-shaped.** They stream
  8 bytes in and 8 bytes out per element; even counting a transcendental at
  ~40 flops they sit around 1-3 flops per byte, well under the ~2 flops/byte
  where a device earns its keep. Measured on a lightly loaded box, `stumpff_c2`
  over 1M elements costs 11.07 ns an element and `E_to_M` 16.05 ns, so the
  entire kernel is bandwidth- and launch-bound rather than compute-bound.
- **The compute-dense kernels are the wrong shape for SIMT.** Izzo, Vallado and
  the three propagators are per-element Newton and bisection loops whose trip
  count depends on the input. A warp runs to the maximum trip count in its
  lanes, so the speedup is bounded by the spread of the iteration counts rather
  than by the arithmetic. The CPU path gets the same work with no divergence,
  because each thread runs its own element.
- **The hot path is transcendental, and libm is the bottleneck, not width.**
  What the kernels are limited by is the number of transcendental calls and
  memory traffic, and a per-lane scalar libm is precisely the workload a
  device handles worst: there is no cheap SIMD equivalent to a correctly
  rounded `sin`/`atan`, so moving the loop to the GPU trades the CPU's
  well-tuned libm for a much weaker approximation and pays the round trip for
  it.

So: no GPU path ships. A full GPU kernel was not benchmarked here — on this
toolchain a kernel using `block_idx`/`thread_idx` failed in the compiler's pass
manager during development — so the numbers above are the host-side and
transfer costs only. That is enough to rule the device out for this calling
convention, and it is the limit of what was actually measured.

## How it works

**FFI.** `src/capi.mojo` is the single compilation unit. It is the only
file that sees the C ABI; the kernels under `src/mojopoliastro/` are ordinary Mojo
that never mention a pointer. `@export("mp_name")` on a `def` with an explicit
`abi("C")` effect is the only way out. Every buffer crosses as an `Int` address
and is rebuilt as a `Pointer[Float64, AnyOrigin[mut=True]]` inside the entry
point, because `@export` rejects parametric functions and a pointer with an
inferred origin is parametric.

**Memory layout.** The caller owns all memory. Kernels allocate nothing, so
nothing can leak, and there is no allocator to keep warm. A call is: a
`Float64` array per argument, laid out row-major, plus a `Float64` output
array whose width the kernel knows. Vectors are interleaved `(x, y, z)`; a
batch of `n` state vectors is `3n` contiguous doubles with no padding, no
strides and no alignment requirement beyond 8 bytes. A `(n, 3)` numpy array
is bit-identical to what the kernel reads, so the Python layer hands
`arr.ctypes.data` straight through with no repacking. Every array crossing the
boundary is forced through `np.ascontiguousarray(..., dtype=np.float64)` and
kept alive by the local that names it, so the address stays valid for the whole
call; there is no dtype a caller can pass that reaches a kernel as anything but
`Float64`.

**Batching.** Upstream is one element per call; the export layer is a map
over the batch. That is the single place the C ABI forces a divergence, and
it is kept in `capi.mojo` rather than pushed into the kernels, so the kernels
read top-to-bottom like the upstream file they came from. The Python layer
broadcasts its arguments, hands over flat addresses, and unwraps the result
back to upstream's scalar-or-array calling convention.

**Parallelism.** Every batched export is a map over independent elements
with disjoint output slots, so above a row threshold the batch divides
cleanly across cores and goes through `max.algorithm.parallelize` with an
explicit worker count. The serial path calls the same worker one element at a
time, so the two sides run identical code.

`parallelize` builds its pool from the machine, not from the worker count, so
in a container the question is not "how many workers for this batch" but "is
there room in the cgroup for a pool at all" — and a launch that runs out of
process ids dies as a `pthread_create failed` abort inside LLVM, which no
caller can catch. Every export therefore takes the worker count as its last
argument, and `_lib.worker_count` decides it: `min(32, cores)` when the cgroup
has roughly two threads per core of headroom, and `1` otherwise. A count of 1
means *do not launch* — the export runs its own serial loop over the same
`work` item, so the numbers are identical either way and the only thing lost
is the cores. Set `MOJO_POLIASTRO_WORKERS` to pin it.

The threshold is not one number, because a parallel launch on this build costs
several milliseconds and the per-element cost here spans three orders of
magnitude. A solver that spends microseconds per element pays that back after
a few thousand rows and uses `PARALLEL_THRESHOLD = 4096`; the vectorised
elementwise maps run tens of nanoseconds per element — measured at 11.07 ns
for `stumpff_c2` on a lightly loaded box — which puts their own break-even near
10^5 rows, and they use `PARALLEL_THRESHOLD_CHEAP = 131072`. That split is
measurable in both directions, not only at 1M. On the same lightly loaded box,
a 131,072-element `E_to_M` call takes 2.35 ms on the serial path and 7.60 ms
once the same call is launched, while 2M elements take 56.66 ms serial
against 27.25 ms parallel. No benchmark row falls between the two thresholds,
and a 131,071-element call is not slow because of the threshold — it is the
same serial loop the 4095-element case runs.

**SIMD.** The genuinely elementwise maps — the seven anomaly conversions, the
parabolic ones, `stumpff_c2`, `stumpff_c3` and `hyp2f1b` — have a vector
body of `simd_width_of[DType.float64]()` lanes. Each work item owns exactly
`W` consecutive elements, so the vector body never has to reason about its
neighbour and only the last item can run short; that one falls through to the
scalar kernel, which is where the tail is handled. The iterative kernels
(`M_to_E`, `M_to_F`, the propagators, both Lambert solvers) keep the scalar
loop, because their trip count varies per element and there is no width to
recover.

The two Stumpff series and `hyp2f1b` are data-dependent, so their vector
bodies run the series until *no* lane is still changing; each lane then stops
adding exactly when it would have stopped alone and lands on the same value.
`tests/test_batching.py` pins this against the scalar kernel at every tail
length and across both dispatch thresholds.

There is no `select` on `SIMD` in this dialect and no bitcast to build one
out of, so the branch is blended arithmetically. That has a sharp edge: a
lane holding an `inf` or a `NaN` that the blend is supposed to discard still
poisons the result, because `inf - res` is `inf` and `inf * 0` is `NaN`. Each
discarded branch is therefore fed an argument that keeps it finite, and
`hyp2f1b`'s `x >= 1` case — which must return `+inf` — is produced by
dividing by a masked zero, since that lane's series input is zero and it
therefore holds exactly 1.0.

**What the SIMD does and does not buy, measured.** The vector bodies are
kept, but it is worth being precise about why, because the obvious
expectation is wrong on this target. A `SIMD` transcendental (`sin`, `cos`,
`atan`, ...) does not use a vector math library here: it de-vectorises to one
scalar libm call per lane. A branch-free arithmetic map (`a*x+b`) is
memory-bound rather than compute-bound and lands close to parity. So the
vector bodies are not a win on the transcendentals themselves; what they do
buy is the masked `any_true` early exit on the two Stumpff series and
`hyp2f1b`, and a slightly cheaper body where the extra lanes really are
independent. Since the batched path is the one a caller sees, the vector body
is kept, and the whole question is recorded here rather than as a claim.

The practical consequence is that the per-element cost here is set by the
number of transcendental calls and by memory traffic, not by vector width.
That is why the optimisation that survives measurement is not a wider vector
but the removal of duplicated transcendentals: `markley_coe` formed its
`f0..f3` Kepler terms from a `sin(E)` and a `cos(E)` that were each evaluated
twice, and now evaluates each once. It is the same floating point operations
in the same order, so the values are unchanged, and
`tests/test_iod_and_propagation.py`
(`test_markley_coe_keeps_the_shared_sin_and_cos`) pins the hoisted form against
the literal double-evaluation one.

**Load order.** `mojo_poliastro/__init__.py` loads the shared library
eagerly. Upstream poliastro JIT-compiles through numba, and numba's LLVM
runtime and the Mojo runtime cannot both be initialised into one process in
the other order. `tests/conftest.py` imports `mojo_poliastro` before any
upstream import, which fixes the order for the whole session.

**Toolchain notes.** `MOJO_NOTES.md` is the list of dialect traps this port
hit — `fn` removed, `UnsafePointer` deprecated, `std.algorithm` gutted in
favour of `max.algorithm`, `@export` now demanding an explicit `abi()`, and so
on. It is checked against the pinned compiler rather than against docs.

## Layout

src/capi.mojo                   the C ABI: 61 exported entry points
src/mojopoliastro/*.mojo        the kernels, one file per upstream module
python/mojo_poliastro/         the package: ctypes glue + the Python API
build/build.sh                 mojo build --emit shared-lib
tests/                       392 tests: parity, per-symbol, batched, coverage
bench/bench.py               benchmarks vs upstream, prints a markdown table

## Licence

MIT, same as upstream poliastro. See `LICENSE`.
