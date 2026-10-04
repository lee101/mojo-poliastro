# Mojo dialect notes

Toolchain these notes describe: `mojo` and `max` at the versions pinned in
`pixi.toml`. These are the traps this port actually hit, in the order it hit
them. Prose is kept to what a reader of `src/` would otherwise get wrong.

## Two things that break everything

- **`MODULAR_HOME` must point at the env's `share/max`.** Without it every
  import fails with `unable to locate module 'std'`, which reads like a missing
  package but is not. `pixi run` sets it. A bare `mojo` on `PATH` does not:
  ```bash
  export MODULAR_HOME="$(dirname "$(dirname "$(which mojo)")")/share/max"
  ```
  `python/mojo_poliastro/_lib.py::_build_env` derives it when unset.
- **`mojo build -o` takes an output FILE path**, not a directory. A directory
  there fails late, at link time.

## Language

- `fn` has been removed. Use `def` for everything, including a function
  *value*'s type: `fn () -> None` is a hard error, not a warning.
- `UnsafePointer` is deprecated in favour of `Pointer`. It still compiles, with
  a warning, so old code builds -- but write
  `Pointer[T, AnyOrigin[mut=True]]`. Note that `Pointer[Float64]` with an
  inferred origin has no `.allocate`.
- `int(x)` / `float(x)` are **not builtins**. Use the type as a constructor.
- `simdwidthof[DType.float64]()` is gone. It is now
  `from std.sys import simd_width_of` then `simd_width_of[DType.float64]()`.
  A hardcoded `comptime W = 4` is always valid.
- `SIMD` has no `.min()` / `.max()`. The free `min(a, b)` / `max(a, b)` work
  on both scalars and `SIMD`.
- Integer-only literal in a float expression does not promote: `1 / 2` is
  integer division, so `sqrt(2 / (1 + ecc))` needs `2.0`. This bit
  `delta_t_from_nu` and `D_to_M_near_parabolic` during the port.
- `struct` methods are move-only in this dialect: `return <local struct>` does
  not parse, so every function upstream returns as an array takes a leading
  `out` argument instead.
- **A `def` cannot be passed as a value.** Upstream's `newton_factory` returns a
  closure; here the two instantiations are written out with the loop body
  unchanged.

## Export / FFI

- `@export("symbol_name")` sits on the line above the def. The ABI is an
  *effect* before the arrow: `def f(a: Int) abi("C") -> Float64:`.
- **`@export` now REQUIRES an explicit `abi()`.** Omitting it is an error:
  `@export requires an explicit 'abi()' effect on the function`.
- An `abi("C")` function **may not be `raises`**.
- `@export` rejects parametric functions, including an inferred pointer origin.
  Buffers therefore cross as `Int` addresses and are rebuilt inside:
  ```mojo
  var q = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=addr)
  ```
- Pointers are NON-NULLABLE: constructing one from address 0 fails a
  compile-time constraint. Take `Int` and construct inside the branch.
- **`@export` lives on the wrapper, not the kernel.** The kernel files under
  `src/mojopoliastro/` are ordinary Mojo that never mention a pointer; the
  batched `def work(i: Int) {imm}:` wrappers and their `@export` are all in
  `src/capi.mojo`. `tests/test_coverage_claims.py` pins that split, because a
  second copy of the sources anywhere else is a kernel that gets edited in one
  place and compiled from the other.

## Memory and SIMD

```mojo
comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()

var i = 0
while i + W <= n:                      # vector body
    p.store(i, p.load[width=W](i) * 2.0)
    i += W
while i < n:                           # scalar tail
    p.store(i, p.unsafe_load(i) * 2.0)
    i += 1
```

- `p.load[width=W](i)`, `p.store(i, v)` and `v.reduce_add()` all work.
- `p.unsafe_load(i)` for a scalar load; positional `p[i]` still compiles but
  warns, so prefer `unsafe_load`.
- `unsafe_load[width=W](i)` and `unsafe_store(i, v)` are the forms this port
  uses, so the load and the store of a vector are always the same width and the
  compiler cannot silently split one into two.
- There is no `select` on `SIMD` in this dialect and no bitcast to build one
  out of, so a branch is blended arithmetically -- which means a discarded
  branch holding an `inf` or a `NaN` still poisons the result (`inf - res` is
  `inf`, `inf * 0` is `NaN`). Feed the discarded branch a finite argument.

## Parallelism -- moved from `std` to `max`

| import | result |
| --- | --- |
| `from max.algorithm import parallelize` | works; this is the live path |
| `from max.algorithm import sync_parallelize` | works |
| `from std.algorithm import parallelize` | `package 'algorithm' does not contain 'parallelize'` |
| `from std.algorithm.functional import parallelize` | gone |
| `from std.sys import parallelize` | gone |
| `from std.threading import ...` | `unable to locate module 'threading'` |
| `from std.algorithm import sort` | `package 'algorithm' does not contain 'sort'` |

`std.algorithm` no longer exporting `sort` is the tell: the module was gutted
while the capability moved to `max`. `max.parallel` and `max.core` do not
exist; `max.algorithm` is the one.

**The `max` conda package is required.** `max.mojoc` / `algorithm.mojoc` ship
in `<prefix>/lib/mojo` from the `max` package, not from `mojo`. Without it the
`max.*` imports fail with `unable to locate module 'max'`. It is released in
lockstep, so `mojo ==X` goes with `max ==matching-version`.

- `parallelize(func, num_work_items, num_workers[, ctx])` works.
  `parallelize[func](n)` does not type-check; the `[func]` form now binds the
  origins overload.
- **The worker must be a plain `def`, not a `@parameter` closure** -- `func`
  has to convert to `def(Int) -> None`, and a `@parameter def` is rejected as
  `capturing thin`. Use a nested `def work(i: Int) {imm}:`. Nested `{imm}`
  calling nested `{imm}` also works.
- **Pass an explicit worker count.** The 2-arg form crashed worker launch when
  called from a ctypes host, and `sync_parallelize` segfaulted from a ctypes
  host entirely. Gate the launch behind a size threshold: on this build a
  `parallelize` launch on a shared box costs about 10 ms, against about 3 ms
  for the same work done serially.
- One trap worth naming: `max`'s caching allocator can request a very large
  chunk (a batched GPU SVD asking for 16.75 GiB gets `CUDA_ERROR_OUT_OF_MEMORY`)
  and a broad `except` will swallow that into a silent CPU fallback. If a GPU
  path "works" but never reports `used_gpu=1`, check rather than assume.

## GPU -- the host API moved from `std.gpu` to `max.gpu`

`from std.gpu import thread_idx` no longer resolves; the capability is in
`max`:

```mojo
from max.gpu import block_idx, thread_idx      # works
from max.gpu.host import DeviceContext         # works
```

The full host surface compiles: `ctx.enqueue_create_buffer[DType.float64](n)`,
`ctx.enqueue_copy(dst, src)`, `ctx.enqueue_function[kernel](...)`,
`ctx.synchronize()`, `ctx.name()`. Three details bite:

- **`DeviceContext` is imported from `max.gpu.host`, not `max.gpu`.** From
  `max.gpu` the error is `package 'gpu' does not contain 'DeviceContext'`.
- Construction needs a `raises` context (`def main() raises:`), and the value
  is not `Writable`, so it cannot be `print`ed. There is no `ctx.devices()`.
- **`std.time` has `perf_counter_ns` and `monotonic`, not `monotonic_ns`.**
  `from time import perf_counter` fails with `unable to locate module 'time'`;
  the spellings are `from std.time import perf_counter_ns`.

`DeviceBuffer` is not subscriptable (`no attribute __getitem__`), and it has no
`enqueue_copy` method -- the copy goes the other way, from a buffer the
`DeviceContext` created:
`host_buffer.enqueue_copy(device_buffer)`.

Practical rule: a GPU path is only worth shipping if you can compile AND run it.
A CPU-only port is a legitimate answer when a kernel is below roughly 2 flops
per byte, but "the host API is missing" is NOT a valid reason -- it is in
`max`. For this port the measured reason is in the README: context
construction costs hundreds of ms and there is no way to cache one across an
`abi("C")` export, which has no module-level mutable globals in this dialect.

## Build

- Build cost is a fixed few seconds regardless of function count. Batch many
  functions into ONE compilation unit.
- `mojo build --emit shared-lib` errors if the file defines `main`.
- `-j` should be capped rather than left at "every core": the compiler's thread
  pool is sized from the machine, and on a busy host the spawn fails outright
  with `LLVM ERROR: pthread_create failed`.

## Diagnosing a moved symbol

Do not guess module paths. A bare unknown name makes the compiler name the
module it expected, which is the whole diagnosis:

```
error: use of unknown declaration 'perf_counter_ns'; did you mean to import it
from 'std.time'? Add 'from std.time import perf_counter_ns'
```

No hint means the symbol is gone, not relocated.
