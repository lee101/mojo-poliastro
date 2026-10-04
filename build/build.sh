#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# `src/capi.mojo` is the single compilation unit; `-I src` resolves the
# `mojopoliastro.*` module imports that hold the kernels, one file per
# upstream module.
#
# `-j` is capped rather than left at "every core": the compiler's thread pool
# is sized from the machine, and on a busy host the spawn fails outright with
# `LLVM ERROR: pthread_create failed`. Compilation is a fixed few seconds, so
# a small pool costs nothing and makes the build survive a loaded machine.
# Override with MOJO_BUILD_JOBS.
mojo build --emit shared-lib -j "${MOJO_BUILD_JOBS:-4}" -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-poliastro.so"
