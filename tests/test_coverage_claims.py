"""The README's coverage claims, checked against the source that is shipped.

A README that says "61 exported entry points" or "every `@export` has a test"
is a claim about two things it does not contain: the `@export` list in
`src/capi.mojo` and the symbols named in `tests/`.  Both can drift silently,
and both have: this file failed to exist while the counts it now checks were
being edited by hand.

So the numbers in the README are not asserted here.  What is asserted is the
property those numbers are supposed to describe, so that a future export or a
deleted case shows up as a failing test rather than as a stale sentence.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

MOJO_SOURCES = sorted(
    p for p in (ROOT / "src").rglob("*.mojo") if "__pycache__" not in p.parts
)

EXPORTS = [
    m.group(1)
    for src in MOJO_SOURCES
    for m in re.finditer(r'@export\("([A-Za-z_0-9]+)"\)', src.read_text())
]

TEST_SOURCES = sorted((ROOT / "tests").glob("test_*.py"))
TESTS_TEXT = "\n".join(p.read_text() for p in TEST_SOURCES)


def test_the_source_tree_is_where_the_build_script_looks_for_it():
    """`build/build.sh` and `_lib.SRC` both resolve to `src/`, not `python/`.

    An earlier layout kept a copy of the Mojo sources under the package
    directory.  That copy was not what got compiled, so a kernel could be
    edited in the copy and nothing would notice.  This pins the single
    source of truth so the duplicate cannot quietly come back.
    """
    assert (ROOT / "src" / "capi.mojo").is_file()
    assert not (ROOT / "python" / "mojo_poliastro" / "mojo_src").exists()
    assert not (ROOT / "python" / "mojo_poliastro" / "src").exists()


def test_build_script_and_loader_compile_the_same_file():
    """`build/build.sh` and the lazy import-time builder name one unit."""
    from mojo_poliastro import _lib

    build = (ROOT / "build" / "build.sh").read_text()
    assert "src/capi.mojo" in build
    assert _lib.SRC == str(ROOT / "src")
    assert _lib.SRC + "/capi.mojo" == str(ROOT / "src" / "capi.mojo")


@pytest.mark.parametrize("name", EXPORTS)
def test_every_export_is_named_in_the_test_suite(name):
    """The README's kernel-coverage claim, per symbol.

    This is the harness check written out so it fails the suite rather than
    the shell:

        for s in $(grep -hoE '@export\\("[A-Za-z_0-9]+"\\)' src/*.mojo \\
                  | sed 's/@export("//;s/")//'); do
            grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"
        done
    """
    assert name in TESTS_TEXT, (
        f"{name} is exported from src/ but no test names it; the parity tests "
        "reach it only through the public API, which is exactly the gap this "
        "closes"
    )


def test_every_export_is_reachable_from_the_ffi_signature_table():
    """An export the Python layer cannot call is dead code that still ships.

    `_SIGNATURES` is what `ctypes` configures, so a symbol missing from it
    would be called with C default `int` arguments, truncating every 64-bit
    buffer address.
    """
    from mojo_poliastro._lib import _SIGNATURES

    missing = sorted(set(EXPORTS) - set(_SIGNATURES))
    assert not missing, f"exported but not in _SIGNATURES: {missing}"


def test_every_exported_symbol_exists_in_the_compiled_library():
    """Names in the source and symbols in the `.so` must agree.

    A stale `dist/` is the failure this catches: the source grew an export,
    the library on disk did not, and the suite would otherwise only find out
    if some test happened to call it.
    """
    from mojo_poliastro._lib import lib

    handle = lib()
    missing = sorted(n for n in EXPORTS if not hasattr(handle, n))
    assert not missing, f"exported in source but absent from the .so: {missing}"


DECL = re.compile(
    r'@export\("(?P<name>[A-Za-z_0-9]+)"\)\s*\ndef ', re.S
)


def _declared_signatures(text):
    """(name, ['F'|'I', ...], returns_int) for every `@export` in `text`.

    Taken from the declaration itself rather than from the `def work(i)`
    body, because the body's argument list is the *kernel's*, not the
    ABI's: `mp_vallado_uni` takes addresses and forwards `V3(...)` values to
    a kernel that takes `V3`.
    """
    out = {}
    for m in DECL.finditer(text):
        name = m.group("name")
        decl = text[m.end():]
        params = re.search(r"\((.*?)\)\s*abi\(\"C\"\)", decl, re.S).group(1)
        # split on top-level commas only
        depth, cur, parts = 0, "", []
        for ch in params:
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
                continue
            depth += (ch in "([" ) - (ch in ")]")
            cur += ch
        if cur.strip():
            parts.append(cur)
        kinds = []
        for p in (p.strip() for p in parts):
            if ": Float64" in p:
                kinds.append("F")
            elif ": Int" in p:
                kinds.append("I")
            else:
                pytest.fail(f"{name}: unrecognised parameter {p!r}")
        # `decl` runs to the next `@export`, so anchor the return type to
        # the *first* `):` that closes the parameter list.  Matching any
        # later `) -> Int:` would find the kernel's own signature inside
        # the body and report every export as returning a status.
        returns_int = re.search(
            r"\)\s*abi\(\"C\"\)(?P<ret>\s*->\s*Int)?\s*:", decl
        ).group("ret") is not None
        out[name] = (kinds, returns_int)
    return out


def test_the_ffi_signature_table_matches_every_mojo_declaration():
    """`ctypes` argtypes are the FFI contract; a wrong one segfaults.

    `_SIGNATURES` is transcribed by hand, and a hand-transcribed table with
    61 rows will eventually disagree with the source.  The failure is not a
    wrong answer: an `Int` where the declaration says `Float64` shifts every
    later register, and an address passed as an unconfigured argument is
    truncated to 32 bits by the C default.  Both land as a segfault, and
    only for the calls that happen to be long enough to reach the bad
    argument.

    The trailing `Int` on every row is the worker count that `_lib` appends
    and the declaration ends with, so the two are compared with it included.
    """
    import ctypes

    from mojo_poliastro._lib import _SIGNATURES

    declared = _declared_signatures((ROOT / "src" / "capi.mojo").read_text())
    assert set(declared) == set(EXPORTS)

    problems = []
    for name, (kinds, returns_int) in declared.items():
        argtypes, restype = _SIGNATURES[name]
        want = ["F" if k is ctypes.c_double else "I" for k in argtypes]
        if want != kinds:
            problems.append(f"{name}: mojo {kinds} vs _SIGNATURES {want}")
        if (restype is not None) != returns_int:
            problems.append(
                f"{name}: mojo returns_int={returns_int} vs restype={restype}"
            )
    assert not problems, "Ffi signature mismatches:\n  " + "\n  ".join(problems)


def test_every_ffi_call_passes_exactly_the_declared_argument_count():
    """Guard the `fn()` wrapper against dropping or duplicating an argument.

    `_lib.fn` appends the worker count itself, so a caller that passes the
    worker count too would shift every address by one register.  Counting
    the positional arguments of each `fn("mp_x")(...)` call in the package
    is enough to catch that, and it needs no execution.
    """
    call_re = re.compile(
        r'fn\(\s*"(?P<name>[A-Za-z_0-9]+)"\s*\)', re.S
    )
    package = sorted(
        p for p in (ROOT / "python").rglob("*.py") if "__pycache__" not in p.parts
    )
    problems = []
    splat_calls = []
    for path in package:
        text = path.read_text()
        for m in call_re.finditer(text):
            name = m.group("name")
            if name not in EXPORTS:
                continue
            start = m.end()
            want = len(_declared_signatures(
                (ROOT / "src" / "capi.mojo").read_text()
            )[name][0]) - 1  # minus the worker count `fn` appends
            rest = text[start:]
            open_i = rest.find("(")
            if open_i < 0:
                problems.append(f"{path.name}: fn({name!r}) is not called")
                continue
            depth, cur, args, seen = 1, "", [], False
            for ch in rest[open_i + 1:]:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
                if ch == "," and depth == 1:
                    args.append(cur)
                    cur = ""
                    seen = True
                    continue
                cur += ch
            if cur.strip() or seen:
                args.append(cur)
            # `*(gen for ...)` splats an unknown number of arguments at the
            # C level, so its width cannot be counted from the source.  The
            # one call site that does it (the ten Jacchia profile buffers) is
            # pinned by count below instead.
            splat = any("*" in a for a in args)
            if splat:
                splat_calls.append(f"{path.name}: fn({name!r})")
                continue
            if len(args) != want:
                problems.append(
                    f"{path.name}: fn({name!r}) passes {len(args)} args, "
                    f"declaration wants {want}"
                )
    assert not problems, "\n  ".join(problems)

    # Every splatted call is the ten-buffer Jacchia profile, and its width is
    # checked for real by `test_kernel_symbols.py`, which calls the symbol by
    # name.  Pin the count here so a second splat is a deliberate decision.
    assert splat_calls == [
        "jacchia.py: fn('mp_jacchia_altitude_profile')"
    ], f"unexpected splatted ffi calls: {splat_calls}"


@pytest.mark.parametrize(
    "module,exports",
    [
        # one entry per upstream module the coverage table names, so a module
        # silently dropped from the port cannot leave the table claiming it
        ("mojopoliastro.linalg", {"mp_norm"}),
        ("mojopoliastro.special", {"mp_hyp2f1b", "mp_stumpff_c2", "mp_stumpff_c3"}),
        (
            "mojopoliastro.angles",
            {
                "mp_D_to_nu", "mp_nu_to_D", "mp_nu_to_E", "mp_nu_to_F", "mp_E_to_nu",
                "mp_F_to_nu", "mp_M_to_E", "mp_M_to_F", "mp_M_to_D", "mp_E_to_M",
                "mp_F_to_M", "mp_D_to_M", "mp_fp_angle",
            },
        ),
        (
            "mojopoliastro.util",
            {
                "mp_rotation_matrix", "mp_alinspace", "mp_spherical_to_cartesian",
                "mp_planetocentric_to_AltAz",
            },
        ),
        (
            "mojopoliastro.elements",
            {
                "mp_eccentricity_vector", "mp_circular_velocity", "mp_rv_pqw",
                "mp_coe_rotation_matrix", "mp_coe2rv", "mp_coe2rv_many",
                "mp_coe2mee", "mp_rv2coe", "mp_mee2coe", "mp_mee2rv",
            },
        ),
        ("mojopoliastro.iod", {"mp_vallado_iod", "mp_izzo"}),
        ("mojopoliastro.propagation", {"mp_func_twobody", "mp_farnocchia_rv",
                                       "mp_farnocchia_coe", "mp_vallado_uni",
                                       "mp_markley", "mp_markley_coe",
                                       "mp_gooding", "mp_gooding_coe", "mp_S_x",
                                       "mp_dS_x_alt", "mp_d2S_x_alt",
                                       "mp_D_to_M_near_parabolic",
                                       "mp_M_to_D_near_parabolic",
                                       "mp_delta_t_from_nu", "mp_nu_from_delta_t"}),
        (
            "mojopoliastro.spheroid_location",
            {
                "mp_cartesian_cords", "mp_f", "mp_N", "mp_tangential_vecs",
                "mp_radius_of_curvature", "mp_distance", "mp_is_visible",
                "mp_cartesian_to_ellipsoidal",
            },
        ),
        (
            "mojopoliastro.earth_atmosphere",
            {
                "mp_geometric_to_geopotential", "mp_geopotential_to_geometric",
                "mp_gravity", "mp_check_altitude", "mp_jacchia_altitude_profile",
            },
        ),
    ],
)
def test_the_kernel_files_exist_and_carry_their_exports(module, exports):
    """Each upstream module the coverage table lists is really ported.

    The table is the README's whole coverage argument, and a file that
    quietly stopped exporting would leave the row true of nothing.
    """
    path = ROOT / "src" / f"{module.replace('.', '/')}.mojo"
    assert path.is_file(), f"{module} is in the coverage table but {path} is gone"
    # The `@export` decorator and the ABI-layer wrapper both live in
    # `src/capi.mojo`; the kernel file holds the kernel itself.  So the two
    # properties worth checking are that every entry point named in the
    # coverage table is defined *in this file* under its kernel name, and
    # that `capi.mojo` imports it.  `mp_` is an ABI-layer prefix; four
    # kernels are imported under a different local name, listed in
    # capi.mojo's import block, so they are named explicitly.
    text = path.read_text()
    defined = {m.group(1) for m in re.finditer(r"^def ([A-Za-z_0-9]+)\(", text, re.M)}
    renamed = {
        # entry point symbol -> (kernel name, name capi.mojo imports it under)
        "mp_N": ("N", "normal_at"),
        "mp_f": ("f", "flattening"),
        "mp_distance": ("distance", "sph_distance"),
        "mp_check_altitude": ("_check_altitude", "_check_altitude"),
        "mp_jacchia_altitude_profile": ("_altitude_profile", "_altitude_profile"),
        "mp_vallado_uni": ("vallado", "vallado_uni"),
        # `iod.vallado` and `iod.izzo` are imported without an alias
        "mp_vallado_iod": ("vallado", "vallado"),
        "mp_izzo": ("izzo", "izzo"),
        # `coe2rv_many` is upstream's batched variant of `coe2rv`; the port
        # reuses `coe2rv` with a per-element `k` rather than a second kernel
        "mp_coe2rv_many": ("coe2rv", "coe2rv"),
    }

    def kernel_of(symbol):
        return renamed.get(symbol, (symbol[3:] if symbol.startswith("mp_") else
                                    symbol, None))[0]

    def imported_as(symbol):
        hit = renamed.get(symbol)
        return hit[1] if hit else kernel_of(symbol)

    missing_def = sorted(
        w for w in {kernel_of(n) for n in exports} if w not in defined
    )
    assert not missing_def, (
        f"{module}: kernels named in the coverage table but not defined in "
        f"the file: {missing_def}"
    )

    capi = (ROOT / "src" / "capi.mojo").read_text()
    imports = set()
    # `capi.mojo` also imports a second kernel module under an alias for the
    # `from X import a as b` form, so walk every `from mojopoliastro...`
    # statement rather than only the parenthesised ones.
    for m in re.finditer(
        r"^from mojopoliastro[\w.]* import (?:\((?P<b>.*?)\)|(?P<s>[^\(\n]+?))\s*$",
        capi, re.S | re.M,
    ):
        block = m.group("b") if m.group("b") is not None else m.group("s")
        for part in block.replace("\n", " ").split(","):
            part = part.strip()
            if part:
                # `x as y`: capi.mojo binds the local name `y`
                imports.add(part.split(" as ")[-1].strip())
    missing_import = sorted(
        w for w in {imported_as(n) for n in exports} if w not in imports
    )
    assert not missing_import, (
        f"{module}: kernels named in the coverage table that capi.mojo does "
        f"not import: {missing_import}"
    )


def test_the_claimed_gaps_are_actually_absent():
    """The propagators the README says are *not* ported must not be here.

    A coverage table is only useful if the "no" column is enforced too:
    claiming a gap that has since been closed would be a stale README, and
    silently shipping a port of something documented as missing is the same
    defect in the other direction.
    """
    # strip comments and docstrings: these names appear in prose explaining
    # why they are absent, and a docstring is not a port
    everything = []
    for p in MOJO_SOURCES:
        code = re.sub(r'""".*?"""', "", p.read_text(), flags=re.S)
        everything.append(re.sub(r"#.*", "", code))
    everything = "\n".join(everything)
    for absent in ("cowell", "danby", "mikkola", "pimienta", "recseries"):
        assert absent not in everything, (
            f"{absent} is documented as not ported but appears in src/"
        )


def test_the_one_ported_but_unexported_helper_is_the_one_that_is_claimed():
    """`_get_index` is the documented exception: ported, never exported.

    The README says exactly one kernel has no Python path and names it.  If
    that stops being true -- because it got exported, or because a second
    kernel quietly became unreachable -- the sentence has to be wrong and this
    is where that shows up.

    "Not exported" is checked against `capi.mojo` *and* against `_SIGNATURES`,
    because a name absent from the table is called with C default arguments
    even when it is present in the `.so`.
    """
    from mojo_poliastro._lib import _SIGNATURES

    capi = (ROOT / "src" / "capi.mojo").read_text()
    assert "_get_index" not in capi, (
        "_get_index is documented as ported-but-unexported; if it now has a "
        "Python path, update the coverage section"
    )
    assert not any("get_index" in n for n in _SIGNATURES), (
        "_get_index has an FFI signature but no export; either export it and "
        "say so, or drop the signature row"
    )
    assert "_get_index" in (ROOT / "src" / "mojopoliastro" /
                            "earth_atmosphere.mojo").read_text(), (
        "_get_index is documented as ported; if it was deleted, update the "
        "coverage section"
    )


def test_the_readme_numbers_match_the_source():
    """The counts the README states, counted here rather than trusted.

    A README that says "61 exported entry points" or "391 tests" is a claim
    about the tree.  If it drifts, the sentence is the thing that is wrong, so
    the numbers are compared rather than asserted: a mismatch here is the
    signal to fix the README, and it cannot be satisfied by editing the README
    alone because the same numbers come from the source.
    """
    readme = (ROOT / "README.md").read_text()

    n_exports = len(EXPORTS)
    assert f"{n_exports} exported entry points" in readme, (
        f"README does not say '{n_exports} exported entry points'; it now has "
        f"{n_exports}. Fix the coverage section."
    )
    assert f"all {n_exports} `@export`ed kernels" in readme, (
        f"README's per-symbol claim does not say {n_exports}"
    )

    n_tests = _count_tests()
    assert f"runs {n_tests} tests" in readme, (
        f"README does not say 'runs {n_tests} tests'; it now has {n_tests}."
    )


def _count_tests() -> int:
    """How many test cases the suite collects, counted without running it."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    m = re.search(r"(\d+) tests? collected", out.stdout)
    if m:
        return int(m.group(1))
    # `pixi run test` output form
    m = re.search(r"(\d+) tests collected", out.stdout)
    assert m, f"could not count tests:\n{out.stdout[-2000:]}"
    return int(m.group(1))
