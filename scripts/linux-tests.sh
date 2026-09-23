#!/usr/bin/env bash
# Builds brass for Linux and runs its tests there: natively on x86_64, and
# for aarch64 under qemu-user so the AArch64 JIT is executed and compared
# against the interpreter exactly as the x64 JIT is.
#
# From Windows (WSL with Debian):   wsl -e bash scripts/linux-tests.sh [cmd...]
# From Linux:                       scripts/linux-tests.sh [cmd...]
#
# Commands (default: "all"):
#   setup          fetch qemu-user-static and the aarch64 cross toolchain
#                  without root (apt-get download + dpkg -x) into $BRASS_XROOT
#   build-a64      configure + build the aarch64 tree ($BRASS_A64_BUILD)
#   test-a64       ctest -L correctness on the aarch64 tree, under qemu
#   fuzz-a64       differential fuzzer (bronze + all pipelines) under qemu
#   build-x64      configure + build the native Linux x86_64 tree
#   test-x64       ctest -L correctness on the x86_64 tree
#   fuzz-x64       differential fuzzer natively on Linux x86_64
#   a64            build-a64 test-a64 fuzz-a64
#   x64            build-x64 test-x64 fuzz-x64
#   all            setup a64 x64
#
# Environment:
#   BRASS_SRC           source tree to build  (default: this checkout)
#   BRASS_XROOT        toolchain root        (default ~/.cache/brass-aarch64)
#   BRASS_A64_BUILD     aarch64 build dir     (default ~/brass-build/aarch64)
#   BRASS_X64_BUILD     x86_64 build dir      (default ~/brass-build/x86_64)
#   FUZZ_SEEDS          programs per pipeline (default 2000)
#   FUZZ_FIRST_SEED     first seed            (default 1)
#   CTEST_JOBS          ctest parallelism     (default: nproc)
#   CTEST_ARGS          extra ctest arguments (e.g. "-R aarch64")
set -euo pipefail

SRC="${BRASS_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
XROOT="${BRASS_XROOT:-$HOME/.cache/brass-aarch64}"
A64_BUILD="${BRASS_A64_BUILD:-$HOME/brass-build/aarch64}"
X64_BUILD="${BRASS_X64_BUILD:-$HOME/brass-build/x86_64}"
FUZZ_SEEDS="${FUZZ_SEEDS:-2000}"
FUZZ_FIRST_SEED="${FUZZ_FIRST_SEED:-1}"
JOBS="${CTEST_JOBS:-$(nproc)}"

# Everything the cross compiler and qemu need that a stock Debian bookworm
# x86_64 install does not already carry.
PACKAGES=(
    qemu-user-static
    binutils-aarch64-linux-gnu
    cpp-12-aarch64-linux-gnu
    gcc-12-aarch64-linux-gnu
    gcc-12-aarch64-linux-gnu-base
    g++-12-aarch64-linux-gnu
    gcc-12-cross-base
    libc6-arm64-cross
    libc6-dev-arm64-cross
    linux-libc-dev-arm64-cross
    libgcc-12-dev-arm64-cross
    libgcc-s1-arm64-cross
    libstdc++-12-dev-arm64-cross
    libstdc++6-arm64-cross
    libatomic1-arm64-cross
    libgomp1-arm64-cross
    libitm1-arm64-cross
    libasan8-arm64-cross
    liblsan0-arm64-cross
    libtsan2-arm64-cross
    libubsan1-arm64-cross
    libhwasan0-arm64-cross
)

log() { printf '\n== %s\n' "$*"; }

# The extracted cross binutils load their libbfd/libopcodes from here.
export LD_LIBRARY_PATH="$XROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BRASS_XROOT="$XROOT"

cmd_setup() {
    if [[ -x "$XROOT/usr/bin/qemu-aarch64-static" && -x "$XROOT/usr/bin/aarch64-linux-gnu-g++-12" ]]; then
        log "toolchain present in $XROOT"
        return
    fi
    log "fetching ${#PACKAGES[@]} packages into $XROOT"
    mkdir -p "$XROOT/debs"
    (cd "$XROOT/debs" && apt-get download "${PACKAGES[@]}")
    for deb in "$XROOT"/debs/*.deb; do
        dpkg -x "$deb" "$XROOT"
    done
    # The toolchain's unversioned names are symlinks in other packages.
    local b
    for b in gcc g++ cpp; do
        ln -sf "aarch64-linux-gnu-$b-12" "$XROOT/usr/bin/aarch64-linux-gnu-$b"
    done
    "$XROOT/usr/bin/aarch64-linux-gnu-g++" --version | head -1
    "$XROOT/usr/bin/qemu-aarch64-static" --version | head -1
}

configure() { # <build-dir> <extra cmake args...>
    local dir="$1"; shift
    cmake -S "$SRC" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release "$@"
}

cmd_build_a64() {
    cmd_setup
    log "building aarch64 ($A64_BUILD)"
    configure "$A64_BUILD" \
        -DCMAKE_TOOLCHAIN_FILE="$SRC/cmake/toolchains/aarch64-linux-gnu.cmake" \
        -DBRASS_XROOT="$XROOT"
    cmake --build "$A64_BUILD" --target brass_unit_tests brass-fuzz -j "$(nproc)"
}

cmd_build_x64() {
    log "building x86_64 ($X64_BUILD)"
    configure "$X64_BUILD"
    cmake --build "$X64_BUILD" --target brass_unit_tests brass-fuzz -j "$(nproc)"
}

run_ctest() { # <build-dir>
    # shellcheck disable=SC2086
    (cd "$1" && ctest -L correctness -j "$JOBS" --output-on-failure ${CTEST_ARGS:-})
}

cmd_test_a64() { log "ctest aarch64 under qemu"; run_ctest "$A64_BUILD"; }
cmd_test_x64() { log "ctest x86_64"; run_ctest "$X64_BUILD"; }

sum_lines() { awk '{ s += $1 } END { print s + 0 }'; }

# Runs both optimizer pipelines, each split over FUZZ_JOBS processes by seed
# range. Logs and reproducers go to <build>/fuzz/<pipeline>-<job>/.
run_fuzz() { # <build-dir> <runner prefix...>
    local dir="$1"; shift
    local jobs="${FUZZ_JOBS:-$(( $(nproc) > 2 ? $(nproc) / 2 : 1 ))}"
    local per=$(( (FUZZ_SEEDS + jobs - 1) / jobs ))
    local rc=0 pipeline j
    rm -rf "$dir/fuzz"
    for pipeline in bronze all; do
        log "fuzz $pipeline: $FUZZ_SEEDS seeds from $FUZZ_FIRST_SEED in $jobs processes"
        local pids=()
        for (( j = 0; j < jobs; j++ )); do
            local first=$(( FUZZ_FIRST_SEED + j * per ))
            local n=$(( FUZZ_SEEDS - j * per < per ? FUZZ_SEEDS - j * per : per ))
            (( n > 0 )) || break
            local out="$dir/fuzz/$pipeline-$j"
            mkdir -p "$out"
            # shellcheck disable=SC2086
            "$@" "$dir/tools/brass-fuzz" --pipeline="$pipeline" --seed="$first" --iterations="$n" \
                --timeout-ms=5000 --chunk-size=100 --repro-dir="$out" ${FUZZ_EXTRA:-} \
                >"$out/summary.log" 2>&1 &
            pids+=($!)
        done
        for j in "${pids[@]}"; do wait "$j" || rc=1; done
        # Each job ends with the chunked-campaign summary; add them up.
        local passed failed crashed
        passed=$(sed -n 's/^ Passed: \([0-9]*\).*/\1/p' "$dir"/fuzz/"$pipeline"-*/summary.log | sum_lines)
        failed=$(sed -n 's/.*Failed: \([0-9]*\).*/\1/p' "$dir"/fuzz/"$pipeline"-*/summary.log | sum_lines)
        crashed=$(sed -n 's/^ Crashed chunks: \([0-9]*\).*/\1/p' "$dir"/fuzz/"$pipeline"-*/summary.log | sum_lines)
        echo "fuzz $pipeline: passed ${passed:-0}, failed ${failed:-0}, crashed chunks ${crashed:-0}"
        grep -h -A50 '^ Failure classes' "$dir"/fuzz/"$pipeline"-*/summary.log | grep '^   ' | sort | uniq -c || true
    done
    return $rc
}

qemu_prefix() {
    echo "$XROOT/usr/bin/qemu-aarch64-static" -L "$XROOT/usr/aarch64-linux-gnu"
}

cmd_fuzz_a64() {
    # brass-fuzz re-executes itself for --chunk-size; QEMU_LD_PREFIX lets the
    # child find the aarch64 loader, and binfmt is not assumed, so the child
    # is started through qemu by the tool's own argv[0] (see run_chunked).
    export QEMU_LD_PREFIX="$XROOT/usr/aarch64-linux-gnu"
    export BRASS_FUZZ_EXEC_PREFIX="$XROOT/usr/bin/qemu-aarch64-static"
    # The interpreter tiers are target-independent (the x86_64 runs cover
    # them); under emulation only the tiers running aarch64 code are compared.
    FUZZ_EXTRA="--jit-only ${FUZZ_EXTRA:-}"
    # shellcheck disable=SC2046
    run_fuzz "$A64_BUILD" $(qemu_prefix)
}

cmd_fuzz_x64() { run_fuzz "$X64_BUILD"; }

[[ $# -eq 0 ]] && set -- all
for c in "$@"; do
    case "$c" in
        setup) cmd_setup ;;
        build-a64) cmd_build_a64 ;;
        test-a64) cmd_test_a64 ;;
        fuzz-a64) cmd_fuzz_a64 ;;
        build-x64) cmd_build_x64 ;;
        test-x64) cmd_test_x64 ;;
        fuzz-x64) cmd_fuzz_x64 ;;
        a64) cmd_build_a64; cmd_test_a64; cmd_fuzz_a64 ;;
        x64) cmd_build_x64; cmd_test_x64; cmd_fuzz_x64 ;;
        all) cmd_setup; cmd_build_a64; cmd_test_a64; cmd_fuzz_a64
             cmd_build_x64; cmd_test_x64; cmd_fuzz_x64 ;;
        *) echo "unknown command: $c" >&2; exit 2 ;;
    esac
done
