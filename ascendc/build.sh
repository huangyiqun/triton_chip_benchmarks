#!/bin/bash
# Build the Ascend C chip benchmark suite.
#
# Requires CANN 9.2.x with bisheng. Include paths cover three trees:
#   asc/include      - Ascend C C API headers
#   asc              - impl/ headers those include by relative path
#   aarch64-linux/include - version/asc_devkit_version.h
#   include          - ACL runtime headers
set -euo pipefail

CANN_HOME="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-9.2.1}"
NPU_ARCH="${NPU_ARCH:-dav-3510}"   # Ascend 950 series

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BIN_DIR="$SCRIPT_DIR/bin"
mkdir -p "$BIN_DIR"

if ! command -v bisheng >/dev/null 2>&1; then
    echo "error: bisheng not on PATH; source ${CANN_HOME}/set_env.sh" >&2
    exit 1
fi

INCLUDES=(
    -I"$CANN_HOME/asc/include"
    -I"$CANN_HOME/asc"
    -I"$CANN_ROOT/aarch64-linux/include"
    -I"$CANN_HOME/include"
    -I"$SRC_DIR"
)

# ascendc_runtime needs the CANN support libraries; this list matches the one
# FindASC.cmake links for kernel-launch executables.
LIBS=(
    -L"$CANN_HOME/lib64"
    -lascendc_runtime -lascendcl -lruntime -lregister -lerror_manager
    -lprofapi -lge_common_base -lunified_dlog -lmmpa -ldl -lascend_dump
    -lc_sec -lstdc++
)

# -Wno-ignored-attributes: __gm__ on a host-side cast expression is meaningless
# to the host compiler but harmless, and the kernel-launch syntax requires it.
FLAGS=(--npu-arch="$NPU_ARCH" -std=c++20 -O2 -Wno-ignored-attributes)

# bench_cube is compiled once per dtype.
#
# Two hard constraints force one kernel per binary: the linker sums static
# on-chip allocations across every kernel in a module (L0A/L0B hold 64 KiB
# each, L0C 128 KiB), and aliasing a single shared pool through casts to work
# around that makes bisheng's on-chip allocator pathological - the same kernel
# compiles in ~4 s with local declarations and does not finish in 25 minutes
# with pool casts.
CUBE_DTYPES=(bf16 fp16 fp32 tf32 int8 fp8 fp4)
# bench_vector is compiled per dtype for the same reason, and because a shared
# UB pool aliased through casts produced nondeterministic chain results.
VECTOR_DTYPES=(fp32 fp16 bf16 int8 fp8 fp4)

vector_selector() {
    case "$1" in
        fp32) echo VECTOR_FP32 ;; fp16) echo VECTOR_FP16 ;;
        bf16) echo VECTOR_BF16 ;; int8) echo VECTOR_INT8 ;;
        fp8)  echo VECTOR_FP8  ;; fp4)  echo VECTOR_FP4  ;;
        *) echo "unknown vector dtype $1" >&2; exit 1 ;;
    esac
}

cube_selector() {
    case "$1" in
        bf16) echo CUBE_BF16 ;; fp16) echo CUBE_FP16 ;;
        fp32) echo CUBE_FP32 ;; tf32) echo CUBE_TF32 ;;
        int8) echo CUBE_INT8 ;; fp8)  echo CUBE_FP8  ;;
        fp4)  echo CUBE_FP4  ;;
        *) echo "unknown cube dtype $1" >&2; exit 1 ;;
    esac
}

build_one() {
    local target="$1"
    case "$target" in
        bench_cube)
            for dtype in "${CUBE_DTYPES[@]}"; do
                echo "[build] bench_cube_$dtype"
                bisheng "${FLAGS[@]}" "${INCLUDES[@]}" \
                    "-DBENCH_CUBE_DTYPE=$(cube_selector "$dtype")" \
                    -x cce "$SRC_DIR/bench_cube.asc" \
                    -o "$BIN_DIR/bench_cube_$dtype" "${LIBS[@]}"
            done
            ;;
        bench_fp64)
            # Capability reports, not benchmarks: one per execution unit.
            for unit in cube vector; do
                echo "[build] bench_fp64_$unit"
                selector=FP64_CUBE
                [ "$unit" = vector ] && selector=FP64_VECTOR
                bisheng "${FLAGS[@]}" "${INCLUDES[@]}" "-DFP64_UNIT=$selector" \
                    -x cce "$SRC_DIR/bench_fp64.asc" \
                    -o "$BIN_DIR/bench_fp64_$unit" "${LIBS[@]}"
            done
            ;;
        bench_vector)
            for dtype in "${VECTOR_DTYPES[@]}"; do
                echo "[build] bench_vector_$dtype"
                bisheng "${FLAGS[@]}" "${INCLUDES[@]}" \
                    "-DBENCH_VECTOR_DTYPE=$(vector_selector "$dtype")" \
                    -x cce "$SRC_DIR/bench_vector.asc" \
                    -o "$BIN_DIR/bench_vector_$dtype" "${LIBS[@]}"
            done
            ;;
        bench_vector_*)
            local dtype="${target#bench_vector_}"
            echo "[build] $target"
            bisheng "${FLAGS[@]}" "${INCLUDES[@]}" \
                "-DBENCH_VECTOR_DTYPE=$(vector_selector "$dtype")" \
                -x cce "$SRC_DIR/bench_vector.asc" \
                -o "$BIN_DIR/$target" "${LIBS[@]}"
            ;;
        bench_cube_*)
            local dtype="${target#bench_cube_}"
            echo "[build] $target"
            bisheng "${FLAGS[@]}" "${INCLUDES[@]}" \
                "-DBENCH_CUBE_DTYPE=$(cube_selector "$dtype")" \
                -x cce "$SRC_DIR/bench_cube.asc" \
                -o "$BIN_DIR/$target" "${LIBS[@]}"
            ;;
        *)
            echo "[build] $target"
            bisheng "${FLAGS[@]}" "${INCLUDES[@]}" -x cce "$SRC_DIR/$target.asc" \
                -o "$BIN_DIR/$target" "${LIBS[@]}"
            ;;
    esac
}

TARGETS=("${@}")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(bench_cube bench_vector bench_memory bench_fp64)
fi

for target in "${TARGETS[@]}"; do
    build_one "$target"
done

echo "Built: ${TARGETS[*]} -> $BIN_DIR"
