#!/usr/bin/env python3
"""Measure matrix-compute peak throughput or sustained dense Triton GEMM.

Peak mode keeps operands resident while repeatedly issuing matrix dot
operations on two independent accumulator chains. GEMM mode measures
application-style matrix multiplication. FP16/BF16/FP8/FP32/TF32 use FP32
accumulators, FP64 uses FP64, and INT8 uses INT32. One multiply plus one add
counts as two operations: FLOPs for floats and integer operations for INT8.
"""

from __future__ import annotations

import os

# FlagGems' KunlunXin event backend reads this during runtime initialization.
# Keep an explicit user setting (for example, 0 while diagnosing event support).
os.environ.setdefault("XPU_EVENT_KL3_ENABLE", "1")

import argparse
import math
import re

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

try:
    from .common import (
        add_common_arguments, available_memory_bytes, emit_results, emit_unsupported,
        full_precision_matmul, get_device, is_flagtree_xpu, measure, setup_device,
    )
except ImportError:
    from common import (
        add_common_arguments, available_memory_bytes, emit_results, emit_unsupported,
        full_precision_matmul, get_device, is_flagtree_xpu, measure, setup_device,
    )


class TensorUnsupportedError(RuntimeError):
    """The requested format or its required execution precision is unavailable."""


# OCP MX FP4: E2M1 elements packed two per byte, with one E8M0 scale factor
# shared by every MX_GROUP_SIZE elements along K. Both constants are fixed by
# the OCP Microscaling Formats v1.0 specification, not by this benchmark.
FP4_PACKING_FACTOR = 2
MX_GROUP_SIZE = 32
# E8M0 code 127 is the biased encoding of 2**0. A unit scale keeps the measured
# arithmetic equal to the raw FP4 products, so the exact reference stays exact,
# while the hardware still runs the full MX dataflow including scale lookup and
# application. A benchmark that disabled scaling would not be measuring the MX
# matrix path the format actually uses.
E8M0_UNIT_SCALE = 127
# The sixteen E2M1 values in code order: sign, two exponent bits, one mantissa
# bit. Used to build exactly representable operands and their FP64 reference.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES = _E2M1_MAGNITUDES + tuple(-value for value in _E2M1_MAGNITUDES)


def _output_dtype(dtype):
    if dtype == torch.int8:
        return torch.int32
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _accumulator_name(dtype: str) -> str:
    return "int32" if dtype == "int8" else "fp64" if dtype == "fp64" else "fp32"


def _rates(operation_count: int, timing: dict, dtype: str) -> dict:
    median = operation_count / (timing["median_ms"] * 1e9)
    peak = operation_count / (timing.get("min_ms", timing["median_ms"]) * 1e9)
    integer = dtype == "int8"
    return {"operation_count": operation_count,
            "operation_type": "integer" if integer else "floating_point",
            "throughput_unit": "TOPS" if integer else "TFLOP/s",
            "median_teraops_per_second": median, "peak_teraops_per_second": peak,
            **({"integer_ops": operation_count, "tops": median, "peak_tops": peak}
               if integer else {"flops": operation_count, "tflops": median, "peak_tflops": peak})}


def _unsupported_reason(exc: Exception) -> str | None:
    """Recognize capability/compiler rejections without hiding runtime failures."""
    errors = []
    current = exc
    while current is not None and all(current is not item for item in errors):
        errors.append(current)
        current = current.__cause__
    # CompilationError.__str__ includes kernel source. Classify diagnostics,
    # never a source line/comment containing a word such as "unsupported".
    messages = [(error.error_message or "") if isinstance(error, CompilationError)
                else str(error) for error in errors]
    diagnostic = "\n".join(messages)
    # Resource exhaustion, device faults, validation and timer errors are not
    # evidence that an input precision is unsupported. Compiler implementation
    # errors must also remain visible, even when wrapped in CompilationError.
    if any(isinstance(error, (triton.OutOfResources, NameError, AttributeError,
                              MemoryError)) for error in errors) or re.search(
        r"out of memory|illegal memory|device.side assert|event time|correctness",
        diagnostic, re.IGNORECASE,
    ):
        return None
    if isinstance(exc, TensorUnsupportedError):
        return str(exc)
    if not isinstance(exc, (CompilationError, RuntimeError, TypeError, ValueError, AssertionError)):
        return None
    format_or_operation = (
        r"\b(?:dtype|data type|operand(?:s| type)?|dot|matmul|matrix|mma|"
        r"fp(?:8\w*|16|32|64)|bf16|tf32|int8|float(?:8\w*|16|32|64)|bfloat16)\b"
    )
    rejection = (r"unsupported|not supported|does not support|not implemented|"
                 r"only .{0,100} supported|invalid (?:input |output )?dtype")
    explicit_rejection = any(
        re.search(format_or_operation, line, re.IGNORECASE)
        and re.search(rejection, line, re.IGNORECASE)
        for message in messages for line in message.splitlines()
    )
    target_rejection = re.search(
        r"no kernel image is available|"
        r"requires .{0,80}(?:compute capability|sm_\d|gfx\d)", diagnostic, re.IGNORECASE)
    if explicit_rejection or target_rejection:
        provider = "Triton" if isinstance(exc, CompilationError) else "Backend"
        return f"{provider} rejected this matrix format/configuration: {diagnostic.strip()}"
    return None


def _emit_tensor_results(metadata: dict, args, rows: list[dict]) -> None:
    """Add an invocation status while keeping the shared result document."""
    successful = any(row.get("status") == "ok" for row in rows)
    unsupported = any(row.get("status") == "unsupported" for row in rows)
    status = "ok" if successful else "unsupported" if unsupported else "error"
    reason = None if successful else "; ".join(dict.fromkeys(
        row.get("reason", "No usable configuration") for row in rows))
    if status == "unsupported":
        emit_unsupported("tensor", metadata, args, reason, details=rows)
    elif status == "error":
        raise RuntimeError(reason)
    else:
        emit_results("tensor", metadata, args, rows)


@triton.jit
def _int8_dot_probe_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """One INT8 tl.dot, used only to detect whether the backend supports it."""
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    inner = tl.arange(0, K)
    a = tl.load(A + rows[:, None] * K + inner[None, :])
    b = tl.load(B + inner[:, None] * N + cols[None, :])
    accumulator = tl.dot(a, b, tl.full((M, N), 0, tl.int32), out_dtype=tl.int32)
    tl.store(C + rows[:, None] * N + cols[None, :], accumulator)


@triton.jit
def _gemm_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, DOT_INPUT_PRECISION: tl.constexpr,
):
    """Contiguous row-major GEMM; grouping improves reuse in L2."""
    program = tl.program_id(0)
    count_m = tl.cdiv(M, BLOCK_M)
    count_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * count_n
    group = program // group_size
    first_m = group * GROUP_M
    actual_group_m = tl.minimum(count_m - first_m, GROUP_M)
    tile_m = first_m + (program % group_size) % actual_group_m
    tile_n = (program % group_size) // actual_group_m

    # Global element offsets may exceed 2**31 for large user-supplied sizes.
    rows = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    inner = tl.arange(0, BLOCK_K)
    if A.dtype.element_ty == tl.int8:
        ACC_DTYPE: tl.constexpr = tl.int32
    elif A.dtype.element_ty == tl.float64:
        ACC_DTYPE: tl.constexpr = tl.float64
    else:
        ACC_DTYPE: tl.constexpr = tl.float32
    accum = tl.full((BLOCK_M, BLOCK_N), 0, ACC_DTYPE)
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k = (k_tile * BLOCK_K + inner).to(tl.int64)
        a = tl.load(A + rows[:, None] * K + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B + k[:, None] * N + cols[None, :],
                    mask=(k[:, None] < K) & (cols[None, :] < N), other=0.0)
        accum = tl.dot(a, b, accum, input_precision=DOT_INPUT_PRECISION, out_dtype=ACC_DTYPE)
    tl.store(C + rows[:, None] * N + cols[None, :], accum,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def _configs(vendor: str, dtype: str = "fp16") -> list[dict[str, int]]:
    # An explicit small sweep keeps compilation/tuning costs visible. Some
    # older GPUs may reject the larger tiles due to shared-memory limits.
    if dtype == "fp64":
        choices = [(32, 32, 16, 4, 1), (32, 64, 16, 4, 1)]
    elif vendor == "amd":
        choices = [(64, 64, 64, 4, 2), (128, 128, 64, 8, 2),
                   (64, 128, 64, 4, 2)]
    elif vendor == "nvidia":
        choices = [(64, 128, 32, 4, 3), (128, 128, 64, 8, 3),
                   (128, 256, 64, 8, 3), (64, 128, 64, 4, 4)]
    else:
        # Mirrors the fallback tile set in `_peak_configs`: 32x32/64x64 alone
        # can sit far below a matrix unit that consumes a whole tile per
        # instruction, and the nvidia/amd branches above already carry
        # 128x128-class shapes for the same reason.
        choices = [(32, 32, 32, 4, 1), (64, 64, 32, 4, 1),
                   (128, 128, 64, 4, 1), (256, 128, 64, 4, 1),
                   (128, 256, 64, 4, 1)]
    return [dict(block_m=m, block_n=n, block_k=k, num_warps=w,
                 num_stages=s) for m, n, k, w, s in choices]


def _launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
            config: dict[str, int], dot_input_precision: str | None = None):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, config["block_m"]) * triton.cdiv(n, config["block_n"]),)
    return _gemm_kernel[grid](
        a, b, c, m, n, k,
        BLOCK_M=config["block_m"], BLOCK_N=config["block_n"],
        BLOCK_K=config["block_k"], GROUP_M=8,
        DOT_INPUT_PRECISION=dot_input_precision,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )


def _inputs(m: int, n: int, k: int, dtype: torch.dtype, device: torch.device):
    # Scaling limits the absolute output magnitude and makes one tolerance
    # useful across large and small K. It does not change the operation count.
    if dtype == torch.int8:
        a = torch.randint(-1, 2, (m, k), dtype=dtype, device=device)
        b = torch.randint(-1, 2, (k, n), dtype=dtype, device=device)
    else:
        generation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        a = torch.randn((m, k), dtype=generation_dtype, device=device).to(dtype)
        b = (torch.randn((k, n), dtype=generation_dtype, device=device)
             * (1.0 / math.sqrt(k))).to(dtype)
    c = torch.empty((m, n), dtype=_output_dtype(dtype), device=device)
    return a, b, c


def _stack2(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Stack two equal-shaped tensors along a new leading axis.

    `torch.stack` is the natural spelling, but not every vendor implements it
    for every storage dtype it otherwise supports: on Ascend, aclnnStack
    rejects float8_e4m3fn while torch.cat of unsqueezed views handles it.
    `cat` of two unsqueezed views is the same tensor layout and the same value
    for every dtype, so this helper is used instead of a vendor branch. The
    copy is explicit either way; no backend inlines it into the timed kernel.
    """
    return torch.cat((first.unsqueeze(0), second.unsqueeze(0)), dim=0)


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The FlagGems backend context disables reduced-precision modes where
    # the selected vendor runtime exposes the corresponding controls.
    with full_precision_matmul():
        if a.dtype == torch.float64:
            return torch.matmul(a, b)
        result = torch.matmul(a.float(), b.float())
        # INT8 test operands are -1/0/1 and K < 2**24, so every partial sum
        # is exactly representable in FP32, including the TF32 product path.
        return result.to(torch.int32) if a.dtype == torch.int8 else result


def _validate(c: torch.Tensor, reference: torch.Tensor,
              dot_input_precision: str | None = None) -> dict:
    tolerance = (0 if c.dtype == torch.int32 else 1e-10 if c.dtype == torch.float64
                 else 1e-2 if dot_input_precision == "tf32" else 2e-3)
    torch.testing.assert_close(c, reference, rtol=tolerance, atol=tolerance)
    max_abs_error = (c - reference).abs().max().item()
    return {"passed": True, "max_abs_error": max_abs_error,
            "rtol": tolerance, "atol": tolerance,
            "reference": ("exact ternary FP32 matmul converted to INT32" if c.dtype == torch.int32
                          else "FP64 torch.matmul" if c.dtype == torch.float64
                          else "FP32 torch.matmul under FlagGems full-precision context")}


def _peak_tolerance(dtype: str, block_k: int, iterations: int) -> float:
    """Bound the accumulated rounding error of the resident peak recurrence.

    Peak operands are multiples of 1/16, so every exact dot entry is a multiple
    of 1/256 with magnitude at most block_k/256, and an INT8 result in INT32
    stays exact. For the floating formats the accumulator rounds once per dot
    and those errors add linearly over the iterations, which bounds the total
    by iterations * block_k/256 * unit_roundoff. The returned bound keeps a
    safety factor and still stays orders of magnitude below 1/256, the smallest
    difference a dropped iteration or a merged accumulator chain produces, so
    the check continues to reject a kernel that skipped requested work.
    """
    if dtype == "int8":
        return 0.0
    if dtype == "fp4":
        # FP4 peak operands are +/-0.5 and +/-1.0, so each dot entry is a sum
        # of block_k products of magnitude at most 1 and every partial sum is
        # an exact multiple of 0.25 bounded by block_k. With iterations
        # accumulations the exact value stays well inside FP32's integer-exact
        # range, so the only rounding is the accumulator's, bounded the same
        # way as the other floating formats.
        return 4.0 * iterations * block_k * 1.0 * 2.0**-24
    unit_roundoff = 2.0**-53 if dtype == "fp64" else 2.0**-24
    return 4.0 * iterations * block_k * (1.0 / 256.0) * unit_roundoff


def _validate_peak(c: torch.Tensor, reference: torch.Tensor, dtype: str,
                   block_k: int, iterations: int) -> dict:
    """Check every resident-peak output against the exact scaled dot reference.

    Demanding bit-exact equality here assumed the accumulator is the only
    rounding step, which holds for an exactly representable FP32 partial sum but
    not for a backend whose MMA rounds its own product internally: MetaX returns
    BF16/FP16/TF32 results a few 2**-33 away from the exact lattice point. The
    analytic bound keeps the measurement while staying far below any difference
    a skipped operation would cause; see flagtree_metax_bug.md.
    """
    tolerance = _peak_tolerance(dtype, block_k, iterations)
    torch.testing.assert_close(c, reference, rtol=0, atol=tolerance)
    return {"passed": True, "max_abs_error": (c - reference).abs().max().item(),
            "rtol": 0, "atol": tolerance,
            "reference": f"exact {_accumulator_name(dtype).upper()} dot * iterations",
            "tolerance_basis": ("exact INT32 equality" if dtype == "int8" else
                                "accumulated accumulator unit roundoff over all iterations, "
                                "far below the 1/256 exact-operand lattice step")}


def _row_multiset_matches(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """True when the expected rows are all present but in a different order."""
    flat_actual = actual.reshape(-1, actual.shape[-1]).cpu().tolist()
    flat_expected = expected.reshape(-1, expected.shape[-1]).cpu().tolist()
    if len(flat_actual) != len(flat_expected):
        return False
    return sorted(map(tuple, flat_actual)) == sorted(map(tuple, flat_expected))


def _normalize_precision_args(parser: argparse.ArgumentParser, args) -> None:
    """Resolve the FP32 dot policy without relying on a backend default."""
    args.requested_dtype = args.dtype
    if args.dtype == "tf32":
        if args.input_precision not in (None, "tf32"):
            parser.error("--dtype tf32 cannot use --input-precision ieee")
        args.dtype, args.input_precision = "fp32", "tf32"
    elif args.dtype == "fp32":
        args.input_precision = args.input_precision or "ieee"
    elif args.input_precision is not None:
        parser.error("--input-precision is only valid with --dtype fp32")


def _dtype_name(dtype: str) -> str:
    # FP4 has no PyTorch scalar dtype that Triton accepts as an operand: OCP MX
    # FP4 operands are carried as uint8 with two E2M1 elements per byte, exactly
    # as tl.dot_scaled documents. The storage dtype therefore differs from the
    # arithmetic format, which _precision_descriptor records separately.
    return {"fp16": "float16", "bf16": "bfloat16", "fp8": "float8_e4m3fn",
            "fp32": "float32", "tf32": "float32", "int8": "int8", "fp64": "float64",
            "fp4": "uint8"}[dtype]


def _precision_descriptor(dtype: str, dot_input_precision: str | None) -> dict:
    storage = "fp32" if dtype == "fp32" else ("e4m3fn" if dtype == "fp8" else
                                              "e2m1" if dtype == "fp4" else dtype)
    if dtype == "fp32":
        math_mode = "tf32" if dot_input_precision == "tf32" else "ieee_fp32"
    elif dtype == "fp4":
        math_mode = "mxfp4_e2m1"
    else:
        math_mode = dtype
    descriptor = {"input_format": storage, "input_storage_dtype": storage,
                  "dot_input_precision": dot_input_precision,
                  "math_mode": math_mode}
    if dtype == "fp4":
        descriptor.update({
            "container_dtype": "uint8",
            "elements_per_container": FP4_PACKING_FACTOR,
            "scale_format": "e8m0",
            "scale_group_size": MX_GROUP_SIZE,
            "scale_value": "2**0 (E8M0 code 127) for every group",
            "microscaled": True,
            "dot_operation": "tl.dot_scaled",
        })
    return descriptor


@triton.jit
def _tensor_peak_kernel(
    A, B, C,
    BLOCKS: tl.constexpr, ITERATIONS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    UNROLL: tl.constexpr = 1, B_TRANSPOSED: tl.constexpr = False,
):
    block = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    # Each operand is read from global memory once per CTA, before the loop.
    # Distinct B matrices prevent merging the two independent accumulators.
    a = tl.load(A + block * BLOCK_M * BLOCK_K + rows[:, None] * BLOCK_K + inner[None, :])
    if B_TRANSPOSED:
        b_offsets = block * BLOCK_K * BLOCK_N + cols[None, :] * BLOCK_K + inner[:, None]
    else:
        b_offsets = block * BLOCK_K * BLOCK_N + inner[:, None] * BLOCK_N + cols[None, :]
    b0 = tl.load(B + b_offsets)
    b1 = tl.load(B + BLOCKS * BLOCK_K * BLOCK_N + b_offsets)
    if A.dtype.element_ty == tl.int8:
        ACC_DTYPE: tl.constexpr = tl.int32
    elif A.dtype.element_ty == tl.float64:
        ACC_DTYPE: tl.constexpr = tl.float64
    else:
        ACC_DTYPE: tl.constexpr = tl.float32
    acc0 = tl.full((BLOCK_M, BLOCK_N), 0, ACC_DTYPE)
    acc1 = tl.full((BLOCK_M, BLOCK_N), 0, ACC_DTYPE)
    # Keep the outer loop dynamic; only expand a small group of dot pairs.
    for _ in range(ITERATIONS // UNROLL):
        # FP32 passes an explicit IEEE/TF32 policy. Other dtypes preserve the
        # backend's native dot policy; forcing exact FP8 accumulation can
        # demote an FP8 operation to FP16 MMA on some targets.
        for _ in tl.static_range(UNROLL):
            acc0 = tl.dot(a, b0, acc0, input_precision=DOT_INPUT_PRECISION, out_dtype=ACC_DTYPE)
            acc1 = tl.dot(a, b1, acc1, input_precision=DOT_INPUT_PRECISION, out_dtype=ACC_DTYPE)
    # Preserve arbitrary iteration counts, including the one-dot lowering probe.
    for _ in tl.static_range(ITERATIONS % UNROLL):
        acc0 = tl.dot(a, b0, acc0, input_precision=DOT_INPUT_PRECISION, out_dtype=ACC_DTYPE)
        acc1 = tl.dot(a, b1, acc1, input_precision=DOT_INPUT_PRECISION, out_dtype=ACC_DTYPE)
    c_offsets = block * BLOCK_M * BLOCK_N + rows[:, None] * BLOCK_N + cols[None, :]
    tl.store(C + c_offsets, acc0)
    tl.store(C + BLOCKS * BLOCK_M * BLOCK_N + c_offsets, acc1)


@triton.jit
def _tensor_peak_scaled_kernel(
    A, SA, B, SB, C,
    BLOCKS: tl.constexpr, ITERATIONS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PACKED: tl.constexpr, GROUP: tl.constexpr, FORMAT: tl.constexpr,
):
    """Resident microscaled (MX) peak, mirroring _tensor_peak_kernel.

    BLOCK_K counts arithmetic elements. Operands hold PACKED of them per
    container byte, so their K extent is BLOCK_K // PACKED, and each operand row
    carries BLOCK_K // GROUP scale factors. Operands, scales and both
    accumulators stay resident; only tl.dot_scaled repeats in the loop.
    """
    block = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    STORED_K: tl.constexpr = BLOCK_K // PACKED
    GROUPS: tl.constexpr = BLOCK_K // GROUP
    stored = tl.arange(0, STORED_K)
    groups = tl.arange(0, GROUPS)
    a = tl.load(A + block * BLOCK_M * STORED_K + rows[:, None] * STORED_K + stored[None, :])
    b_offsets = block * STORED_K * BLOCK_N + stored[:, None] * BLOCK_N + cols[None, :]
    b0 = tl.load(B + b_offsets)
    b1 = tl.load(B + BLOCKS * STORED_K * BLOCK_N + b_offsets)
    # Scales are indexed [M, K/GROUP] for A and [N, K/GROUP] for B; the B scale
    # is deliberately not transposed, as tl.dot_scaled requires.
    sa = tl.load(SA + block * BLOCK_M * GROUPS + rows[:, None] * GROUPS + groups[None, :])
    sb = tl.load(SB + block * BLOCK_N * GROUPS + cols[:, None] * GROUPS + groups[None, :])
    acc0 = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    acc1 = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for _ in range(ITERATIONS):
        acc0 = tl.dot_scaled(a, sa, FORMAT, b0, sb, FORMAT, acc0)
        acc1 = tl.dot_scaled(a, sa, FORMAT, b1, sb, FORMAT, acc1)
    c_offsets = block * BLOCK_M * BLOCK_N + rows[:, None] * BLOCK_N + cols[None, :]
    tl.store(C + c_offsets, acc0)
    tl.store(C + BLOCKS * BLOCK_M * BLOCK_N + c_offsets, acc1)


def _peak_configs(vendor: str, dtype: str = "fp16",
                  dot_input_precision: str | None = None) -> list[dict[str, int]]:
    # FP4 needs K to be a multiple of the 32-element MX group, and the Ascend
    # MX path rejects K=64 outright (its meta-op library has no 64-element FP4
    # fractal), so FP4's smallest usable K is 128.
    k = (128 if dtype == "fp4" else 64 if dtype in ("fp8", "int8")
         else 16 if dtype == "fp64" else 32)
    if dtype == "fp64":
        choices = [(32, 32, k, 4), (32, 64, k, 4)]
    elif dtype == "fp32" and dot_input_precision == "ieee":
        # IEEE FP32 can lower to scalar/vector FMA, where smaller accumulator
        # tiles avoid excessive register pressure. AMD may still select MFMA.
        choices = [(32, 32, k, 4), (32, 64, k, 4),
                   (64, 32, k, 4), (64, 64, k, 4)]
    elif vendor in ("nvidia", "amd"):
        choices = [(64, 64, k, 4), (64, 128, k, 4), (128, 64, k, 4)]
    elif vendor == "thead":
        # T-Head PPU: the AIU executes m16n16k16/m16n16k8/m16n16k32 MMA
        # instructions, but its throughput is governed by the accumulator
        # footprint per warp. On PPU-ZW810E every shape holding
        # block_m * block_n / num_warps == 2048 output elements reaches
        # 120+ TFLOP/s FP16, while 4096 spills the register file (256 regs)
        # and collapses to about 20.
        # Like the nvidia/amd branch above, these shapes are shared across
        # dtypes and only BLOCK_K varies with k: the best tile differs per
        # dtype, so the sweep has to choose it rather than the vendor branch.
        # Narrow N is the secondary factor, and it does not rank the same way
        # for every dtype, because the MMA shape differs. FP16/BF16 issue
        # m16n16k16 and peak at N=32 (128x32x32 measures 128 TFLOP/s, against
        # 109 at N=16). INT8 issues m16n16k32, which carries twice the MACs
        # per instruction but halves the issue rate at N=32 (7.4 G/s, down
        # from 13.3 at N=16) and so cancels its own advantage: 256x16x64
        # measures 218 TOP/s, against 121 at N=32. Keeping both widths in one
        # list lets each dtype reach its own peak, and keeps the INT8:BF16
        # ratio near the 2x the instruction shapes imply.
        # IEEE FP32 and FP64 keep the SIMT tile sets of the branches above;
        # FP64 needs them, since wide-M shapes spill its FP64 accumulators.
        choices = [(128, 32, k, 2), (256, 32, k, 4),
                   (256, 16, k, 2), (128, 16, k, 1),
                   (512, 32, 2 * k, 8), (128, 64, k, 4)]
    else:
        # Backends whose matrix unit consumes a whole tile per instruction need
        # an accumulator tile large enough to cover the instruction's issue
        # latency, so the 32x32/64x64 pair alone can land a long way below the
        # achievable rate. On Ascend950DT every dtype gains about 3x between
        # 64x64 and 128x128 and then flattens: BF16/FP16 measure 137 TFLOP/s at
        # 64x64 against 430 at 256x128x2k, INT8 155 TOP/s against 859, FP8 271
        # against 861, TF32 101 TFLOP/s against 215. The small tiles are kept
        # because they remain the best choice where the matrix unit is fed per
        # warp instead, and because a narrow tile is the only one that fits
        # when the accumulator is wide.
        # Shapes stay expressed in multiples of the per-dtype k so the sweep
        # picks the winner per dtype rather than a vendor branch choosing it.
        # 256x256 and 512x128 are deliberately absent: both exceed the
        # Ascend950DT unified-buffer budget and fail to compile at every k.
        choices = [(32, 32, k, 4), (64, 64, k, 4),
                   (128, 128, 2 * k, 4), (256, 128, 2 * k, 4),
                   (128, 256, 2 * k, 4)]
        if dtype == "fp4":
            # FP4's k is already 128, so `2 * k` would be 256 everywhere. Keep
            # the same tile footprints, add a K=128 variant, and extend to
            # K=512: FP4 packs two elements per byte, so a deeper K costs the
            # same operand bytes as half that K in FP8 and keeps the MX scale
            # fetch amortized over more MACs. On Ascend950DT that is where the
            # format approaches its ceiling -- 256x128x512 and 128x256x512
            # measure 1725 TFLOP/s against 1570 at K=256, which is 99.7% of
            # the 1730 TFLOP/s the card's 32 Cube cores imply and matches what
            # a native Ascend C kernel reaches on the same card.
            # 512x128, 128x512 and 256x256 are absent at every K: all three
            # exceed the Ascend950DT buffer budget, exactly as for the other
            # formats.
            choices = [(32, 32, k, 4), (64, 64, k, 4),
                       (128, 128, k, 4), (128, 128, 2 * k, 4),
                       (256, 128, 2 * k, 4), (128, 256, 2 * k, 4),
                       (256, 128, 4 * k, 4), (128, 256, 4 * k, 4),
                       (128, 128, 8 * k, 4)]
    configs = [dict(block_m=m, block_n=n, block_k=k, num_warps=w,
                 num_stages=1, chains=2) for m, n, k, w in choices]
    if vendor == "thead" and dtype == "int8":
        # PPU-ZW810E resident INT8 peak; retain original candidates as baselines.
        configs.append(dict(block_m=128, block_n=16, block_k=128, num_warps=2,
                            num_stages=2, chains=2, unroll=4, b_transposed=True))
    return configs


def _launch_peak(a, b, c, iterations: int, config: dict[str, int],
                 dot_input_precision: str | None = None, scales=None):
    """Launch the resident peak kernel; `scales` selects the microscaled path."""
    blocks = a.shape[0]
    if scales is not None:
        scale_a, scale_b = scales
        return _tensor_peak_scaled_kernel[(blocks,)](
            a, scale_a, b, scale_b, c, BLOCKS=blocks, ITERATIONS=iterations,
            BLOCK_M=config["block_m"], BLOCK_N=config["block_n"],
            BLOCK_K=config["block_k"], PACKED=FP4_PACKING_FACTOR,
            GROUP=MX_GROUP_SIZE, FORMAT="e2m1",
            num_warps=config["num_warps"], num_stages=config["num_stages"],
        )
    return _tensor_peak_kernel[(blocks,)](
        a, b, c, BLOCKS=blocks, ITERATIONS=iterations,
        BLOCK_M=config["block_m"], BLOCK_N=config["block_n"], BLOCK_K=config["block_k"],
        DOT_INPUT_PRECISION=dot_input_precision,
        UNROLL=config.get("unroll", 1), B_TRANSPOSED=b.stride(-2) == 1,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )


def _pack_e2m1_pairs(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 nibble codes two per byte along the last dimension.

    tl.dot_scaled specifies the first element in the lower bits, so an even
    index k goes to the low nibble and k+1 to the high nibble.
    """
    low = codes[..., 0::2].to(torch.int16)
    high = codes[..., 1::2].to(torch.int16)
    return ((high << 4) | low).to(torch.uint8)


def _fp4_peak_inputs(blocks: int, config: dict[str, int], device):
    """Build packed MX FP4 operands, unit scales, and their exact FP64 values.

    Operands are drawn from the four E2M1 codes for +/-0.5 and +/-1.0. Each dot
    entry is then a sum of block_k products of magnitude at most 1, which FP32
    accumulates exactly, and the repeated accumulation over the iteration range
    stays exactly representable. Returning the FP64 element values alongside the
    packed bytes lets the reference be computed from what was really stored,
    rather than from a second independent construction that could silently
    disagree with it.
    """
    m, n, k = (config[f"block_{axis}"] for axis in ("m", "n", "k"))
    # Codes 1, 2 (+0.5, +1.0) and 9, 10 (-0.5, -1.0).
    palette = torch.tensor([1, 2, 9, 10], dtype=torch.uint8)
    codes_a = palette[torch.randint(len(palette), (blocks, m, k))]
    codes_b = palette[torch.randint(len(palette), (2, blocks, k, n))]
    lookup = torch.tensor(E2M1_VALUES, dtype=torch.float64)
    values_a = lookup[codes_a.long()]
    values_b = lookup[codes_b.long()]
    a = _pack_e2m1_pairs(codes_a).to(device)
    # B is [K, N] and K is the packed axis, so pack along K by transposing the
    # last two dimensions, packing, and transposing back.
    b = _pack_e2m1_pairs(codes_b.transpose(-1, -2).contiguous())
    b = b.transpose(-1, -2).contiguous().to(device)
    groups = k // MX_GROUP_SIZE
    scale_a = torch.full((blocks, m, groups), E8M0_UNIT_SCALE,
                         dtype=torch.uint8, device=device)
    scale_b = torch.full((blocks, n, groups), E8M0_UNIT_SCALE,
                         dtype=torch.uint8, device=device)
    c = torch.empty((2, blocks, m, n), dtype=torch.float32, device=device)
    return a, b, c, (scale_a, scale_b), (values_a, values_b)


def _peak_inputs(blocks: int, config: dict[str, int], dtype: torch.dtype, device):
    m, n, k = (config[f"block_{axis}"] for axis in ("m", "n", "k"))
    # Products and partial sums lie on an exact 1/256 lattice. The iteration
    # bound ensures all FP32 integer coefficients remain exactly representable.
    if str(dtype).startswith("torch.float8_"):
        # Native Hopper FP8 internal accumulation has fewer precision bits
        # than FP32. Random one-hot rows make every dot +/-1/256, keeping the
        # repeated result exactly representable for the permitted iteration
        # range. Hardware still executes dense MMA, with no sparsity metadata.
        a = torch.zeros((blocks, m, k), device=device)
        a.scatter_(2, torch.randint(k, (blocks, m, 1), device=device), 1 / 16)
        a = a.to(dtype)
        b = torch.randint(0, 2, (2, blocks, k, n), device=device).float()
        b = b.mul_(2).sub_(1).mul_(1 / 16).to(dtype)
    elif dtype == torch.int8:
        a = torch.randint(-1, 2, (blocks, m, k), device=device, dtype=dtype)
        b = torch.randint(-1, 2, (2, blocks, k, n), device=device, dtype=dtype)
    else:
        a = torch.randint(-1, 2, (blocks, m, k), device=device).float().mul_(1 / 16).to(dtype)
        b = torch.randint(-1, 2, (2, blocks, k, n), device=device).float().mul_(1 / 16).to(dtype)
    if config.get("b_transposed", False):
        # Preserve logical [chain, block, K, N] for the unchanged reference.
        # Layout preparation is outside the resident-kernel timing.
        b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
    c = torch.empty((2, blocks, m, n), dtype=_output_dtype(dtype), device=device)
    return a, b, c


def _validate_fp32_precision(device, dot_input_precision: str,
                             mode: str = "gemm") -> dict:
    """Run an untimed probe whose exact IEEE and TF32 answers differ."""
    size = 32
    ieee_tail = 1.0 + 2.0**-12
    tf32_mantissa = 1.0 + 2.0**-9
    tf32_exponent = 2.0**20
    config = {"block_m": size, "block_n": size, "block_k": size,
              "num_warps": 4, "num_stages": 1}
    a = torch.full((size, size), ieee_tail, dtype=torch.float32, device=device)
    a[0, 1] = tf32_mantissa
    a[0, 2] = tf32_exponent
    b = torch.eye(size, dtype=torch.float32, device=device)
    if mode == "peak":
        peak_a = a.unsqueeze(0)
        peak_b = _stack2(b, b).unsqueeze(1)
        c = torch.empty((2, 1, size, size), dtype=torch.float32, device=device)
        _launch_peak(peak_a, peak_b, c, 1, config, dot_input_precision)
    else:
        c = torch.empty((size, size), dtype=torch.float32, device=device)
        _launch(a, b, c, config, dot_input_precision)
    expected = a.clone()
    if dot_input_precision == "tf32":
        expected.fill_(1.0)
        expected[0, 1] = tf32_mantissa
        expected[0, 2] = tf32_exponent
    expected_output = expected if mode == "gemm" else _stack2(expected, expected).unsqueeze(1)
    max_abs_error = (c - expected_output).abs().max().item()
    if max_abs_error != 0.0:
        observed = (c if mode == "gemm" else c[0, 0])[0, 0].item()
        raise TensorUnsupportedError(
            f"The backend did not honor tl.dot input_precision={dot_input_precision!r}: "
            f"precision probe expected first value {expected[0, 0].item()}, observed {observed}, "
            f"max absolute error {max_abs_error}."
        )
    return {"passed": True, "max_abs_error": max_abs_error,
            "probe_values": [ieee_tail, tf32_mantissa, tf32_exponent],
            "kernel_mode": mode,
            "description": ("A contains 1+2^-12, 1+2^-9, and 2^20; B=identity; "
                            "distinguishes IEEE, TF32, BF16, and FP16")}


def _validate_fp64_precision(device, mode: str) -> dict:
    """Reject a backend that accepts FP64 storage but computes it incorrectly.

    Two independent faults are checked. The precision rows carry 1+2^-40, which
    an FP32 fallback cannot represent. The row-identity rows carry a distinct
    value per row, which catches a backend that computes every product
    correctly but delivers it to the wrong output row: with uniform rows such a
    permutation is invisible, and MetaX has exactly that defect, so a
    uniform-only probe passed and the sweep then failed its own output check
    with no capability diagnosis. See flagtree_metax_bug.md.
    """
    size = 32
    inner = 16 if mode == "peak" else size
    config = {"block_m": size, "block_n": size, "block_k": inner,
              "num_warps": 4, "num_stages": 1}
    # Column 0 distinguishes the rows; the FP64-only tail occupies the rest.
    a = torch.full((size, inner), 1.0 + 2.0**-40, dtype=torch.float64, device=device)
    a[:, 0] = torch.arange(1, size + 1, dtype=torch.float64, device=device)
    b = torch.eye(inner, size, dtype=torch.float64, device=device)
    if mode == "peak":
        c = torch.empty((2, 1, size, size), dtype=torch.float64, device=device)
        _launch_peak(a.unsqueeze(0), _stack2(b, b).unsqueeze(1), c, 1, config)
    else:
        c = torch.empty((size, size), dtype=torch.float64, device=device)
        _launch(a, b, c, config)
    # b selects the first `inner` columns of A and zeroes the rest.
    expected = torch.cat((a, torch.zeros((size, size - inner), dtype=torch.float64,
                                         device=device)), dim=1)
    if mode == "peak":
        expected = _stack2(expected, expected).unsqueeze(1)
    error = (c - expected).abs().max().item()
    if error != 0.0:
        permuted = _row_multiset_matches(c, expected)
        detail = ("the returned rows are a permutation of the correct rows, so products are "
                  "computed but written to the wrong output row"
                  if permuted else f"probe error={error}")
        raise TensorUnsupportedError(
            f"The backend did not compute FP64 tl.dot correctly: {detail}")
    return {"passed": True, "max_abs_error": error, "kernel_mode": mode,
            "description": "FP64 (1+2^-40) with a per-row marker times identity; "
                           "rejects FP32 fallback and permuted output rows"}


def _validate_fp8_storage(device, dtype: torch.dtype) -> dict:
    """Reject devices that expose the FP8 dtype but cannot hold FP8 tensors."""
    # Multiples of 1/16 are exact in E4M3FN, so a faithful storage and
    # conversion path round-trips them bit for bit. Allocation, conversion or
    # copy failures here are capability rejections, not measurement failures;
    # resource exhaustion and device faults stay visible through the cause
    # chain, which _unsupported_reason inspects before trusting this class.
    probe = torch.tensor([1 / 16, -1 / 16, 1.0], dtype=torch.float32, device=device)
    try:
        restored = probe.to(dtype).float()
    except Exception as exc:
        raise TensorUnsupportedError(
            f"The backend cannot materialize {dtype} tensors on this device: {exc}") from exc
    error = (restored - probe).abs().max().item()
    if error != 0.0:
        raise TensorUnsupportedError(
            f"The backend did not preserve {dtype} storage; round-trip error={error}")
    return {"passed": True, "max_abs_error": error,
            "probe_values": [1 / 16, -1 / 16, 1.0],
            "description": "FP8 E4M3FN storage round-trip of exactly representable values"}


def _validate_dot_lowering(device, dtype: torch.dtype, dtype_name: str, mode: str,
                           candidates: list[dict], dot_input_precision: str | None = None) -> dict:
    """Compile a single-tile dot to confirm the backend lowers this format.

    Holding a dtype in memory does not mean `tl.dot` has a lowering for it. The
    MACA backend accepts FP8 tensors but fails in ConvertTritonGPUToLLVM with an
    `i8`/`f32` element-type mismatch, and its `PassManager::run failed` carries
    no dtype name, so the sweep could not classify it. Probing first turns that
    into an explicit unsupported status instead of an unhandled traceback.

    Every tile shape the sweep would try is probed, and only a format that no
    shape can compile is rejected. A single hardcoded shape could fail for a
    reason specific to that shape and wrongly discard a measurable format on a
    backend whose MMA has different tile constraints.
    """
    failures = []
    for config in candidates:
        probe = {**config, "num_stages": 1}
        m, n, k = probe["block_m"], probe["block_n"], probe["block_k"]
        try:
            a = torch.zeros((m, k), device=device).to(dtype)
            b = torch.zeros((k, n), device=device).to(dtype)
            if mode == "peak":
                c = torch.empty((2, 1, m, n), dtype=_output_dtype(dtype), device=device)
                _launch_peak(a.unsqueeze(0), _stack2(b, b).unsqueeze(1), c, 1,
                             probe, dot_input_precision)
            else:
                c = torch.empty((m, n), dtype=_output_dtype(dtype), device=device)
                _launch(a, b, c, probe, dot_input_precision)
        except triton.OutOfResources as exc:
            # A shape too large to launch says nothing about the dtype.
            failures.append(f"{m}x{n}x{k}: resource limit ({exc})")
            continue
        except Exception as exc:
            first_line = (str(exc).strip().splitlines()[0] if str(exc).strip()
                          else type(exc).__name__)
            failures.append(f"{m}x{n}x{k}: {first_line}")
            continue
        return {"passed": True, "kernel_mode": mode,
                "probed_tile": f"{m}x{n}x{k}",
                "rejected_tiles": failures,
                "description": f"{dtype_name.upper()} tl.dot lowering probe"}
    raise TensorUnsupportedError(
        f"The backend cannot lower tl.dot for {dtype_name.upper()} operands at any "
        f"tested tile shape: {'; '.join(failures)}")


def _validate_fp4_arithmetic(device, mode: str, candidates: list[dict]) -> dict:
    """Confirm the MX FP4 path computes real E2M1 products, bit for bit.

    A lowering that compiles proves nothing about what it computes. FP4 is the
    one format whose operands are carried in a container dtype (uint8), so a
    backend that ignored the E2M1 interpretation and multiplied the raw bytes,
    or dropped the scale factors, would still compile and still produce
    plausible-looking numbers. This probe multiplies known E2M1 codes and
    demands exact equality with the FP64 reference, then raises the A scale from
    2**0 to 2**1 and demands the result exactly double: that rejects a path
    which silently ignores the per-group scale.
    """
    failures = []
    for config in candidates:
        probe = {**config, "num_stages": 1}
        m, n, k = probe["block_m"], probe["block_n"], probe["block_k"]
        try:
            torch.manual_seed(0)
            a, b, c, scales, values = _fp4_peak_inputs(1, probe, device)
            scale_a, scale_b = scales
            values_a, values_b = values
            _launch_peak(a, b, c, 1, probe, None, scales)
            reference = _stack2(
                values_a[0].matmul(values_b[0, 0]), values_a[0].matmul(values_b[1, 0]),
            ).unsqueeze(1)
            error = (c.cpu().double() - reference).abs().max().item()
            if error != 0.0:
                raise TensorUnsupportedError(
                    "The backend did not compute MX FP4 tl.dot_scaled correctly: "
                    f"max_abs_error={error} against the exact E2M1 reference")
            # E8M0 code 128 encodes 2**1, so doubling A's scale must double C.
            doubled = torch.empty_like(c)
            _launch_peak(a, b, doubled, 1, probe, None,
                         (torch.full_like(scale_a, E8M0_UNIT_SCALE + 1), scale_b))
            scale_error = (doubled.cpu().double() - 2.0 * reference).abs().max().item()
            if scale_error != 0.0:
                raise TensorUnsupportedError(
                    "The backend did not apply the E8M0 scale factor: raising A's scale "
                    f"from 2**0 to 2**1 changed the result by max_abs_error={scale_error} "
                    "from the exactly doubled reference")
        except triton.OutOfResources as exc:
            failures.append(f"{m}x{n}x{k}: resource limit ({exc})")
            continue
        except TensorUnsupportedError:
            raise
        except Exception as exc:
            first_line = (str(exc).strip().splitlines()[0] if str(exc).strip()
                          else type(exc).__name__)
            failures.append(f"{m}x{n}x{k}: {first_line}")
            continue
        return {"passed": True, "kernel_mode": mode, "probed_tile": f"{m}x{n}x{k}",
                "rejected_tiles": failures, "max_abs_error": 0.0,
                "scale_sensitivity_verified": True,
                "description": "MX FP4 (E2M1 packed two per uint8, E8M0 unit scales) "
                               "tl.dot_scaled lowering, exact-product and scale-application probe"}
    raise TensorUnsupportedError(
        "The backend cannot lower tl.dot_scaled for MX FP4 (E2M1) operands at any "
        f"tested tile shape: {'; '.join(failures)}")


def _instruction_name(line: str) -> str:
    for piece in line.strip().split():
        if piece.startswith("@") or piece.endswith(":"):
            continue
        return piece.rstrip(";")
    return ""


def _amd_matrix_format(line: str, accumulator: str, operand: str) -> bool:
    """Accept LLVM's legacy K+type spelling and the normalized K_type spelling."""
    return re.fullmatch(
        rf"v_(?:mfma|wmma)_{accumulator}_\d+x\d+x\d+_?{operand}(?:_e(?:32|64))?",
        _instruction_name(line),
    ) is not None


# T-Head PPU emits no PTX: its AIU matrix ops reach LLVM IR as inline asm
# ("ppu.mma.sync.aligned.<shape>.row.col[.satfinite].<acc>.<a>.<b>.<acc> {...}"),
# so the audit reads "llir" and matches the operand/accumulator suffix the same
# way the nvidia branch matches PTX .f32.bf16.bf16 forms. The mnemonic sits
# inside the asm string rather than at the start of the line, so it is matched
# directly instead of through _instruction_name.
_PPU_MMA_RE = re.compile(r"ppu\.mma\.sync\.aligned\.[a-z0-9]+\.row\.col\.[a-z0-9.]+")
# The SIMT path reaches LLVM IR as llvm.fmuladd, scalar or vectorized
# (llvm.fmuladd.v16f32 in the peak kernel, llvm.fmuladd.f32 in the smaller
# probes), so both spellings have to count as FP32/FP64 FMA evidence.
_PPU_FMA_RE = re.compile(r"llvm\.fmuladd\.(?:v\d+)?f(32|64)\b")


def _ppu_fma_width(line: str, bits: str) -> bool:
    match = _PPU_FMA_RE.search(line)
    return match is not None and match.group(1) == bits
_PPU_MMA_FORMATS = {
    "int8": ".s32.s8.s8.s32",
    "fp8": ".f32.e4m3.e4m3.f32",
    "bf16": ".f32.bf16.bf16.f32",
    "fp16": ".f32.f16.f16.f32",
    "tf32": ".f32.tf32.tf32.f32",
    "fp32": ".f32.f32.f32.f32",
    "fp64": ".f64.f64.f64.f64",
}


def _ppu_mma_mnemonic(line: str) -> str:
    match = _PPU_MMA_RE.search(line)
    return match.group(0).rstrip(".") if match else ""


def _ppu_matrix_format(line: str, dtype_key: str) -> bool:
    return _ppu_mma_mnemonic(line).endswith(_PPU_MMA_FORMATS[dtype_key])


def _mnemonic(line: str, vendor: str) -> str:
    """Instruction name for reporting; PPU hides its mnemonic inside inline asm
    or an LLVM intrinsic call rather than at the start of the line."""
    if vendor != "thead":
        return _instruction_name(line)
    mma = _ppu_mma_mnemonic(line)
    if mma:
        return mma
    fma = _PPU_FMA_RE.search(line)
    return fma.group(0) if fma else _instruction_name(line)


def _compiled_info(compiled, vendor: str, dtype: str = "fp16",
                   dot_input_precision: str | None = None) -> dict:
    # Device type is not a vendor identifier: several FlagGems vendors expose
    # a CUDA-compatible torch module while emitting a different instruction set.
    assemblies = getattr(compiled, "asm", {}) or {}
    tensor_lines = []
    scalar_fp32_lines = []
    scalar_fp64_lines = []
    audit = "unavailable"
    execution_kind = "unverified"
    instruction_precision_verified = False
    native_tensor_verified = False
    if vendor in ("nvidia", "amd", "thead"):
        assembly = assemblies.get(
            "ptx" if vendor == "nvidia" else "amdgcn" if vendor == "amd" else "llir", "")
        if vendor == "nvidia":
            tensor_lines = [line for line in assembly.splitlines()
                            if "mma.sync." in line or "wgmma.mma_async." in line
                            or "tcgen05.mma." in line]
            scalar_fp32_lines = [line for line in assembly.splitlines()
                                 if "fma.rn.f32" in line or "mad.rn.f32" in line]
        elif vendor == "thead":
            tensor_lines = [line for line in assembly.splitlines()
                            if _PPU_MMA_RE.search(line)]
            scalar_fp32_lines = [line for line in assembly.splitlines()
                                 if _ppu_fma_width(line, "32")]
        else:
            tensor_lines = [line for line in assembly.splitlines()
                            if "v_mfma_" in line or "v_wmma_" in line]
            scalar_fp32_lines = [line for line in assembly.splitlines()
                                 if any(name in line for name in
                                        ("v_fma_f32", "v_fmac_f32", "v_mac_f32"))]

        if dtype == "fp64":
            scalar_fp64_lines = [line for line in assembly.splitlines()
                                 if ("fma.rn.f64" in line if vendor == "nvidia" else
                                     _ppu_fma_width(line, "64") if vendor == "thead" else
                                     "v_fma_f64" in line)]
            correct_tensor = (all(".f64.f64.f64.f64" in line for line in tensor_lines)
                              if vendor == "nvidia" else
                              all(_ppu_matrix_format(line, "fp64")
                                  for line in tensor_lines)
                              if vendor == "thead" else
                              all(_amd_matrix_format(line, "f64", "f64")
                                  for line in tensor_lines))
            if not correct_tensor or not (tensor_lines or scalar_fp64_lines):
                raise TensorUnsupportedError(
                    f"Cannot verify FP64 matrix or scalar FMA instructions for {vendor}")
            native_tensor_verified = bool(tensor_lines)
            execution_kind = "tensor" if tensor_lines else "simt"
            instruction_precision_verified = True
        elif dtype == "fp32":
            reduced_lines = [line for line in tensor_lines
                             if "tf32" in line.lower() or "xf32" in line.lower()]
            if dot_input_precision == "tf32":
                if not tensor_lines or len(reduced_lines) != len(tensor_lines):
                    instructions = sorted({_mnemonic(line, vendor) for line in tensor_lines})
                    raise TensorUnsupportedError(
                        f"Cannot verify native TF32 matrix instructions for {vendor}; "
                        f"compiled tensor instructions: {instructions or 'none'}."
                    )
                execution_kind = "tensor"
                native_tensor_verified = True
            else:
                if reduced_lines:
                    instructions = sorted({_mnemonic(line, vendor) for line in reduced_lines})
                    raise TensorUnsupportedError(
                        f"Requested IEEE FP32 but found reduced-precision instructions: {instructions}."
                    )
                if vendor == "nvidia":
                    invalid_tensor_lines = tensor_lines
                elif vendor == "thead":
                    invalid_tensor_lines = [
                        line for line in tensor_lines
                        if not _ppu_matrix_format(line, "fp32")
                    ]
                else:
                    invalid_tensor_lines = [
                        line for line in tensor_lines
                        if not _amd_matrix_format(line, "f32", "f32")
                    ]
                if invalid_tensor_lines:
                    instructions = sorted(
                        {_mnemonic(line, vendor) for line in invalid_tensor_lines}
                    )
                    raise TensorUnsupportedError(
                        f"Requested IEEE FP32 but found non-FP32 matrix instructions: "
                        f"{instructions}."
                    )
                if not tensor_lines and not scalar_fp32_lines:
                    raise TensorUnsupportedError(
                        f"No recognized IEEE FP32 matrix or scalar FMA instructions for {vendor}."
                    )
                execution_kind = "tensor" if tensor_lines else "simt"
                native_tensor_verified = bool(tensor_lines)
            instruction_precision_verified = True
        else:
            if not tensor_lines:
                raise TensorUnsupportedError(
                    f"No recognized {vendor} tensor MMA instructions in the compiled kernel"
                )
            native_tensor_verified = True
            execution_kind = "tensor"
            if dtype == "fp8":
                native_format = (all(".f32.e4m3.e4m3" in line for line in tensor_lines)
                              if vendor == "nvidia" else
                              all(_ppu_matrix_format(line, "fp8") for line in tensor_lines)
                              if vendor == "thead" else
                              all("fp8_fp8" in line for line in tensor_lines))
            elif dtype == "int8":
                native_format = (all(".s32.s8.s8" in line for line in tensor_lines)
                              if vendor == "nvidia" else
                              all(_ppu_matrix_format(line, "int8") for line in tensor_lines)
                              if vendor == "thead" else all(
                                  _amd_matrix_format(line, "i32", "i8")
                                  for line in tensor_lines))
            elif dtype == "bf16":
                native_format = (all(".f32.bf16.bf16" in line for line in tensor_lines)
                                 if vendor == "nvidia" else
                                 all(_ppu_matrix_format(line, "bf16") for line in tensor_lines)
                                 if vendor == "thead" else
                                 all("bf16" in line for line in tensor_lines))
            elif dtype == "fp16":
                native_format = (all(".f32.f16.f16" in line for line in tensor_lines)
                                 if vendor == "nvidia" else
                                 all(_ppu_matrix_format(line, "fp16") for line in tensor_lines)
                                 if vendor == "thead" else
                                 all("f16" in line and "bf16" not in line for line in tensor_lines))
            else:
                native_format = True
            if not native_format:
                instructions = sorted({_mnemonic(line, vendor) for line in tensor_lines})
                raise TensorUnsupportedError(
                    f"Cannot verify native {dtype} MMA for {vendor}; compiled instructions: "
                    f"{instructions}. The selected target may not support this format "
                    "or this compiler may lower it to another dtype."
                )
            instruction_precision_verified = True
        audit = "verified"
    instructions = sorted({_mnemonic(line, vendor) for line in tensor_lines})
    scalar_fp32_instructions = sorted({_mnemonic(line, vendor) for line in scalar_fp32_lines})
    if audit == "verified" and execution_kind == "simt":
        note = f"{dtype.upper()} execution verified as scalar/vector FMA; no native tensor MMA"
    elif audit == "verified":
        note = None
    else:
        note = (f"No instruction auditor for FlagGems vendor {vendor}; "
                "execution unit and instruction precision are unverified")
    return {"registers_per_thread": getattr(compiled, "n_regs", None),
            "spill_count": getattr(compiled, "n_spills", None),
            "shared_memory_bytes": getattr(getattr(compiled, "metadata", None), "shared", None),
            "assembly_audit": audit, "execution_kind": execution_kind,
            "instruction_precision_verified": instruction_precision_verified,
            "native_tensor_verified": native_tensor_verified,
            "assembly_audit_note": note,
            "tensor_instruction_types": instructions,
            "static_tensor_instruction_count": len(tensor_lines) if audit == "verified" else None,
            "fp32_fma_instruction_types": scalar_fp32_instructions,
            "fp64_fma_instruction_types": sorted({_mnemonic(line, vendor) for line in scalar_fp64_lines}),
            "static_fp32_fma_instruction_count": len(scalar_fp32_lines)
            if audit == "verified" else None}


def run_peak(args, metadata: dict, dtype: torch.dtype, device) -> None:
    vendor = metadata["vendor_name"]
    if not metadata.get("sm_count"):
        raise RuntimeError(
            f"The {vendor} runtime did not report the compute-unit count; "
            "provide --compute-units for the resident tensor peak grid."
        )
    candidates = _peak_configs(vendor, args.dtype, args.input_precision)
    if args.quick:
        candidates = candidates[:1]
        args.blocks_per_sm = [1]
        args.iterations = min(args.iterations, 64)
    blocks_per_sm_values = list(args.blocks_per_sm)
    if vendor == "thead" and blocks_per_sm_values == [1, 2, 4, 8]:
        # Default sweep: the PPU AIU keeps gaining with more resident CTAs
        # per SM and plateaus around 16-32 blocks/SM (measured on
        # PPU-ZW810E), so extend the default occupancy sweep for this vendor.
        blocks_per_sm_values += [16, 32, 64]
        if args.dtype == "int8":
            blocks_per_sm_values += [128]
    precision_text = (f"FP32 storage, {args.input_precision.upper()} dot precision"
                      if args.dtype == "fp32" else
                      "MX FP4 (E2M1) operands with E8M0 group scales"
                      if args.dtype == "fp4" else f"{args.dtype} operands")
    accumulator = _accumulator_name(args.dtype)
    unit = "TOPS" if args.dtype == "int8" else "TFLOP/s"
    print(f"\nMatrix peak: resident {precision_text}, two independent {accumulator.upper()} accumulator chains")
    if args.dtype == "fp8":
        print("FP8 E4M3FN: backend accumulation policy (internal precision is hardware-dependent); "
              "exact-check inputs")
    if args.dtype == "fp4":
        print(f"MX FP4 E2M1 via tl.dot_scaled: {FP4_PACKING_FACTOR} elements per uint8 container, "
              f"one E8M0 scale per {MX_GROUP_SIZE} elements along K, all scales 2**0; "
              "exact-check inputs. tile K counts E2M1 elements, not container bytes.")
    if vendor not in ("nvidia", "amd", "thead"):
        print(f"Instruction audit unavailable for {vendor}; native tensor execution is unverified.")
    print(f"Iterations={args.iterations}; operations=2*tile_M*tile_N*tile_K*iterations*chains*blocks")
    print(f" tile M/N/K  blocks/SM  warps  median ms    min ms  median {unit}  peak {unit}  regs")
    rows = []
    for config in candidates:
        for blocks_per_sm in blocks_per_sm_values:
            blocks = blocks_per_sm * metadata["sm_count"]
            m, n, k = (config[f"block_{axis}"] for axis in ("m", "n", "k"))
            free_bytes = available_memory_bytes()
            if free_bytes is not None and blocks * (m * k + 2 * k * n + 8 * m * n) * 8 > free_bytes * 0.8:
                raise RuntimeError("Peak validation exceeds available memory; reduce --blocks-per-sm")
            try:
                scales = None
                if args.dtype == "fp4":
                    a, b, c, scales, fp4_values = _fp4_peak_inputs(blocks, config, device)
                else:
                    a, b, c = _peak_inputs(blocks, config, dtype, device)
                c.fill_(-2**31 if dtype == torch.int8 else float("nan"))
                compiled = _launch_peak(
                    a, b, c, args.iterations, config, args.input_precision, scales)
                info = _compiled_info(
                    compiled, vendor, args.dtype, args.input_precision)
            except triton.OutOfResources as exc:
                print(f"  config={config}: skipped (resource limit: {exc})")
                rows.append({"mode": "peak", "config": config, "blocks_per_sm": blocks_per_sm,
                             "status": "skipped", "reason": str(exc)})
                continue
            except Exception as exc:
                reason = _unsupported_reason(exc)
                if reason is None:
                    raise
                rows.append({"mode": "peak", "dtype": args.dtype, "config": config,
                             "blocks_per_sm": blocks_per_sm,
                             "status": "unsupported", "reason": reason})
                print(f"  config={config}: unsupported ({reason})")
                continue
            if args.dtype == "fp4":
                # The reference uses the FP64 values of the codes that were
                # actually packed into the operands, so a packing mistake cannot
                # cancel itself out between kernel and reference. Every exact
                # entry is a multiple of 0.25 bounded by block_k, so narrowing
                # the FP64 reference to the accumulator's FP32 is lossless.
                values_a, values_b = fp4_values
                reference = _stack2(values_a.matmul(values_b[0]),
                                    values_a.matmul(values_b[1]))
                reference = reference.to(torch.float32).to(c.device)
            else:
                reference = _stack2(_reference(a, b[0]), _reference(a, b[1]))
            reference.mul_(args.iterations)
            correctness = _validate_peak(c, reference, args.dtype, k, args.iterations)
            if info["spill_count"]:
                rows.append({"mode": "peak", "dtype": args.dtype, "config": config,
                             "blocks_per_sm": blocks_per_sm, "status": "skipped",
                             "reason": "configuration spills registers"})
                del a, b, c, reference
                continue
            timing = measure(lambda: _launch_peak(
                a, b, c, args.iterations, config, args.input_precision, scales), args)
            operations = 2 * m * n * k * args.iterations * config["chains"] * blocks
            rates = _rates(operations, timing, args.dtype)
            descriptor = _precision_descriptor(args.dtype, args.input_precision)
            if args.dtype in ("fp32", "fp64"):
                measurement_kind = (f"{descriptor['math_mode']}_tensor_peak"
                                    if info["native_tensor_verified"] else
                                    f"{descriptor['math_mode']}_matrix_throughput")
            else:
                measurement_kind = ("native_tensor_peak" if info["native_tensor_verified"]
                                    else "tl_dot_throughput_unverified")
            row = {"mode": "peak", "dtype": args.dtype, "config": config,
                   "measurement_kind": measurement_kind,
                   "execution_kind": info["execution_kind"],
                   "precision_verified": bool(metadata.get("tensor_precision_probe"))
                   if args.dtype in ("fp32", "fp64") else info["instruction_precision_verified"],
                   "accumulator_dtype": accumulator, "output_dtype": accumulator, "blocks": blocks,
                   "blocks_per_sm": blocks_per_sm, "iterations": args.iterations,
                   **rates,
                   **descriptor,
                   "input_pattern": "random one-hot A; random signed B; dense tl.dot operations"
                   if args.dtype == "fp8" else "random ternary integers" if args.dtype == "int8"
                   else "random E2M1 codes for +/-0.5 and +/-1.0, packed two per uint8; "
                        "unit E8M0 scales; dense tl.dot_scaled operations"
                   if args.dtype == "fp4"
                   else "random ternary exact binary fractions",
                   "accumulation_policy": "backend FP8 accumulation; internal precision is hardware-dependent"
                   if args.dtype == "fp8" else
                   "MX E2M1 multiply with E8M0 group scaling and FP32 accumulation"
                   if args.dtype == "fp4" else
                   "TF32 multiply with FP32 accumulation" if args.input_precision == "tf32" else
                   "IEEE FP32 multiply and accumulation" if args.dtype == "fp32" else
                   "exact INT32 accumulation (no overflow)" if args.dtype == "int8" else
                   "FP64 multiply and accumulation" if args.dtype == "fp64" else
                   "FP32 tensor accumulation",
                   "precision_probe": metadata.get("tensor_precision_probe"),
                   "correctness": correctness,
                   "compiled": info, "status": "ok", **timing}
            rows.append(row)
            tile = f"{m}/{n}/{k}"
            registers = info["registers_per_thread"]
            registers_text = "?" if registers is None else str(registers)
            print(f"{tile:>11}  {blocks_per_sm:9d}  {config['num_warps']:5d} "
                  f"{timing['median_ms']:10.4f}  {timing.get('min_ms', timing['median_ms']):8.4f} "
                  f"{rates['median_teraops_per_second']:12.2f} "
                  f"{rates['peak_teraops_per_second']:10.2f} {registers_text:>5}", flush=True)
            del a, b, c, reference
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        _emit_tensor_results(metadata, args, rows)
        return
    best = max(successful, key=lambda row: row["peak_teraops_per_second"])
    if args.dtype == "fp32" and args.input_precision == "ieee":
        label = "Measured IEEE FP32 matrix throughput"
    elif args.dtype == "fp32":
        label = ("Measured native TF32 tensor peak"
                 if best["compiled"]["native_tensor_verified"] else
                 "Measured TF32 matrix throughput (execution unit unverified)")
    elif args.dtype == "fp64":
        label = ("Measured native FP64 tensor peak" if best["compiled"]["native_tensor_verified"]
                 else "Measured FP64 matrix throughput")
    else:
        label = ("Measured tensor peak" if best["compiled"]["native_tensor_verified"] else
                 "Measured tl.dot throughput (native execution unverified)")
    print(f"{label}: {best['peak_teraops_per_second']:.2f} {unit} "
          f"(same configuration median {best['median_teraops_per_second']:.2f} {unit}; "
          f"best observed timing sample, dense {best['math_mode']})")
    _emit_tensor_results(metadata, args, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("peak", "gemm"), default="peak",
                        help="resident matrix-compute peak (default), or application-style GEMM")
    parser.add_argument("--iterations", type=int, default=1024,
                        help="resident MMA repeats, 1..262144; FP8 maximum 4096 (default: 1024)")
    parser.add_argument("--blocks-per-sm", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="launched CTAs per SM/CU (default: 1 2 4 8)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2048, 4096, 8192],
                        help="square GEMM dimensions M=N=K (default: 2048 4096 8192)")
    parser.add_argument("--dtype",
                        choices=("fp16", "bf16", "fp8", "fp4", "fp32", "tf32", "int8", "fp64"),
                        default="fp16",
                        help="matrix input format; TF32 uses FP32 storage; FP8 is E4M3FN "
                             "and FP4 is OCP MX E2M1 via tl.dot_scaled (both peak only)")
    parser.add_argument("--input-precision", choices=("ieee", "tf32"),
                        help="tl.dot math mode for --dtype fp32 (default: ieee); TF32 requires backend support")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: one tile and 64 repeats, or GEMM dimensions 256/513")
    add_common_arguments(parser)
    return parser


def _backend_fp8_dtypes() -> tuple[str, ...] | None:
    """The fp8 types the active Triton backend declares it can lower.

    Returns None when the backend does not publish the list, which is not
    evidence either way.
    """
    try:
        backend = triton.compiler.compiler.make_backend(
            triton.runtime.driver.active.get_current_target())
        options = backend.parse_options({})
    except Exception:
        return None
    declared = getattr(options, "supported_fp8_dtypes", None)
    return None if declared is None else tuple(declared)


def _xpu_int8_dot_reason(device) -> str | None:
    """Check whether this backend can run tl.dot on INT8 operands at all.

    The probe is deliberately tiny and runs before any large allocation: a
    failure leaves the device context unusable, so nothing may depend on the
    device afterwards.
    """
    size = 32
    a = torch.randint(-1, 2, (size, size), device=device, dtype=torch.int8)
    b = torch.randint(-1, 2, (size, size), device=device, dtype=torch.int8)
    c = torch.zeros((size, size), device=device, dtype=torch.int32)
    try:
        _int8_dot_probe_kernel[(1,)](a, b, c, M=size, N=size, K=size)
        torch.cuda.synchronize(device)
    except Exception as exc:
        summary = str(exc).strip().splitlines()
        detail = summary[0][:160] if summary else type(exc).__name__
        return ("The FlagTree XPU backend does not support tl.dot with INT8 operands: "
                f"a minimal {size}x{size}x{size} INT8 dot fails ({detail}). "
                "This is a backend limitation, not a determination of chip support")
    if not torch.equal(c, torch.matmul(a.float(), b.float()).to(torch.int32)):
        return ("The FlagTree XPU backend returns incorrect results for tl.dot with INT8 "
                "operands; no verified native INT8 matrix path is available. "
                "This is a backend limitation, not a determination of chip support")
    return None


def _capability_reason(args, metadata: dict) -> str | None:
    if getattr(torch, _dtype_name(args.dtype), None) is None:
        return f"This PyTorch build does not expose {_dtype_name(args.dtype)}"
    if args.dtype == "bf16" and metadata.get("supports_bf16") is False:
        return f"FlagGems reports no BF16 support for {metadata['vendor_name']}"
    if args.dtype == "fp64" and metadata.get("supports_fp64") is False:
        return f"FlagGems reports no FP64 support for {metadata['vendor_name']}"
    if args.dtype == "fp8" and args.mode != "peak":
        return "This benchmark implements FP8 E4M3FN in --mode peak only"
    if args.dtype == "fp4" and args.mode != "peak":
        return "This benchmark implements MX FP4 (E2M1) in --mode peak only"
    if args.dtype == "fp4" and not hasattr(tl, "dot_scaled"):
        # FP4 operands are only defined for the microscaled dot: OCP MX pairs
        # every group of elements with a shared scale, and there is no unscaled
        # tl.dot overload for a 4-bit format.
        return ("This Triton build does not provide tl.dot_scaled, which MX FP4 (E2M1) "
                "requires. This is a toolchain limitation, not a determination of chip "
                "support")
    if (args.dtype == "fp8" and is_flagtree_xpu(metadata)
            and _backend_fp8_dtypes() == ()):
        # tl.dot rejects fp8 operands during compilation when the backend
        # declares no fp8 type, so no FP8 matrix rate can be measured here.
        # Other vendors reach their own storage and instruction probes, which
        # give a more specific diagnosis than this declaration check.
        return (f"The Triton backend for {metadata['vendor_name']} declares no supported FP8 "
                "dtypes, so tl.dot cannot accept FP8 operands. This is a backend limitation, "
                "not a determination of chip support")
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _normalize_precision_args(parser, args)
    if any(size <= 0 for size in args.sizes):
        parser.error("--sizes values must be positive")
    if not 1 <= args.iterations <= 262144:
        parser.error("--iterations must be between 1 and 262144 for exact FP32 validation")
    if any(value <= 0 for value in args.blocks_per_sm):
        parser.error("--blocks-per-sm values must be positive")
    if args.dtype == "fp8" and args.iterations > 4096:
        parser.error("native FP8 peak limits --iterations to 4096 for exact internal accumulation")
    if args.dtype == "fp4" and args.iterations > 4096:
        parser.error("MX FP4 peak limits --iterations to 4096 for exact FP32 accumulation")
    if args.dtype == "int8" and args.mode == "gemm" and any(size >= 2**24 for size in args.sizes):
        parser.error("INT8 GEMM requires K < 2**24 for an exact FP32 reference")
    if args.quick:
        args.sizes = [256, 513]
    metadata = setup_device(args)
    metadata.update(_precision_descriptor(args.dtype, args.input_precision))
    reason = _capability_reason(args, metadata)
    if reason is not None:
        emit_unsupported("tensor", metadata, args, reason)
        return
    dtype_name = _dtype_name(args.dtype)
    dtype = getattr(torch, dtype_name)
    device = get_device(args)
    torch.manual_seed(0)
    # A failing INT8 dot faults the device context, so probe before allocating
    # operands and report the backend limitation instead of a runtime error.
    if args.dtype == "int8" and is_flagtree_xpu(metadata):
        reason = _xpu_int8_dot_reason(device)
        if reason is not None:
            emit_unsupported("tensor", metadata, args, reason)
            return
    try:
        if args.dtype == "fp32":
            metadata["tensor_precision_probe"] = _validate_fp32_precision(
                device, args.input_precision, args.mode)
        elif args.dtype == "fp64":
            metadata["tensor_precision_probe"] = _validate_fp64_precision(device, args.mode)
        elif args.dtype == "fp8":
            metadata["tensor_storage_probe"] = _validate_fp8_storage(device, dtype)
            # Storage alone does not imply a dot lowering; probe before sweeping.
            shapes = (_peak_configs(metadata["vendor_name"], args.dtype, args.input_precision)
                      if args.mode == "peak"
                      else _configs(metadata["vendor_name"], args.dtype))
            metadata["tensor_dot_lowering_probe"] = _validate_dot_lowering(
                device, dtype, args.dtype, args.mode, shapes, args.input_precision)
        elif args.dtype == "fp4":
            # Compiling is not enough for FP4: the operands ride in a uint8
            # container, so a backend that ignored the E2M1 interpretation or
            # the E8M0 scales would still compile. Probe arithmetic exactness
            # and scale application before sweeping.
            metadata["tensor_dot_lowering_probe"] = _validate_fp4_arithmetic(
                device, args.mode,
                _peak_configs(metadata["vendor_name"], args.dtype, args.input_precision))
    except Exception as exc:
        reason = _unsupported_reason(exc)
        if reason is None:
            raise
        emit_unsupported("tensor", metadata, args, reason)
        return
    if args.mode == "peak":
        run_peak(args, metadata, dtype, device)
        return
    run_gemm(args, metadata, dtype, device)


def run_gemm(args, metadata: dict, dtype: torch.dtype, device) -> None:
    candidates = _configs(metadata["vendor_name"], args.dtype)
    if args.quick:
        candidates = candidates[:1]

    precision_text = (f"FP32 storage, {args.input_precision.upper()} dot precision"
                      if args.dtype == "fp32" else f"{args.dtype} inputs")
    accumulator = _accumulator_name(args.dtype)
    unit = "TOPS" if args.dtype == "int8" else "TFLOP/s"
    print(f"\nMatrix GEMM: {precision_text}, {accumulator.upper()} accumulate/output; 2*M*N*K operations")
    print(f"  N        tile M/N/K    warps stages   median ms  median {unit}  peak {unit}   max abs error")
    rows = []
    for size in args.sizes:
        # Inputs, output, FP32 reference and conversion/validation temporaries.
        # Query currently available memory rather than total device capacity.
        required = size * size * (64 if args.dtype == "fp64" else 32)
        free_bytes = available_memory_bytes()
        if free_bytes is not None and required > free_bytes * 0.8:
            raise RuntimeError(
                f"N={size} validation needs about {required / 2**30:.2f} GiB; "
                f"only {free_bytes / 2**30:.2f} GiB is currently free. "
                "Reduce --sizes."
            )
        try:
            a, b, c = _inputs(size, size, size, dtype, device)
        except Exception as exc:
            reason = _unsupported_reason(exc)
            if reason is None:
                raise
            rows.append({"mode": "gemm", "dtype": args.dtype, "m": size, "n": size, "k": size,
                         "status": "unsupported", "reason": reason})
            continue
        reference = _reference(a, b)
        for config in candidates:
            try:
                c.fill_(-2**31 if dtype == torch.int8 else float("nan"))
                compiled = _launch(
                    a, b, c, config, args.input_precision)  # Compile before validation and timing.
                info = _compiled_info(
                    compiled, metadata["vendor_name"], args.dtype, args.input_precision)
            except triton.OutOfResources as exc:
                print(f"  N={size} config={config}: skipped (resource limit: {exc})")
                rows.append({"m": size, "n": size, "k": size, "dtype": args.dtype,
                             "config": config, "status": "skipped", "reason": str(exc)})
                continue
            except Exception as exc:
                reason = _unsupported_reason(exc)
                if reason is None:
                    raise
                rows.append({"mode": "gemm", "m": size, "n": size, "k": size,
                             "dtype": args.dtype, "config": config,
                             "status": "unsupported", "reason": reason})
                print(f"  N={size} config={config}: unsupported ({reason})")
                continue
            correctness = _validate(c, reference, args.input_precision)
            timing = measure(lambda: _launch(a, b, c, config, args.input_precision), args)
            rates = _rates(2 * size**3, timing, args.dtype)
            descriptor = _precision_descriptor(args.dtype, args.input_precision)
            measurement_kind = (f"{descriptor['math_mode']}_gemm"
                                if args.dtype in ("fp32", "fp64") else "dense_tensor_gemm")
            row = {"mode": "gemm", "m": size, "n": size, "k": size, "dtype": args.dtype,
                   "measurement_kind": measurement_kind,
                   "execution_kind": info["execution_kind"],
                   "precision_verified": bool(metadata.get("tensor_precision_probe"))
                   if args.dtype in ("fp32", "fp64") else info["instruction_precision_verified"],
                   "accumulator_dtype": accumulator, "output_dtype": accumulator,
                   **descriptor,
                   **rates, "config": config, "status": "ok",
                   "correctness": correctness,
                   "precision_probe": metadata.get("tensor_precision_probe"),
                   "compiled": info, **timing}
            rows.append(row)
            tile = f"{config['block_m']}/{config['block_n']}/{config['block_k']}"
            print(f"{size:5d}   {tile:>14}   {config['num_warps']:5d} "
                  f"{config['num_stages']:6d}   {timing['median_ms']:9.4f} "
                  f"{rates['median_teraops_per_second']:11.2f} "
                  f"{rates['peak_teraops_per_second']:10.2f}   {correctness['max_abs_error']:.3g}", flush=True)
        del a, b, c, reference

    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        _emit_tensor_results(metadata, args, rows)
        return
    best = max(successful, key=lambda row: row["median_teraops_per_second"])
    print(f"Best measured matrix throughput: {best['median_teraops_per_second']:.2f} {unit} "
          f"(N={best['n']}, {best['math_mode']}; sustained dense GEMM)")
    _emit_tensor_results(metadata, args, rows)


if __name__ == "__main__":
    main()
