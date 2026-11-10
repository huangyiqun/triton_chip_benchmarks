#!/usr/bin/env python3
"""Measure typed vector arithmetic throughput; report unsupported types explicitly."""

import os

# Enable the KunlunXin KL3 event backend before accelerator packages are loaded.
os.environ.setdefault("XPU_EVENT_KL3_ENABLE", "1")
# FlagTree's hcu (Hygon DCU) backend lowers masked loads to buffer resources via
# llvm.amdgcn.make.buffer.rsrc, which the clang-18 shipped in DTK 26.04 rejects
# ("Intrinsic has incorrect argument type"). Every masked tl.load in this suite
# hits it, so the default would fail the whole table at compile time rather than
# measure anything. Buffer ops are an addressing optimization, not an
# arithmetic path: disabling them changes how operands are fetched, not which
# matrix or FMA instruction executes, and the instruction audits still verify
# the arithmetic that ran. An explicit user setting is preserved, so the knob
# can be turned back on once a fixed toolchain is installed.
os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")

import argparse
import math
import re

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

from common import (add_common_arguments, available_memory_bytes, emit_not_implemented,
                    emit_results, emit_unsupported, get_device, get_device_api,
                    is_flagtree_xpu, measure, positive_int, reference_device, setup_device)


CHAINS = 8
DEFAULT_ITERATIONS = 4096

# Types whose loop-carried state is narrower than 32 bits. The FlagTree XPU
# compiler cannot lower an scf.for that carries them, so these run fully
# unrolled; see FULLY_STATIC in vector_fma_kernel.
XPU_NARROW_DTYPES = ("fp16", "bf16", "int8")

# A fully unrolled body makes compile time grow with the iteration count: at
# the 4096 default, FP16 takes ~101 s and INT8 ~230 s per configuration on a
# P800, which exceeds the suite's per-metric timeout once a sweep compiles
# several. 512 keeps compiles near 6 s and still measures a steady-state rate
# within ~15% of the longer run.
XPU_STATIC_ITERATIONS = 512
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
          "int8": torch.int8, "fp64": torch.float64,
          "fp8": getattr(torch, "float8_e4m3fn", None),
          # FP4 has no arithmetic scalar dtype anywhere in this stack. PyTorch
          # exposes float4_e2m1fn_x2, which is a *container* holding two E2M1
          # elements per byte rather than a scalar type, and Triton has no FP4
          # element type at all. It is listed so --dtype fp4 is a recognized
          # request that reports a documented status instead of a KeyError.
          "fp4": getattr(torch, "float4_e2m1fn_x2", None)}

# Narrow float types that a backend may only be able to compute by extending to
# FP32. For these, an explicitly widened FMA keeping the narrow state is a valid
# fallback when tl.fma does not lower; see backend_supports_widened_fma.
WIDENED_FMA_DTYPES = ("bf16", "fp16")

# Typed INT8 arithmetic needs the explicit builtins to pass sanitize_overflow.
# Some Triton/FlagTree versions do not re-export every one of them from
# triton.language, so fall back to the same builtins in triton.language.core.
_TL_MUL = getattr(tl, "mul", None) or tl.core.mul
_TL_ADD = getattr(tl, "add", None) or tl.core.add

# A vector benchmark must not execute on the matrix unit. The PPU spells its
# AIU matrix ops as ppu.mma.* inline asm in LLVM IR, the way NVIDIA spells
# them mma./wgmma./tcgen05. in PTX.
_PPU_VECTOR_MMA_RE = re.compile(r"ppu\.mma\.")
# FlagGems vendors whose Triton backend emits final AMDGCN assembly. Hygon's
# DCU is a separate vendor with its own ISA dialect, but it compiles through
# the same AMDGCN path and publishes the assembly under the same key, so its
# vector arithmetic can be audited against real ISA rather than left unverified.
VECTOR_AMDGCN_VENDORS = ("amd", "hygon")


class UnsupportedVectorDtype(RuntimeError):
    """The requested arithmetic cannot be provided by this benchmark/backend."""


class NotImplementedVectorDtype(RuntimeError):
    """This benchmark lacks a native path for the requested arithmetic."""


@triton.jit
def _float_update(x, a, b, USE_EXPLICIT_FMA: tl.constexpr):
    if USE_EXPLICIT_FMA:
        return tl.fma(x, a, b)
    return x * a + b


@triton.jit
def _widened_fma_update(x, a, b):
    # `a` and `b` arrive already widened, so their conversions stay outside the
    # loop. Only the accumulator is widened per iteration, and the result is
    # rounded back to the narrow type so the state keeps that dtype. One
    # rounding per update, matching tl.fma on backends that lower it natively.
    return tl.fma(x.to(tl.float32), a, b).to(x.dtype)


@triton.jit
def _fma_probe_kernel(Input, Output, BLOCK_SIZE: tl.constexpr):
    # Operands are loaded at runtime so the probe cannot be constant-folded
    # away before the backend has to lower a real fused multiply-add.
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(Input + offsets)
    a = tl.load(Input + offsets + BLOCK_SIZE)
    b = tl.load(Input + offsets + 2 * BLOCK_SIZE)
    tl.store(Output + offsets, tl.fma(x, a, b))


@triton.jit
def _fp8_probe_kernel(X, A, B, Output, BLOCK_SIZE: tl.constexpr):
    # Operands are loaded at runtime so the multiply-add cannot be constant
    # folded before the backend has to lower real FP8 arithmetic.
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    tl.store(Output + offsets, tl.fma(x, a, b))


@triton.jit
def _widened_fma_probe_kernel(Input, Output, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(Input + offsets)
    a = tl.load(Input + offsets + BLOCK_SIZE).to(tl.float32)
    b = tl.load(Input + offsets + 2 * BLOCK_SIZE).to(tl.float32)
    tl.store(Output + offsets, _widened_fma_update(x, a, b))


@triton.jit
def _dp4a_update(x, a, b, GCN: tl.constexpr = False):
    # The same four-product-plus-four-accumulate operation has two spellings.
    # NVIDIA's PTX names it dp4a; the AMDGCN path spells it v_dot4_i32_i8 and
    # needs VGPR operands ("v") rather than the scalar-class "r" constraint.
    if GCN:
        return tl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3;", constraints="=v,v,v,v",
            args=[x, a, b], dtype=tl.int32, is_pure=True, pack=1)
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, $3;", constraints="=r,r,r,r",
        args=[x, a, b], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _dp4a_probe_kernel(Input, Output, BLOCK_SIZE: tl.constexpr,
                       GCN: tl.constexpr = False):
    # Operands load at runtime so the probe cannot be constant folded before
    # the backend has to emit a real DP4A / v_dot4_i32_i8.
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(Input + offsets)
    a = tl.load(Input + offsets + BLOCK_SIZE)
    b = tl.load(Input + offsets + 2 * BLOCK_SIZE)
    tl.store(Output + offsets, _dp4a_update(x, a, b, GCN))


@triton.jit
def _int8_mul_add_update(x, a, b):
    # XPU has a distinct typed INT8 path, without packed INT32 emulation.
    # Overflow is intentional modulo 256. The XPU default sanitizer would
    # otherwise widen the computation and insert overflow assertions.
    product = _TL_MUL(x, a, sanitize_overflow=False).to(tl.int8)
    return _TL_ADD(product, b, sanitize_overflow=False).to(tl.int8)


@triton.jit
def _chain_step(x, a, b, IS_INT8: tl.constexpr, USE_DP4A: tl.constexpr,
                USE_EXPLICIT_FMA: tl.constexpr, USE_WIDENED_FMA: tl.constexpr = False,
                GCN_DP4A: tl.constexpr = False):
    if IS_INT8 and USE_DP4A:
        return _dp4a_update(x, a, b, GCN_DP4A)
    if IS_INT8:
        return _int8_mul_add_update(x, a, b)
    if USE_WIDENED_FMA:
        return _widened_fma_update(x, a, b)
    return _float_update(x, a, b, USE_EXPLICIT_FMA)


@triton.jit
def vector_fma_kernel(Input, Output, Coefficients, ELEMENTS: tl.constexpr,
                      ITERATIONS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                      UNROLL: tl.constexpr, USE_EXPLICIT_FMA: tl.constexpr,
                      IS_INT8: tl.constexpr = False, USE_DP4A: tl.constexpr = False,
                      FULLY_STATIC: tl.constexpr = False,
                      USE_WIDENED_FMA: tl.constexpr = False,
                      BROADCAST_COEFFICIENTS: tl.constexpr = False,
                      GCN_DP4A: tl.constexpr = False):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Runtime input loads keep all lanes and chains independent. Repeating a
    # formula every warp/program could let the compiler merge equal chains.
    x0 = tl.load(Input + offsets + 0 * ELEMENTS)
    x1 = tl.load(Input + offsets + 1 * ELEMENTS)
    x2 = tl.load(Input + offsets + 2 * ELEMENTS)
    x3 = tl.load(Input + offsets + 3 * ELEMENTS)
    x4 = tl.load(Input + offsets + 4 * ELEMENTS)
    x5 = tl.load(Input + offsets + 5 * ELEMENTS)
    x6 = tl.load(Input + offsets + 6 * ELEMENTS)
    x7 = tl.load(Input + offsets + 7 * ELEMENTS)
    if BROADCAST_COEFFICIENTS:
        # Read each loop-invariant coefficient from its own BLOCK_SIZE-wide
        # run in an expanded buffer, instead of as a scalar. The host fills
        # every run with a single repeated value, so the arithmetic below is
        # unchanged; only the shape of the load differs.
        #
        # The Ascend bishengir compiler aborts with "LLVM ERROR: unsupported
        # type cast" on FP8 whenever a coefficient reaches elementwise
        # arithmetic at scalar shape. Re-reading one address as a vector
        # (`Coefficients + (lanes - lanes)`) hits the same abort, so the
        # operand has to come from genuinely contiguous memory. These loads
        # stay outside the timed loop, so the measured work is the same.
        lanes = tl.arange(0, BLOCK_SIZE)
        a = tl.load(Coefficients + 0 * BLOCK_SIZE + lanes)
        b0 = tl.load(Coefficients + 1 * BLOCK_SIZE + lanes)
        b1 = tl.load(Coefficients + 2 * BLOCK_SIZE + lanes)
        b2 = tl.load(Coefficients + 3 * BLOCK_SIZE + lanes)
        b3 = tl.load(Coefficients + 4 * BLOCK_SIZE + lanes)
        b4 = tl.load(Coefficients + 5 * BLOCK_SIZE + lanes)
        b5 = tl.load(Coefficients + 6 * BLOCK_SIZE + lanes)
        b6 = tl.load(Coefficients + 7 * BLOCK_SIZE + lanes)
        b7 = tl.load(Coefficients + 8 * BLOCK_SIZE + lanes)
    else:
        a = tl.load(Coefficients)
        b0 = tl.load(Coefficients + 1)
        b1 = tl.load(Coefficients + 2)
        b2 = tl.load(Coefficients + 3)
        b3 = tl.load(Coefficients + 4)
        b4 = tl.load(Coefficients + 5)
        b5 = tl.load(Coefficients + 6)
        b6 = tl.load(Coefficients + 7)
        b7 = tl.load(Coefficients + 8)

    if USE_WIDENED_FMA:
        # Widen the loop-invariant coefficients once, here. A backend that
        # emulates narrow arithmetic by extending to FP32 would otherwise
        # re-extend them on every iteration; the MACA backend does not hoist
        # those conversions itself. See flagtree_metax_bug.md.
        a = a.to(tl.float32)
        b0 = b0.to(tl.float32)
        b1 = b1.to(tl.float32)
        b2 = b2.to(tl.float32)
        b3 = b3.to(tl.float32)
        b4 = b4.to(tl.float32)
        b5 = b5.to(tl.float32)
        b6 = b6.to(tl.float32)
        b7 = b7.to(tl.float32)

    if FULLY_STATIC:
        # No scf.for at all. The FlagTree XPU compiler fails to lower a loop
        # that carries sub-32-bit values across iterations (f16/bf16/i8 hit
        # arith.mulf/arith.muli/scf.for type mismatches), so the only form it
        # accepts for those types is a fully unrolled body. Code size grows
        # with ITERATIONS, which is why the caller lowers the iteration count.
        for _static in tl.static_range(ITERATIONS):
            x0 = _chain_step(x0, a, b0, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x1 = _chain_step(x1, a, b1, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x2 = _chain_step(x2, a, b2, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x3 = _chain_step(x3, a, b3, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x4 = _chain_step(x4, a, b4, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x5 = _chain_step(x5, a, b5, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x6 = _chain_step(x6, a, b6, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
            x7 = _chain_step(x7, a, b7, IS_INT8, USE_DP4A, USE_EXPLICIT_FMA, USE_WIDENED_FMA, GCN_DP4A)
    else:
        # Explicitly unroll a small group of iterations to amortize loop control.
        # The CLI requires exact divisibility, so each lane still executes exactly
        # ITERATIONS * 8 updates. Independent chains hide dependency latency.
        for _ in tl.range(0, ITERATIONS // UNROLL, loop_unroll_factor=1):
            # The inner index keeps a distinct name: reusing the outer loop's name
            # rebinds its loop-carried value to a constexpr, which some frontends
            # reject when they check loop-carried types.
            for _unrolled in tl.static_range(UNROLL):
                if IS_INT8 and USE_DP4A:
                    x0 = _dp4a_update(x0, a, b0, GCN_DP4A)
                    x1 = _dp4a_update(x1, a, b1, GCN_DP4A)
                    x2 = _dp4a_update(x2, a, b2, GCN_DP4A)
                    x3 = _dp4a_update(x3, a, b3, GCN_DP4A)
                    x4 = _dp4a_update(x4, a, b4, GCN_DP4A)
                    x5 = _dp4a_update(x5, a, b5, GCN_DP4A)
                    x6 = _dp4a_update(x6, a, b6, GCN_DP4A)
                    x7 = _dp4a_update(x7, a, b7, GCN_DP4A)
                elif IS_INT8:
                    x0 = _int8_mul_add_update(x0, a, b0)
                    x1 = _int8_mul_add_update(x1, a, b1)
                    x2 = _int8_mul_add_update(x2, a, b2)
                    x3 = _int8_mul_add_update(x3, a, b3)
                    x4 = _int8_mul_add_update(x4, a, b4)
                    x5 = _int8_mul_add_update(x5, a, b5)
                    x6 = _int8_mul_add_update(x6, a, b6)
                    x7 = _int8_mul_add_update(x7, a, b7)
                elif USE_WIDENED_FMA:
                    x0 = _widened_fma_update(x0, a, b0)
                    x1 = _widened_fma_update(x1, a, b1)
                    x2 = _widened_fma_update(x2, a, b2)
                    x3 = _widened_fma_update(x3, a, b3)
                    x4 = _widened_fma_update(x4, a, b4)
                    x5 = _widened_fma_update(x5, a, b5)
                    x6 = _widened_fma_update(x6, a, b6)
                    x7 = _widened_fma_update(x7, a, b7)
                else:
                    x0 = _float_update(x0, a, b0, USE_EXPLICIT_FMA)
                    x1 = _float_update(x1, a, b1, USE_EXPLICIT_FMA)
                    x2 = _float_update(x2, a, b2, USE_EXPLICIT_FMA)
                    x3 = _float_update(x3, a, b3, USE_EXPLICIT_FMA)
                    x4 = _float_update(x4, a, b4, USE_EXPLICIT_FMA)
                    x5 = _float_update(x5, a, b5, USE_EXPLICIT_FMA)
                    x6 = _float_update(x6, a, b6, USE_EXPLICIT_FMA)
                    x7 = _float_update(x7, a, b7, USE_EXPLICIT_FMA)

    # Every accumulator is observable, preventing dead-chain elimination.
    tl.store(Output + offsets + 0 * ELEMENTS, x0)
    tl.store(Output + offsets + 1 * ELEMENTS, x1)
    tl.store(Output + offsets + 2 * ELEMENTS, x2)
    tl.store(Output + offsets + 3 * ELEMENTS, x3)
    tl.store(Output + offsets + 4 * ELEMENTS, x4)
    tl.store(Output + offsets + 5 * ELEMENTS, x5)
    tl.store(Output + offsets + 6 * ELEMENTS, x6)
    tl.store(Output + offsets + 7 * ELEMENTS, x7)


def check_output(inputs, output, coefficients, iterations, use_explicit_fma=True):
    """Check every output against a double-precision closed-form recurrence.

    FP16/BF16 use an exactly representable two-cycle, with zero tolerance.
    FP32/FP64 use 0 < a < 1, bounded by the initial value and b/(1-a). Summing
    per-operation rounding errors as a geometric series bounds accumulated
    error using the selected dtype's unit roundoff. Separate source Mul/Add
    receives twice the FMA bound to allow two rounding steps.
    """
    elements = output.shape[1]
    # Copy first: several accelerators do not support FP64 device arithmetic.
    coeff = coefficients.cpu().double()
    validation_device = reference_device(output.device)
    a = coeff[0].item()
    exact_cycle = a == -1.0
    decay = None if exact_cycle else math.exp(iterations * math.log(a))
    geometric_sum = (None if exact_cycle else
                     -math.expm1(iterations * math.log(a)) / (1.0 - a))
    b = coeff[1:, None].to(validation_device)
    max_error, max_tolerance = 0.0, 0.0
    # Bound temporary FP64 allocations even for unusually large user grids.
    for start in range(0, elements, 131072):
        initial = inputs[:, start:start + 131072].to(validation_device).double()
        actual = output[:, start:start + 131072].to(validation_device).double()
        if exact_cycle:
            # Low-precision inputs and coefficients are multiples of 1/8 in
            # [-1, 1]. Both -x+b and its inverse are exactly representable.
            reference = b - initial if iterations % 2 else initial
            tolerance = torch.zeros_like(reference)
        else:
            reference = decay * initial + b * geometric_sum
            max_magnitude = torch.maximum(initial.abs(), (b / (1.0 - a)).abs())
            rounding_factor = 2.0 if use_explicit_fma else 4.0
            unit_roundoff = 2.0**(-53 if inputs.dtype == torch.float64 else -24)
            tolerance = (rounding_factor * unit_roundoff * max_magnitude
                         * geometric_sum + 16 * unit_roundoff)
        error = (actual - reference).abs()
        # Comparisons reject NaNs and infinities as well as inaccurate values.
        if not torch.all(error <= tolerance).item():
            raise RuntimeError("Vector correctness failed: non-finite or inaccurate output; "
                               f"max_abs_error={error.max().item():.6g}, "
                               f"max_tolerance={tolerance.max().item():.6g}")
        max_error = max(max_error, error.max().item())
        max_tolerance = max(max_tolerance, tolerance.max().item())
    return {
        "correctness": "all elements passed analytic FP64 reference and finite check",
        "max_abs_error": max_error,
        "max_abs_tolerance": max_tolerance,
        "reference_device": str(validation_device),
        "validation_recurrence": "exact involution" if exact_cycle else "contractive affine",
    }


def _signed_bytes(word):
    return [((int(word) >> shift) & 255) -
            (256 if ((int(word) >> shift) & 255) >= 128 else 0)
            for shift in (0, 8, 16, 24)]


def _dot4_reference(words, weights):
    result = torch.zeros_like(words, dtype=torch.int64)
    for shift, weight in zip((0, 8, 16, 24), weights):
        lane = (words >> shift) & 255
        lane = torch.where(lane >= 128, lane - 256, lane)
        result += lane * weight
    return result


def check_int8_output(inputs, output, coefficients, iterations):
    """Exact full validation using a bounded transition table and binary lifting.

    After one update the state is bounded by 128*sum(abs(weights))+abs(bias).
    Composing this small state table avoids a CPU loop over every output for
    every benchmark iteration. All arithmetic here is signed INT64 on CPU.
    """
    coeff = coefficients.cpu().long()
    weights = _signed_bytes(coeff[0].item())
    bias = coeff[1:, None]
    bound = 128 * sum(abs(value) for value in weights) + int(bias.abs().max())
    states = torch.arange(-bound, bound + 1).expand(CHAINS, -1)
    power = _dot4_reference(states, weights) + bias + bound
    composed = torch.arange(2 * bound + 1).expand(CHAINS, -1)
    remaining = iterations - 1
    while remaining:
        if remaining & 1:
            composed = power.gather(1, composed)
        remaining >>= 1
        if remaining:
            power = power.gather(1, power)
    for start in range(0, inputs.shape[1], 131072):
        initial = inputs[:, start:start + 131072].cpu().long()
        first = _dot4_reference(initial, weights) + bias
        reference = composed.gather(1, first + bound) - bound
        actual = output[:, start:start + 131072].cpu().long()
        if not torch.equal(actual, reference):
            raise RuntimeError("Vector INT8 correctness failed: output differs from exact "
                               "signed INT8 dot4 / INT32 state reference")
    return {"correctness": "all elements passed exact INT64 reference",
            "max_abs_error": 0, "max_abs_tolerance": 0, "reference_device": "cpu",
            "validation_recurrence": "signed INT8 dot4 with INT32 bias/state"}


def check_int8_mul_add_output(inputs, output, coefficients, iterations):
    """Check every INT8 result exactly against the affine map modulo 256."""
    if any(value.dtype != torch.int8 for value in (inputs, output, coefficients)):
        raise RuntimeError("INT8 multiply-add validation requires INT8 input, output and coefficients")
    coeff = coefficients.cpu().long()
    # Compose affine maps by exponentiation: (a*x+b)^iterations modulo 256.
    # All intermediates remain small signed INT64 values, even for long runs.
    multiplier, bias = 1, torch.zeros((CHAINS, 1), dtype=torch.int64)
    step_multiplier, step_bias = int(coeff[0]) % 256, coeff[1:, None] % 256
    remaining = iterations
    while remaining:
        if remaining & 1:
            bias = (step_multiplier * bias + step_bias) % 256
            multiplier = (step_multiplier * multiplier) % 256
        step_bias = (step_multiplier * step_bias + step_bias) % 256
        step_multiplier = (step_multiplier * step_multiplier) % 256
        remaining >>= 1
    for start in range(0, inputs.shape[1], 131072):
        initial = inputs[:, start:start + 131072].cpu().long()
        reference = (multiplier * initial + bias + 128) % 256 - 128
        actual = output[:, start:start + 131072].cpu().long()
        if not torch.equal(actual, reference):
            raise RuntimeError("Vector INT8 correctness failed: output differs from exact "
                               "modular INT8 multiply-add reference")
    return {"correctness": "all elements passed exact modulo-256 INT64 reference",
            "max_abs_error": 0, "max_abs_tolerance": 0, "reference_device": "cpu",
            "validation_recurrence": "INT8 multiply-add with modulo-256 state"}


def _inspect_xpu_artifacts(compiled, dtype):
    artifacts = getattr(compiled, "asm", {})
    assembly = artifacts.get("asm", "")
    assembly = assembly if isinstance(assembly, str) else ""
    # Evidence must be executable final assembly, not quoted IR asm templates,
    # comments, or a filename. Unknown mnemonic suffixes remain unverified.
    assembly = re.sub(r"/\*.*?\*/", "", assembly, flags=re.DOTALL)
    assembly = re.sub(r"(?m)(?://|#).*?$", "", assembly)
    opcodes = re.findall(
        r"(?m)^\s*(?:[0-9a-fA-F]+:\s+)?(v(?:mac|mul|add)(?:\.[a-zA-Z0-9_]+)+)\b",
        assembly)
    float32 = [op for op in opcodes if op.split(".")[1] == "f"]
    if dtype in ("bf16", "fp16") and float32:
        raise UnsupportedVectorDtype(
            f"XPU final assembly uses FP32 vector arithmetic for {dtype.upper()} inputs; "
            f"this promoted path is not reported as a native {dtype.upper()} peak")
    result = {"xpu_assembly_available": bool(assembly.strip()),
              "xpu_arithmetic_instruction_types": sorted(set(opcodes)),
              "native_arithmetic_verified": False, "hardware_fusion_verified": False}
    if dtype == "int8":
        evidence = {}
        for kind in ("ttir", "ttxir", "llir"):
            text = artifacts.get(kind, "")
            text = text if isinstance(text, str) else ""
            if kind == "llir":
                evidence[kind] = all(re.search(r"\b" + op + r"\s+<\s*\d+\s+x\s+i8\s*>", text)
                                     for op in ("mul", "add"))
            else:
                evidence[kind] = all(any(op in line and "vector<" in line
                                          and re.search(r"\bi8\b|xi8\b", line)
                                          for line in text.splitlines())
                                     for op in ("vvmuli", "vvaddi"))
        result["xpu_ir_vector_i8_evidence"] = evidence
        typed_ops = [op for op in opcodes if set(op.split(".")[1:]) & {"i8", "s8", "u8"}
                     and not set(op.split(".")[1:]) & {"wide", "sat", "saturate"}]
        result["native_arithmetic_verified"] = all(
            any(op.startswith(prefix + ".") for op in typed_ops) for prefix in ("vmul", "vadd"))
        if result["native_arithmetic_verified"]:
            result["assembly_check"] = "passed (final XPU ISA contains explicitly typed 8-bit vector multiply and add)"
        else:
            result["assembly_check"] = ("unverified: final XPU ISA does not prove 8-bit vector multiply/add; "
                                        "IR evidence is recorded separately; resource metadata is placeholder data")
    elif dtype in ("fp16", "fp32"):
        suffix = "hf" if dtype == "fp16" else "f"
        mac = [op for op in opcodes if op.startswith("vmac." + suffix + ".") or op == "vmac." + suffix]
        if mac:
            result.update(native_arithmetic_verified=True, hardware_fusion_verified=True,
                          assembly_check=f"passed (final XPU ISA contains vmac.{suffix})")
    return result


_LLVM_ARITHMETIC_WIDTHS = {
    "fp16": ("half", "f16"), "bf16": ("bfloat", "bf16"),
    "fp32": ("float", "f32"), "fp64": ("double", "f64"),
}


def _inspect_maca_artifacts(compiled, dtype):
    """Record which arithmetic width MetaX actually emits for this dtype.

    The MACA backend ships no final-ISA disassembler here, so LLVM IR is the
    last artifact available. It still answers the question that matters: a
    BF16 kernel whose only arithmetic is `fmul float`/`fadd float` reached
    FP32 execution units, which is not a native BF16 rate. IR is not final ISA
    evidence, so this never sets native_arithmetic_verified.
    """
    artifacts = getattr(compiled, "asm", {})
    llir = artifacts.get("llir", "")
    llir = llir if isinstance(llir, str) else ""
    result = {"maca_llir_available": bool(llir.strip()),
              "native_arithmetic_verified": False}
    if not llir.strip():
        result["assembly_check"] = ("unavailable; the MACA backend exposed no LLVM IR, "
                                    "so arithmetic width is unaudited")
        return result
    if dtype == "int8":
        return result
    llvm_type, _ = _LLVM_ARITHMETIC_WIDTHS[dtype]
    arithmetic = sorted(set(re.findall(
        r"\b(?:fmul|fadd|fsub)\s+(?:[a-z]+\s+)*(?:<\s*\d+\s+x\s+)?"
        r"(half|bfloat|float|double)\b", llir)))
    intrinsics = sorted(set(re.findall(
        r"@llvm\.(?:mxc\.)?(?:pk\.)?(?:fma|fmuladd)\.[a-z0-9.]+", llir)))
    result["maca_arithmetic_llvm_types"] = arithmetic
    result["maca_arithmetic_intrinsics"] = intrinsics
    # A narrow dtype computed in FP32 must round every result back. Counting
    # those conversions shows the recurrence state really kept the narrow type
    # instead of staying in FP32 and only converting once at the end.
    if dtype in ("bf16", "fp16"):
        narrow = "bf16" if dtype == "bf16" else "f16"
        result["maca_narrowing_conversion_sites"] = len(re.findall(
            r"@llvm\.mxc\.cvt\.(?:pk\.)?f32to" + narrow + r"\b", llir))
    requested_present = llvm_type in arithmetic or any(
        intrinsic.endswith(_LLVM_ARITHMETIC_WIDTHS[dtype][1]) for intrinsic in intrinsics)
    wider = [found for found in arithmetic if found != llvm_type]
    wider_intrinsics = [intrinsic for intrinsic in intrinsics
                        if not intrinsic.endswith(_LLVM_ARITHMETIC_WIDTHS[dtype][1])]
    if not requested_present and (wider or wider_intrinsics):
        # Keep the measurement, but state plainly that the arithmetic was widened.
        result["maca_arithmetic_promoted"] = True
        result["assembly_check"] = (
            f"unverified: MACA LLVM IR contains no {dtype.upper()} arithmetic; the kernel was "
            f"promoted to {', '.join(wider + wider_intrinsics)}, so this is not a native "
            f"{dtype.upper()} vector rate")
    else:
        result["maca_arithmetic_promoted"] = not requested_present
        result["assembly_check"] = (
            f"unverified: MACA LLVM IR keeps {dtype.upper()} arithmetic "
            f"({', '.join(arithmetic + intrinsics) or 'no explicit arithmetic found'}), "
            "but LLVM IR is not final ISA evidence of a native instruction")
    return result


def _inspect_ascend_artifacts(compiled, dtype):
    """Record which arithmetic width the Ascend backend keeps for this dtype.

    triton-ascend emits MLIR (`ttir` before lowering, `bcmlir` after) and then
    a `npubin` binary with no disassembler available here, so MLIR is the last
    readable artifact. It still answers the question that matters: arithmetic
    carrying `tensor<...xf8E4M3FN>` stayed in the narrow type, while a kernel
    promoted to FP32 shows `tensor<...xf32>` arithmetic plus `arith.extf`
    widening and an `arith.truncf` back. Both forms compute correct values and
    only the IR distinguishes them.

    MLIR is not final ISA evidence, so this never sets
    native_arithmetic_verified; the suite reports such rows `unverified`.
    """
    artifacts = getattr(compiled, "asm", {})
    mlir = artifacts.get("bcmlir", "") or artifacts.get("ttir", "")
    mlir = mlir if isinstance(mlir, str) else ""
    result = {"ascend_mlir_available": bool(mlir.strip()),
              "native_arithmetic_verified": False}
    if not mlir.strip():
        result["assembly_check"] = ("unavailable; the Ascend backend exposed no MLIR, "
                                    "so arithmetic width is unaudited")
        return result
    # The cube (matrix) unit is selected per kernel. A vector benchmark that
    # reached it would not be measuring vector throughput.
    mix_mode = re.findall(r'mix_mode\s*=\s*"(\w+)"', mlir)
    result["ascend_mix_mode"] = mix_mode[0] if mix_mode else None
    if any(mode == "aic" for mode in mix_mode):
        raise RuntimeError(
            "Ascend selected the cube/matrix unit (mix_mode=aic) in a vector benchmark")
    if dtype == "int8":
        return result
    element = {"fp8": "f8E4M3FN", "fp16": "f16", "bf16": "bf16",
               "fp32": "f32", "fp64": "f64"}[dtype]
    arithmetic_re = r"(?:math\.fma|arith\.mulf|arith\.addf)[^\n]*?(?:tensor<[\dx]*|vector<[\dx]*)"
    requested = len(re.findall(arithmetic_re + re.escape(element) + r"\b", mlir))
    widened = [found for found in sorted(set(re.findall(
        arithmetic_re + r"(f32|f64)\b", mlir))) if found != element]
    result.update({"ascend_requested_arithmetic_sites": requested,
                   "ascend_widened_arithmetic": widened,
                   "ascend_extf_sites": len(re.findall(r"\barith\.extf\b", mlir))})
    if requested == 0 and widened:
        # Keep the measurement, but state plainly that the arithmetic widened.
        result["ascend_arithmetic_promoted"] = True
        result["assembly_check"] = (
            f"unverified: Ascend MLIR contains no {dtype.upper()} arithmetic; the kernel was "
            f"promoted to {', '.join(widened)}, so this is not a native "
            f"{dtype.upper()} vector rate")
    else:
        result["ascend_arithmetic_promoted"] = requested == 0
        result["assembly_check"] = (
            f"unverified: Ascend MLIR keeps {dtype.upper()} arithmetic "
            f"({requested} site(s) on {element}), but MLIR is not final ISA "
            "evidence of a native instruction")
    return result


def inspect_assembly(compiled, vendor_name, dtype="fp32", backend=None):
    """Check NVIDIA PTX; retain optional metadata on other FlagGems devices."""
    result = {"registers_per_thread": getattr(compiled, "n_regs", None),
              "register_spills": getattr(compiled, "n_spills", None),
              "assembly_check": "unavailable for this vendor",
              "native_arithmetic_verified": False}
    if vendor_name == "kunlunxin" or backend == "xpu":
        # FlagTree's XPU loader currently returns literal zero placeholders for
        # both values, so they cannot prove register residency or no spilling.
        result.update({
            "registers_per_thread": None,
            "register_spills": None,
            "assembly_check": "unavailable; XPU register metadata is placeholder data",
        })
        result.update(_inspect_xpu_artifacts(compiled, dtype))
    elif vendor_name == "metax" or backend == "maca":
        result.update(_inspect_maca_artifacts(compiled, dtype))
    elif vendor_name == "ascend" or backend == "npu":
        result.update(_inspect_ascend_artifacts(compiled, dtype))
    elif vendor_name == "thead":
        # T-Head PPU emits no PTX; its vector arithmetic reaches LLVM IR as
        # vectorized intrinsics (@llvm.fma.v16bf16 and friends). The checks
        # mirror the nvidia branch: reject tensor instructions in a vector
        # benchmark, require a native per-dtype FMA, and reject a dtype that
        # the compiler promoted to FP32/FP64 instead of executing natively.
        llir = (compiled.asm or {}).get("llir", "")
        tensor_ops = bool(_PPU_VECTOR_MMA_RE.search(llir))
        if tensor_ops:
            raise RuntimeError("Generated LLVM IR contains tensor instructions in a vector benchmark")
        if dtype == "int8":
            # The PPU has no DP4A, so the INT8 path is typed multiply/add.
            # Count the integer arithmetic that INT8_MUL_ADD_MOD256 lowers to.
            instruction_count = len(re.findall(r"\bmul\b", llir))
            instruction = "INT8 integer multiply/add (no DP4A on this target)"
        else:
            width = {"fp16": "f16", "bf16": "bf16", "fp32": "f32", "fp64": "f64"}[dtype]
            instruction_count = len(re.findall(
                rf"@llvm\.(?:fma|fmuladd)\.(?:v\d+)?{width}\b", llir))
            instruction = f"{dtype.upper()} FMA"
            if dtype in ("fp16", "bf16"):
                promoted = re.search(r"@llvm\.(?:fma|fmuladd)\.(?:v\d+)?f(?:32|64)\b", llir)
                if promoted or re.search(r"\bfpext\b", llir):
                    raise UnsupportedVectorDtype(
                        f"Compiler promotes {dtype.upper()} vector arithmetic to FP32/FP64; "
                        "promoted arithmetic is not reported as the requested native peak")
        if instruction_count == 0:
            raise UnsupportedVectorDtype(
                f"Generated LLVM IR does not contain native {instruction}")
        result.update({
            "llir_arithmetic_instruction_sites": instruction_count,
            "llir_contains_tensor_instructions": tensor_ops,
            "native_arithmetic_verified": True,
            "assembly_check": f"passed ({instruction}; no tensor instructions)",
        })
        if dtype == "fp32":
            result["llir_fp32_fma_instruction_sites"] = instruction_count
    elif vendor_name in VECTOR_AMDGCN_VENDORS or backend == "hip":
        # AMD and Hygon both compile through the AMDGCN path and expose the
        # final assembly under "amdgcn", so this is real ISA evidence rather
        # than IR. The checks mirror the nvidia branch: reject matrix
        # instructions in a vector benchmark, require a per-dtype native FMA,
        # and reject a narrow dtype the compiler widened to FP32 instead of
        # executing natively. Hygon spells its matrix instruction v_mmac_*
        # where AMD spells it v_mfma_*/v_wmma_*; all three are rejected here.
        gcn = (compiled.asm or {}).get("amdgcn", "")
        if not gcn:
            return result
        tensor_ops = bool(re.search(r"\bv_(?:mfma|wmma|mmac)_", gcn))
        if tensor_ops:
            raise RuntimeError("Generated AMDGCN contains matrix instructions in a vector benchmark")
        promoted_from = None
        native_dp4a = False
        if dtype == "int8":
            # The INT8 kernel is either the native DP4A or the typed modulo-256
            # multiply plus add, depending on what the target assembles; the
            # listing decides which. First count the integer arithmetic the
            # emulated path lowers to. Width matters: the DCU keeps that path in
            # 16-bit integer ops (v_mul_lo_u16, v_mad_legacy_u16), which still
            # carry the INT8 values the kernel declared, so the 16- and 32-bit
            # spellings both count as typed integer arithmetic. On the other
            # non-DP4A vendors that is not final proof of a native INT8
            # execution unit, so the suite keeps that rate unverified.
            instruction_count = len(re.findall(
                r"\bv_(?:mul_(?:lo_|hi_)?|mad_(?:legacy_)?)[iu](?:8|16|32)(?:_e(?:32|64))?\b",
                gcn))
            instruction = "typed integer multiply/add (no DP4A on this target)"
            # gfx936 assembles v_dot4_i32_i8, the AMDGCN spelling of DP4A: four
            # signed INT8 products plus four INT32 additions in one instruction.
            # When the kernel actually issued it, the rate is a native INT8 rate
            # and the earlier "no DP4A here" note was a stale vendor-based guess.
            dp4a_sites = len(re.findall(r"\bv_dot4_i32_i8\b", gcn))
            if dp4a_sites:
                instruction_count = dp4a_sites
                instruction = "signed INT8 DP4A (v_dot4_i32_i8) with INT32 output"
                native_dp4a = True
            else:
                native_dp4a = False
        else:
            width = {"fp16": "f16", "bf16": "bf16", "fp32": "f32", "fp64": "f64"}[dtype]
            instruction_count = len(re.findall(
                rf"\bv_(?:pk_)?(?:fma|fmac|mac|mad)_{width}(?:_e(?:32|64))?\b", gcn))
            instruction = f"{dtype.upper()} FMA"
            if dtype in ("fp16", "bf16") and instruction_count == 0:
                # No narrow FMA at all: the compiler widened this dtype to FP32.
                # The DCU does this for BF16, which has no v_fma_bf16. Follow the
                # MetaX precedent rather than the NVIDIA one and keep the rate:
                # the arithmetic really ran and the recurrence state really kept
                # the narrow dtype, so the number describes what the chip does
                # with BF16 data. It is not a native BF16 execution rate, so it
                # is recorded as promoted and never marked verified.
                widened = sorted(set(re.findall(
                    r"\bv_(?:pk_)?(?:fma|fmac|mac|mad)_f(?:32|64)(?:_e(?:32|64))?\b", gcn)))
                if widened:
                    promoted_from = widened
                    instruction_count = sum(gcn.count(name) for name in widened)
        if instruction_count == 0:
            raise UnsupportedVectorDtype(
                f"Generated AMDGCN does not contain native {instruction}")
        result.update({
            "amdgcn_arithmetic_instruction_sites": instruction_count,
            "amdgcn_contains_matrix_instructions": tensor_ops,
            # Only a floating FMA naming the requested element width proves the
            # requested dtype executed natively. A widened narrow dtype and the
            # INT8 integer path both fail that test.
            "native_arithmetic_verified": (dtype == "int8" and native_dp4a)
                                         or (dtype != "int8" and promoted_from is None),
        })
        if promoted_from is not None:
            result.update({
                "amdgcn_arithmetic_promoted": True,
                "amdgcn_promoted_instructions": promoted_from,
                "assembly_check": (
                    f"unverified: AMDGCN contains no native {dtype.upper()} FMA; the kernel "
                    f"was promoted to {', '.join(promoted_from)}, so this is not a native "
                    f"{dtype.upper()} vector rate"),
            })
        elif dtype == "int8" and native_dp4a:
            result["assembly_check"] = (
                f"passed ({instruction}; no matrix instructions)")
        elif dtype == "int8":
            result["assembly_check"] = (
                f"unverified: AMDGCN contains {instruction_count} typed integer "
                "multiply/add site(s), but this target has no DP4A and the integer "
                "ops are wider than INT8, so native INT8 arithmetic is not established")
        else:
            result["assembly_check"] = f"passed ({instruction}; no matrix instructions)"
        if dtype == "fp32":
            result["amdgcn_fp32_fma_instruction_sites"] = instruction_count
    # Some non-NVIDIA runtimes also expose a backend named "cuda".
    elif vendor_name == "nvidia":
        ptx = compiled.asm["ptx"]
        tensor_ops = bool(re.search(r"\b(?:mma|wgmma|tcgen05)\.", ptx))
        if tensor_ops:
            raise RuntimeError("Generated PTX contains tensor instructions in a vector benchmark")
        if dtype == "int8":
            instruction_count = len(re.findall(r"\bdp4a\.s32\.s32\b", ptx))
            instruction = "signed INT8 DP4A with INT32 output"
        else:
            suffix = {"fp16": "f16(?:x2)?", "bf16": "bf16(?:x2)?",
                      "fp32": "f32", "fp64": "f64", "fp8": "e4m3(?:x[24])?"}[dtype]
            instruction_count = len(re.findall(r"\bfma\.(?:\w+\.)*" + suffix + r"\b", ptx))
            instruction = f"{dtype.upper()} FMA"
            if dtype in ("fp16", "bf16", "fp8") and re.search(r"\bfma\.(?:\w+\.)*f(?:32|64)\b", ptx):
                raise UnsupportedVectorDtype(
                    f"Compiler promotes {dtype.upper()} vector arithmetic to FP32/FP64; "
                    "promoted arithmetic is not reported as the requested native peak")
        if instruction_count == 0:
            raise UnsupportedVectorDtype(f"Generated PTX does not contain native {instruction}")
        result.update({
            "ptx_arithmetic_instruction_sites": instruction_count,
            "ptx_contains_tensor_instructions": tensor_ops,
            "native_arithmetic_verified": True,
            "assembly_check": f"passed ({instruction}; no tensor instructions)",
        })
        if dtype == "fp32":
            result["ptx_fp32_fma_instruction_sites"] = instruction_count
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp32",
                        help="Arithmetic type (default: fp32); unsupported types emit a status JSON")
    parser.add_argument("--iterations", type=positive_int, default=DEFAULT_ITERATIONS,
                        help=f"Multiply-add iterations per accumulator "
                             f"(default: {DEFAULT_ITERATIONS}; "
                             f"{XPU_STATIC_ITERATIONS} for FP16/BF16/INT8 on FlagTree XPU)")
    parser.add_argument("--blocks-per-sm", type=positive_int, nargs="+",
                        help="Grid multipliers per compute unit (XPU default: 1 2; "
                             "other backends: 8 16)")
    parser.add_argument("--block-sizes", type=positive_int, nargs="+",
                        help="Elements per program, powers of two (XPU INT8: 4096 8192; "
                             "other XPU: 1024 2048; other backends: 256 512)")
    parser.add_argument("--num-warps", type=int, choices=[4, 8], nargs="+",
                        help="Warps per program (default: 4 8; ignored by FlagTree XPU)")
    parser.add_argument("--unroll", type=int, choices=[1, 2, 4, 8, 16], nargs="+",
                        help="Explicit loop unroll factors (XPU default: 1 4; "
                             "other backends: 1 4 8)")
    parser.add_argument("--quick", action="store_true",
                        help="One backend-aware smoke configuration; not a peak result")
    return parser


_is_flagtree_xpu = is_flagtree_xpu


def needs_fully_static_loop(dtype, is_xpu):
    return bool(is_xpu) and dtype in XPU_NARROW_DTYPES


def backend_supports_explicit_fma(dtype, device):
    """Compile one tiny kernel to see whether `tl.fma` lowers for this dtype.

    A backend can carry a dtype through loads, stores and `*`/`+` while having
    no fused multiply-add lowering for it. On MetaX the BF16 `tl.fma` fails in
    ConvertTritonGPUToLLVM because the backend represents BF16 as `i16`, which
    `llvm.intr.fma` rejects. That is a missing fused-op lowering, not a missing
    dtype, so the sweep falls back to source multiply plus add instead of
    reporting the whole metric as unsupported. Returns the probe outcome and a
    reason string for the JSON record.
    """
    torch_dtype = DTYPES[dtype]
    if torch_dtype is None:
        return False, f"This PyTorch build does not expose {dtype.upper()} storage"
    block = 32
    try:
        probe_input = _filled((3 * block,), 1.0, torch_dtype, device)
        probe_output = torch.empty((block,), dtype=torch_dtype, device=device)
        _fma_probe_kernel[(1,)](probe_input, probe_output, block, num_warps=4)
    except Exception as exc:  # noqa: BLE001 - any lowering failure means no fused path
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return False, (f"Backend cannot lower tl.fma for {dtype.upper()}; "
                       f"using source multiply plus add: {first_line}")
    finally:
        probe_input = probe_output = None
    return True, None


def backend_supports_dp4a(device, gcn=False):
    """Compile one tiny kernel to see whether this target assembles DP4A.

    The selector used to enable DP4A only for the NVIDIA vendor name, which is a
    vendor test, not a capability test. gfx936 assembles `v_dot4_i32_i8`, the
    AMDGCN spelling of the same four-int8-products-plus-INT32-accumulate
    operation, so the emulated typed path was measuring roughly a quarter of
    what the hardware executes. Probing keeps a target whose assembler rejects
    the mnemonic on the emulated path instead of guessing from the vendor name.
    Returns the probe outcome and a reason string for the JSON record.
    """
    block = 32
    try:
        probe_input = torch.randint(-100, 100, (3 * block,), dtype=torch.int32,
                                    device=device)
        probe_output = torch.empty((block,), dtype=torch.int32, device=device)
        _dp4a_probe_kernel[(1,)](probe_input, probe_output, block,
                                 GCN=gcn, num_warps=4)
    except Exception as exc:  # noqa: BLE001 - any failure means no native DP4A here
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return False, (f"Backend cannot assemble DP4A on this target; using the typed "
                       f"modulo-256 INT8 path instead: {first_line}")
    finally:
        probe_input = probe_output = None
    return True, None


def _filled(shape, value, torch_dtype, device):
    """Allocate a tensor holding `value` without relying on a typed fill.

    `torch.ones`/`torch.full`/`Tensor.fill_` go through aclnnInplaceFillScalar
    on Ascend, which rejects float8_e4m3fn even though the same device holds
    and converts FP8 fine. Building the value in FP32 and converting keeps the
    requested dtype and works on every backend, so this is not a vendor
    branch. Only used for untimed probe and sentinel buffers.
    """
    return torch.full(shape, value, device=device).to(torch_dtype)


def _fill_sentinel(tensor, value):
    """Overwrite `tensor` with `value` through a copy instead of a typed fill.

    Same reason as `_filled`: the in-place scalar fill has no FP8 kernel on
    Ascend, while `copy_` from an FP32 source does. The sentinel marks the
    output buffer before each timed launch so an unwritten element is caught.
    """
    tensor.copy_(torch.full(tensor.shape, value, device=tensor.device))


@triton.jit
def _scalar_coefficient_probe_kernel(X, A, B, Output, BLOCK_SIZE: tl.constexpr):
    # Mirrors `vector_fma_kernel`'s use of a scalar (0-D) coefficient feeding
    # elementwise arithmetic, which is what distinguishes it from the other
    # probes: those load coefficients at the same shape as the data.
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    a = tl.load(A)
    b = tl.load(B)
    tl.store(Output + offsets, tl.fma(x, a, b))


def needs_broadcast_coefficients(dtype, device):
    """Detect a backend that cannot lower scalar coefficients for this dtype.

    The Ascend bishengir compiler aborts with "LLVM ERROR: unsupported type
    cast" when an FP8 scalar is combined with an FP8 vector, while the same
    arithmetic with a vector-shaped (broadcast) coefficient compiles and runs.
    That is a backend lowering defect, not a hardware property, so it is
    discovered by compiling rather than by naming the vendor: a backend that
    lowers the scalar form keeps using it. Returns (needed, reason).
    """
    torch_dtype = DTYPES[dtype]
    if torch_dtype is None:
        return False, None
    block = 32
    try:
        api = get_device_api()
    except (ImportError, ModuleNotFoundError, SystemExit):
        # No device runtime to probe: keep the default scalar form rather than
        # changing behaviour on the strength of a probe that never ran.
        return False, None
    try:
        probe_input = _filled((block,), 1.0, torch_dtype, device)
        probe_a = _filled((1,), 0.5, torch_dtype, device)
        probe_b = _filled((1,), 0.25, torch_dtype, device)
        probe_output = torch.empty((block,), dtype=torch_dtype, device=device)
        _scalar_coefficient_probe_kernel[(1,)](probe_input, probe_a, probe_b,
                                               probe_output, BLOCK_SIZE=block)
        api.synchronize()
    except Exception as exc:  # noqa: BLE001 - a failure here is a lowering gap
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        crash = "unsupported type cast" in str(exc)
        return True, (f"Backend cannot compile scalar coefficients for {dtype.upper()}"
                      f"{' (compiler abort: unsupported type cast)' if crash else ''}; "
                      f"coefficients are loaded as broadcast vectors instead: {first_line[:160]}")
    return False, None


def backend_supports_widened_fma(dtype, device):
    """Compile a narrow-state/FP32-coefficient FMA probe for this dtype.

    When a backend has no native arithmetic for a narrow float type it emulates
    it by extending to FP32. `tl.fma` may still fail to lower (MetaX BF16), yet
    an explicitly widened FMA whose result is rounded back to the narrow type
    does lower, and it keeps the same one-rounding-per-update semantics that
    `tl.fma` has elsewhere. Only offered for narrow float types: for FP32/FP64 a
    widened FMA would either be a no-op or silently narrow the arithmetic.
    """
    if dtype not in WIDENED_FMA_DTYPES:
        return False, None
    torch_dtype = DTYPES[dtype]
    if torch_dtype is None:
        return False, f"This PyTorch build does not expose {dtype.upper()} storage"
    block = 32
    try:
        probe_input = _filled((3 * block,), 1.0, torch_dtype, device)
        probe_output = torch.empty((block,), dtype=torch_dtype, device=device)
        _widened_fma_probe_kernel[(1,)](probe_input, probe_output, block, num_warps=4)
    except Exception as exc:  # noqa: BLE001 - any lowering failure rules the path out
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return False, (f"Backend cannot lower an FP32-widened FMA for {dtype.upper()}: "
                       f"{first_line}")
    finally:
        probe_input = probe_output = None
    return True, None


def configure_sweep(args, metadata):
    """Fill only unspecified tuning dimensions with backend-appropriate values."""
    is_xpu = is_flagtree_xpu(metadata)
    fully_static = needs_fully_static_loop(args.dtype, is_xpu)
    if args.quick:
        args.iterations = 32
        args.blocks_per_sm = [1]
        args.block_sizes = [1024] if is_xpu else [128]
        args.num_warps = [None] if is_xpu else [4]
        args.unroll = [1]
    else:
        if fully_static and args.iterations == DEFAULT_ITERATIONS:
            args.iterations = XPU_STATIC_ITERATIONS
            print(f"FlagTree XPU {args.dtype.upper()} requires a fully unrolled loop; "
                  f"using {args.iterations} iterations to bound compile time "
                  f"(override with --iterations)")
        args.blocks_per_sm = args.blocks_per_sm or ([1, 2] if is_xpu else [8, 16])
        # 2048 elements per program is unusable for two XPU types: INT8 fails
        # instruction selection for the 64-lane vector add, and BF16 silently
        # miscompiles into non-finite results that output validation rejects.
        # 1024 is exact for both.
        xpu_blocks = [1024] if args.dtype in ("int8", "bf16") else [1024, 2048]
        # 256/512 elements per program leaves some vector units short of a full
        # issue window: on Ascend950DT BF16 measures 7.41 TFLOP/s at 512 and
        # 8.89 at 2048. 4096 is past the ceiling there, where the kernel stops
        # compiling for want of unified-buffer space, so the sweep stops at
        # 2048 and the larger sizes stay available through --block-sizes.
        args.block_sizes = args.block_sizes or (xpu_blocks if is_xpu else [256, 512, 1024, 2048])
        if is_xpu:
            if args.num_warps is not None:
                print("FlagTree XPU does not use Triton num_warps; ignoring --num-warps")
            args.num_warps = [None]
        else:
            args.num_warps = args.num_warps or [4, 8]
        # A fully unrolled loop has no strip-mining factor to vary; every
        # unroll value would compile the identical kernel.
        if fully_static:
            args.unroll = [1]
        else:
            args.unroll = args.unroll or ([1, 4] if is_xpu else [1, 4, 8, 16])
    return is_xpu


def launch_vector(kernel, grid, inputs, output, coefficients, elements, iterations,
                  block_size, unroll, use_explicit_fma, num_warps,
                  is_int8=False, use_dp4a=False, fully_static=False,
                  use_widened_fma=False, broadcast_coefficients=False,
                  gcn_dp4a=False):
    launch_options = {} if num_warps is None else {"num_warps": num_warps}
    if is_int8:
        launch_options.update(IS_INT8=True, USE_DP4A=use_dp4a)
        if gcn_dp4a:
            launch_options.update(GCN_DP4A=True)
    if fully_static:
        launch_options.update(FULLY_STATIC=True)
    if use_widened_fma:
        launch_options.update(USE_WIDENED_FMA=True)
    if broadcast_coefficients:
        launch_options.update(BROADCAST_COEFFICIENTS=True)
    return kernel[grid](
        inputs, output, coefficients, elements, iterations, block_size, unroll,
        use_explicit_fma, **launch_options,
    )


def unsupported_dtype_reason(dtype, metadata):
    if dtype == "fp4":
        # Checked before the storage test below: the blocker is the absence of
        # an FP4 arithmetic element type, which holds whether or not this
        # PyTorch build ships the packed container dtype.
        return f"{_FP4_VECTOR_REJECTION} (checked: {_fp4_vector_rejection_detail()['detail']})"
    if DTYPES[dtype] is None:
        return f"This PyTorch build does not expose {dtype.upper()} storage"
    if dtype == "fp64" and metadata.get("supports_fp64") is False:
        return "FlagGems reports that this device does not support FP64 arithmetic"
    if dtype == "bf16" and metadata.get("supports_bf16") is False:
        return "FlagGems reports that this device does not support BF16 arithmetic"
    return None


_FP8_VECTOR_REJECTION = (
    "This benchmark has no implemented and verified native FP8 vector arithmetic "
    "path; FP8 storage, conversions, or tensor instructions do not measure native "
    "FP8 vector throughput. This does not determine chip support")

_FP4_VECTOR_REJECTION = (
    "FP4 (E2M1) is not an arithmetic element type in this stack: Triton exposes no "
    "FP4 scalar dtype, so no FP4 multiply-add can be written, and PyTorch's "
    "float4_e2m1fn_x2 is a two-element container rather than a scalar type. FP4 "
    "storage, packing, conversions, or microscaled matrix instructions do not "
    "measure native FP4 vector throughput. This does not determine chip support; "
    "where a chip executes FP4 natively on its matrix unit, that rate is the "
    "tensor_fp4 metric")


def _fp4_vector_rejection_detail():
    """Name the specific layer that blocks an FP4 vector arithmetic path.

    Reported alongside the refusal so the status is traceable to a checked fact
    rather than to a hardcoded opinion. Both checks are local to the installed
    toolchain and need no device.
    """
    triton_fp4 = [name for name in dir(tl)
                  if name.startswith("float4") or name.startswith("fp4")]
    container = DTYPES["fp4"]
    return {
        "triton_fp4_scalar_dtypes": triton_fp4,
        "triton_provides_fp4_element_type": bool(triton_fp4),
        "torch_fp4_container_dtype": None if container is None else str(container),
        "torch_container_elements_per_item": None if container is None else 2,
        "detail": ("triton.language exposes no FP4 element type, so tl.fma/tl.mul have "
                   "no FP4 overload to lower" if not triton_fp4 else
                   "triton.language exposes an FP4 element type; re-examine this refusal"),
    }


def _fp8_native_elementwise_probe(device):
    """Compile a real FP8 multiply-add and report whether it stayed in FP8.

    FP8 storage or FP8 tensor-core support does not imply native elementwise
    FP8 arithmetic, so the benchmark refuses the metric unless this probe
    shows the compiler both accepted FP8 operands and emitted arithmetic that
    is still typed FP8. A backend that silently widens to FP32 is rejected
    here: its rate would be an FP32 rate wearing an FP8 label.

    Returns (supported, detail). Any compilation or runtime failure means the
    backend has no such path, which is reported, not raised.
    """
    dtype = DTYPES["fp8"]
    if dtype is None:
        return False, "this PyTorch build does not expose FP8 storage"
    size = 256
    try:
        x = torch.full((size,), 0.5, device=device).to(dtype)
        a = torch.full((size,), 0.5, device=device).to(dtype)
        b = torch.full((size,), 0.25, device=device).to(dtype)
        out = torch.zeros((size,), device=device).to(dtype)
        compiled = _fp8_probe_kernel[(1,)](x, a, b, out, BLOCK_SIZE=size)
        get_device_api().synchronize()
    except Exception as exc:
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        return False, f"the backend cannot compile FP8 elementwise multiply-add: {first[:200]}"
    # 0.5*0.5+0.25 = 0.5 is exact in E4M3FN, so a correct native or widened
    # path both return it; this only rejects a path that does not compute.
    produced = out.float()
    if not torch.equal(produced, torch.full_like(produced, 0.5)):
        return False, (f"the FP8 elementwise probe returned {produced[0].item()} "
                       "instead of the exact value 0.5")
    artifacts = getattr(compiled, "asm", {}) or {}
    mlir = artifacts.get("bcmlir", "") or artifacts.get("ttir", "")
    if not isinstance(mlir, str) or not mlir.strip():
        return False, ("the backend exposes no readable IR for the FP8 probe, so native "
                       "FP8 arithmetic cannot be distinguished from widened FP32 math")
    narrow = re.search(r"(?:math\.fma|arith\.mulf|arith\.addf)[^\n]*f8E4M3FN\b", mlir)
    if not narrow:
        return False, ("the compiler did not keep FP8 arithmetic for an FP8 multiply-add; "
                       "widened arithmetic is not reported as a native FP8 vector rate")
    return True, "FP8 multiply-add compiles and its IR arithmetic is still typed FP8"


def not_implemented_dtype_reason(dtype, device=None):
    """Report why a dtype has no implemented path, or None if it has one.

    `device` may be a device or a zero-argument callable returning one. The
    callable form keeps this usable before any accelerator runtime is loaded:
    it is only invoked for a dtype whose decision needs a compile probe.
    """
    if dtype != "fp8":
        return None
    # FP8 tensor-core/storage support does not establish native elementwise
    # FP8 arithmetic. Do not publish widened math as an FP8 vector peak.
    # Without a device to probe, keep the unconditional refusal.
    if device is None:
        return _FP8_VECTOR_REJECTION
    try:
        target = device() if callable(device) else device
    except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError, SystemExit) as exc:
        # No reachable device runtime means no FP8 path can be established.
        # `common.get_flag_gems` raises SystemExit when no backend is installed.
        return f"{_FP8_VECTOR_REJECTION} (probe: the device is unreachable: {exc})"
    supported, detail = _fp8_native_elementwise_probe(target)
    if supported:
        return None
    return f"{_FP8_VECTOR_REJECTION} (probe: {detail})"


def _expand_coefficients(coefficients, block_size):
    """Repeat each coefficient into its own `block_size`-wide run.

    `vector_fma_kernel` reads coefficients this way when the backend cannot
    lower scalar operands for the dtype (see BROADCAST_COEFFICIENTS), so every
    lane still sees exactly the same value it saw as a scalar. Total extra
    memory is (CHAINS + 1) * block_size elements, which is negligible next to
    Expanded in FP32 first: Ascend's repeat_interleave has no FP8 kernel, and
    converting afterwards is exact because each value is copied, not rounded.
    Total extra memory is (CHAINS + 1) * block_size elements, which is
    negligible next to the input and output buffers.
    """
    expanded = coefficients.float().repeat_interleave(block_size)
    return expanded.to(coefficients.dtype).contiguous()


def _make_inputs(dtype, elements, device, *, int8_mul_add=False):
    if dtype == "int8" and int8_mul_add:
        inputs = torch.randint(-128, 128, (CHAINS, elements), dtype=torch.int8)
        coefficients = torch.tensor([3] + [2 * chain + 1 for chain in range(CHAINS)],
                                    dtype=torch.int8)
    elif dtype == "int8":
        inputs = torch.randint(-(2**31), 2**31, (CHAINS, elements), dtype=torch.int32)
        packed = sum((value & 255) << shift
                     for value, shift in zip((3, -2, 1, -1), (0, 8, 16, 24)))
        if packed >= 2**31:
            packed -= 2**32
        coefficients = torch.tensor([packed] + list(range(1, CHAINS + 1)), dtype=torch.int32)
    elif dtype in ("fp16", "bf16", "fp8"):
        # Exact arithmetic in every supported floating format. Coefficients are
        # loaded from memory, so the compiler cannot fold the two-cycle away.
        # Multiples of 1/8 in [-1, 1] are exact in E4M3FN too, so FP8 uses the
        # same exactly representable involution and keeps zero tolerance.
        inputs = (torch.randint(1, 8, (CHAINS, elements)).float() / 8).to(DTYPES[dtype])
        coefficients = torch.tensor([-1.0] + [(i + 1) / 8 for i in range(CHAINS)],
                                    dtype=DTYPES[dtype])
    else:
        inputs = torch.empty((CHAINS, elements), dtype=DTYPES[dtype]).uniform_(0.125, 1.0)
        coefficients = torch.tensor([1.0 - 2.0**-14] +
                                    [(chain + 1) * 2.0**-16 for chain in range(CHAINS)],
                                    dtype=DTYPES[dtype])
    return inputs.to(device), coefficients.to(device)


def _compile_failure_is_unsupported(exc):
    """Recognize explicit arithmetic capability diagnostics, not compiler bugs.

    CompilationError also covers bad launch parameters, frontend bugs, and
    invalid user code. Its source excerpt is deliberately excluded: a quoted
    source string must not change an unrelated error into an unsupported type.
    """
    message = str((exc.error_message or "") if isinstance(exc, CompilationError) else exc).lower()
    dtype = r"(?:fp16|float16|f16|bf16|bfloat16|fp32|float32|f32|fp64|float64|f64|int8|s8|fp8|float8|e4m3|e5m2)"
    patterns = (
        r"unsupported\s+(?:data\s+|scalar\s+)?(?:dtype|type)\s*[:=]?\s*['\"]?" + dtype + r"\b",
        r"\b" + dtype + r"\s+(?:(?:arithmetic|dtype|type)\s+)?(?:is\s+)?(?:unsupported|not supported)\b",
        r"(?:does not support|doesn't support)\s+(?:the\s+)?(?:(?:dtype|type)\s+)?['\"]?" + dtype + r"\b",
        r"\b(?:dp4a|fma(?:\.[a-z0-9]+)*|bf16|f16|f64)\b[^\n]*requires\s+['\"]?(?:\.target\s+)?sm_\d+",
    )
    return any(re.search(pattern, message) for pattern in patterns)


def _compile_failure_is_resource_limit(exc):
    """Recognize a per-configuration capacity limit reported as text.

    `triton.OutOfResources` covers the cases Triton itself detects, but a
    vendor compiler that runs after Triton can reject a tile for want of
    stack, scratch, or local memory and surface it only as a backend fatal
    error. Ascend's BiShengIR reports "The total stack object size (20256)
    exceeded vf stack size (6144)" for FP32 at block=2048/unroll=16. That is
    the same class of exclusion as OutOfResources -- the configuration is too
    large, the dtype is not unsupported -- so it must skip that point in the
    sweep rather than abort the whole dtype. Kept deliberately narrow: only
    explicit capacity-exceeded wording matches, so a genuine compiler bug
    still raises.
    """
    message = str((exc.error_message or "") if isinstance(exc, CompilationError) else exc).lower()
    patterns = (
        r"exceeded\s+\w+\s+stack size",
        r"stack object size\s*\(\d+\)\s*exceeded",
        r"\b(?:ub|local memory|scratch|shared memory)\b[^\n]*\b(?:overflow|exceeded|exhausted)\b",
        r"requires \d+ bits while \d+ bits available",
    )
    return any(re.search(pattern, message) for pattern in patterns)


def _storage_failure_is_unsupported(exc):
    # Missing dtype allocation/copy support is distinct from memory exhaustion
    # and runtime/device faults, which must remain errors.
    return isinstance(exc, NotImplementedError) or any(
        term in str(exc).lower() for term in (
            "not implemented for", "unsupported dtype", "unsupported data type",
            "unsupported scalar type", "does not support float64", "does not support bfloat16",
        ))


def throughput_fields(dtype, elements, iterations, timing, *, int8_mul_add=False):
    per_update = 8 if dtype == "int8" and not int8_mul_add else 2
    operations = per_update * CHAINS * elements * iterations
    median = operations / (timing["median_ms"] * 1e9)
    peak = operations / (timing["min_ms"] * 1e9)
    if dtype == "int8":
        return {"counted_integer_ops": operations, "tops": median, "peak_tops": peak,
                "throughput_unit": "TOPS", "ops_per_accumulator_update": per_update}
    return {"counted_flops": operations, "tflops": median, "peak_tflops": peak,
            "throughput_unit": "TFLOP/s", "ops_per_accumulator_update": 2}


def run_benchmark(args, metadata, is_xpu):
    reason = unsupported_dtype_reason(args.dtype, metadata)
    if reason:
        raise UnsupportedVectorDtype(reason)
    # Resolve the device lazily: the status paths below must stay usable
    # without an accelerator runtime, so only a dtype that needs a compile
    # probe may touch it.
    reason = not_implemented_dtype_reason(
        args.dtype, (lambda: get_device(args)) if args.dtype == "fp8" else None)
    if reason:
        raise NotImplementedVectorDtype(reason)
    if not metadata.get("sm_count"):
        raise SystemExit("The runtime did not expose the compute-unit count; "
                         "supply the chip's count with --compute-units")
    device = get_device(args)
    is_int8 = args.dtype == "int8"
    # DP4A is not NVIDIA-only. gfx936 (Hygon BW1000) assembles
    # v_dot4_i32_i8, the AMDGCN spelling of the same four-int8-products-plus-
    # INT32-accumulate instruction, so the native path is offered there too
    # instead of silently falling back to the typed modulo-256 emulation that
    # measures roughly a quarter of the native rate. The probe decides, not the
    # vendor name: a target whose assembler rejects the mnemonic keeps the
    # emulated path and its unverified status.
    gcn_dp4a = (is_int8 and not is_xpu
                and metadata["vendor_name"] in VECTOR_AMDGCN_VENDORS)
    use_dp4a = is_int8 and not is_xpu and (
        metadata["vendor_name"] == "nvidia" or gcn_dp4a)
    dp4a_reason = None
    if use_dp4a:
        use_dp4a, dp4a_reason = backend_supports_dp4a(device, gcn_dp4a)
        if not use_dp4a:
            gcn_dp4a = False
    # On backends without NVIDIA DP4A, try typed INT8 arithmetic. If final ISA
    # cannot be audited, keep the numeric result but mark it unverified.
    int8_mul_add = is_int8 and not use_dp4a
    use_explicit_fma = not is_xpu and not is_int8
    fma_fallback_reason = None
    use_widened_fma = False
    widened_fma_reason = None
    if use_explicit_fma:
        # Probe before the sweep: a backend may support the dtype everywhere
        # except in a fused multiply-add. Falling back keeps a measurable
        # source multiply/add path instead of discarding the metric.
        use_explicit_fma, fma_fallback_reason = backend_supports_explicit_fma(args.dtype, device)
        if not use_explicit_fma:
            # tl.fma did not lower. Before settling for source multiply/add,
            # try an FP32-widened FMA with narrow state: on a backend that
            # emulates this dtype in FP32 anyway, that is the same arithmetic
            # the native path would perform, with the invariant conversions
            # hoisted out of the loop.
            use_widened_fma, widened_fma_reason = backend_supports_widened_fma(args.dtype, device)
    fully_static = needs_fully_static_loop(args.dtype, is_xpu)
    # Independent of which arithmetic form was selected above: the defect is in
    # lowering a scalar operand of this dtype, not in the multiply-add itself.
    broadcast_coefficients, broadcast_reason = (False, None) if is_int8 else \
        needs_broadcast_coefficients(args.dtype, device)
    source_operation = ("INT8_MUL_ADD_MOD256" if int8_mul_add else "DP4A") if is_int8 else (
        "FMA" if use_explicit_fma else "WIDENED_FMA" if use_widened_fma else "MUL_ADD")
    median_key, peak_key = ("tops", "peak_tops") if is_int8 else ("tflops", "peak_tflops")
    unit = "TOPS" if is_int8 else "TFLOP/s"
    print(f"{args.dtype.upper()} vector: {CHAINS} independent {source_operation} chains, "
          f"{args.iterations} iterations; {8 if use_dp4a else 2} operations per update")
    if int8_mul_add:
        print("Typed INT8 path uses INT8 input, coefficients, state and output with modulo-256 multiply/add; "
              "native instruction verification requires explicit final ISA evidence")
        if dp4a_reason:
            print(dp4a_reason)
    elif is_int8:
        print("INT8 lanes are packed in INT32 words; each dot4 uses four signed products "
              "and an INT32 bias, with its result supplying the next packed input")
    elif is_xpu:
        print("XPU source uses multiply plus add for FlagTree vectorization; "
              "hardware fusion and native SIMD instructions are unverified")
    if fma_fallback_reason:
        print(fma_fallback_reason)
        print("Hardware fusion is therefore unverified: the measured rate reflects the "
              "source multiply/add workload, not a verified fused native peak")
    if use_widened_fma:
        print(f"Using an FP32-widened FMA with {args.dtype.upper()} state: one rounding per "
              "update, and the loop-invariant coefficients are widened once outside the loop")
        print("This is the arithmetic a backend without native "
              f"{args.dtype.upper()} units performs anyway; it is still not a native "
              f"{args.dtype.upper()} vector peak")
    elif widened_fma_reason:
        print(widened_fma_reason)
    if fully_static:
        print(f"Loop is fully unrolled: the FlagTree XPU compiler cannot lower an scf.for "
              f"carrying {args.dtype.upper()} state, so every iteration is emitted inline")
    if broadcast_reason:
        print(broadcast_reason)
    if args.quick:
        print("QUICK smoke mode: throughput is not a representative performance result")
    print(f"blocks/unit  block  warps  unroll  registers  median_ms  median_{unit}  peak_{unit}")
    rows = []
    excluded = []

    def exclude(kind, reason, blocks_per_sm, block_size, num_warps, unroll):
        config = {"blocks_per_sm": blocks_per_sm, "block_size": block_size,
                  "num_warps": num_warps, "unroll": unroll}
        excluded.append({"status": kind, "reason": str(reason), **config})
        print(f"Excluding vector configuration ({kind}): {config}: {reason}")
    for blocks_per_sm in args.blocks_per_sm:
        blocks = metadata["sm_count"] * blocks_per_sm
        for block_size in args.block_sizes:
            elements = blocks * block_size
            itemsize = 4 if use_dp4a else torch.empty((), dtype=DTYPES[args.dtype]).element_size()
            required_bytes = 2 * CHAINS * elements * itemsize
            free_bytes = available_memory_bytes()
            if free_bytes is not None and required_bytes > free_bytes * 0.5:
                raise SystemExit(f"Vector input/output need {required_bytes / 2**30:.2f} GiB; "
                                 "reduce --blocks-per-sm or --block-sizes")
            try:
                inputs, coefficients = _make_inputs(args.dtype, elements, device, int8_mul_add=int8_mul_add)
                # The kernel may need coefficients expanded, but the reference
                # indexes them by chain, so validation keeps the compact form.
                kernel_coefficients = (_expand_coefficients(coefficients, block_size)
                                       if broadcast_coefficients else coefficients)
                output = torch.empty_like(inputs)
            except Exception as exc:
                if _storage_failure_is_unsupported(exc):
                    raise UnsupportedVectorDtype(
                        f"Backend storage/copy support for {args.dtype.upper()} is unavailable: {exc}") from exc
                raise
            for num_warps in args.num_warps:
                for unroll in args.unroll:
                    _fill_sentinel(output, -128 if int8_mul_add else -(2**31) if is_int8 else float("nan"))

                    def run(iterations=args.iterations, unroll_value=unroll):
                        return launch_vector(
                            vector_fma_kernel, (blocks,), inputs, output, kernel_coefficients,
                            elements, iterations, block_size, unroll_value,
                            use_explicit_fma, num_warps, is_int8, use_dp4a,
                            fully_static, use_widened_fma, broadcast_coefficients,
                            gcn_dp4a,
                        )

                    try:
                        compiled = run()
                        assembly = inspect_assembly(compiled, metadata["vendor_name"], args.dtype,
                                                    metadata.get("backend"))
                    except triton.OutOfResources as exc:
                        exclude("resource_limit", exc, blocks_per_sm, block_size, num_warps, unroll)
                        continue
                    except Exception as exc:
                        if isinstance(exc, UnsupportedVectorDtype) or _compile_failure_is_unsupported(exc):
                            exclude("unsupported", exc, blocks_per_sm, block_size, num_warps, unroll)
                            continue
                        if _compile_failure_is_resource_limit(exc):
                            exclude("resource_limit", exc, blocks_per_sm, block_size, num_warps, unroll)
                            continue
                        if isinstance(exc, CompilationError):
                            exclude("compilation_error", exc, blocks_per_sm, block_size, num_warps, unroll)
                            continue
                        raise
                    if assembly["register_spills"]:
                        exclude("register_spills", f"spills={assembly['register_spills']}",
                                blocks_per_sm, block_size, num_warps, unroll)
                        continue
                    if int8_mul_add:
                        correctness = check_int8_mul_add_output(inputs, output, coefficients, args.iterations)
                        # INT8 has no NaN sentinel. Revalidate the same run with
                        # a different poison to detect every unwritten lane.
                        output.fill_(127)
                        run()
                        check_int8_mul_add_output(inputs, output, coefficients, args.iterations)
                        # Modular recurrences can cycle back to their inputs;
                        # an adjacent iteration count catches an identity copy.
                        output.fill_(-128)
                        run(args.iterations + 1, 1)
                        check_int8_mul_add_output(inputs, output, coefficients, args.iterations + 1)
                        correctness["write_coverage_probe"] = "both INT8 poison values passed"
                        correctness["adjacent_iteration_probe"] = "all elements passed"
                    elif is_int8:
                        correctness = check_int8_output(inputs, output, coefficients, args.iterations)
                    else:
                        # A widened FMA rounds once per update, like tl.fma, so
                        # it earns the same single-rounding error bound.
                        single_rounding = use_explicit_fma or use_widened_fma
                        correctness = check_output(inputs, output, coefficients, args.iterations,
                                                   single_rounding)
                        if args.dtype in ("bf16", "fp16") and args.iterations % 2 == 0:
                            output.fill_(float("nan"))
                            run(args.iterations + 1, 1)
                            check_output(inputs, output, coefficients, args.iterations + 1,
                                         single_rounding)
                            correctness["odd_iteration_probe"] = "all elements passed"
                    timing = measure(run, args)
                    # Count only repeated arithmetic. Initialization, loop control,
                    # input/coefficient loads and final stores remain in elapsed time.
                    rates = throughput_fields(args.dtype, elements, args.iterations, timing,
                                              int8_mul_add=int8_mul_add)
                    arithmetic_dtype = ("int8" if assembly["native_arithmetic_verified"] else "unverified") if int8_mul_add else (
                        "int8 products with int32 sum" if is_int8 else args.dtype)
                    row = {
                        "status": "ok", "requested_dtype": args.dtype,
                        "dtype": str(DTYPES[args.dtype]).removeprefix("torch."),
                        "input_storage_dtype": "int8 packed in int32" if use_dp4a else args.dtype,
                        "accumulator_dtype": "int32" if use_dp4a else args.dtype,
                        "source_arithmetic_dtype": args.dtype,
                        "arithmetic_dtype": arithmetic_dtype,
                        "includes_unpacking": False,
                        "operation": source_operation,
                        "loop_structure": "fully_static" if fully_static else "strip_mined",
                        "coefficient_load": ("broadcast_vector" if broadcast_coefficients
                                            else "scalar"),
                        "hardware_fusion_verified": bool(use_explicit_fma
                                                         and assembly["native_arithmetic_verified"]),
                        "explicit_fma_supported": use_explicit_fma,
                        "dp4a_used": use_dp4a,
                        "dp4a_probe_reason": dp4a_reason,
                        "fma_fallback_reason": fma_fallback_reason,
                        "widened_fma_used": use_widened_fma,
                        "widened_fma_reason": widened_fma_reason,
                        "roundings_per_update": 1 if (use_explicit_fma or use_widened_fma) else 2,
                        "accumulator_chains": CHAINS,
                        "iterations": args.iterations,
                        "blocks_per_sm": blocks_per_sm,
                        "blocks": blocks,
                        "block_size": block_size,
                        "num_warps": num_warps,
                        "unroll": unroll,
                        "elements_per_chain": elements,
                        "smoke_only": args.quick,
                        **rates, **timing, **correctness, **assembly,
                    }
                    rows.append(row)
                    registers = assembly["registers_per_thread"]
                    registers_label = str(registers) if registers is not None else "n/a"
                    warps_label = str(num_warps) if num_warps is not None else "n/a"
                    print(f"{blocks_per_sm:11d}  {block_size:5d}  {warps_label:>5}  {unroll:6d}  "
                          f"{registers_label:>9}  {timing['median_ms']:9.5f}  "
                          f"{rates[median_key]:11.3f}  {rates[peak_key]:9.3f}")
            del inputs, output, coefficients
    if not rows:
        reasons = "; ".join(dict.fromkeys(item["reason"] for item in excluded))
        if excluded and all(item["status"] == "unsupported" for item in excluded):
            raise UnsupportedVectorDtype(
                f"No tested vector configuration provides the requested {args.dtype.upper()} "
                f"arithmetic path: {reasons}")
        raise RuntimeError("No vector configuration produced a valid measurement; "
                           f"compiler/configuration/resource failures: {reasons}")
    best = max(rows, key=lambda row: row[peak_key])
    tuning = (f"blocks/unit={best['blocks_per_sm']}, block={best['block_size']}, "
              f"unroll={best['unroll']}")
    if best["num_warps"] is not None:
        tuning += f", warps={best['num_warps']}"
    print(f"Peak measured {args.dtype.upper()} vector workload throughput: {best[peak_key]:.3f} {unit} "
          f"(median {best[median_key]:.3f} {unit}; {tuning})")
    if int8_mul_add and not best["native_arithmetic_verified"]:
        print("This numeric result measures the typed INT8 source workload; "
              "it is not a verified native INT8 vector peak")
    result_name = f"vector_{args.dtype}_" + ("dot4" if use_dp4a else
                                            "fma" if use_explicit_fma else
                                            "widened_fma" if use_widened_fma else "mul_add")
    result_metadata = {**metadata, "vector_excluded_configurations": excluded}
    emit_results(result_name, result_metadata, args, rows)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.block_sizes is not None and any(size & (size - 1) for size in args.block_sizes):
        parser.error("--block-sizes must contain only powers of two")
    if args.iterations >= 2**31:
        parser.error("--iterations must be less than 2**31")
    if args.unroll is not None and any(args.iterations % unroll for unroll in args.unroll):
        parser.error("--iterations must be divisible by every --unroll factor")
    metadata = setup_device(args)
    is_xpu = configure_sweep(args, metadata)
    if any(args.iterations % unroll for unroll in args.unroll):
        parser.error("--iterations must be divisible by every --unroll factor")
    reason = unsupported_dtype_reason(args.dtype, metadata)
    if reason:
        if args.dtype == "fp4":
            # Record the checked toolchain facts next to the refusal so the
            # status can be re-verified from the result file alone.
            metadata["vector_fp4_capability_probe"] = _fp4_vector_rejection_detail()
        emit_unsupported(f"vector_{args.dtype}", metadata, args, reason)
        return
    # Resolve the device lazily: the status paths below must stay usable
    # without an accelerator runtime, so only a dtype that needs a compile
    # probe may touch it.
    reason = not_implemented_dtype_reason(
        args.dtype, (lambda: get_device(args)) if args.dtype == "fp8" else None)
    if reason:
        emit_not_implemented(f"vector_{args.dtype}", metadata, args, reason)
        return
    try:
        run_benchmark(args, metadata, is_xpu)
    except UnsupportedVectorDtype as exc:
        emit_unsupported(f"vector_{args.dtype}", metadata, args, str(exc))
    except NotImplementedVectorDtype as exc:
        emit_not_implemented(f"vector_{args.dtype}", metadata, args, str(exc))


if __name__ == "__main__":
    main()
