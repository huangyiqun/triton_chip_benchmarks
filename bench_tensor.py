#!/usr/bin/env python3
"""Measure tensor-unit peak throughput or sustained dense Triton GEMM.

Peak mode keeps operands resident while repeatedly issuing matrix MMA
instructions on two independent accumulator chains. GEMM mode measures
application-style matrix multiplication. FP16/BF16/FP8 inputs use FP32
accumulator registers/output; native FP8 internal precision is hardware
dependent. One multiply plus one add counts as two FLOPs.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl

try:
    from .common import (
        add_common_arguments, available_memory_bytes, emit_results,
        full_precision_matmul, get_device, measure, setup_device,
    )
except ImportError:
    from common import (
        add_common_arguments, available_memory_bytes, emit_results,
        full_precision_matmul, get_device, measure, setup_device,
    )


@triton.jit
def _gemm_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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
    accum = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k = (k_tile * BLOCK_K + inner).to(tl.int64)
        a = tl.load(A + rows[:, None] * K + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B + k[:, None] * N + cols[None, :],
                    mask=(k[:, None] < K) & (cols[None, :] < N), other=0.0)
        accum = tl.dot(a, b, accum)
    tl.store(C + rows[:, None] * N + cols[None, :], accum,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def _configs(vendor: str) -> list[dict[str, int]]:
    # An explicit small sweep keeps compilation/tuning costs visible. Some
    # older GPUs may reject the larger tiles due to shared-memory limits.
    if vendor == "amd":
        choices = [(64, 64, 64, 4, 2), (128, 128, 64, 8, 2),
                   (64, 128, 64, 4, 2)]
    elif vendor == "nvidia":
        choices = [(64, 128, 32, 4, 3), (128, 128, 64, 8, 3),
                   (128, 256, 64, 8, 3), (64, 128, 64, 4, 4)]
    else:
        choices = [(32, 32, 32, 4, 1), (64, 64, 32, 4, 1)]
    return [dict(block_m=m, block_n=n, block_k=k, num_warps=w,
                 num_stages=s) for m, n, k, w, s in choices]


def _launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
            config: dict[str, int]):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, config["block_m"]) * triton.cdiv(n, config["block_n"]),)
    return _gemm_kernel[grid](
        a, b, c, m, n, k,
        BLOCK_M=config["block_m"], BLOCK_N=config["block_n"],
        BLOCK_K=config["block_k"], GROUP_M=8,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )


def _inputs(m: int, n: int, k: int, dtype: torch.dtype, device: torch.device):
    # Scaling limits the absolute output magnitude and makes one tolerance
    # useful across large and small K. It does not change the operation count.
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device) * (1.0 / math.sqrt(k))
    c = torch.empty((m, n), dtype=torch.float32, device=device)
    return a, b, c


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The FlagGems backend context disables reduced-precision modes where
    # the selected vendor runtime exposes the corresponding controls.
    with full_precision_matmul():
        return torch.matmul(a.float(), b.float())


def _validate(c: torch.Tensor, reference: torch.Tensor) -> dict:
    torch.testing.assert_close(c, reference, rtol=2e-3, atol=2e-3)
    max_abs_error = (c - reference).abs().max().item()
    return {"passed": True, "max_abs_error": max_abs_error,
            "rtol": 2e-3, "atol": 2e-3,
            "reference": "FP32 torch.matmul under FlagGems full-precision context"}


@triton.jit
def _tensor_peak_kernel(
    A, B, C,
    BLOCKS: tl.constexpr, ITERATIONS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    # Each operand is read from global memory once per CTA, before the loop.
    # Distinct B matrices prevent merging the two independent accumulators.
    a = tl.load(A + block * BLOCK_M * BLOCK_K + rows[:, None] * BLOCK_K + inner[None, :])
    b_offsets = block * BLOCK_K * BLOCK_N + inner[:, None] * BLOCK_N + cols[None, :]
    b0 = tl.load(B + b_offsets)
    b1 = tl.load(B + BLOCKS * BLOCK_K * BLOCK_N + b_offsets)
    acc0 = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
    acc1 = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
    # Keep a real runtime loop (not static_range) to avoid code-size growth.
    for _ in range(ITERATIONS):
        # Keep each vendor backend's native dot policy. For example, forcing
        # max_num_imprecise_acc=0 on Hopper can demote FP8 to FP16 MMA.
        acc0 = tl.dot(a, b0, acc0)
        acc1 = tl.dot(a, b1, acc1)
    c_offsets = block * BLOCK_M * BLOCK_N + rows[:, None] * BLOCK_N + cols[None, :]
    tl.store(C + c_offsets, acc0)
    tl.store(C + BLOCKS * BLOCK_M * BLOCK_N + c_offsets, acc1)


def _peak_configs(vendor: str, dtype: str = "fp16") -> list[dict[str, int]]:
    k = 64 if dtype == "fp8" else 32
    if vendor in ("nvidia", "amd"):
        choices = [(64, 64, k, 4), (64, 128, k, 4), (128, 64, k, 4)]
    else:
        choices = [(32, 32, k, 4), (64, 64, k, 4)]
    return [dict(block_m=m, block_n=n, block_k=k, num_warps=w,
                 num_stages=1, chains=2) for m, n, k, w in choices]


def _launch_peak(a, b, c, iterations: int, config: dict[str, int]):
    blocks = a.shape[0]
    return _tensor_peak_kernel[(blocks,)](
        a, b, c, BLOCKS=blocks, ITERATIONS=iterations,
        BLOCK_M=config["block_m"], BLOCK_N=config["block_n"], BLOCK_K=config["block_k"],
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )


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
    else:
        a = torch.randint(-1, 2, (blocks, m, k), device=device).float().mul_(1 / 16).to(dtype)
        b = torch.randint(-1, 2, (2, blocks, k, n), device=device).float().mul_(1 / 16).to(dtype)
    c = torch.empty((2, blocks, m, n), dtype=torch.float32, device=device)
    return a, b, c


def _compiled_info(compiled, vendor: str, dtype: str = "fp16") -> dict:
    # Device type is not a vendor identifier: several FlagGems vendors expose
    # a CUDA-compatible torch module while emitting a different instruction set.
    assemblies = getattr(compiled, "asm", {}) or {}
    instructions = []
    audit = "unavailable"
    if vendor in ("nvidia", "amd"):
        assembly = assemblies.get("ptx" if vendor == "nvidia" else "amdgcn", "")
        if vendor == "nvidia":
            instructions = [line.strip().split()[0] for line in assembly.splitlines()
                            if "mma.sync." in line or "wgmma.mma_async." in line
                            or "tcgen05.mma." in line]
        else:
            instructions = [line.strip().split()[0] for line in assembly.splitlines()
                            if "v_mfma_" in line or "v_wmma_" in line]
        if not instructions:
            raise RuntimeError(f"No recognized {vendor} tensor MMA instructions in the compiled peak kernel")
        if dtype == "fp8":
            native_fp8 = (all("e4m3" in name for name in instructions) if vendor == "nvidia"
                          else all("fp8_fp8" in name for name in instructions))
            if not native_fp8:
                raise RuntimeError(
                    f"Cannot verify native E4M3 FP8 MMA for {vendor}; compiled instructions: "
                    f"{sorted(set(instructions))}. The selected target may not support E4M3FN "
                    "or this compiler may lower it to another dtype."
                )
        audit = "verified"
    return {"registers_per_thread": getattr(compiled, "n_regs", None),
            "spill_count": getattr(compiled, "n_spills", None),
            "shared_memory_bytes": getattr(getattr(compiled, "metadata", None), "shared", None),
            "assembly_audit": audit, "native_tensor_verified": audit == "verified",
            "assembly_audit_note": None if audit == "verified" else
            f"No instruction auditor for FlagGems vendor {vendor}; native tensor execution is unverified",
            "tensor_instruction_types": sorted(set(instructions)),
            "static_tensor_instruction_count": len(instructions) if audit == "verified" else None}


def run_peak(args, metadata: dict, dtype: torch.dtype, device) -> None:
    vendor = metadata["vendor_name"]
    if not metadata.get("sm_count"):
        raise RuntimeError(
            f"The {vendor} runtime did not report the compute-unit count; "
            "provide --compute-units for the resident tensor peak grid."
        )
    candidates = _peak_configs(vendor, args.dtype)
    if args.quick:
        candidates = candidates[:1]
        args.blocks_per_sm = [1]
        args.iterations = min(args.iterations, 64)
    print(f"\nTensor peak: resident {args.dtype} operands, two independent FP32 MMA chains")
    if args.dtype == "fp8":
        print("FP8 E4M3FN: backend accumulation policy (internal precision is hardware-dependent); "
              "exact-check inputs")
    if vendor not in ("nvidia", "amd"):
        print(f"Instruction audit unavailable for {vendor}; native tensor execution is unverified.")
    print(f"Iterations={args.iterations}; FLOPs=2*tile_M*tile_N*tile_K*iterations*chains*blocks")
    print(" tile M/N/K  blocks/SM  warps  median ms    min ms  median TF/s  peak TF/s  regs")
    rows = []
    for config in candidates:
        for blocks_per_sm in args.blocks_per_sm:
            blocks = blocks_per_sm * metadata["sm_count"]
            m, n, k = (config[f"block_{axis}"] for axis in ("m", "n", "k"))
            free_bytes = available_memory_bytes()
            if free_bytes is not None and blocks * (m * k + 2 * k * n + 8 * m * n) * 8 > free_bytes * 0.8:
                raise RuntimeError("Peak validation exceeds available memory; reduce --blocks-per-sm")
            a, b, c = _peak_inputs(blocks, config, dtype, device)
            c.fill_(float("nan"))
            try:
                compiled = _launch_peak(a, b, c, args.iterations, config)
                info = _compiled_info(compiled, vendor, args.dtype)
                reference = torch.stack((_reference(a, b[0]), _reference(a, b[1])))
                reference.mul_(args.iterations)
                torch.testing.assert_close(c, reference, rtol=0, atol=0)
                if info["spill_count"]:
                    raise RuntimeError(f"Peak configuration spills registers: {config}, {info}")
                timing = measure(lambda: _launch_peak(a, b, c, args.iterations, config), args)
            except triton.OutOfResources as exc:
                print(f"  config={config}: skipped (resource limit: {exc})")
                rows.append({"mode": "peak", "config": config, "blocks_per_sm": blocks_per_sm,
                             "status": "skipped", "reason": str(exc)})
                del a, b, c
                continue
            flops = 2 * m * n * k * args.iterations * config["chains"] * blocks
            tflops = flops / (timing["median_ms"] * 1e9)
            peak_tflops = flops / (timing.get("min_ms", timing["median_ms"]) * 1e9)
            row = {"mode": "peak", "dtype": args.dtype, "config": config,
                   "measurement_kind": "native_tensor_peak" if info["native_tensor_verified"]
                   else "tl_dot_throughput_unverified",
                   "accumulator_dtype": "fp32", "output_dtype": "fp32", "blocks": blocks,
                   "blocks_per_sm": blocks_per_sm, "iterations": args.iterations,
                   "flops": flops, "tflops": tflops, "peak_tflops": peak_tflops,
                   "input_format": "e4m3fn" if args.dtype == "fp8" else args.dtype,
                   "input_pattern": "random one-hot A; random signed B; dense tl.dot operations"
                   if args.dtype == "fp8" else "random ternary exact binary fractions",
                   "accumulation_policy": "backend FP8 accumulation; internal precision is hardware-dependent"
                   if args.dtype == "fp8" else "FP32 tensor accumulation",
                   "correctness": {"passed": True, "max_abs_error": 0.0,
                                   "rtol": 0, "atol": 0, "reference": "exact FP32 dot * iterations"},
                   "compiled": info, "status": "ok", **timing}
            rows.append(row)
            tile = f"{m}/{n}/{k}"
            registers = info["registers_per_thread"]
            registers_text = "?" if registers is None else str(registers)
            print(f"{tile:>11}  {blocks_per_sm:9d}  {config['num_warps']:5d} "
                  f"{timing['median_ms']:10.4f}  {timing.get('min_ms', timing['median_ms']):8.4f} "
                  f"{tflops:12.2f} {peak_tflops:10.2f} {registers_text:>5}", flush=True)
            del a, b, c, reference
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("No supported tensor peak configuration")
    best = max(successful, key=lambda row: row["peak_tflops"])
    label = "Measured tensor peak" if best["compiled"]["native_tensor_verified"] else "Measured tl.dot throughput (native execution unverified)"
    print(f"{label}: {best['peak_tflops']:.2f} TFLOP/s "
          f"(same configuration median {best['tflops']:.2f} TFLOP/s; "
          f"best observed timing sample, dense {args.dtype})")
    emit_results("tensor", metadata, args, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("peak", "gemm"), default="peak",
                        help="resident MMA peak (default), or application-style GEMM")
    parser.add_argument("--iterations", type=int, default=1024,
                        help="resident MMA repeats, 1..262144; FP8 maximum 4096 (default: 1024)")
    parser.add_argument("--blocks-per-sm", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="launched CTAs per SM/CU (default: 1 2 4 8)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2048, 4096, 8192],
                        help="square GEMM dimensions M=N=K (default: 2048 4096 8192)")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp8"), default="fp16",
                        help="input dtype; fp8 means E4M3FN (peak mode, requires backend support); output FP32")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: one tile and 64 repeats, or GEMM dimensions 256/513")
    add_common_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes):
        parser.error("--sizes values must be positive")
    if not 1 <= args.iterations <= 262144:
        parser.error("--iterations must be between 1 and 262144 for exact FP32 validation")
    if any(value <= 0 for value in args.blocks_per_sm):
        parser.error("--blocks-per-sm values must be positive")
    if args.dtype == "fp8" and args.mode != "peak":
        parser.error("FP8 E4M3FN is supported in --mode peak; GEMM supports fp16/bf16")
    if args.dtype == "fp8" and args.iterations > 4096:
        parser.error("native FP8 peak limits --iterations to 4096 for exact internal accumulation")
    if args.quick:
        args.sizes = [256, 513]
    metadata = setup_device(args)
    dtype_name = {"fp16": "float16", "bf16": "bfloat16", "fp8": "float8_e4m3fn"}[args.dtype]
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        parser.error(f"this PyTorch build does not expose {dtype_name}")
    device = get_device(args)
    if dtype == torch.bfloat16 and metadata.get("supports_bf16") is False:
        parser.error(f"the selected {metadata['vendor_name']} device does not support BF16")
    torch.manual_seed(0)
    if args.mode == "peak":
        run_peak(args, metadata, dtype, device)
        return
    candidates = _configs(metadata["vendor_name"])
    if args.quick:
        candidates = candidates[:1]

    print(f"\nTensor GEMM: {args.dtype} inputs, FP32 accumulate/output; 2*M*N*K FLOPs")
    print("  N        tile M/N/K    warps stages   median ms  median TF/s  peak TF/s   max abs error")
    rows = []
    for size in args.sizes:
        # Inputs, output, FP32 reference and conversion/validation temporaries.
        # Query currently available memory rather than total device capacity.
        required = size * size * 32
        free_bytes = available_memory_bytes()
        if free_bytes is not None and required > free_bytes * 0.8:
            raise RuntimeError(
                f"N={size} validation needs about {required / 2**30:.2f} GiB; "
                f"only {free_bytes / 2**30:.2f} GiB is currently free. "
                "Reduce --sizes."
            )
        a, b, c = _inputs(size, size, size, dtype, device)
        reference = _reference(a, b)
        valid_rows = []
        for config in candidates:
            try:
                c.fill_(float("nan"))  # Detect unwritten output after any config.
                _launch(a, b, c, config)  # Compile before validation and timing.
                correctness = _validate(c, reference)
                timing = measure(lambda: _launch(a, b, c, config), args)
            except triton.OutOfResources as exc:
                print(f"  N={size} config={config}: skipped (resource limit: {exc})")
                rows.append({"m": size, "n": size, "k": size, "dtype": args.dtype,
                             "config": config, "status": "skipped", "reason": str(exc)})
                continue
            tflops = 2 * size**3 / (timing["median_ms"] * 1e9)
            peak_tflops = 2 * size**3 / (timing.get("min_ms", timing["median_ms"]) * 1e9)
            row = {"mode": "gemm", "m": size, "n": size, "k": size, "dtype": args.dtype,
                   "accumulator_dtype": "fp32", "output_dtype": "fp32",
                   "flops": 2 * size**3, "config": config, "status": "ok",
                   "correctness": correctness, "tflops": tflops,
                   "peak_tflops": peak_tflops, **timing}
            rows.append(row)
            valid_rows.append(row)
            tile = f"{config['block_m']}/{config['block_n']}/{config['block_k']}"
            print(f"{size:5d}   {tile:>14}   {config['num_warps']:5d} "
                  f"{config['num_stages']:6d}   {timing['median_ms']:9.4f} "
                  f"{tflops:11.2f} {peak_tflops:10.2f}   {correctness['max_abs_error']:.3g}", flush=True)
        if not valid_rows:
            raise RuntimeError(f"No supported GEMM configuration for N={size}")
        del a, b, c, reference

    successful = [row for row in rows if row["status"] == "ok"]
    best = max(successful, key=lambda row: row["tflops"])
    print(f"Best measured tensor throughput: {best['tflops']:.2f} TFLOP/s "
          f"(N={best['n']}, {args.dtype}; sustained dense GEMM)")
    emit_results("tensor", metadata, args, rows)


if __name__ == "__main__":
    main()
