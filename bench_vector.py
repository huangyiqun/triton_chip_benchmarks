#!/usr/bin/env python3
"""Measure sustained FP32 vector FMA throughput with eight independent chains."""

import argparse
import math
import re

import torch
import triton
import triton.language as tl

from common import (add_common_arguments, available_memory_bytes, emit_results,
                    get_device, measure, positive_int, reference_device, setup_device)


CHAINS = 8


@triton.jit
def vector_fma_kernel(Input, Output, Coefficients, ELEMENTS: tl.constexpr,
                      ITERATIONS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                      UNROLL: tl.constexpr):
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
    a = tl.load(Coefficients)
    b0 = tl.load(Coefficients + 1)
    b1 = tl.load(Coefficients + 2)
    b2 = tl.load(Coefficients + 3)
    b3 = tl.load(Coefficients + 4)
    b4 = tl.load(Coefficients + 5)
    b5 = tl.load(Coefficients + 6)
    b6 = tl.load(Coefficients + 7)
    b7 = tl.load(Coefficients + 8)

    # Explicitly unroll a small group of iterations to amortize loop control.
    # The CLI requires exact divisibility, so each lane still executes exactly
    # ITERATIONS * 8 FP32 FMAs. Independent chains hide dependent-FMA latency.
    for _ in tl.range(0, ITERATIONS // UNROLL, loop_unroll_factor=1):
        for _ in tl.static_range(UNROLL):
            x0 = tl.fma(x0, a, b0)
            x1 = tl.fma(x1, a, b1)
            x2 = tl.fma(x2, a, b2)
            x3 = tl.fma(x3, a, b3)
            x4 = tl.fma(x4, a, b4)
            x5 = tl.fma(x5, a, b5)
            x6 = tl.fma(x6, a, b6)
            x7 = tl.fma(x7, a, b7)

    # Every accumulator is observable, preventing dead-chain elimination.
    tl.store(Output + offsets + 0 * ELEMENTS, x0)
    tl.store(Output + offsets + 1 * ELEMENTS, x1)
    tl.store(Output + offsets + 2 * ELEMENTS, x2)
    tl.store(Output + offsets + 3 * ELEMENTS, x3)
    tl.store(Output + offsets + 4 * ELEMENTS, x4)
    tl.store(Output + offsets + 5 * ELEMENTS, x5)
    tl.store(Output + offsets + 6 * ELEMENTS, x6)
    tl.store(Output + offsets + 7 * ELEMENTS, x7)


def check_output(inputs, output, coefficients, iterations):
    """Check every output against a double-precision closed-form recurrence.

    With 0 < a < 1, the recurrence stays between its initial value and b/(1-a).
    Each FP32 FMA incurs at most u * max_magnitude absolute rounding error,
    where u = 2**-24. Summing a geometric series bounds the accumulated error.
    A factor of two allows for the rounding of intermediate recurrence values.
    """
    elements = output.shape[1]
    # Copy first: several accelerators do not support FP64 device arithmetic.
    coeff = coefficients.cpu().double()
    validation_device = reference_device(output.device)
    a = coeff[0].item()
    decay = math.exp(iterations * math.log(a))
    geometric_sum = -math.expm1(iterations * math.log(a)) / (1.0 - a)
    b = coeff[1:, None].to(validation_device)
    max_error, max_tolerance = 0.0, 0.0
    # Bound temporary FP64 allocations even for unusually large user grids.
    for start in range(0, elements, 131072):
        initial = inputs[:, start:start + 131072].to(validation_device).double()
        actual = output[:, start:start + 131072].to(validation_device).double()
        reference = decay * initial + b * geometric_sum
        max_magnitude = torch.maximum(initial.abs(), (b / (1.0 - a)).abs())
        tolerance = 2.0 * (2.0**-24) * max_magnitude * geometric_sum + 1e-7
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
    }


def inspect_assembly(compiled, vendor_name):
    """Check NVIDIA PTX; retain optional metadata on other FlagGems devices."""
    result = {"registers_per_thread": getattr(compiled, "n_regs", None),
              "register_spills": getattr(compiled, "n_spills", None),
              "assembly_check": "unavailable for this vendor"}
    # Some non-NVIDIA runtimes also expose a backend named "cuda".
    if vendor_name == "nvidia":
        ptx = compiled.asm["ptx"]
        fma_count = len(re.findall(r"\bfma\.(?:rn\.)?(?:ftz\.)?f32\b", ptx))
        tensor_ops = bool(re.search(r"\b(?:mma|wgmma|tcgen05)\.", ptx))
        if fma_count == 0 or tensor_ops:
            raise RuntimeError("Generated PTX failed vector instruction check: "
                               f"FP32 FMA instructions={fma_count}, tensor ops={tensor_ops}")
        result.update({
            "ptx_fp32_fma_instruction_sites": fma_count,
            "ptx_contains_tensor_instructions": tensor_ops,
            "assembly_check": "passed (FP32 FMA; no tensor instructions)",
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--iterations", type=positive_int, default=4096,
                        help="FMA iterations per accumulator (default: 4096)")
    parser.add_argument("--blocks-per-sm", type=positive_int, nargs="+", default=[8, 16],
                        help="Grid size multipliers for SM/CU count (default: 8 16)")
    parser.add_argument("--block-sizes", type=positive_int, nargs="+", default=[256, 512],
                        help="Elements per program, powers of two (default: 256 512)")
    parser.add_argument("--num-warps", type=int, choices=[4, 8], nargs="+", default=[4, 8],
                        help="Warps per program to sweep (default: 4 8)")
    parser.add_argument("--unroll", type=int, choices=[1, 2, 4, 8, 16], nargs="+",
                        default=[1, 4, 8],
                        help="Explicit FMA loop unroll factors (default: 1 4 8)")
    parser.add_argument("--quick", action="store_true",
                        help="One smoke configuration: 32 iterations, block=128, warps=4, unroll=1")
    args = parser.parse_args()
    if args.quick:
        args.iterations, args.blocks_per_sm, args.block_sizes = 32, [1], [128]
        args.num_warps, args.unroll = [4], [1]
    if any(size & (size - 1) for size in args.block_sizes):
        parser.error("--block-sizes must contain only powers of two")
    if args.iterations >= 2**31:
        parser.error("--iterations must be less than 2**31")
    if any(args.iterations % unroll for unroll in args.unroll):
        parser.error("--iterations must be divisible by every --unroll factor")
    metadata = setup_device(args)
    if not metadata.get("sm_count"):
        raise SystemExit("The runtime did not expose the compute-unit count; "
                         "supply the chip's count with --compute-units")
    device = get_device(args)
    # Exact binary coefficients: fixed points b/(1-a) range from 0.25 to 2.0,
    # so even very long runs cannot overflow or collapse to a trivial zero.
    coefficients = torch.tensor([1.0 - 2.0**-14] +
                                [(chain + 1) * 2.0**-16 for chain in range(CHAINS)],
                                device=device, dtype=torch.float32)
    print(f"FP32 vector: {CHAINS} independent register FMA chains, "
          f"{args.iterations} iterations; 1 FMA = 2 FLOPs")
    if args.quick:
        print("QUICK smoke mode: throughput is not a representative performance result")
    print("blocks/SM  block  warps  unroll  registers  median_ms  median_TF/s  peak_TF/s")
    rows = []
    for blocks_per_sm in args.blocks_per_sm:
        blocks = metadata["sm_count"] * blocks_per_sm
        for block_size in args.block_sizes:
            elements = blocks * block_size
            required_bytes = 2 * CHAINS * elements * 4
            free_bytes = available_memory_bytes()
            if free_bytes is not None and required_bytes > free_bytes * 0.5:
                raise SystemExit(f"Vector input/output need {required_bytes / 2**30:.2f} GiB; "
                                 "reduce --blocks-per-sm or --block-sizes")
            inputs = torch.empty((CHAINS, elements), device=device, dtype=torch.float32)
            inputs.uniform_(0.125, 1.0)
            output = torch.empty_like(inputs)
            for num_warps in args.num_warps:
                for unroll in args.unroll:
                    output.fill_(float("nan"))

                    def run():
                        return vector_fma_kernel[(blocks,)](
                            inputs, output, coefficients, elements, args.iterations, block_size,
                            unroll, num_warps=num_warps,
                        )

                    compiled = run()
                    assembly = inspect_assembly(compiled, metadata["vendor_name"])
                    if assembly["register_spills"]:
                        print(f"Skipping spilled kernel: blocks/SM={blocks_per_sm}, "
                              f"block={block_size}, warps={num_warps}, unroll={unroll}, "
                              f"spills={assembly['register_spills']}")
                        continue
                    correctness = check_output(inputs, output, coefficients, args.iterations)
                    timing = measure(run, args)
                    # Count only repeated arithmetic. Initialization, loop control,
                    # input/coefficient loads and final stores remain in elapsed time.
                    flops = 2 * CHAINS * elements * args.iterations
                    tflops = flops / (timing["median_ms"] * 1e9)
                    peak_tflops = flops / (timing["min_ms"] * 1e9)
                    row = {
                        "dtype": "float32",
                        "operation": "FMA",
                        "accumulator_chains": CHAINS,
                        "iterations": args.iterations,
                        "blocks_per_sm": blocks_per_sm,
                        "blocks": blocks,
                        "block_size": block_size,
                        "num_warps": num_warps,
                        "unroll": unroll,
                        "elements_per_chain": elements,
                        "counted_flops": flops,
                        "tflops": tflops,
                        "peak_tflops": peak_tflops,
                        "smoke_only": args.quick,
                        **timing, **correctness, **assembly,
                    }
                    rows.append(row)
                    registers = assembly["registers_per_thread"]
                    registers_label = str(registers) if registers is not None else "n/a"
                    print(f"{blocks_per_sm:9d}  {block_size:5d}  {num_warps:5d}  {unroll:6d}  "
                          f"{registers_label:>9}  {timing['median_ms']:9.5f}  "
                          f"{tflops:11.3f}  {peak_tflops:9.3f}")
            del inputs, output
    if not rows:
        raise SystemExit("No vector configuration passed the register-spill check; "
                         "reduce --block-sizes or --unroll")
    best = max(rows, key=lambda row: row["peak_tflops"])
    print(f"Peak measured FP32 vector throughput: {best['peak_tflops']:.3f} TFLOP/s "
          f"(median {best['tflops']:.3f} TFLOP/s; "
          f"blocks/SM={best['blocks_per_sm']}, block={best['block_size']}, "
          f"warps={best['num_warps']}, unroll={best['unroll']})")
    emit_results("vector_fp32_fma", metadata, args, rows)


if __name__ == "__main__":
    main()
