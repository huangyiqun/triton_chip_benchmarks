#!/usr/bin/env python3
"""Search for peak achieved global-memory copy and read bandwidth with Triton."""

import argparse
import math

import torch
import triton
import triton.language as tl

from common import (add_common_arguments, available_memory_bytes, emit_results,
                    get_device, measure, positive_int, reference_device, setup_device)


@triton.jit
def copy_kernel(SRC, DST, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(SRC + offsets, mask=offsets < N, other=0)
    tl.store(DST + offsets, values, mask=offsets < N)


def launch_copy(src, dst, block_size, num_warps):
    return copy_kernel[(triton.cdiv(src.numel(), block_size),)](
        src, dst, src.numel(), BLOCK_SIZE=block_size, num_warps=num_warps,
    )


@triton.jit
def read_kernel(SRC, SINK, N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                CACHE_MODIFIER: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(SRC + offsets, mask=offsets < N, other=0,
                     cache_modifier=CACHE_MODIFIER)
    # Each loaded value contributes to an observable result, preventing the
    # compiler from deleting the read. The small sink traffic is counted too.
    total = tl.sum(values.to(tl.float32), axis=0)
    tl.store(SINK + tl.program_id(0), total)


def launch_read(src, sink, block_size, num_warps, vendor_name):
    return read_kernel[(triton.cdiv(src.numel(), block_size),)](
        src, sink, src.numel(), BLOCK_SIZE=block_size,
        CACHE_MODIFIER=".cg" if vendor_name == "nvidia" else "", num_warps=num_warps,
    )


def check_read(src, sink, block_size):
    """Validate every block sum against FP64 without a full-size FP64 copy."""
    maximum_error, maximum_tolerance = 0.0, 0.0
    validation_device = reference_device(src.device)
    # tl.sum uses a tree reduction. Bound FP32 rounding by its tree depth
    # times the sum of absolute inputs, with a factor of two for headroom.
    rounding_factor = 2 * torch.finfo(torch.float32).eps * max(1, math.log2(block_size))
    for start in range(0, src.numel(), 1 << 20):
        # Transfer the bounded FP32/FP16/BF16 chunk before any FP64 work when
        # the selected accelerator lacks double-precision arithmetic.
        chunk = src[start:start + (1 << 20)].to(validation_device)
        if chunk.numel() % block_size:
            matrix = torch.zeros((triton.cdiv(chunk.numel(), block_size), block_size),
                                 device=validation_device, dtype=src.dtype)
            matrix.view(-1)[:chunk.numel()].copy_(chunk)
        else:
            matrix = chunk.view(-1, block_size)
        reference = matrix.sum(dim=1, dtype=torch.float64)
        tolerance = matrix.abs().sum(dim=1, dtype=torch.float64) * rounding_factor + 1e-6
        actual = sink[start // block_size:start // block_size + reference.numel()]
        actual = actual.to(validation_device).double()
        error = (actual - reference).abs()
        if not torch.all(error <= tolerance).item():
            raise AssertionError("Read reduction validation failed: non-finite or inaccurate "
                                 f"output; max_abs_error={error.max().item():.6g}")
        maximum_error = max(maximum_error, error.max().item())
        maximum_tolerance = max(maximum_tolerance, tolerance.max().item())
    return {"correctness": "passed (all block sums against FP64, plus masked tail)",
            "max_abs_error": maximum_error, "max_abs_tolerance": maximum_tolerance,
            "reference_device": str(validation_device)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--mode", choices=["copy", "read", "all"], default="all",
                        help="Access patterns to measure separately (default: all)")
    parser.add_argument("--sizes-mib", type=positive_int, nargs="+", default=[256, 512, 1024, 2048],
                        help="Size of EACH source/destination buffer in MiB")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--block-sizes", type=positive_int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--num-warps", type=positive_int, nargs="+", default=[4, 8])
    parser.add_argument("--quick", action="store_true",
                        help="Smoke check: 8 MiB, one configuration per mode; not an HBM result")
    args = parser.parse_args()
    if args.quick:
        args.sizes_mib, args.block_sizes, args.num_warps = [8], [1024], [4]
    if any(block & (block - 1) or block > 65536 for block in args.block_sizes):
        parser.error("--block-sizes must be powers of two no greater than 65536")
    if any(warps not in (4, 8) for warps in args.num_warps):
        parser.error("--num-warps must be 4 or 8")
    return args


def main():
    args = parse_args()
    metadata = setup_device(args)
    device = get_device(args)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.dtype == "bf16" and metadata.get("supports_bf16") is False:
        raise SystemExit("The selected accelerator does not support BF16; choose --dtype fp16 or fp32")
    element_size = torch.empty((), dtype=dtype).element_size()
    l2_bytes = metadata["l2_cache_bytes"]
    modes = ["copy", "read"] if args.mode == "all" else [args.mode]
    configurations = [(block, warps) for block in args.block_sizes for warps in args.num_warps]

    # Exercise a masked tail independently of the aligned throughput workloads.
    for block, warps in configurations:
        small_src = torch.randn(block + 17, device=device, dtype=dtype)
        if "copy" in modes:
            small_dst = torch.full_like(small_src, float("nan"))
            launch_copy(small_src, small_dst, block, warps)
            if not torch.equal(small_src, small_dst):
                raise AssertionError(f"Copy tail validation failed: block={block}, warps={warps}")
            del small_dst
        if "read" in modes:
            small_sink = torch.full((triton.cdiv(small_src.numel(), block),), float("nan"),
                                    device=device, dtype=torch.float32)
            launch_read(small_src, small_sink, block, warps, metadata["vendor_name"])
            check_read(small_src, small_sink, block)
            del small_sink
    del small_src
    print("Validation: masked tails passed; every measured output is checked in full.")
    print("Copy traffic = 2 * buffer bytes; read traffic = buffer bytes + FP32 reduction sink. "
          "GB/s and TB/s use decimal units; peak uses the shortest timed sample.")
    print("mode   buffer MiB   block  warps      min ms   median ms   peak GB/s  median GB/s")
    rows = []
    for size_mib in args.sizes_mib:
        buffer_bytes = size_mib * 2**20
        n = buffer_bytes // element_size
        free_bytes = available_memory_bytes()
        # Keep room for the sink, timing cache and bounded validation temporaries.
        max_sink_bytes = triton.cdiv(n, min(args.block_sizes)) * 4 if "read" in modes else 0
        required_bytes = (2 if "copy" in modes else 1) * buffer_bytes + max_sink_bytes
        if free_bytes is not None and required_bytes + 512 * 2**20 > free_bytes:
            raise SystemExit(f"Insufficient free GPU memory for {size_mib} MiB buffers; "
                             "reduce --sizes-mib.")
        regime = ("unknown_l2" if not l2_bytes else
                  "larger_than_l2" if buffer_bytes >= 4 * l2_bytes else "cache_sensitive")
        peak_eligible = regime == "larger_than_l2" and not args.quick
        if not peak_eligible:
            reason = ("smoke mode" if args.quick else "L2 capacity is unavailable"
                      if not l2_bytes else "buffer is smaller than 4x L2")
            print(f"Note: {size_mib} MiB per buffer excluded from the HBM-sized "
                  f"peak summary: {reason}.")
        src = torch.empty(n, device=device, dtype=dtype).normal_()
        dst = torch.empty_like(src) if "copy" in modes else None
        for block, warps in configurations:
            for mode in modes:
                sink = None
                if mode == "copy":
                    # A sentinel catches missing writes when reusing dst.
                    dst.fill_(float("nan"))
                    run = lambda: launch_copy(src, dst, block, warps)
                    run()
                    if not torch.equal(src, dst):
                        raise AssertionError(f"Copy validation failed: size={size_mib}, "
                                             f"block={block}, warps={warps}")
                    correctness = {"correctness": "passed (full output and masked tail)"}
                    write_bytes = buffer_bytes
                else:
                    sink = torch.full((triton.cdiv(n, block),), float("nan"),
                                      device=device, dtype=torch.float32)
                    run = lambda: launch_read(src, sink, block, warps, metadata["vendor_name"])
                    run()
                    correctness = check_read(src, sink, block)
                    write_bytes = sink.numel() * sink.element_size()
                timings = measure(run, args)
                traffic_bytes = buffer_bytes + write_bytes
                gbps = traffic_bytes / (timings["median_ms"] * 1e6)
                peak_gbps = traffic_bytes / (timings["min_ms"] * 1e6)
                rows.append({
                    "mode": f"device_global_memory_{mode}", "pattern": mode, "dtype": args.dtype,
                    "buffer_mib": size_mib, "buffer_bytes": buffer_bytes,
                    "elements": n, "read_bytes": buffer_bytes, "write_bytes": write_bytes,
                    "traffic_bytes": traffic_bytes, "block_size": block, "num_warps": warps,
                    "cache_regime": regime, "peak_eligible": peak_eligible,
                    "smoke_only": args.quick, **timings,
                    "gbps": gbps, "tbps": gbps / 1000,
                    "peak_gbps": peak_gbps, "peak_tbps": peak_gbps / 1000,
                    **correctness,
                })
                print(f"{mode:4s} {size_mib:11d} {block:7d} {warps:6d} "
                      f"{timings['min_ms']:11.6f} {timings['median_ms']:11.6f} "
                      f"{peak_gbps:11.2f} {gbps:12.2f}", flush=True)
                del sink
        del src, dst
    eligible = [row for row in rows if row["peak_eligible"]]
    for mode in modes:
        candidates = [row for row in (eligible or rows) if row["pattern"] == mode]
        best = max(candidates, key=lambda row: row["peak_gbps"])
        label = "Best measured" if eligible else "Best cache-sensitive/smoke"
        print(f"{label} {mode} bandwidth: {best['peak_gbps']:.2f} GB/s "
              f"({best['peak_tbps']:.3f} TB/s), {best['buffer_mib']} MiB/buffer, "
              f"block={best['block_size']}, warps={best['num_warps']}.")
    if eligible:
        best = max(eligible, key=lambda row: row["peak_gbps"])
        print(f"Highest measured HBM-sized pattern: {best['pattern']}, "
              f"{best['peak_tbps']:.3f} TB/s (best observed timing, not a theoretical limit).")
    else:
        print("No HBM-sized peak candidate: eligibility requires a known L2 capacity, "
              "buffers >=4x L2 and a run without --quick.")
    print("These are effective bytes requested by the kernels; read includes a block reduction. "
          "Physical DRAM traffic and other access patterns can differ.")
    emit_results("bandwidth", metadata, args, rows)


if __name__ == "__main__":
    main()
