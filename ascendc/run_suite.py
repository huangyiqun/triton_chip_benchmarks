#!/usr/bin/env python3
"""Run the complete Ascend 950DT chip metric table.

Seventeen rows: the fifteen the Triton chip-benchmark suite defines, plus the
two FP4 rows this chip adds (FP4 tensor and FP4 vector).

Every row keeps its own status. A metric with no numeric value is reported as
unsupported (the hardware lacks the instruction), not_built (the binary is
missing), or error (it ran and failed) - never as zero. Each row also records
whether the number came from a native instruction for that format or from a
widened/converted path, because on this architecture several vector formats
have no native arithmetic instruction at all.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"
# Results land in the repository-wide results/ directory, beside the Triton
# suite's tables for the same chips, so the two implementations can be
# compared without copying files between trees.
RESULTS = ROOT.parent / "results"


class Metric:
    __slots__ = ("key", "group", "precision", "unit", "binary", "extra")

    def __init__(self, key, group, precision, unit, binary, extra=()):
        self.key = key
        self.group = group
        self.precision = precision
        self.unit = unit
        self.binary = binary
        self.extra = list(extra)


def ops_unit(precision):
    return "TOP/s" if precision == "int8" else "TFLOP/s"


METRICS = [
    Metric("memory_capacity", "Memory", None, "GiB", "bench_memory", ["--mode", "copy"]),
    Metric("memory_bandwidth", "Memory", None, "GB/s", "bench_memory", ["--mode", "all"]),
    # Tensor (Cube matrix unit): the 7 Triton formats plus FP4.
    # FP64 has no Cube instruction on this chip, so it is a capability report
    # rather than a measurement; see src/bench_fp64.asc.
    *(Metric(f"tensor_{dtype}", "Tensor", dtype, ops_unit(dtype),
             "bench_fp64_cube" if dtype == "fp64" else f"bench_cube_{dtype}",
             ["--dtype", dtype])
      for dtype in ("bf16", "fp16", "fp32", "tf32", "int8", "fp64", "fp8", "fp4")),
    # Vector (AIV): the 6 Triton formats plus FP4.
    *(Metric(f"vector_{dtype}", "Vector", dtype, ops_unit(dtype),
             "bench_fp64_vector" if dtype == "fp64" else f"bench_vector_{dtype}",
             ["--dtype", dtype])
      for dtype in ("bf16", "fp16", "fp32", "int8", "fp64", "fp8", "fp4")),
]


def empty_metric(metric, status="pending", reason=None):
    row = {
        "metric": metric.key,
        "group": metric.group,
        "precision": metric.precision,
        "unit": metric.unit,
        "status": status,
        "value": None,
        "median_value": None,
    }
    if reason:
        row["reason"] = reason
    return row


def is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def summarize(metric, document):
    """Pull one table row out of a benchmark's JSON document."""
    row = empty_metric(metric)

    # Capacity and bandwidth live in nested objects of the memory document.
    if metric.key == "memory_capacity":
        capacity = document.get("memory_capacity") or {}
        row["status"] = capacity.get("status", "error")
        row["value"] = capacity.get("value")
        row["measurement_kind"] = capacity.get("measurement_kind")
        row["capacity_bytes"] = capacity.get("capacity_bytes")
        return row
    if metric.key == "memory_bandwidth":
        bandwidth = document.get("memory_bandwidth") or {}
        row["status"] = bandwidth.get("status", "error")
        row["value"] = bandwidth.get("value")
        row["median_value"] = bandwidth.get("median_value")
        for key in ("pattern", "best_buffer_mib", "best_blocks", "best_chunk_elements",
                    "read_peak_gbps", "read_median_gbps",
                    "read_best_blocks", "read_best_chunk_elements"):
            if key in bandwidth:
                row[key] = bandwidth[key]
        if "reason" in bandwidth:
            row["reason"] = bandwidth["reason"]
        return row

    # Compute metrics report at the document's top level.
    row["status"] = document.get("status", "error")
    row["value"] = document.get("value")
    row["median_value"] = document.get("median_value")
    for key in ("reason", "measurement_kind", "arithmetic_kind", "arithmetic_note",
                "native_arithmetic", "native_instruction", "accumulator_dtype",
                "iterations", "best_block_m", "best_block_n", "best_block_k",
                "best_blocks", "chains", "lanes_per_register"):
        if key in document:
            row[key] = document[key]
    return row


def run_metric(metric, args, directory):
    output = directory / f"{metric.key}.json"
    log = directory / f"{metric.key}.log"
    binary = BIN / metric.binary

    if not binary.exists():
        return empty_metric(
            metric, "not_built",
            f"{metric.binary} is not in {BIN}; run ./build.sh"), None

    command = [str(binary), *metric.extra, "--device", str(args.device),
               "--output", str(output)]
    if args.iterations and metric.group == "Tensor":
        command += ["--iterations", str(args.iterations)]
    if args.quick:
        command.append("--quick")

    # A stale output must never be mistaken for a fresh run.
    output.unlink(missing_ok=True)

    environment = dict(os.environ)
    # Both binaries print the physical device they opened; pinning visibility
    # here keeps a shared machine's other cards out of the measurement.
    if args.visible_devices is not None:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices

    try:
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=False,
                                       env=environment)
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        row = empty_metric(metric, "error", f"exceeded {args.timeout:g} s; see log")
        row.update(command=command, log_file=str(log), returncode=None)
        return row, None
    except OSError as exc:
        row = empty_metric(metric, "error", f"cannot execute: {exc}")
        row.update(command=command, log_file=str(log), returncode=None)
        return row, None

    document = None
    if output.exists():
        try:
            document = json.loads(output.read_text())
            row = summarize(metric, document)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row = empty_metric(metric, "error", f"invalid benchmark JSON: {exc}")
    else:
        row = empty_metric(metric, "error", "benchmark wrote no result JSON")

    # A nonzero exit invalidates a numeric claim unless the benchmark itself
    # reported an explicit capability absence.
    if returncode and row["status"] not in ("unsupported", "not_built"):
        partial = row if is_number(row.get("value")) else None
        row = empty_metric(metric, "error", f"exited with code {returncode}; see log")
        if partial:
            row["partial_measurement"] = partial

    row.update(command=command, source_json=str(output), log_file=str(log),
               returncode=returncode)
    return row, document


STATUS_LABEL = {
    "ok": "",
    "measured_widened": "widened",
    "unsupported": "unsupported",
    "not_built": "not built",
    "error": "error",
    "pending": "pending",
}


def format_table(rows):
    header = f"{'Metric':<18} {'Group':<7} {'Precision':<10} {'Peak':>12} {'Median':>12}  {'Unit':<8} Note"
    lines = [header, "-" * len(header)]
    for row in rows:
        value = row.get("value")
        median = row.get("median_value")
        peak_text = f"{value:,.2f}" if is_number(value) else "-"
        median_text = f"{median:,.2f}" if is_number(median) else "-"
        note = STATUS_LABEL.get(row["status"], row["status"])
        # Name the execution path when it is not a native instruction for the
        # reported format, so a widened number is never read as a native peak.
        if row["status"] in ("ok", "measured_widened"):
            kind = row.get("arithmetic_kind") or row.get("measurement_kind") or ""
            if row.get("native_arithmetic") is False or row["status"] == "measured_widened":
                note = f"widened: {kind}" if kind else "widened"
            elif kind.startswith("native"):
                note = kind
        elif row.get("reason"):
            note = f"{note}: {row['reason'][:60]}"
        lines.append(f"{row['metric']:<18} {row['group']:<7} "
                     f"{(row['precision'] or '-'):<10} {peak_text:>12} {median_text:>12}  "
                     f"{row['unit']:<8} {note}")
    return "\n".join(lines)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0,
                        help="device index passed to each benchmark (default: 0)")
    parser.add_argument("--visible-devices",
                        help="value for ASCEND_RT_VISIBLE_DEVICES, to pin the "
                             "physical card on a shared machine")
    parser.add_argument("--iterations", type=int,
                        help="override the Cube MMAD iteration count")
    parser.add_argument("--timeout", type=float, default=900,
                        help="seconds allowed per metric (default: 900)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run; these rates are not peak measurements")
    parser.add_argument("--only", nargs="+", metavar="METRIC",
                        help="run only these metric keys")
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "ascend950dt_ascendc_metrics.json",
                        help="complete table JSON, checkpointed after every metric")
    parser.add_argument("--results-dir", type=Path,
                        help="per-metric JSON and logs (default: OUTPUT_STEM_runs)")
    args = parser.parse_args()

    selected = METRICS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {metric.key for metric in METRICS}
        if unknown:
            parser.error(f"unknown metric(s): {', '.join(sorted(unknown))}")
        selected = [metric for metric in METRICS if metric.key in wanted]

    args.output = args.output.resolve()
    directory = (args.results_dir
                 or args.output.with_name(args.output.stem + "_runs")).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "benchmark": "ascendc_chip_metric_suite",
        "implementation": "Ascend C (SIMD C API), compiled with bisheng for dav-3510",
        "status": "running",
        "smoke_only": args.quick,
        "device": {},
        "measurement_policy":
            "Highest observed valid throughput, with the median from the same "
            "configuration. Capacity is the runtime-reported device capacity. "
            "Bandwidth is effective copy read+write traffic with buffers of at "
            "least 4x L2; the read rate is an auxiliary detail. Every "
            "configuration is validated against an exact host reference before "
            "it is timed. A metric with no numeric value is unsupported, "
            "not_built or error, never zero. Rows whose format has no native "
            "arithmetic instruction on this chip are marked widened.",
        "results": [empty_metric(metric) for metric in selected],
    }
    write_json(args.output, report)

    for index, metric in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] {metric.key} ...", flush=True)
        row, document = run_metric(metric, args, directory)
        report["results"][index] = row
        if document and document.get("device") and not report["device"]:
            report["device"] = document["device"]
        write_json(args.output, report)
        detail = f": {row['reason']}" if row.get("reason") else ""
        value = row.get("value")
        if is_number(value):
            detail = f": {value:,.2f} {metric.unit}"
        print(f"  {row['status']}{detail}", flush=True)

    report["status"] = ("complete_with_errors"
                        if any(row["status"] == "error" for row in report["results"])
                        else "complete")
    write_json(args.output, report)

    print("\n" + format_table(report["results"]))
    print(f"\nJSON: {args.output}")
    print(f"Per-metric JSON and logs: {directory}")
    return 1 if report["status"] == "complete_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
