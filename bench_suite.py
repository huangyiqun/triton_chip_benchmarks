#!/usr/bin/env python3
"""Run the complete chip metric table, retaining unsupported and failed entries."""

import os

# Child benchmarks inherit this setting; set it before any device runtime starts.
os.environ.setdefault("XPU_EVENT_KL3_ENABLE", "1")

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

try:
    from .suite_report import METRICS, empty_metric, format_table, summarize
except ImportError:
    from suite_report import METRICS, empty_metric, format_table, summarize


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timing", choices=("auto", "graph", "events"), default="auto")
    parser.add_argument("--warmup", type=positive_float, default=100)
    parser.add_argument("--rep", type=positive_float, default=200)
    parser.add_argument("--batch-ms", type=positive_float, default=10)
    parser.add_argument("--rounds", type=positive_int, default=10)
    parser.add_argument("--compute-units", type=positive_int)
    parser.add_argument("--l2-mib", type=positive_float)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke checks; these rates are not peak measurements")
    parser.add_argument("--output", type=Path, default=Path("results/chip_metrics.json"),
                        help="Complete table JSON, checkpointed after every metric")
    parser.add_argument("--results-dir", type=Path,
                        help="Per-metric raw JSON and logs (default: OUTPUT_STEM_runs)")
    parser.add_argument("--timeout", type=positive_float, default=900,
                        help="Maximum seconds per metric, including compilation (default: 900)")
    parser.add_argument("--probe-device", action="store_true", help=argparse.SUPPRESS)
    return parser


def metric_command(metric, args, output):
    """Use the active Python and repository-local scripts without a shell."""
    if metric.key == "memory_capacity":
        script, extra = "bench_suite.py", ["--probe-device"]
    elif metric.key == "memory_bandwidth":
        script, extra = "bench_bandwidth.py", ["--mode", "all", "--dtype", "fp32"]
    elif metric.group == "Tensor":
        script = "bench_tensor.py"
        extra = ["--mode", "peak", "--dtype",
                 "fp32" if metric.precision == "tf32" else metric.precision]
        if metric.precision in ("fp32", "tf32"):
            extra += ["--input-precision", "tf32" if metric.precision == "tf32" else "ieee"]
    else:
        script, extra = "bench_vector.py", ["--dtype", metric.precision]
    command = [sys.executable, "-u", str(ROOT / script), *extra,
               "--device", str(args.device), "--timing", args.timing,
               "--warmup", str(args.warmup), "--rep", str(args.rep),
               "--batch-ms", str(args.batch_ms), "--rounds", str(args.rounds),
               "--output", str(output)]
    for flag, value in (("--compute-units", args.compute_units), ("--l2-mib", args.l2_mib)):
        if value is not None:
            command += [flag, str(value)]
    if args.quick:
        command.append("--quick")
    return command


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def probe_device(args):
    # Only the isolated child loads accelerator packages. Suite/report tooling
    # and --help remain usable on a machine without an accelerator SDK.
    try:
        from .common import setup_device
    except ImportError:
        from common import setup_device
    metadata = setup_device(args)
    write_json(args.output, {"status": "ok", "device": metadata, "results": []})


def run_metric(metric, args, directory):
    output = directory / f"{metric.key}.json"
    log = directory / f"{metric.key}.log"
    command = metric_command(metric, args, output)
    # An old successful output must never masquerade as a new failed run.
    output.unlink(missing_ok=True)
    document = None
    result = None
    returncode = None
    try:
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=args.timeout,
                                       check=False)
        returncode = completed.returncode
        if output.exists():
            try:
                document = json.loads(output.read_text())
                if not isinstance(document, dict):
                    raise ValueError("expected a JSON object")
                result = summarize(metric, document)
                if (args.quick and result["status"] in ("ok", "unverified")
                        and metric.group != "Memory"):
                    result["smoke_only"] = True
            except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                result = empty_metric(metric, "error", f"Invalid benchmark JSON: {exc}")
        else:
            result = empty_metric(metric, "error", "Benchmark did not write result JSON")
        if returncode and result["status"] not in ("unsupported", "not_implemented"):
            partial = result if result["status"] in ("ok", "unverified") else None
            result = empty_metric(metric, "error", f"Benchmark exited with code {returncode}; see log")
            if partial is not None:
                result["partial_measurement"] = partial
    except subprocess.TimeoutExpired:
        result = empty_metric(metric, "error", f"Exceeded {args.timeout:g} seconds; see log")
    except OSError as exc:
        result = empty_metric(metric, "error", f"Cannot execute benchmark: {exc}")
    result.update(command=command, source_json=str(output), log_file=str(log), returncode=returncode)
    return result, document


def run_suite(args):
    args.output = args.output.resolve()
    directory = (args.results_dir or args.output.with_name(args.output.stem + "_runs")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "benchmark": "chip_metric_suite",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "device": {}, "smoke_only": args.quick,
        "measurement_policy": "Highest observed valid throughput; median from the same configuration. "
                              "Capacity is reported device capacity. Bandwidth is effective copy read+write "
                              "traffic with HBM-sized buffers; read rates are auxiliary details. "
                              "Unsupported and not-implemented metrics have no numeric value; "
                              "errors do not imply unsupported hardware.",
        "results": [empty_metric(metric) for metric in METRICS],
    }
    write_json(args.output, report)
    for index, metric in enumerate(METRICS):
        print(f"[{index + 1}/{len(METRICS)}] {metric.key} ...", flush=True)
        result, document = run_metric(metric, args, directory)
        report["results"][index] = result
        if document and document.get("device") and not report["device"]:
            report["device"] = document["device"]
        write_json(args.output, report)
        print(f"  {result['status']}" + (f": {result['reason']}" if result.get("reason") else ""),
              flush=True)
    report["status"] = "complete_with_errors" if any(
        item["status"] == "error" for item in report["results"]) else "complete"
    write_json(args.output, report)
    print("\n" + format_table(report["results"]))
    print(f"\nJSON: {args.output}")
    print(f"Raw results and logs: {directory}")
    return report


def main():
    args = build_parser().parse_args()
    if args.probe_device:
        probe_device(args)
    else:
        report = run_suite(args)
        return 1 if report["status"] == "complete_with_errors" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
