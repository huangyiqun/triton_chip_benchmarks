"""Backend-independent summaries for the complete chip metric table."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Metric:
    key: str
    group: str
    precision: str | None
    unit: str


METRICS = (
    Metric("memory_capacity", "Memory", None, "GiB"),
    Metric("memory_bandwidth", "Memory", None, "GB/s"),
    *(Metric(f"tensor_{dtype}", "Tensor", dtype,
             "TOP/s" if dtype == "int8" else "TFLOP/s")
      for dtype in ("bf16", "fp16", "fp32", "tf32", "int8", "fp64", "fp8")),
    *(Metric(f"vector_{dtype}", "Vector", dtype,
             "TOP/s" if dtype == "int8" else "TFLOP/s")
      for dtype in ("bf16", "fp16", "fp32", "int8", "fp64", "fp8")),
)


def empty_metric(metric, status="pending", reason=None):
    result = {"metric": metric.key, "group": metric.group,
              "precision": metric.precision, "unit": metric.unit,
              "status": status, "value": None, "median_value": None}
    if reason:
        result["reason"] = reason
    return result


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _rates(metric, row):
    if metric.key == "memory_bandwidth":
        return row.get("peak_gbps"), row.get("gbps")
    peak = row.get("peak_teraops_per_second")
    median = row.get("median_teraops_per_second")
    suffix = "tops" if metric.precision == "int8" else "tflops"
    return (row.get(f"peak_{suffix}") if peak is None else peak,
            row.get(suffix) if median is None else median)


def _row_details(metric, row):
    peak, median = _rates(metric, row)
    details = {"value": peak, "median_value": median,
               "smoke_only": bool(row.get("smoke_only", False))}
    # Keep verification separate from successful measurement: IEEE FP32 matrix
    # arithmetic can legitimately use SIMT, and some vendors lack an auditor.
    compiled = row.get("compiled") or {}
    for key in ("execution_kind", "native_tensor_verified", "native_arithmetic_verified",
                "instruction_precision_verified", "precision_verified", "assembly_check",
                "assembly_audit", "assembly_audit_note", "arithmetic_dtype", "includes_unpacking",
                "input_storage_dtype", "accumulator_dtype", "output_dtype", "operation",
                "measurement_kind", "timing_method", "timing_fallback_reason"):
        if key in row or key in compiled:
            details[key] = row.get(key, compiled.get(key))
    details["configuration"] = {
        key: row[key] for key in ("config", "block_size", "num_warps", "unroll",
                                  "blocks_per_sm", "blocks", "iterations", "pattern",
                                  "buffer_mib", "cache_regime", "peak_eligible") if key in row
    }
    return details


def _verification_reason(metric, details):
    if metric.group == "Tensor":
        execution = details.get("execution_kind")
        known_execution = (execution == "simt" or
                           (execution == "tensor" and details.get("native_tensor_verified") is True))
        if details.get("instruction_precision_verified") is not True or not known_execution:
            return "Measured matrix rate retained; native instruction precision or execution unit is unverified"
    elif metric.group == "Vector" and details.get("native_arithmetic_verified") is not True:
        if metric.precision == "int8" and details.get("includes_unpacking"):
            return ("Measured portable INT8-input path uses INT32 arithmetic and includes unpacking; "
                    "native INT8 execution is unverified")
        return "Measured vector rate retained; native arithmetic instruction precision is unverified"
    return None


def summarize(metric, document):
    """Summarize a child's declared status and validated measurement rows.

    Missing measurements are never represented as zero. Exceptions and compiler
    failures are unsupported only when the child explicitly reports that status.
    """
    result = empty_metric(metric)
    status = document.get("status")
    if status in ("unsupported", "error", "unavailable"):
        result.update(status=status, reason=document.get("reason", status))
        return result
    if metric.key == "memory_capacity":
        capacity = document.get("device", {}).get("total_memory_bytes")
        if _positive(capacity):
            result.update(status="ok", value=capacity / 2**30,
                          capacity_bytes=capacity, measurement_kind="reported_device_capacity")
        else:
            result.update(status="unavailable", reason="Backend did not report total memory capacity")
        return result

    rows = document.get("results", [])
    if not isinstance(rows, list):
        return empty_metric(metric, "error", "Benchmark results must be a list")
    successful = [row for row in rows if isinstance(row, dict)
                  and row.get("status", "ok") == "ok"
                  and all(_positive(value) for value in _rates(metric, row))]
    if not successful:
        rejected = [row for row in rows if isinstance(row, dict)]
        unsupported = rejected and all(row.get("status") == "unsupported" for row in rejected)
        reasons = list(dict.fromkeys(str(row.get("reason", "No valid throughput"))
                                    for row in rejected))
        return empty_metric(metric, "unsupported" if unsupported else "error",
                            "; ".join(reasons) or "Benchmark produced no valid throughput measurements")

    if metric.key == "memory_bandwidth":
        patterns = {}
        for pattern in ("copy", "read"):
            candidates = [row for row in successful if row.get("pattern") == pattern]
            eligible = [row for row in candidates if row.get("peak_eligible") is True
                        and not row.get("smoke_only", False)]
            if candidates:
                best = max(eligible or candidates, key=lambda row: _rates(metric, row)[0])
                patterns[pattern] = _row_details(metric, best)
        result["patterns"] = patterns
        successful = [row for row in successful if row.get("pattern") == "copy"
                      and row.get("peak_eligible") is True and not row.get("smoke_only", False)]
        if not successful:
            result.update(status="not_measured", reason=(
                "No HBM-sized copy candidate: use a non-quick run with known L2 capacity "
                "and buffers at least 4x L2; observed pattern rates remain in patterns"))
            return result

    best = max(successful, key=lambda row: _rates(metric, row)[0])
    result.update(status="ok", **_row_details(metric, best))
    if document.get("arguments", {}).get("quick") and metric.group != "Memory":
        result["smoke_only"] = True
    reason = _verification_reason(metric, result)
    if reason:
        result.update(status="unverified", reason=reason)
    result["successful_configurations"] = len(successful)
    return result


def format_table(results):
    """Render one row for every requested metric, including missing results."""
    lines = [f"{'Metric':<20} {'Peak / capacity':>17} {'Median':>12} {'Unit':<8} Status",
             "-" * 86]
    notes = []
    for item in results:
        value = "-" if item.get("value") is None else f"{item['value']:.3f}"
        median = "-" if item.get("median_value") is None else f"{item['median_value']:.3f}"
        lines.append(f"{item['metric']:<20} {value:>17} {median:>12} "
                     f"{item['unit']:<8} {item['status']}")
        if item.get("reason"):
            notes.append(f"{item['metric']}: {item['reason']}")
        if item.get("smoke_only"):
            notes.append(f"{item['metric']}: smoke test; not a peak result")
        if item.get("execution_kind") == "simt":
            notes.append(f"{item['metric']}: measured matrix rate using SIMT scalar/vector instructions "
                         "(no native tensor MMA)")
    if notes:
        lines.extend(("", *notes))
    return "\n".join(lines)
