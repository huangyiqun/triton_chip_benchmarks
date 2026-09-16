"""FlagGems device management, backend events and peak-throughput reporting."""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

import torch
import triton

flag_gems = None


def get_flag_gems():
    """Load FlagGems only when hardware access is needed.

    Keeping this lazy lets ``--help`` and source inspection work on login or
    build nodes that cannot access an accelerator.
    """
    global flag_gems
    if flag_gems is None:
        try:
            import flag_gems as loaded_flag_gems
        except ModuleNotFoundError as exc:
            if exc.name != "flag_gems":
                raise
            raise SystemExit(
                "FlagGems is required to run a benchmark. Install it for your chip backend."
            ) from exc
        flag_gems = loaded_flag_gems
    return flag_gems


def get_device_api():
    """The active vendor module, supplied by FlagGems (not inferred from its name)."""
    return get_flag_gems().runtime.torch_device_fn


def get_device(args):
    index = args if isinstance(args, int) else args.device
    return torch.device(get_flag_gems().device, index)


def reference_device(device):
    """Use bounded CPU references on devices that cannot calculate in FP64."""
    gems = get_flag_gems()
    return device if getattr(gems.runtime.device, "support_fp64", False) else torch.device("cpu")


def _optional_query(obj, name, *args):
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return fn(*args)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        # Optional queries are absent or unimplemented on some vendor modules.
        return None


def available_memory_bytes():
    info = _optional_query(get_device_api(), "mem_get_info")
    return int(info[0]) if info is not None else None


@contextmanager
def full_precision_matmul():
    """Temporarily disable TF32 through FlagGems' backend settings, if exposed."""
    backend = getattr(get_flag_gems().runtime, "torch_backend_device", None)
    matmul = getattr(backend, "matmul", None)
    if matmul is None:
        yield
        return

    # PyTorch 2.9 deprecated allow_tf32 in favor of the string-valued
    # fp32_precision setting. Probe the modern API first without touching the
    # deprecated property, whose getter alone emits a warning.
    try:
        previous = matmul.fp32_precision
    except AttributeError:
        try:
            previous = matmul.allow_tf32
        except AttributeError:
            yield
            return
        setting, full_precision = "allow_tf32", False
    else:
        setting, full_precision = "fp32_precision", "ieee"

    setattr(matmul, setting, full_precision)
    try:
        yield
    finally:
        setattr(matmul, setting, previous)


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


def add_common_arguments(parser):
    parser.add_argument("--device", type=int, default=0, help="Accelerator device index (default: 0)")
    parser.add_argument("--warmup", type=positive_float, default=100,
                        help="Approximate device warmup duration in ms (default: 100)")
    parser.add_argument("--rep", type=positive_float, default=200,
                        help="Approximate measurement budget in ms (default: 200)")
    parser.add_argument("--timing", choices=("auto", "graph", "events"), default="auto",
                        help="Auto selects backend Graph if available, otherwise device-event batches")
    parser.add_argument("--batch-ms", type=positive_float, default=10,
                        help="Target batch duration in ms (default: 10)")
    parser.add_argument("--rounds", type=positive_int, default=10,
                        help="Minimum number of measured batches (default: 10)")
    parser.add_argument("--compute-units", type=positive_int,
                        help="Override SM/CU/core count when the backend cannot report it")
    parser.add_argument("--l2-mib", type=positive_float,
                        help="Known L2 capacity in MiB when the backend cannot report it")
    parser.add_argument("--output", type=Path, help="Save metadata and all samples as JSON")


def _property(props, *names):
    for name in names:
        value = props.get(name) if isinstance(props, dict) else getattr(props, name, None)
        if value is not None:
            return value
    return None


def _first_property(sources, *names):
    for _, props in sources:
        value = _property(props, *names)
        if value is not None:
            return value
    return None


def _compute_units(sources):
    for source, props in sources:
        value = _property(props, "multi_processor_count", "multiProcessorCount", "sm_count",
                          "compute_unit_count", "num_vectorcore", "cube_core_num",
                          "num_aicore", "core_count", "core_num", "num_cores")
        if value is not None and int(value) > 0:
            return int(value), source
        clusters = _property(props, "cluster_num", "cluster_count")
        cores = _property(props, "core_num_per_cluster", "cores_per_cluster")
        if clusters and cores:
            return int(clusters) * int(cores), source + " (clusters * cores/cluster)"
    # FlagGems exports vendor-specific counts for backends without SM properties.
    gems = get_flag_gems()
    backend = getattr(gems.runtime, "backend", None)
    module = _optional_query(backend, "get_vendor_module", gems.vendor_name)
    for name in ("TOTAL_CORE_NUM", "CORE_NUM"):
        value = getattr(module, name, None)
        if value is not None and int(value) > 0:
            return int(value), f"FlagGems backend.{name}"
    return None, "unavailable; use --compute-units for compute benchmarks"


def setup_device(args):
    gems = get_flag_gems()
    api = get_device_api()
    detector = gems.runtime.device
    if _optional_query(api, "is_available") is False:
        raise SystemExit(f"FlagGems backend {gems.vendor_name!r} has no available device")
    count = _optional_query(api, "device_count")
    if count is None:
        count = getattr(detector, "device_count", 0)
    if not 0 <= args.device < count:
        raise SystemExit(f"Invalid --device {args.device}; available indices: 0..{count - 1}")
    setter = getattr(api, "set_device", None)
    if callable(setter):
        setter(args.device)
    elif _optional_query(api, "current_device") != args.device:
        raise SystemExit(f"{gems.vendor_name} cannot select device {args.device}")
    torch.manual_seed(0)
    sources = [("FlagGems device API", _optional_query(api, "get_device_properties", args.device))]
    target = None
    try:
        driver = triton.runtime.driver.active
        target = driver.get_current_target()
        sources.append(("active Triton backend", _optional_query(
            getattr(driver, "utils", None), "get_device_properties", args.device)))
    except (AttributeError, NotImplementedError, RuntimeError):
        pass
    units, unit_source = _compute_units(sources)
    if getattr(args, "compute_units", None) is not None:
        units, unit_source = args.compute_units, "--compute-units"
    l2 = _first_property(sources, "L2_cache_size", "l2_cache_size", "l2_cache_bytes") or 0
    if getattr(args, "l2_mib", None) is not None:
        l2 = int(args.l2_mib * 2**20)
    total = _first_property(sources, "total_memory", "total_memory_bytes")
    name = (_first_property(sources, "name", "device_name")
        or _optional_query(api, "get_device_name", args.device) or gems.vendor_name)
    bf16 = getattr(detector, "support_bf16", None)
    detected_bf16 = _optional_query(api, "is_bf16_supported")
    if detected_bf16 is not None:
        bf16 = bool(detected_bf16) and bf16 is not False
    metadata = {
        "device_index": args.device, "device_type": gems.device,
        "device_name": str(name), "vendor_name": gems.vendor_name,
        "backend": str(getattr(target, "backend", "unknown")),
        "architecture": str(getattr(target, "arch", "unknown")),
        "sm_count": units, "compute_units_source": unit_source,
        "total_memory_bytes": int(total) if total else None, "l2_cache_bytes": int(l2),
        "supports_fp64": bool(getattr(detector, "support_fp64", False)),
        "supports_bf16": bf16,
        "torch_version": torch.__version__, "triton_version": triton.__version__,
        "flaggems_version": getattr(gems, "__version__", "unknown"),
        "device_api": getattr(api, "__name__", type(api).__name__),
    }
    memory = f"{total / 2**30:.1f} GiB" if total else "unknown"
    print(f"Device {get_device(args)}: {name} | vendor={gems.vendor_name} | "
          f"Triton={metadata['backend']}/{metadata['architecture']} | "
          f"compute units={units or 'unknown'} | memory={memory}")
    print(f"FlagGems {metadata['flaggems_version']}, PyTorch {torch.__version__}, "
          f"Triton {triton.__version__}; timing={args.timing}, "
          f"warmup={args.warmup:g} ms, rep={args.rep:g} ms")
    return metadata


def _quantile(samples, quantile):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class GraphUnavailable(RuntimeError):
    """The active backend does not offer a usable graph capture implementation."""


_INITIAL_CALIBRATION_LAUNCHES = 5
_MAX_BATCH_LAUNCHES = 2048
_MIN_RESOLVED_EVENT_TICKS = 20


def _graph_factory(api, device_type):
    if not all(callable(getattr(api, name, None)) for name in ("graph", "Stream", "stream")):
        return None, None
    # Resolve exported classes on the FlagGems device module. A CUDA-compatible
    # class name is not evidence that its vendor is NVIDIA.
    for name in dict.fromkeys((device_type.upper() + "Graph", "Graph", "CUDAGraph",
                              "NPUGraph", "MUSAGraph", "MLUGraph")):
        factory = getattr(api, name, None)
        if callable(factory):
            return factory, name
    return None, None


def _stream_context(api):
    if callable(getattr(api, "Stream", None)) and callable(getattr(api, "stream", None)):
        try:
            stream = api.Stream()
            return api.stream(stream)
        except (AttributeError, NotImplementedError):
            pass  # Device events can still record on the current/default stream.
    return nullcontext()


def _zero_event_error(launches=None):
    gems = get_flag_gems()
    workload = f" for {launches} launches" if launches is not None else ""
    message = (f"Invalid backend event time: 0 ms{workload} on "
               f"{gems.vendor_name}/{gems.device}; timed events are unavailable, "
               "disabled, or too coarse for this workload")
    if gems.vendor_name == "kunlunxin":
        if os.environ.get("XPU_EVENT_KL3_ENABLE") != "1":
            message += ("; FlagGems' KunlunXin setup exports XPU_EVENT_KL3_ENABLE=1, "
                        "so set it before starting Python and retry")
        else:
            message += "; verify the KunlunXin SDK, driver, and event implementation"
    return RuntimeError(message)


def _event_elapsed(api, start, end, submit, *, allow_zero=False):
    start.record()
    submit()
    end.record()
    # Use the FlagGems device-level synchronization path, matching Triton's
    # backend-neutral benchmark flow before it reads elapsed event time.
    api.synchronize()
    elapsed = float(start.elapsed_time(end))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError(f"Invalid backend event time: {elapsed!r} ms")
    if elapsed == 0 and not allow_zero:
        raise _zero_event_error()
    return elapsed


def _next_launch_count(launches):
    return min(_MAX_BATCH_LAUNCHES, launches * 2)


def _coarse_event_error(launches, elapsed_ms, required_ms):
    gems = get_flag_gems()
    return RuntimeError(
        f"Backend event resolution is too coarse on {gems.vendor_name}/{gems.device}: "
        f"{launches} launches measured {elapsed_ms:g} ms, but at least "
        f"{required_ms:g} ms is needed for a reliable peak measurement")


def _calibrate_events(api, start, end, direct):
    """Find a direct-launch batch large enough for the event timer to resolve."""
    launches = _INITIAL_CALIBRATION_LAUNCHES
    saw_zero = False
    while True:
        elapsed = _event_elapsed(
            api, start, end, lambda: direct(launches), allow_zero=True)
        if elapsed > 0:
            minimum_launches = launches if saw_zero else 1
            minimum_batch_ms = (elapsed * _MIN_RESOLVED_EVENT_TICKS
                                if saw_zero else 0.0)
            return elapsed / launches, launches, minimum_launches, minimum_batch_ms
        if launches == _MAX_BATCH_LAUNCHES:
            raise _zero_event_error(launches)
        saw_zero = True
        launches = _next_launch_count(launches)


def _batch_samples(fn, args, use_graph):
    gems = get_flag_gems()
    api = get_device_api()
    graph_factory, graph_name = _graph_factory(api, gems.device) if use_graph else (None, None)
    if use_graph and graph_factory is None:
        raise GraphUnavailable(f"{gems.vendor_name}/{gems.device} exposes no usable Graph API")
    with _stream_context(api):
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)

        def direct(count):
            for _ in range(count):
                fn()

        (estimate_ms, calibration_launches, minimum_launches,
         minimum_batch_ms) = _calibrate_events(api, start, end, direct)
        estimate_ms = max(estimate_ms, 0.001)
        target_ms = min(args.batch_ms, args.rep / args.rounds)
        launches = max(minimum_launches,
                       min(_MAX_BATCH_LAUNCHES, math.ceil(target_ms / estimate_ms)))

        def build_submit(count):
            if not use_graph:
                return lambda: direct(count)
            try:
                captured = graph_factory()
                # The vendor context manages its capture stream. Replay and
                # both timing events run on our active stream below.
                with api.graph(captured):
                    direct(count)
            except NotImplementedError as exc:
                raise GraphUnavailable(f"{gems.vendor_name} graph capture: {exc}") from exc
            except RuntimeError as exc:
                message = str(exc).lower()
                if any(word in message for word in ("not supported", "unsupported", "not implemented")):
                    raise GraphUnavailable(f"{gems.vendor_name} graph capture: {exc}") from exc
                raise
            return captured.replay  # Bound method keeps the captured graph alive.

        def time_batch(count, *, zero_seen=False):
            nonlocal minimum_batch_ms
            while True:
                submission = build_submit(count)
                elapsed = _event_elapsed(
                    api, start, end, submission, allow_zero=True)
                if elapsed > 0 and zero_seen and minimum_batch_ms == 0:
                    minimum_batch_ms = elapsed * _MIN_RESOLVED_EVENT_TICKS
                if elapsed > 0 and elapsed >= minimum_batch_ms:
                    return count, submission, elapsed
                if count == _MAX_BATCH_LAUNCHES:
                    error = (_zero_event_error(count) if elapsed == 0 else
                             _coarse_event_error(count, elapsed, minimum_batch_ms))
                    if use_graph:
                        raise GraphUnavailable(f"{gems.vendor_name} graph timing: {error}")
                    raise error
                zero_seen = zero_seen or elapsed == 0
                count = _next_launch_count(count)

        launches, submit, batch_ms = time_batch(launches)
        for _ in range(2):
            refined = max(minimum_launches,
                          min(_MAX_BATCH_LAUNCHES,
                              math.ceil(launches * target_ms / batch_ms)))
            if refined == launches or 0.5 * target_ms <= batch_ms <= 2 * target_ms:
                break
            launches, submit, batch_ms = time_batch(refined)
        for _ in range(max(1, math.ceil(args.warmup / batch_ms))):
            submit()
        api.synchronize()
        launches, submit, batch_ms = time_batch(launches)
        rounds = max(args.rounds, math.ceil(args.rep / batch_ms))
        batches = []
        while len(batches) < rounds:
            elapsed = _event_elapsed(api, start, end, submit, allow_zero=True)
            if elapsed > 0 and elapsed >= minimum_batch_ms:
                batches.append(elapsed)
                continue
            # If timer quantization produces an intermittent zero, enlarge the
            # batch and restart so every sample has the same denominator.
            if launches == _MAX_BATCH_LAUNCHES:
                error = (_zero_event_error(launches) if elapsed == 0 else
                         _coarse_event_error(launches, elapsed, minimum_batch_ms))
                if use_graph:
                    raise GraphUnavailable(f"{gems.vendor_name} graph timing: {error}")
                raise error
            launches, submit, batch_ms = time_batch(
                _next_launch_count(launches), zero_seen=elapsed == 0)
            rounds = max(args.rounds, math.ceil(args.rep / batch_ms))
            batches.clear()
    return [elapsed / launches for elapsed in batches], {
        "launches_per_batch": launches, "batch_samples_ms": batches,
        "sample_count": len(batches), "measured_total_ms": sum(batches),
        "graph_api": graph_name, "calibration_launches": calibration_launches,
        "minimum_resolved_batch_ms": minimum_batch_ms,
    }


def _graph_samples(fn, args):
    return _batch_samples(fn, args, use_graph=True)


def _event_samples(fn, args):
    return _batch_samples(fn, args, use_graph=False)


def measure(fn, args):
    """GPU/NPU event timing through FlagGems only; never a CPU wall-clock fallback."""
    gems = get_flag_gems()
    api = get_device_api()
    if not all(callable(getattr(api, name, None)) for name in ("Event", "synchronize")):
        raise RuntimeError(f"{gems.vendor_name} does not expose timed device events through FlagGems")
    fn()  # Compile before any graph capture or timed samples.
    api.synchronize()
    requested = args.timing
    selected = "events" if requested == "events" else "graph"
    fallback = None
    if selected == "graph":
        try:
            samples, details = _graph_samples(fn, args)
        except (GraphUnavailable, NotImplementedError) as exc:
            if requested == "graph":
                raise
            api.synchronize()
            fallback = str(exc)
            selected = "events"
            samples, details = _event_samples(fn, args)
    else:
        samples, details = _event_samples(fn, args)
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise RuntimeError("Invalid device timing; increase the workload/--rep")
    notice = f"Timing: {selected} via FlagGems {gems.vendor_name} device events"
    if fallback:
        notice += f" (auto fallback: {fallback}; includes host dispatch gaps)"
    if getattr(args, "_timing_notice", None) != notice:
        print(notice)
        args._timing_notice = notice
    return {
        "median_ms": statistics.median(samples), "p20_ms": _quantile(samples, 0.2),
        "p80_ms": _quantile(samples, 0.8), "min_ms": min(samples), "max_ms": max(samples),
        "timing_method": selected, "timing_requested": requested,
        "timing_fallback_reason": fallback, "samples_ms_per_launch": samples, **details,
    }


def emit_results(name, metadata, args, rows):
    if not rows:
        raise RuntimeError("No valid measurements were produced")
    if args.output is None:
        return
    methods = sorted({row["timing_method"] for row in rows if "timing_method" in row})
    document = {
        "benchmark": name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": metadata,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if not key.startswith("_")},
        "timing": {
            "provider": "flag_gems.runtime.torch_device_fn.Event",
            "requested": args.timing, "methods_used": methods,
            "method": "backend events around batches; elapsed / actual launch count; no cache flush",
            "peak_statistic": "minimum sampled batch-average device time",
            "sustained_statistic": "median sampled batch-average device time",
            "includes": "kernel work, device scheduling and amortized event overhead; "
                        "events mode additionally includes host dispatch gaps",
            "excludes": "JIT compilation, allocation, initialization, validation, capture and warmup",
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"JSON: {args.output.resolve()}")
