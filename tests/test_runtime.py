"""CPU-only tests of the FlagGems runtime boundary, without mock math kernels.

The fake accelerator advances a device clock when work executes. Its events
and graph objects implement the same timing contract as vendor device APIs;
neither PyTorch nor an accelerator SDK is imported by these tests.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


@dataclass(frozen=True, init=False)
class FakeDevice:
    type: str
    index: int | None

    def __init__(self, device_type, index=None):
        if isinstance(device_type, FakeDevice):
            name, parsed_index = device_type.type, device_type.index
        else:
            pieces = str(device_type).split(":", 1)
            name = pieces[0]
            parsed_index = int(pieces[1]) if len(pieces) == 2 else None
        object.__setattr__(self, "type", name)
        object.__setattr__(self, "index", parsed_index if index is None else index)

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeAccelerator:
    """A device API intentionally exposing no CUDA namespace."""

    kernel_ms = 0.25

    def __init__(self, device_type="npu", graph_name=None, properties=None, memory=None):
        self.device_type = device_type
        self.now_ms = 0.0
        self.synchronized_ms = 0.0
        self.selected_device = 0
        self.synchronize_calls = 0
        self.elapsed_time_calls = 0
        self.replay_calls = 0
        self.capturing = None
        self.active_stream = None
        if properties is not None:
            self.get_device_properties = lambda *args: properties
        if memory is not None:
            self.mem_get_info = lambda *args: memory
        if graph_name is not None:
            api = self

            class DeviceGraph:
                def __init__(self):
                    self.launches = 0

                def replay(self):
                    api.replay_calls += 1
                    api.now_ms += self.launches * api.kernel_ms

            DeviceGraph.__name__ = graph_name
            setattr(self, graph_name, DeviceGraph)

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def set_device(self, index):
        self.selected_device = index.index if isinstance(index, FakeDevice) else index

    def current_device(self):
        return self.selected_device

    def get_device_name(self, *args):
        return f"Fake {self.device_type.upper()}"

    def synchronize(self, *args):
        self.synchronize_calls += 1
        self.synchronized_ms = self.now_ms

    def Event(self, enable_timing=False):
        if not enable_timing:
            raise AssertionError("Device timing must request enable_timing=True")
        api = self

        class DeviceEvent:
            def record(self, stream=None):
                self.timestamp_ms = api.now_ms

            def synchronize(self):
                api.synchronize()

            def elapsed_time(self, end):
                if api.synchronized_ms < end.timestamp_ms:
                    raise AssertionError("Read elapsed time before device completion")
                api.elapsed_time_calls += 1
                return end.timestamp_ms - self.timestamp_ms

        return DeviceEvent()

    def Stream(self, *args, **kwargs):
        return SimpleNamespace(synchronize=self.synchronize)

    @contextmanager
    def stream(self, stream):
        previous = self.active_stream
        self.active_stream = stream
        try:
            yield stream
        finally:
            self.active_stream = previous

    @contextmanager
    def graph(self, graph, stream=None):
        if self.capturing is not None:
            raise AssertionError("Nested graph capture")
        self.capturing = graph
        try:
            yield graph
        finally:
            self.capturing = None

    def launch_work(self):
        # Represents submission to a device, not any tensor/math operation.
        if self.capturing is not None:
            self.capturing.launches += 1
        else:
            self.now_ms += self.kernel_ms


def load_common_without_accelerator_imports():
    torch = ModuleType("torch")
    torch.__version__ = "test"
    torch.version = SimpleNamespace(cuda=None, hip=None)
    torch.device = FakeDevice
    torch.manual_seed = lambda seed: None
    torch.backends = SimpleNamespace()

    def missing_torch_attribute(name):
        if name == "cuda":
            raise AssertionError("Portable runtime accessed torch.cuda")
        raise AttributeError(name)

    torch.__getattr__ = missing_torch_attribute
    triton = ModuleType("triton")
    triton.__version__ = "test"
    triton.__path__ = []
    testing = ModuleType("triton.testing")
    triton.testing = testing
    gems = ModuleType("flag_gems")
    gems.__version__ = "test"
    path = Path(__file__).resolve().parents[1] / "common.py"
    spec = importlib.util.spec_from_file_location("benchmark_common_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"torch": torch, "triton": triton,
                                      "triton.testing": testing, "flag_gems": gems}):
        spec.loader.exec_module(module)
    return module


common = load_common_without_accelerator_imports()


class RuntimeTests(unittest.TestCase):
    def configure(self, api=None, vendor="ascend", support_fp64=False, driver_properties=None):
        api = api or FakeAccelerator()
        device_info = SimpleNamespace(name=api.device_type, vendor_name=vendor,
                                      support_fp64=support_fp64, device_count=2)
        common.flag_gems = SimpleNamespace(
            device=api.device_type, vendor_name=vendor, __version__="test",
            runtime=SimpleNamespace(device=device_info, torch_device_fn=api,
                                    torch_backend_device=None),
        )
        target = SimpleNamespace(backend=api.device_type, arch="test-arch", warp_size=32)
        driver = SimpleNamespace(
            get_current_target=lambda: target,
            utils=SimpleNamespace(get_device_properties=lambda *args: driver_properties or {}),
        )
        common.triton.runtime = SimpleNamespace(driver=SimpleNamespace(active=driver))
        return api

    def arguments(self, **overrides):
        values = dict(device=1, warmup=1.0, rep=8.0, timing="auto", batch_ms=2.0,
                      rounds=2, compute_units=None, l2_mib=None, output=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def set_up_device(self, args=None):
        with redirect_stdout(io.StringIO()):
            return common.setup_device(args or self.arguments())

    def measure(self, api, **overrides):
        # A missing vendor event implementation must fail, not silently switch
        # to a host wall clock that includes Python dispatch time.
        failure = AssertionError("Host wall-clock timing is forbidden")
        with mock.patch("time.perf_counter", side_effect=failure), \
             mock.patch("time.monotonic", side_effect=failure), \
             mock.patch("time.time", side_effect=failure), \
             redirect_stdout(io.StringIO()):
            return common.measure(api.launch_work, self.arguments(**overrides))

    def test_non_cuda_device_selection_uses_flag_gems_api(self):
        api = self.configure(FakeAccelerator("npu"))
        metadata = self.set_up_device()
        self.assertEqual(api.selected_device, 1)
        self.assertEqual(str(common.get_device(self.arguments())), "npu:1")
        self.assertEqual(metadata["vendor_name"], "ascend")
        self.assertEqual(metadata["device_type"], "npu")

    def test_cuda_device_type_does_not_imply_nvidia_vendor(self):
        self.configure(FakeAccelerator("cuda"), vendor="metax")
        metadata = self.set_up_device()
        self.assertEqual(metadata["device_type"], "cuda")
        self.assertEqual(metadata["vendor_name"], "metax")

    def test_missing_optional_properties_and_memory_remain_unknown(self):
        self.configure(FakeAccelerator("mlu"), vendor="cambricon")
        metadata = self.set_up_device()
        self.assertIsNone(metadata["sm_count"])
        self.assertIn(metadata["l2_cache_bytes"], (None, 0))
        self.assertIn(metadata["total_memory_bytes"], (None, 0))
        self.assertIsNone(common.available_memory_bytes())
        self.assertEqual(str(common.reference_device(FakeDevice("mlu", 1))), "cpu")

    def test_compute_and_l2_overrides_are_recorded(self):
        self.configure()
        metadata = self.set_up_device(self.arguments(compute_units=40, l2_mib=12.5))
        self.assertEqual(metadata["sm_count"], 40)
        self.assertEqual(metadata["l2_cache_bytes"], int(12.5 * 2**20))

    def test_dictionary_device_properties_are_supported(self):
        properties = {"name": "Dictionary NPU", "multi_processor_count": 32,
                      "total_memory": 16 * 2**30, "L2_cache_size": 8 * 2**20}
        self.configure(FakeAccelerator(properties=properties, memory=(12 * 2**30, 16 * 2**30)))
        metadata = self.set_up_device()
        self.assertEqual(metadata["sm_count"], 32)
        self.assertEqual(metadata["l2_cache_bytes"], 8 * 2**20)
        self.assertEqual(common.available_memory_bytes(), 12 * 2**30)

    def test_vendor_driver_core_counts_do_not_require_sm_properties(self):
        cases = (("npu", "ascend", {"num_vectorcore": 40}, 40),
                 ("mlu", "cambricon", {"cluster_num": 8, "core_num_per_cluster": 4}, 32))
        for device_type, vendor, properties, expected in cases:
            with self.subTest(device_type=device_type):
                self.configure(FakeAccelerator(device_type), vendor=vendor,
                               driver_properties=properties)
                metadata = self.set_up_device()
                self.assertEqual(metadata["sm_count"], expected)
                self.assertIn("Triton", metadata["compute_units_source"])

    def test_unimplemented_optional_queries_do_not_block_device_setup(self):
        api = self.configure()

        def unavailable(*args):
            raise NotImplementedError("SDK does not implement this optional query")

        api.get_device_properties = unavailable
        api.mem_get_info = unavailable
        metadata = self.set_up_device()
        self.assertIsNone(metadata["sm_count"])
        self.assertIsNone(common.available_memory_bytes())

    def test_missing_fp64_capability_defaults_to_cpu_reference(self):
        self.configure()
        del common.flag_gems.runtime.device.support_fp64
        self.assertEqual(str(common.reference_device(FakeDevice("npu", 1))), "cpu")

    def test_supported_fp64_keeps_reference_on_selected_device(self):
        self.configure(support_fp64=True)
        selected = FakeDevice("npu", 1)
        self.assertEqual(common.reference_device(selected), selected)

    def test_precision_context_uses_and_restores_flag_gems_backend_settings(self):
        self.configure()
        # Missing optional matmul settings are valid on non-CUDA devices.
        with common.full_precision_matmul():
            pass
        matmul = SimpleNamespace(allow_tf32=True)
        common.flag_gems.runtime.torch_backend_device = SimpleNamespace(matmul=matmul)
        with self.assertRaisesRegex(ValueError, "reference failed"):
            with common.full_precision_matmul():
                self.assertFalse(matmul.allow_tf32)
                raise ValueError("reference failed")
        self.assertTrue(matmul.allow_tf32)

    def test_graph_aliases_use_vendor_api_without_cuda(self):
        cases = (("npu", "ascend", "NPUGraph"), ("mlu", "cambricon", "MLUGraph"),
                 ("musa", "mthreads", "MUSAGraph"))
        for device_type, vendor, graph_name in cases:
            with self.subTest(device_type=device_type):
                api = self.configure(FakeAccelerator(device_type, graph_name), vendor=vendor)
                factory, name = common._graph_factory(api, device_type)
                self.assertEqual(name, graph_name)
                self.assertIs(factory, getattr(api, graph_name))
                result = self.measure(api)
                self.assertEqual(result["timing_method"], "graph")
                self.assertGreater(api.replay_calls, 0)
                self.assertGreater(result["launches_per_batch"], 1)
                self.assertTrue(all(sample == api.kernel_ms
                                    for sample in result["samples_ms_per_launch"]))
                self.assertTrue(all(batch / result["launches_per_batch"] == api.kernel_ms
                                    for batch in result["batch_samples_ms"]))

    def test_missing_graph_auto_uses_device_event_batches(self):
        api = self.configure(FakeAccelerator("npu"))
        result = self.measure(api)
        self.assertEqual(result["timing_method"], "events")
        self.assertGreater(api.elapsed_time_calls, 0)
        self.assertEqual(api.replay_calls, 0)
        self.assertEqual(result["median_ms"], api.kernel_ms)
        self.assertEqual(result["min_ms"], api.kernel_ms)
        self.assertTrue(result.get("timing_fallback_reason"))

    def test_event_only_sdk_without_stream_api_can_measure(self):
        api = self.configure(FakeAccelerator("mlu"), vendor="cambricon")
        api.Stream = None
        api.stream = None
        result = self.measure(api)
        self.assertEqual(result["timing_method"], "events")
        self.assertEqual(result["median_ms"], api.kernel_ms)
        self.assertGreater(result["launches_per_batch"], 1)

    def test_stub_stream_constructor_still_allows_device_event_timing(self):
        api = self.configure()

        def unavailable_stream():
            raise NotImplementedError("Stream constructor is an SDK stub")

        api.Stream = unavailable_stream
        result = self.measure(api, timing="events")
        self.assertEqual(result["timing_method"], "events")
        self.assertEqual(result["median_ms"], api.kernel_ms)

    def test_present_but_unimplemented_graph_reports_auto_fallback_reason(self):
        api = self.configure(FakeAccelerator("npu", "NPUGraph"))

        def unavailable_graph():
            raise NotImplementedError("Graph capture unavailable in this SDK build")

        api.NPUGraph = unavailable_graph
        result = self.measure(api)
        self.assertEqual(result["timing_method"], "events")
        self.assertIn("SDK build", result["timing_fallback_reason"])
        self.assertEqual(result["median_ms"], api.kernel_ms)

    def test_graph_out_of_memory_is_not_hidden_as_a_capability_fallback(self):
        api = self.configure(FakeAccelerator("npu", "NPUGraph"))

        def broken_graph():
            raise RuntimeError("device out of memory")

        api.NPUGraph = broken_graph
        with self.assertRaisesRegex(RuntimeError, "device out of memory"):
            self.measure(api)

    def test_explicit_graph_request_does_not_silently_change_timer(self):
        api = self.configure()
        with self.assertRaisesRegex((SystemExit, RuntimeError), "[Gg]raph"):
            self.measure(api, timing="graph")

    def test_explicit_events_uses_device_timestamps_even_when_graph_exists(self):
        api = self.configure(FakeAccelerator("mlu", "MLUGraph"), vendor="cambricon")
        result = self.measure(api, timing="events")
        self.assertEqual(result["timing_method"], "events")
        self.assertEqual(api.replay_calls, 0)
        self.assertGreater(api.elapsed_time_calls, 0)
        self.assertEqual(result["median_ms"], api.kernel_ms)

    def test_missing_device_events_fails_without_host_timer_fallback(self):
        api = self.configure()
        api.Event = None
        with self.assertRaisesRegex((SystemExit, RuntimeError), "[Ee]vent|timing"):
            self.measure(api, timing="events")

    def test_stub_event_constructor_fails_without_host_timer_fallback(self):
        api = self.configure()

        def unavailable_event(**kwargs):
            raise NotImplementedError("Timed events unavailable in this SDK")

        api.Event = unavailable_event
        with self.assertRaisesRegex(NotImplementedError, "Timed events unavailable"):
            self.measure(api, timing="events")

    def test_zero_and_nonfinite_device_timestamps_are_rejected(self):
        for invalid in (0.0, float("nan")):
            with self.subTest(kernel_ms=invalid):
                api = self.configure()
                api.kernel_ms = invalid
                with self.assertRaisesRegex(RuntimeError, "[Ii]nvalid.*event"):
                    self.measure(api, timing="events")


if __name__ == "__main__":
    unittest.main()
