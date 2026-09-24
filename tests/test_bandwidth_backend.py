"""CPU-only launch-contract checks for device-memory bandwidth benchmarks."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest import mock

import bench_bandwidth as bandwidth


class RecordingKernel:
    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **kwargs):
            self.kwargs = kwargs
            return "compiled"

        return launch


class BandwidthBackendTests(unittest.TestCase):
    def test_xpu_omits_warps_and_runs_each_block_once(self):
        args = SimpleNamespace(num_warps=[4, 8], block_sizes=[1024, 4096], quick=False)
        with redirect_stdout(io.StringIO()):
            configurations = bandwidth.configure_launches(
                args, {"backend": "xpu", "vendor_name": "kunlunxin"})
        self.assertEqual(configurations, [(1024, None), (4096, None)])
        for name in ("copy_kernel", "read_kernel"):
            kernel = RecordingKernel()
            source = SimpleNamespace(numel=lambda: 1031)
            with mock.patch.object(bandwidth, name, kernel):
                if name == "copy_kernel":
                    bandwidth.launch_copy(source, "dst", 1024, None)
                else:
                    bandwidth.launch_read(source, "sink", 1024, None, "kunlunxin")
                    self.assertEqual(kernel.kwargs["CACHE_MODIFIER"], "")
            self.assertEqual(kernel.grid, (2,))
            self.assertNotIn("num_warps", kernel.kwargs)

    def test_cuda_name_does_not_select_nvidia_cache_modifier(self):
        kernel = RecordingKernel()
        with mock.patch.object(bandwidth, "read_kernel", kernel):
            bandwidth.launch_read(SimpleNamespace(numel=lambda: 1024), "sink", 1024, 4, "metax")
        self.assertEqual(kernel.kwargs["CACHE_MODIFIER"], "")
        self.assertEqual(kernel.kwargs["num_warps"], 4)

    def test_other_backend_defaults_and_quick_mode_remain_valid(self):
        for quick, expected in ((False, [4, 8]), (True, [4])):
            with self.subTest(quick=quick):
                args = SimpleNamespace(num_warps=None, block_sizes=[1024], quick=quick)
                bandwidth.configure_launches(args, {"backend": "cuda", "vendor_name": "nvidia"})
                self.assertEqual(args.num_warps, expected)

    def test_unsupported_bf16_emits_status_before_allocating(self):
        args = SimpleNamespace(dtype="bf16")
        metadata = {"supports_bf16": False}
        with mock.patch.object(bandwidth, "parse_args", return_value=args), \
             mock.patch.object(bandwidth, "setup_device", return_value=metadata), \
             mock.patch.object(bandwidth, "get_device", return_value="fake:0"), \
             mock.patch.object(bandwidth, "emit_unsupported") as emit, \
             mock.patch.object(bandwidth.torch, "empty", side_effect=AssertionError("must not allocate")):
            bandwidth.main()
        self.assertEqual(emit.call_args.args[:3], ("bandwidth", metadata, args))
        self.assertIn("does not support BF16", emit.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
