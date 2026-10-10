"""CPU-only checks for backend-specific vector benchmark configuration."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest

import bench_vector as vector


class RecordingKernel:
    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "compiled"

        return launch


class VectorBackendTests(unittest.TestCase):
    def parse(self, *arguments):
        return vector.build_parser().parse_args(arguments)

    def test_xpu_defaults_match_cluster_vector_width(self):
        args = self.parse()
        is_xpu = vector.configure_sweep(
            args, {"backend": "xpu", "vendor_name": "kunlunxin"})
        self.assertTrue(is_xpu)
        self.assertEqual(args.blocks_per_sm, [1, 2])
        self.assertEqual(args.block_sizes, [1024, 2048])
        self.assertEqual(args.num_warps, [None])
        self.assertEqual(args.unroll, [1, 4])
        # FP32 carries 32-bit state, so it keeps the strip-mined loop.
        self.assertEqual(args.iterations, vector.DEFAULT_ITERATIONS)

    def test_flagtree_xpu_reporting_cuda_backend_is_recognized(self):
        """FlagTree's XPU driver reports backend='cuda', like NVIDIA does."""
        args = self.parse()
        is_xpu = vector.configure_sweep(
            args, {"backend": "cuda", "vendor_name": "kunlunxin"})
        self.assertTrue(is_xpu)
        self.assertEqual(args.block_sizes, [1024, 2048])
        self.assertEqual(args.num_warps, [None])

    def test_xpu_narrow_dtypes_use_fully_static_loop_and_fewer_iterations(self):
        for dtype, expected_blocks in (("fp16", [1024, 2048]),
                                       ("bf16", [1024]),
                                       ("int8", [1024])):
            with self.subTest(dtype=dtype):
                args = self.parse("--dtype", dtype)
                with redirect_stdout(io.StringIO()):
                    is_xpu = vector.configure_sweep(
                        args, {"backend": "cuda", "vendor_name": "kunlunxin"})
                self.assertTrue(vector.needs_fully_static_loop(dtype, is_xpu))
                self.assertEqual(args.iterations, vector.XPU_STATIC_ITERATIONS)
                self.assertEqual(args.block_sizes, expected_blocks)
                # A fully unrolled body has no strip-mining factor to sweep.
                self.assertEqual(args.unroll, [1])

    def test_explicit_iterations_survive_the_static_loop_default(self):
        args = self.parse("--dtype", "fp16", "--iterations", "128")
        with redirect_stdout(io.StringIO()):
            vector.configure_sweep(args, {"backend": "cuda", "vendor_name": "kunlunxin"})
        self.assertEqual(args.iterations, 128)

    def test_other_backends_never_use_the_fully_static_loop(self):
        for dtype in ("fp16", "bf16", "int8", "fp32"):
            with self.subTest(dtype=dtype):
                args = self.parse("--dtype", dtype)
                is_xpu = vector.configure_sweep(
                    args, {"backend": "cuda", "vendor_name": "nvidia"})
                self.assertFalse(is_xpu)
                self.assertFalse(vector.needs_fully_static_loop(dtype, is_xpu))
                self.assertEqual(args.iterations, vector.DEFAULT_ITERATIONS)
                self.assertEqual(args.block_sizes, [256, 512])

    def test_other_backend_defaults_remain_unchanged(self):
        args = self.parse()
        is_xpu = vector.configure_sweep(
            args, {"backend": "cuda", "vendor_name": "nvidia"})
        self.assertFalse(is_xpu)
        self.assertEqual(args.blocks_per_sm, [8, 16])
        self.assertEqual(args.block_sizes, [256, 512])
        self.assertEqual(args.num_warps, [4, 8])
        self.assertEqual(args.unroll, [1, 4, 8])

    def test_explicit_xpu_grid_and_block_are_preserved(self):
        args = self.parse("--blocks-per-sm", "3", "--block-sizes", "4096",
                          "--unroll", "2", "--num-warps", "8")
        with redirect_stdout(io.StringIO()) as output:
            vector.configure_sweep(
                args, {"backend": "xpu", "vendor_name": "kunlunxin"})
        self.assertEqual(args.blocks_per_sm, [3])
        self.assertEqual(args.block_sizes, [4096])
        self.assertEqual(args.unroll, [2])
        self.assertEqual(args.num_warps, [None])
        self.assertIn("ignoring --num-warps", output.getvalue())

    def test_xpu_launch_omits_num_warps_and_selects_mul_add(self):
        kernel = RecordingKernel()
        result = vector.launch_vector(
            kernel, (12,), "input", "output", "coeff", 1024, 4096, 1024, 1,
            False, None,
        )
        self.assertEqual(result, "compiled")
        self.assertEqual(kernel.grid, (12,))
        self.assertFalse(kernel.args[-1])
        self.assertNotIn("num_warps", kernel.kwargs)

    def test_nvidia_launch_keeps_explicit_fma_and_num_warps(self):
        kernel = RecordingKernel()
        vector.launch_vector(
            kernel, (1,), "input", "output", "coeff", 128, 32, 128, 1,
            True, 4,
        )
        self.assertTrue(kernel.args[-1])
        self.assertEqual(kernel.kwargs["num_warps"], 4)

    def test_xpu_int8_uses_typed_arithmetic_without_dp4a_or_warps(self):
        args = self.parse("--dtype", "int8")
        with redirect_stdout(io.StringIO()):
            vector.configure_sweep(args, {"backend": "xpu", "vendor_name": "kunlunxin"})
        # Larger tiles make the backend fail to select a 64-lane INT8 add.
        self.assertEqual(args.block_sizes, [1024])
        kernel = RecordingKernel()
        vector.launch_vector(kernel, (12,), "input", "output", "coeff", 1024, 32,
                             1024, 1, False, None, is_int8=True, use_dp4a=False)
        self.assertEqual(kernel.kwargs, {"IS_INT8": True, "USE_DP4A": False})
        self.assertFalse(kernel.args[-1])

    def test_fully_static_flag_reaches_the_kernel_only_when_requested(self):
        kernel = RecordingKernel()
        vector.launch_vector(kernel, (8,), "input", "output", "coeff", 1024, 512,
                             1024, 1, False, None, fully_static=True)
        self.assertTrue(kernel.kwargs["FULLY_STATIC"])
        kernel = RecordingKernel()
        vector.launch_vector(kernel, (8,), "input", "output", "coeff", 1024, 512,
                             1024, 1, True, 4)
        self.assertNotIn("FULLY_STATIC", kernel.kwargs)

    def test_xpu_placeholder_resource_counts_are_unknown(self):
        compiled = SimpleNamespace(n_regs=0, n_spills=0, asm={})
        info = vector.inspect_assembly(compiled, "kunlunxin")
        self.assertIsNone(info["registers_per_thread"])
        self.assertIsNone(info["register_spills"])
        self.assertIn("placeholder", info["assembly_check"])


if __name__ == "__main__":
    unittest.main()
