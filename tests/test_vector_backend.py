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

    def test_missing_fma_lowering_falls_back_to_multiply_add(self):
        """A backend may carry a dtype but reject tl.fma; keep a measurable path."""
        def failing_probe(dtype, device):
            raise RuntimeError("PassManager::run failed")

        original = vector._fma_probe_kernel
        vector._fma_probe_kernel = SimpleNamespace(
            __getitem__=lambda self, grid: failing_probe)
        try:
            supported, reason = vector.backend_supports_explicit_fma("bf16", "cpu")
        finally:
            vector._fma_probe_kernel = original
        self.assertFalse(supported)
        self.assertIn("cannot lower tl.fma for BF16", reason)

    def test_unexposed_dtype_storage_reports_no_fma_support(self):
        original = vector.DTYPES["fp8"]
        vector.DTYPES["fp8"] = None
        try:
            supported, reason = vector.backend_supports_explicit_fma("fp8", "cpu")
        finally:
            vector.DTYPES["fp8"] = original
        self.assertFalse(supported)
        self.assertIn("does not expose FP8 storage", reason)

    def test_maca_promoted_bf16_arithmetic_is_not_native(self):
        """MetaX widens BF16 to FP32, so the rate must not be called native."""
        compiled = SimpleNamespace(n_regs=30, n_spills=0, asm={
            "llir": "%41 = fmul float %37, %40\n%49 = fadd float %45, %48\n"})
        info = vector.inspect_assembly(compiled, "metax", "bf16", backend="maca")
        self.assertFalse(info["native_arithmetic_verified"])
        self.assertTrue(info["maca_arithmetic_promoted"])
        self.assertEqual(info["maca_arithmetic_llvm_types"], ["float"])
        self.assertIn("not a native BF16 vector rate", info["assembly_check"])
        # Register metadata stays usable for spill-based exclusion.
        self.assertEqual(info["registers_per_thread"], 30)
        self.assertEqual(info["register_spills"], 0)

    def test_maca_retained_dtype_arithmetic_stays_unverified(self):
        """Matching IR width is still not final ISA proof of a native instruction."""
        compiled = SimpleNamespace(n_regs=40, n_spills=0, asm={
            "llir": "%4 = call float @llvm.fma.f32(float %1, float %2, float %3)\n"})
        info = vector.inspect_assembly(compiled, "metax", "fp32", backend="maca")
        self.assertFalse(info["native_arithmetic_verified"])
        self.assertFalse(info["maca_arithmetic_promoted"])
        self.assertIn("not final ISA evidence", info["assembly_check"])

    def test_maca_without_llir_reports_unaudited_arithmetic(self):
        compiled = SimpleNamespace(n_regs=12, n_spills=0, asm={})
        info = vector.inspect_assembly(compiled, "metax", "bf16", backend="maca")
        self.assertFalse(info["maca_llir_available"])
        self.assertFalse(info["native_arithmetic_verified"])
        self.assertIn("unaudited", info["assembly_check"])

    def test_widened_fma_probe_offered_only_for_narrow_floats(self):
        """FP32/FP64 must not be "widened": that would narrow the arithmetic."""
        for dtype in ("fp32", "fp64", "int8"):
            supported, reason = vector.backend_supports_widened_fma(dtype, "cpu")
            self.assertFalse(supported)
            self.assertIsNone(reason)

    def test_widened_fma_probe_failure_is_reported(self):
        def failing_probe(*args, **kwargs):
            raise RuntimeError("PassManager::run failed")

        original = vector._widened_fma_probe_kernel
        vector._widened_fma_probe_kernel = SimpleNamespace(
            __getitem__=lambda self, grid: failing_probe)
        try:
            supported, reason = vector.backend_supports_widened_fma("bf16", "cpu")
        finally:
            vector._widened_fma_probe_kernel = original
        self.assertFalse(supported)
        self.assertIn("cannot lower an FP32-widened FMA for BF16", reason)

    def test_widened_fma_launch_sets_the_constexpr(self):
        kernel = RecordingKernel()
        vector.launch_vector(kernel, (4,), "in", "out", "coeff", 128, 32, 64, 1,
                             False, 4, use_widened_fma=True)
        self.assertTrue(kernel.kwargs["USE_WIDENED_FMA"])
        self.assertEqual(kernel.kwargs["num_warps"], 4)

    def test_mul_add_launch_omits_the_widened_constexpr(self):
        kernel = RecordingKernel()
        vector.launch_vector(kernel, (4,), "in", "out", "coeff", 128, 32, 64, 1,
                             False, 4)
        self.assertNotIn("USE_WIDENED_FMA", kernel.kwargs)

    def test_maca_counts_narrowing_conversions_for_narrow_floats(self):
        """Conversions back to BF16 prove the state kept the narrow dtype."""
        compiled = SimpleNamespace(n_regs=30, n_spills=0, asm={"llir": (
            "%1 = call <2 x float> @llvm.mxc.pk.fma.f32(<2 x float> %a)\n"
            "%2 = call i16 @llvm.mxc.cvt.f32tobf16(float %1)\n"
            "%3 = call i16 @llvm.mxc.cvt.f32tobf16(float %2)\n")})
        info = vector.inspect_assembly(compiled, "metax", "bf16", backend="maca")
        self.assertEqual(info["maca_narrowing_conversion_sites"], 2)
        self.assertIn("@llvm.mxc.pk.fma.f32", info["maca_arithmetic_intrinsics"])
        # Widened arithmetic is still not a native BF16 rate.
        self.assertTrue(info["maca_arithmetic_promoted"])
        self.assertFalse(info["native_arithmetic_verified"])


if __name__ == "__main__":
    unittest.main()
