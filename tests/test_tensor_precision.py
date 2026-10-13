"""CPU-only checks for explicit IEEE FP32 and TF32 tensor policies."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest import mock

import bench_tensor as tensor


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


def compiled(ptx="", amdgcn=""):
    return SimpleNamespace(
        asm={"ptx": ptx, "amdgcn": amdgcn},
        n_regs=32,
        n_spills=0,
        metadata=SimpleNamespace(shared=0),
    )


class TensorPrecisionTests(unittest.TestCase):
    def parse(self, *arguments):
        parser = tensor.build_parser()
        args = parser.parse_args(arguments)
        tensor._normalize_precision_args(parser, args)
        return args

    def test_fp32_defaults_to_ieee_and_tf32_is_explicit(self):
        ieee = self.parse("--dtype", "fp32")
        tf32 = self.parse("--dtype", "fp32", "--input-precision", "tf32")
        self.assertEqual(ieee.input_precision, "ieee")
        self.assertEqual(tf32.input_precision, "tf32")
        self.assertEqual(tensor._dtype_name(ieee.dtype), "float32")
        self.assertEqual(tensor._dtype_name(tf32.dtype), "float32")
        self.assertEqual(
            tensor._precision_descriptor("fp32", "ieee")["math_mode"],
            "ieee_fp32",
        )
        self.assertEqual(
            tensor._precision_descriptor("fp32", "tf32")["input_storage_dtype"],
            "fp32",
        )

    def test_low_precision_inputs_reject_fp32_precision_option(self):
        parser = tensor.build_parser()
        args = parser.parse_args(("--dtype", "fp16", "--input-precision", "tf32"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tensor._normalize_precision_args(parser, args)

    def test_both_launch_paths_forward_explicit_precision(self):
        config = {"block_m": 32, "block_n": 32, "block_k": 32,
                  "num_warps": 4, "num_stages": 1}
        matrix_a = SimpleNamespace(shape=(32, 32))
        matrix_b = SimpleNamespace(shape=(32, 32))
        matrix_c = SimpleNamespace(shape=(32, 32))
        peak_a = SimpleNamespace(shape=(3, 32, 32))
        peak_b = SimpleNamespace(shape=(2, 3, 32, 32))
        peak_c = SimpleNamespace(shape=(2, 3, 32, 32))

        for precision in ("ieee", "tf32"):
            with self.subTest(precision=precision):
                gemm_kernel = RecordingKernel()
                peak_kernel = RecordingKernel()
                with mock.patch.object(tensor, "_gemm_kernel", gemm_kernel), \
                     mock.patch.object(tensor, "_tensor_peak_kernel", peak_kernel):
                    tensor._launch(matrix_a, matrix_b, matrix_c, config, precision)
                    tensor._launch_peak(peak_a, peak_b, peak_c, 8, config, precision)
                self.assertEqual(gemm_kernel.kwargs["DOT_INPUT_PRECISION"], precision)
                self.assertEqual(peak_kernel.kwargs["DOT_INPUT_PRECISION"], precision)

    def test_tf32_probe_rejects_bf16_and_fp16_fallbacks(self):
        def gemm_fallback(storage_dtype):
            def launch(a, _b, c, _config, _precision):
                c.copy_(a.to(storage_dtype).to(tensor.torch.float32))
            return launch

        def peak_fallback(storage_dtype):
            def launch(a, _b, c, _iterations, _config, _precision):
                converted = a.to(storage_dtype).to(tensor.torch.float32)
                c[0].copy_(converted)
                c[1].copy_(converted)
            return launch

        for mode in ("gemm", "peak"):
            for storage_dtype in (tensor.torch.bfloat16, tensor.torch.float16):
                target = "_launch" if mode == "gemm" else "_launch_peak"
                fallback = (gemm_fallback(storage_dtype) if mode == "gemm" else
                            peak_fallback(storage_dtype))
                with self.subTest(mode=mode, storage_dtype=storage_dtype), \
                     mock.patch.object(tensor, target, side_effect=fallback), \
                     self.assertRaisesRegex(RuntimeError, "did not honor"):
                    tensor._validate_fp32_precision(
                        tensor.torch.device("cpu"), "tf32", mode)

    def test_nvidia_ieee_scalar_fma_is_verified_but_not_tensor(self):
        info = tensor._compiled_info(
            compiled(ptx="fma.rn.f32 %f0, %f1, %f2, %f3;"),
            "nvidia", "fp32", "ieee",
        )
        self.assertEqual(info["execution_kind"], "simt")
        self.assertTrue(info["instruction_precision_verified"])
        self.assertFalse(info["native_tensor_verified"])

    def test_nvidia_tf32_mma_is_verified_as_tensor(self):
        instruction = (
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%f0}, {%r0}, {%r1}, {%f1};"
        )
        info = tensor._compiled_info(
            compiled(ptx=instruction), "nvidia", "fp32", "tf32"
        )
        self.assertEqual(info["execution_kind"], "tensor")
        self.assertTrue(info["instruction_precision_verified"])
        self.assertTrue(info["native_tensor_verified"])

    def test_instruction_audit_rejects_fp32_tf32_mismatches(self):
        tf32_instruction = "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32;"
        fp16_instruction = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;"
        with self.assertRaisesRegex(RuntimeError, "Requested IEEE FP32"):
            tensor._compiled_info(
                compiled(ptx=tf32_instruction), "nvidia", "fp32", "ieee"
            )
        with self.assertRaisesRegex(RuntimeError, "native TF32"):
            tensor._compiled_info(
                compiled(ptx=fp16_instruction), "nvidia", "fp32", "tf32"
            )
        for instruction in (
            fp16_instruction,
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;",
        ):
            with self.subTest(instruction=instruction), \
                 self.assertRaisesRegex(RuntimeError, "IEEE FP32"):
                tensor._compiled_info(
                    compiled(ptx=instruction), "nvidia", "fp32", "ieee"
                )

    def test_amd_ieee_and_tf32_matrix_instructions_are_distinguished(self):
        ieee = tensor._compiled_info(
            compiled(amdgcn="v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[0:15]"),
            "amd", "fp32", "ieee",
        )
        tf32 = tensor._compiled_info(
            compiled(amdgcn="v_mfma_f32_32x32x4_xf32 v[0:15], v0, v1, v[0:15]"),
            "amd", "fp32", "tf32",
        )
        self.assertTrue(ieee["native_tensor_verified"])
        self.assertTrue(tf32["native_tensor_verified"])
        self.assertNotEqual(ieee["tensor_instruction_types"],
                            tf32["tensor_instruction_types"])
        with self.assertRaisesRegex(RuntimeError, "IEEE FP32"):
            tensor._compiled_info(
                compiled(amdgcn="v_mfma_f32_32x32x8_f16 v[0:15], v0, v1, v[0:15]"),
                "amd", "fp32", "ieee",
            )

    def test_unknown_vendor_does_not_claim_instruction_verification(self):
        info = tensor._compiled_info(compiled(), "ascend", "fp32", "tf32")
        self.assertEqual(info["assembly_audit"], "unavailable")
        self.assertEqual(info["execution_kind"], "unverified")
        self.assertFalse(info["instruction_precision_verified"])
        self.assertFalse(info["native_tensor_verified"])

    def test_peak_tolerance_stays_far_below_a_dropped_iteration(self):
        """An MMA that rounds internally must pass; a skipped operation must not."""
        iterations, block_k = 1024, 32
        tolerance = tensor._peak_tolerance("bf16", block_k, iterations)
        # The smallest fault the exact check existed to catch: one of the
        # iterations missing changes an entry by at least one lattice step.
        self.assertLess(tolerance * 64, 1 / 256)
        # The rounding actually observed on a backend whose MMA rounds products.
        self.assertGreater(tolerance, 1.9073486328125e-06)

    def test_peak_tolerance_is_exact_for_integer_accumulation(self):
        self.assertEqual(tensor._peak_tolerance("int8", 64, 4096), 0.0)

    def test_peak_tolerance_scales_with_accumulator_width(self):
        self.assertLess(tensor._peak_tolerance("fp64", 16, 1024),
                        tensor._peak_tolerance("fp32", 16, 1024))

    def test_validate_peak_rejects_a_difference_above_the_bound(self):
        import torch

        reference = torch.zeros((4, 4), dtype=torch.float32)
        actual = reference.clone()
        actual[0, 0] = tensor._peak_tolerance("bf16", 32, 1024) * 10
        with self.assertRaises(AssertionError):
            tensor._validate_peak(actual, reference, "bf16", 32, 1024)

    def test_validate_peak_accepts_bounded_rounding_and_reports_it(self):
        import torch

        reference = torch.zeros((4, 4), dtype=torch.float32)
        actual = reference.clone()
        actual[0, 0] = 1.9073486328125e-06
        result = tensor._validate_peak(actual, reference, "bf16", 32, 1024)
        self.assertTrue(result["passed"])
        self.assertEqual(result["rtol"], 0)
        self.assertGreater(result["atol"], 0)
        self.assertAlmostEqual(result["max_abs_error"], 1.9073486328125e-06)

    def test_row_multiset_detects_a_permutation_but_not_a_value_error(self):
        import torch

        expected = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        permuted = expected[[0, 2, 1, 3], :]
        self.assertTrue(tensor._row_multiset_matches(permuted, expected))
        wrong = expected.clone()
        wrong[0, 0] += 1
        self.assertFalse(tensor._row_multiset_matches(wrong, expected))

    def test_dot_lowering_failure_is_reported_as_unsupported(self):
        """A dtype can be storable while tl.dot has no lowering for it."""
        import torch

        def failing(*args, **kwargs):
            raise RuntimeError("PassManager::run failed")

        shapes = [{"block_m": 16, "block_n": 16, "block_k": 64, "num_warps": 4}]
        with mock.patch.object(tensor, "_launch_peak", failing):
            with self.assertRaises(tensor.TensorUnsupportedError) as caught:
                tensor._validate_dot_lowering(
                    torch.device("cpu"), torch.float32, "fp8", "peak", shapes)
        self.assertIn("cannot lower tl.dot for FP8", str(caught.exception))

    def test_dot_lowering_accepts_a_format_any_tile_can_compile(self):
        """One bad tile shape must not discard a measurable format."""
        import torch

        calls = []

        def sometimes(a, b, c, iterations, config, precision=None):
            calls.append(config["block_m"])
            if config["block_m"] == 32:
                raise RuntimeError("shape-specific lowering failure")
            return "compiled"

        shapes = [{"block_m": 32, "block_n": 32, "block_k": 64, "num_warps": 4},
                  {"block_m": 64, "block_n": 64, "block_k": 64, "num_warps": 4}]
        with mock.patch.object(tensor, "_launch_peak", sometimes):
            result = tensor._validate_dot_lowering(
                torch.device("cpu"), torch.float32, "fp8", "peak", shapes)
        self.assertTrue(result["passed"])
        self.assertEqual(result["probed_tile"], "64x64x64")
        self.assertEqual(len(result["rejected_tiles"]), 1)
        self.assertEqual(calls, [32, 64])

    def test_dot_lowering_does_not_blame_dtype_for_resource_limits(self):
        """A tile too large to launch is not evidence about the dtype."""
        import torch
        import triton

        def oom(*args, **kwargs):
            raise triton.OutOfResources(2048, 1024, "shared memory")

        shapes = [{"block_m": 64, "block_n": 64, "block_k": 64, "num_warps": 4}]
        with mock.patch.object(tensor, "_launch_peak", oom):
            with self.assertRaises(tensor.TensorUnsupportedError) as caught:
                tensor._validate_dot_lowering(
                    torch.device("cpu"), torch.float32, "fp8", "peak", shapes)
        self.assertIn("resource limit", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
