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


if __name__ == "__main__":
    unittest.main()
