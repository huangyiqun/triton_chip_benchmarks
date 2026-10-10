"""CPU regression checks for tensor formats, accounting and unsupported results."""

from contextlib import nullcontext, redirect_stdout
import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bench_tensor as tensor
from test_tensor_precision import compiled


class TensorFormatTests(unittest.TestCase):
    def arguments(self, *cli):
        parser = tensor.build_parser()
        args = parser.parse_args(cli)
        tensor._normalize_precision_args(parser, args)
        return args

    def test_all_requested_formats_and_tf32_alias(self):
        for name in ("bf16", "fp16", "fp32", "tf32", "int8", "fp64", "fp8"):
            with self.subTest(dtype=name):
                args = self.arguments("--dtype", name)
                self.assertEqual(args.requested_dtype, name)
                self.assertIsNotNone(getattr(tensor.torch, tensor._dtype_name(args.dtype)))
        args = self.arguments("--dtype", "tf32")
        self.assertEqual((args.dtype, args.input_precision), ("fp32", "tf32"))

    def test_integer_rates_do_not_claim_floating_point_operations(self):
        timing = {"median_ms": 2.0, "min_ms": 1.0}
        integer = tensor._rates(4_000_000_000, timing, "int8")
        floating = tensor._rates(4_000_000_000, timing, "fp64")
        self.assertEqual((integer["tops"], integer["peak_tops"]), (2.0, 4.0))
        self.assertEqual(integer["throughput_unit"], "TOPS")
        self.assertNotIn("flops", integer)
        self.assertNotIn("peak_tflops", integer)
        self.assertEqual(floating["throughput_unit"], "TFLOP/s")
        self.assertEqual(floating["peak_tflops"], 4.0)

    def test_int8_peak_inputs_and_reference_preserve_integer_math(self):
        cfg = tensor._peak_configs("nvidia", "int8")[0]
        a, b, c = tensor._peak_inputs(2, cfg, tensor.torch.int8, "cpu")
        self.assertEqual((a.dtype, b.dtype, c.dtype),
                         (tensor.torch.int8, tensor.torch.int8, tensor.torch.int32))
        self.assertGreater(a.abs().sum().item(), 0)
        with mock.patch.object(tensor, "full_precision_matmul", nullcontext):
            expected = tensor._reference(a, b[0])
        integer_reference = tensor.torch.matmul(a.to(tensor.torch.int64), b[0].to(tensor.torch.int64))
        tensor.torch.testing.assert_close(expected.to(tensor.torch.int64), integer_reference)
        self.assertLess(cfg["block_k"] * 262144, 2**31)

    def test_fp64_inputs_and_reference_remain_double(self):
        cfg = tensor._peak_configs("nvidia", "fp64")[0]
        a, b, c = tensor._peak_inputs(1, cfg, tensor.torch.float64, "cpu")
        self.assertEqual(c.dtype, tensor.torch.float64)
        with mock.patch.object(tensor, "full_precision_matmul", nullcontext):
            expected = tensor._reference(a, b[0])
        self.assertEqual(expected.dtype, tensor.torch.float64)

    def test_fp64_probe_accepts_double_and_rejects_float32_fallback(self):
        for mode in ("peak", "gemm"):
            for reduce_precision in (False, True):
                def gemm(a, b, c, *_args):
                    if reduce_precision:
                        a, b = a.float(), b.float()
                    c.copy_(tensor.torch.matmul(a, b))

                def peak(a, b, c, *_args):
                    if reduce_precision:
                        a, b = a.float(), b.float()
                    c[0].copy_(tensor.torch.matmul(a, b[0]))
                    c[1].copy_(tensor.torch.matmul(a, b[1]))

                with self.subTest(mode=mode, reduce_precision=reduce_precision), \
                     mock.patch.object(tensor, "_launch", side_effect=gemm), \
                     mock.patch.object(tensor, "_launch_peak", side_effect=peak):
                    if reduce_precision:
                        with self.assertRaises(tensor.TensorUnsupportedError):
                            tensor._validate_fp64_precision("cpu", mode)
                    else:
                        self.assertTrue(tensor._validate_fp64_precision("cpu", mode)["passed"])

    def test_native_instruction_audit_matches_each_requested_dtype(self):
        instructions = {
            "fp16": "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16;",
            "bf16": "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16;",
            "fp8": "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3;",
            "int8": "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8;",
            "fp64": "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64;",
        }
        for requested, correct_instruction in instructions.items():
            info = tensor._compiled_info(compiled(ptx=correct_instruction), "nvidia", requested)
            self.assertTrue(info["native_tensor_verified"])
            self.assertTrue(info["instruction_precision_verified"])
            for other, wrong_instruction in instructions.items():
                if requested == other:
                    continue
                with self.subTest(requested=requested, actual=other), \
                     self.assertRaises(tensor.TensorUnsupportedError):
                    tensor._compiled_info(compiled(ptx=wrong_instruction), "nvidia", requested)

    def test_fp64_scalar_fallback_is_not_called_tensor_peak(self):
        info = tensor._compiled_info(compiled(ptx="fma.rn.f64 %d0, %d1, %d2, %d3;"),
                                     "nvidia", "fp64")
        self.assertEqual(info["execution_kind"], "simt")
        self.assertFalse(info["native_tensor_verified"])
        self.assertTrue(info["instruction_precision_verified"])

    def test_fp8_mixed_operand_formats_do_not_pass_e4m3_audit(self):
        with self.assertRaises(tensor.TensorUnsupportedError):
            tensor._compiled_info(compiled(
                ptx="wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e5m2;"), "nvidia", "fp8")

    def test_amd_legacy_and_normalized_matrix_opcodes(self):
        for dtype, accumulator, shape, operand in (
            ("int8", "i32", "32x32x32", "i8"),
            ("fp64", "f64", "16x16x4", "f64"),
            ("fp32", "f32", "32x32x2", "f32"),
        ):
            for separator in ("", "_"):
                opcode = f"v_mfma_{accumulator}_{shape}{separator}{operand}"
                instruction = f"{opcode} v[0:15], v0, v1, v[0:15]"
                with self.subTest(dtype=dtype, opcode=opcode):
                    info = tensor._compiled_info(compiled(amdgcn=instruction), "amd", dtype,
                                                 "ieee" if dtype == "fp32" else None)
                    self.assertTrue(info["native_tensor_verified"])
                    self.assertTrue(info["instruction_precision_verified"])
                    self.assertEqual(info["tensor_instruction_types"], [opcode])

    def test_amd_legacy_spelling_still_rejects_wrong_operand_precision(self):
        for dtype, opcode in (
            ("fp32", "v_mfma_f32_32x32x4xf32"),
            ("fp32", "v_mfma_f32_32x32x8f16"),
            ("fp64", "v_mfma_f64_16x16x4f32"),
            ("int8", "v_mfma_i32_32x32x32ui8"),
        ):
            with self.subTest(dtype=dtype, opcode=opcode), \
                 self.assertRaises(tensor.TensorUnsupportedError):
                tensor._compiled_info(compiled(amdgcn=opcode), "amd", dtype,
                                      "ieee" if dtype == "fp32" else None)

    def test_unsupported_classification_preserves_real_failures(self):
        for exc in (NotImplementedError("dtype not implemented"),
                    tensor.CompilationError(None, ast.Pass(), "Unsupported dot operands"),
                    tensor.TensorUnsupportedError("required precision not available"),
                    RuntimeError("FP64 dot is not supported on this target")):
            self.assertIsNotNone(tensor._unsupported_reason(exc))
        for exc in (RuntimeError("CUDA out of memory"), RuntimeError("illegal memory access"),
                    RuntimeError("Invalid backend event time"),
                    AssertionError(), RuntimeError("PassManager::run failed"),
                    NotImplementedError("compiler visitor not implemented"),
                    tensor.CompilationError(None, ast.Pass(), "PassManager::run failed"),
                    tensor.CompilationError(None, ast.Pass(), "NameError: unknown variable"),
                    AssertionError("Tensor-likes are not close!")):
            self.assertIsNone(tensor._unsupported_reason(exc))

    def test_compiler_source_excerpt_is_not_capability_evidence(self):
        source = "# FP64 dot unsupported on some targets\nx = missing_name\n"
        node = ast.parse(source).body[0]
        exc = tensor.CompilationError(source, node, "NameError: missing_name is not defined")
        self.assertIn("unsupported", str(exc))
        self.assertIsNone(tensor._unsupported_reason(exc))

    def test_wrapped_compiler_error_uses_explicit_cause_diagnostic(self):
        unsupported = tensor.CompilationError(None, ast.Pass(), None)
        unsupported.__cause__ = AssertionError("Only int8 supported!")
        self.assertIsNotNone(tensor._unsupported_reason(unsupported))
        bug = tensor.CompilationError(None, ast.Pass(), "Unsupported dot operands")
        bug.__cause__ = NameError("missing compiler helper")
        self.assertIsNone(tensor._unsupported_reason(bug))

    def test_capability_failure_writes_structured_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with mock.patch("sys.argv", ["bench_tensor.py", "--dtype", "fp64", "--output", str(path)]), \
                 mock.patch.object(tensor, "setup_device", return_value={
                     "vendor_name": "example", "supports_fp64": False}), \
                 mock.patch.object(tensor, "run_peak") as run, redirect_stdout(io.StringIO()):
                tensor.main()
            document = json.loads(path.read_text())
            self.assertEqual(document["status"], "unsupported")
            self.assertEqual(document["results"], [])
            self.assertIn("FP64", document["reason"])
            run.assert_not_called()

    def test_fp8_storage_probe_accepts_a_faithful_round_trip(self):
        probe = tensor._validate_fp8_storage("cpu", tensor.torch.float8_e4m3fn)
        self.assertTrue(probe["passed"])
        self.assertEqual(probe["max_abs_error"], 0.0)

    def test_fp8_allocation_failure_is_reported_as_unsupported_not_error(self):
        failure = RuntimeError("call aclnnInplaceCopy failed, error code is 561103")
        with mock.patch.object(tensor.torch.Tensor, "to", side_effect=failure):
            with self.assertRaises(tensor.TensorUnsupportedError) as caught:
                tensor._validate_fp8_storage("cpu", tensor.torch.float8_e4m3fn)
        self.assertIsNotNone(tensor._unsupported_reason(caught.exception))

    def test_fp8_storage_failure_writes_unsupported_and_skips_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            failure = RuntimeError("call aclnnInplaceCopy failed, error code is 561103")
            with mock.patch("sys.argv", ["bench_tensor.py", "--dtype", "fp8",
                                         "--output", str(path)]), \
                 mock.patch.object(tensor, "setup_device",
                                   return_value={"vendor_name": "example"}), \
                 mock.patch.object(tensor, "get_device", return_value="cpu"), \
                 mock.patch.object(tensor.torch.Tensor, "to", side_effect=failure), \
                 mock.patch.object(tensor, "run_peak") as run, redirect_stdout(io.StringIO()):
                tensor.main()
            document = json.loads(path.read_text())
            self.assertEqual(document["status"], "unsupported")
            self.assertEqual(document["results"], [])
            self.assertIn("float8_e4m3fn", document["reason"])
            run.assert_not_called()

    def test_fp8_storage_probe_rejects_a_lossy_round_trip(self):
        original = tensor.torch.Tensor.float

        def lossy(self, *args, **kwargs):
            return original(self, *args, **kwargs) + 2.0**-4

        with mock.patch.object(tensor.torch.Tensor, "float", lossy):
            with self.assertRaises(tensor.TensorUnsupportedError) as caught:
                tensor._validate_fp8_storage("cpu", tensor.torch.float8_e4m3fn)
        self.assertIn("did not preserve", str(caught.exception))

    def test_fp8_device_fault_during_probe_is_not_called_unsupported(self):
        failure = RuntimeError("NPU illegal memory access was encountered")
        with mock.patch.object(tensor.torch.Tensor, "to", side_effect=failure):
            with self.assertRaises(tensor.TensorUnsupportedError) as caught:
                tensor._validate_fp8_storage("cpu", tensor.torch.float8_e4m3fn)
        self.assertIsNone(tensor._unsupported_reason(caught.exception))

    def test_unsupported_compile_never_enters_timer(self):
        args = self.arguments("--dtype", "int8", "--quick")
        with mock.patch.object(tensor, "available_memory_bytes", return_value=None), \
             mock.patch.object(tensor, "_launch_peak", side_effect=NotImplementedError("INT8 dot unsupported")), \
             mock.patch.object(tensor, "measure") as timer, \
             mock.patch.object(tensor, "emit_unsupported") as emit, redirect_stdout(io.StringIO()):
            tensor.run_peak(args, {"vendor_name": "example", "sm_count": 1}, tensor.torch.int8, "cpu")
        timer.assert_not_called()
        self.assertIn("unsupported", emit.call_args.args[3])

    def test_generic_compilation_error_propagates_without_unsupported_result(self):
        args = self.arguments("--dtype", "int8", "--quick")
        failure = tensor.CompilationError(None, ast.Pass(), "PassManager::run failed")
        with mock.patch.object(tensor, "available_memory_bytes", return_value=None), \
             mock.patch.object(tensor, "_launch_peak", side_effect=failure), \
             mock.patch.object(tensor, "emit_unsupported") as emit, \
             redirect_stdout(io.StringIO()), self.assertRaises(tensor.CompilationError):
            tensor.run_peak(args, {"vendor_name": "example", "sm_count": 1}, tensor.torch.int8, "cpu")
        emit.assert_not_called()

    def test_correctness_failure_is_not_reclassified_as_unsupported(self):
        args = self.arguments("--dtype", "int8", "--quick")
        def bad_kernel(_a, _b, c, *_args):
            c.fill_(-2**31)
            return compiled()

        with mock.patch.object(tensor, "available_memory_bytes", return_value=None), \
             mock.patch.object(tensor, "full_precision_matmul", nullcontext), \
             mock.patch.object(tensor, "_launch_peak", side_effect=bad_kernel), \
             mock.patch.object(tensor, "emit_unsupported") as emit, \
             redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
            tensor.run_peak(args, {"vendor_name": "example", "sm_count": 1}, tensor.torch.int8, "cpu")
        emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
