"""CPU-only validation of vector dtype semantics, counting, and status output."""

from contextlib import redirect_stdout
import ast
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import bench_vector as vector


class VectorDtypeTests(unittest.TestCase):
    def test_all_requested_types_parse_and_fp32_remains_default(self):
        parser = vector.build_parser()
        self.assertEqual(parser.parse_args([]).dtype, "fp32")
        for dtype in ("bf16", "fp16", "fp32", "int8", "fp64", "fp8"):
            self.assertEqual(parser.parse_args(["--dtype", dtype]).dtype, dtype)

    def test_integer_ops_use_tops_and_floats_use_tflops(self):
        timing = {"median_ms": 2.0, "min_ms": 1.0}
        integer = vector.throughput_fields("int8", 17, 32, timing)
        floating = vector.throughput_fields("fp64", 17, 32, timing)
        self.assertEqual(integer["counted_integer_ops"], 8 * 8 * 17 * 32)
        self.assertEqual(floating["counted_flops"], 2 * 8 * 17 * 32)
        self.assertEqual(integer["peak_tops"], integer["counted_integer_ops"] / 1e9)
        self.assertNotIn("peak_tflops", integer)
        self.assertNotIn("peak_tops", floating)
        xpu = vector.throughput_fields("int8", 17, 32, timing, int8_mul_add=True)
        self.assertEqual(xpu["counted_integer_ops"], 2 * 8 * 17 * 32)
        self.assertEqual(xpu["ops_per_accumulator_update"], 2)

    def test_xpu_modular_int8_reference_checks_wraparound_and_all_lanes(self):
        inputs, coeff = vector._make_inputs("int8", 256, "cpu", int8_mul_add=True)
        inputs[:] = torch.arange(-128, 128).to(torch.int8)
        self.assertEqual(inputs.dtype, torch.int8)
        self.assertEqual(coeff.dtype, torch.int8)
        actual = inputs.long()
        for iteration in range(1, 4098):
            actual = (actual * int(coeff[0]) + coeff[1:, None].long() + 128) % 256 - 128
            if iteration in (1, 2, 3, 32, 4096, 4097):
                info = vector.check_int8_mul_add_output(inputs, actual.to(torch.int8), coeff, iteration)
                self.assertEqual(info["max_abs_error"], 0)
        actual[3, 200] = (actual[3, 200] + 129) % 256 - 128
        with self.assertRaisesRegex(RuntimeError, "modular INT8"):
            vector.check_int8_mul_add_output(inputs, actual.to(torch.int8), coeff, 4097)

    def test_xpu_integer_ir_is_not_final_native_isa_evidence(self):
        artifacts = {"ttxir": '%0 = ttx.vvmuli %a, %b : tensor<1xvector<64xi8>>\n'
                               '%1 = ttx.vvaddi %0, %c : tensor<1xvector<64xi8>>',
                     "llir": '%0 = mul <64 x i8> %a, %b\n%1 = add <64 x i8> %0, %c'}
        compiled = SimpleNamespace(asm=artifacts, n_regs=0, n_spills=0)
        info = vector.inspect_assembly(compiled, "kunlunxin", "int8")
        self.assertFalse(info["native_arithmetic_verified"])
        self.assertTrue(info["xpu_ir_vector_i8_evidence"]["ttxir"])
        self.assertTrue(info["xpu_ir_vector_i8_evidence"]["llir"])
        for assembly in ('vmul.b vr0, vr1, vr2\nvadd.b vr0, vr0, vr3',
                         '# vmul.i8 vr0, vr1, vr2\n# vadd.i8 vr0, vr0, vr3'):
            artifacts["asm"] = assembly
            self.assertFalse(vector.inspect_assembly(compiled, "kunlunxin", "int8")["native_arithmetic_verified"])
        artifacts["asm"] = 'vmul.i8 vr0, vr1, vr2\nvadd.i8 vr0, vr0, vr3'
        self.assertTrue(vector.inspect_assembly(compiled, "kunlunxin", "int8")["native_arithmetic_verified"])

    def test_xpu_float_mac_audit_and_bf16_promotion_rejection(self):
        for dtype, opcode in (("fp16", "vmac.hf.mz.rn"), ("fp32", "vmac.f.mz.rn")):
            info = vector.inspect_assembly(SimpleNamespace(asm={"asm": opcode + ' vr0, vr1, vr2'}),
                                           "kunlunxin", dtype)
            self.assertTrue(info["native_arithmetic_verified"])
            self.assertTrue(info["hardware_fusion_verified"])
        with self.assertRaisesRegex(vector.UnsupportedVectorDtype, "FP32.*BF16"):
            vector.inspect_assembly(SimpleNamespace(asm={"asm": 'vmac.f.mz.rn vr0, vr1, vr2'}),
                                    "kunlunxin", "bf16")
        info = vector.inspect_assembly(SimpleNamespace(asm={}), "kunlunxin", "bf16")
        self.assertFalse(info["native_arithmetic_verified"])

    def test_xpu_typed_int8_measurement_remains_explicitly_unverified_without_isa(self):
        args = vector.build_parser().parse_args([
            "--dtype", "int8", "--block-sizes", "64", "--blocks-per-sm", "1",
            "--unroll", "1", "--iterations", "4"])
        metadata = {"vendor_name": "kunlunxin", "backend": "xpu", "sm_count": 1}
        vector.configure_sweep(args, metadata)

        def launch(kernel, grid, inputs, output, coeff, elements, iterations, *unused):
            self.assertEqual(inputs.dtype, torch.int8)
            self.assertEqual(output.dtype, torch.int8)
            state = inputs.clone()
            for _ in range(iterations):
                state = state * coeff[0] + coeff[1:, None]
            output.copy_(state)
            return SimpleNamespace(asm={}, n_regs=0, n_spills=0)

        with mock.patch.object(vector, "get_device", return_value=torch.device("cpu")), \
             mock.patch.object(vector, "available_memory_bytes", return_value=None), \
             mock.patch.object(vector, "launch_vector", side_effect=launch), \
             mock.patch.object(vector, "measure", return_value={"min_ms": 1.0, "median_ms": 2.0}), \
             mock.patch.object(vector, "emit_results") as emit, \
             redirect_stdout(io.StringIO()) as printed:
            vector.run_benchmark(args, metadata, True)
        name, _, _, rows = emit.call_args.args
        self.assertEqual(name, "vector_int8_mul_add")
        row = rows[0]
        self.assertEqual(row["operation"], "INT8_MUL_ADD_MOD256")
        self.assertEqual(row["accumulator_dtype"], "int8")
        self.assertEqual(row["arithmetic_dtype"], "unverified")
        self.assertEqual(row["source_arithmetic_dtype"], "int8")
        self.assertFalse(row["native_arithmetic_verified"])
        self.assertEqual(row["counted_integer_ops"], 2 * 8 * 64 * 4)
        self.assertIn("not a verified native INT8 vector peak", printed.getvalue())

    def test_low_precision_recurrence_is_exact_and_odd_probe_rejects_identity(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                inputs, coeff = vector._make_inputs(dtype, 31, "cpu")
                current = inputs.clone()
                with mock.patch.object(vector, "reference_device", return_value=torch.device("cpu")):
                    for iteration in range(1, 34):
                        current = current * coeff[0] + coeff[1:, None]
                        info = vector.check_output(inputs, current, coeff, iteration)
                        self.assertEqual(info["max_abs_error"], 0)
                        self.assertEqual(info["max_abs_tolerance"], 0)
                    with self.assertRaisesRegex(RuntimeError, "correctness failed"):
                        vector.check_output(inputs, inputs, coeff, 33)

    def test_fp64_reference_has_fp64_scale_tolerance_and_rejects_corruption(self):
        inputs, coeff = vector._make_inputs("fp64", 31, "cpu")
        actual = inputs.clone()
        for _ in range(64):
            actual = actual * coeff[0] + coeff[1:, None]
        with mock.patch.object(vector, "reference_device", return_value=torch.device("cpu")):
            info = vector.check_output(inputs, actual, coeff, 64)
            self.assertLess(info["max_abs_tolerance"], 1e-10)
            actual[0, 0] += 1e-8
            with self.assertRaisesRegex(RuntimeError, "correctness failed"):
                vector.check_output(inputs, actual, coeff, 64)

    def test_int8_reference_matches_signed_naive_recurrence(self):
        inputs, coeff = vector._make_inputs("int8", 43, "cpu")
        weights = vector._signed_bytes(coeff[0].item())
        self.assertEqual(weights, [3, -2, 1, -1])
        actual = inputs.long()
        for iteration in range(1, 4097):
            actual = vector._dot4_reference(actual, weights) + coeff[1:, None]
            if iteration in (1, 2, 3, 32, 4096):
                info = vector.check_int8_output(inputs, actual.int(), coeff, iteration)
                self.assertEqual(info["max_abs_error"], 0)
        actual[3, 7] += 1
        with self.assertRaisesRegex(RuntimeError, "INT8 correctness failed"):
            vector.check_int8_output(inputs, actual.int(), coeff, 4096)

    def test_nvidia_audit_requires_requested_native_arithmetic(self):
        cases = {"fp16": "fma.rn.f16x2", "bf16": "fma.rn.bf16x2",
                 "fp32": "fma.rn.f32", "fp64": "fma.rn.f64",
                 "int8": "dp4a.s32.s32"}
        for dtype, instruction in cases.items():
            with self.subTest(dtype=dtype):
                compiled = SimpleNamespace(asm={"ptx": instruction + " %r1, %r2, %r3, %r4;"})
                result = vector.inspect_assembly(compiled, "nvidia", dtype)
                self.assertTrue(result["native_arithmetic_verified"])
        compiled = SimpleNamespace(asm={"ptx": "fma.rn.f32 %f1, %f2, %f3, %f4;"})
        with self.assertRaises(vector.UnsupportedVectorDtype):
            vector.inspect_assembly(compiled, "nvidia", "bf16")
        with self.assertRaises(vector.UnsupportedVectorDtype):
            vector.inspect_assembly(compiled, "nvidia", "int8")

    def test_unknown_vendor_can_report_compiled_unverified_arithmetic(self):
        info = vector.inspect_assembly(SimpleNamespace(asm={}), "other_vendor", "bf16")
        self.assertFalse(info["native_arithmetic_verified"])
        self.assertIsNone(vector.unsupported_dtype_reason("bf16", {}))

    def test_missing_storage_support_does_not_hide_oom_or_device_faults(self):
        self.assertTrue(vector._storage_failure_is_unsupported(
            RuntimeError('"copy_" not implemented for BFloat16')))
        for message in ("out of memory", "illegal memory access", "device lost"):
            self.assertFalse(vector._storage_failure_is_unsupported(RuntimeError(message)))

    def test_fp8_and_missing_fp64_write_structured_unsupported(self):
        metadata = {"vendor_name": "kunlunxin", "backend": "xpu", "supports_fp64": False}
        with TemporaryDirectory() as directory:
            for dtype in ("fp8", "fp64"):
                path = Path(directory) / f"{dtype}.json"
                with mock.patch("sys.argv", ["bench_vector.py", "--dtype", dtype,
                                             "--output", str(path)]), \
                     mock.patch.object(vector, "setup_device", return_value=metadata), \
                     mock.patch.object(vector, "run_benchmark") as run, \
                     redirect_stdout(io.StringIO()) as printed:
                    vector.main()
                run.assert_not_called()
                document = json.loads(path.read_text())
                self.assertEqual(document["status"], "unsupported")
                self.assertEqual(document["results"], [])
                self.assertTrue(document["reason"])
                self.assertIn("UNSUPPORTED:", printed.getvalue())

    def test_non_xpu_non_nvidia_int8_is_unsupported_before_launch_or_timing(self):
        with TemporaryDirectory() as directory:
            for vendor in ("kunlunxin", "amd", "metax", "other_vendor"):
                metadata = {"vendor_name": vendor, "backend": "cuda"}
                path = Path(directory) / f"{vendor}.json"
                with self.subTest(vendor=vendor), \
                     mock.patch("sys.argv", ["bench_vector.py", "--dtype", "int8",
                                             "--output", str(path)]), \
                     mock.patch.object(vector, "setup_device", return_value=metadata), \
                     mock.patch.object(vector, "run_benchmark") as run, \
                     redirect_stdout(io.StringIO()):
                    vector.main()
                run.assert_not_called()
                document = json.loads(path.read_text())
                self.assertEqual(document["status"], "unsupported")
                self.assertEqual(document["results"], [])
                self.assertIn("benchmark-path limitation", document["reason"])
        args = vector.build_parser().parse_args(["--dtype", "int8"])
        with mock.patch.object(vector, "get_device") as device, \
             self.assertRaises(vector.UnsupportedVectorDtype):
            vector.run_benchmark(args, {"vendor_name": "kunlunxin"}, True)
        device.assert_not_called()
        self.assertIsNone(vector.unsupported_dtype_reason(
            "int8", {"vendor_name": "kunlunxin", "backend": "xpu"}))

    def test_compile_classification_requires_explicit_arithmetic_capability(self):
        unsupported = (
            vector.CompilationError(None, ast.Pass(), "Unsupported dtype: bfloat16"),
            RuntimeError("Feature 'dp4a' requires .target sm_61 or higher"),
            NotImplementedError("float64 arithmetic is not supported"),
        )
        for error in unsupported:
            with self.subTest(error=str(error)):
                self.assertTrue(vector._compile_failure_is_unsupported(error))
        actual_errors = (
            vector.CompilationError(None, ast.Pass(), "invalid block size"),
            vector.CompilationError(None, ast.Pass(), "layout conversion verification failed"),
            vector.CompilationError('x = "unsupported dtype fp16"', ast.Pass(), "bad launch parameter"),
            vector.CompilationError('x = "unsupported dtype fp16"', ast.Pass()),
            NotImplementedError("compiler lowering pass has not been implemented"),
            RuntimeError("unsupported type: pointer<tensor<32xf32>>"),
            RuntimeError("no kernel image is available"),
            RuntimeError("out of memory"),
        )
        for error in actual_errors:
            with self.subTest(error=str(error)):
                self.assertFalse(vector._compile_failure_is_unsupported(error))

    def _run_mock_sweep(self, launch_results):
        args = vector.build_parser().parse_args([
            "--dtype", "fp32", "--block-sizes", "128", "256", "--blocks-per-sm", "1",
            "--num-warps", "4", "--unroll", "1", "--iterations", "32"])
        metadata = {"vendor_name": "other_vendor", "backend": "other", "sm_count": 1}
        vector.configure_sweep(args, metadata)
        with mock.patch.object(vector, "get_device", return_value=torch.device("cpu")), \
             mock.patch.object(vector, "available_memory_bytes", return_value=None), \
             mock.patch.object(vector, "launch_vector", side_effect=launch_results) as launch, \
             mock.patch.object(vector, "check_output", return_value={}), \
             mock.patch.object(vector, "measure", return_value={"min_ms": 1.0, "median_ms": 2.0}) as timing, \
             mock.patch.object(vector, "emit_results") as emit, \
             redirect_stdout(io.StringIO()):
            vector.run_benchmark(args, metadata, False)
        return launch, timing, emit

    def test_config_compile_errors_and_native_rejections_try_remaining_configs(self):
        compiled = SimpleNamespace(asm={}, n_regs=32, n_spills=0)
        errors = (
            (vector.CompilationError(None, ast.Pass(), "bad layout for this block"), "compilation_error"),
            (vector.CompilationError(None, ast.Pass(), "Unsupported dtype: float32"), "unsupported"),
        )
        for error, expected_status in errors:
            with self.subTest(status=expected_status):
                launch, timing, emit = self._run_mock_sweep([error, compiled])
                self.assertEqual(launch.call_count, 2)
                self.assertEqual(timing.call_count, 1)
                _, metadata, _, rows = emit.call_args.args
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["block_size"], 256)
                self.assertEqual(metadata["vector_excluded_configurations"][0]["status"], expected_status)

    def test_all_compile_errors_stay_errors_and_only_capability_failures_are_unsupported(self):
        bug = vector.CompilationError(None, ast.Pass(), "internal layout verification failed")
        unsupported = vector.CompilationError(None, ast.Pass(), "Unsupported dtype: float32")
        for errors in ([bug, bug], [unsupported, bug]):
            with self.subTest(errors=[str(error) for error in errors]), \
                 self.assertRaisesRegex(RuntimeError, "compiler/configuration/resource") as raised:
                self._run_mock_sweep(errors)
            self.assertNotIsInstance(raised.exception, vector.UnsupportedVectorDtype)
        with self.assertRaisesRegex(vector.UnsupportedVectorDtype, "No tested vector configuration"):
            self._run_mock_sweep([unsupported, unsupported])

    def test_correctness_and_timing_errors_are_not_labeled_unsupported(self):
        metadata = {"vendor_name": "other_vendor", "backend": "other"}
        for message in ("correctness failed", "Invalid backend event time", "out of memory"):
            with self.subTest(message=message), \
                 mock.patch("sys.argv", ["bench_vector.py", "--dtype", "fp16"]), \
                 mock.patch.object(vector, "setup_device", return_value=metadata), \
                 mock.patch.object(vector, "run_benchmark", side_effect=RuntimeError(message)), \
                 mock.patch.object(vector, "emit_unsupported") as emit, \
                 self.assertRaisesRegex(RuntimeError, message):
                vector.main()
            emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
