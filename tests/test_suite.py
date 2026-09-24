"""CPU-only tests of metric selection, status reporting, and suite isolation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import bench_suite as suite
from suite_report import METRICS, empty_metric, format_table, summarize


def metric(key):
    return next(item for item in METRICS if item.key == key)


class SuiteReportTests(unittest.TestCase):
    def test_complete_metric_table_has_all_requested_precisions(self):
        self.assertEqual(len(METRICS), 15)
        self.assertEqual({item.precision for item in METRICS if item.group == "Tensor"},
                         {"bf16", "fp16", "fp32", "tf32", "int8", "fp64", "fp8"})
        self.assertEqual({item.precision for item in METRICS if item.group == "Vector"},
                         {"bf16", "fp16", "fp32", "int8", "fp64", "fp8"})

    def test_capacity_is_reported_in_gib_and_missing_capacity_is_explicit(self):
        item = metric("memory_capacity")
        result = summarize(item, {"device": {"total_memory_bytes": 96 * 2**30}})
        self.assertEqual(result["value"], 96)
        self.assertEqual(result["unit"], "GiB")
        missing = summarize(item, {"device": {"total_memory_bytes": None}})
        self.assertEqual(missing["status"], "unavailable")
        self.assertIsNone(missing["value"])

    def test_bandwidth_uses_eligible_copy_and_keeps_read_auxiliary(self):
        rows = [
            {"pattern": "copy", "peak_gbps": 11000, "gbps": 10000, "peak_eligible": False},
            {"pattern": "copy", "peak_gbps": 3500, "gbps": 3300, "peak_eligible": True},
            {"pattern": "read", "peak_gbps": 3900, "gbps": 3700, "peak_eligible": True},
        ]
        result = summarize(metric("memory_bandwidth"), {"results": rows})
        self.assertEqual(result["value"], 3500)
        self.assertEqual(result["median_value"], 3300)
        self.assertEqual(result["patterns"]["read"]["value"], 3900)
        self.assertEqual(result["patterns"]["copy"]["value"], 3500)
        self.assertEqual(result["status"], "ok")

    def test_cache_sensitive_or_smoke_rates_never_become_hbm_peak(self):
        for eligible, smoke in ((False, False), (True, True)):
            with self.subTest(eligible=eligible, smoke=smoke):
                result = summarize(metric("memory_bandwidth"), {"results": [{
                    "pattern": "copy", "peak_gbps": 10000, "gbps": 9000,
                    "peak_eligible": eligible, "smoke_only": smoke}]})
                self.assertEqual(result["status"], "not_measured")
                self.assertIsNone(result["value"])
                self.assertEqual(result["patterns"]["copy"]["value"], 10000)

    def test_integer_ops_do_not_use_float_rates(self):
        item = metric("vector_int8")
        result = summarize(item, {"results": [{"peak_tops": 80, "tops": 75}]})
        self.assertEqual((result["value"], result["unit"]), (80, "TOP/s"))
        wrong_units = summarize(item, {"results": [{"peak_tflops": 80, "tflops": 75}]})
        self.assertEqual(wrong_units["status"], "error")

    def test_simt_matrix_result_preserves_value_and_verification(self):
        result = summarize(metric("tensor_fp32"), {"results": [{
            "status": "ok", "peak_tflops": 20, "tflops": 19,
            "execution_kind": "simt", "compiled": {"native_tensor_verified": False,
            "instruction_precision_verified": True, "assembly_audit": "verified"}}]})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["value"], 20)
        self.assertFalse(result["native_tensor_verified"])
        self.assertIn("measured matrix rate", format_table([result]))
        self.assertIn("SIMT scalar/vector instructions", format_table([result]))
        self.assertIn("no native tensor MMA", format_table([result]))

    def test_tensor_unaudited_precision_or_execution_preserves_unverified_value(self):
        for info in (
            {"execution_kind": "unverified", "native_tensor_verified": False,
             "instruction_precision_verified": False},
            {"execution_kind": "tensor", "native_tensor_verified": True,
             "instruction_precision_verified": False},
            {"execution_kind": "unverified", "native_tensor_verified": False,
             "instruction_precision_verified": True},
            {},
        ):
            with self.subTest(compiled=info):
                result = summarize(metric("tensor_fp32"), {
                    "device": {"vendor_name": "kunlunxin"}, "arguments": {"quick": True},
                    "results": [{"peak_tflops": 42, "tflops": 40,
                                 "precision_verified": True, "compiled": info}]})
                self.assertEqual(result["status"], "unverified")
                self.assertEqual((result["value"], result["median_value"]), (42, 40))
                self.assertTrue(result["smoke_only"])
                self.assertIn("unverified", format_table([result]))

    def test_audited_native_tensor_measurement_is_ok(self):
        result = summarize(metric("tensor_bf16"), {"results": [{
            "peak_tflops": 140, "tflops": 139,
            "compiled": {"execution_kind": "tensor", "native_tensor_verified": True,
                         "instruction_precision_verified": True}}]})
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("reason", result)

    def test_vector_unverified_arithmetic_keeps_rates_and_portable_int8_details(self):
        result = summarize(metric("vector_int8"), {"results": [{
            "peak_tops": 12, "tops": 11, "native_arithmetic_verified": False,
            "arithmetic_dtype": "int32", "includes_unpacking": True}]})
        self.assertEqual(result["status"], "unverified")
        self.assertEqual((result["value"], result["median_value"]), (12, 11))
        self.assertEqual(result["arithmetic_dtype"], "int32")
        self.assertTrue(result["includes_unpacking"])
        self.assertIn("INT32 arithmetic and includes unpacking", result["reason"])
        native = summarize(metric("vector_fp16"), {"results": [{
            "peak_tflops": 70, "tflops": 69, "native_arithmetic_verified": True}]})
        self.assertEqual(native["status"], "ok")

    def test_memory_results_do_not_require_compute_instruction_audits(self):
        result = summarize(metric("memory_bandwidth"), {"results": [{
            "pattern": "copy", "peak_gbps": 3500, "gbps": 3400,
            "peak_eligible": True, "native_arithmetic_verified": False,
            "instruction_precision_verified": False}]})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["value"], 3500)

    def test_unsupported_is_never_zero_or_guessed_from_failure_text(self):
        item = metric("vector_fp8")
        declared = summarize(item, {"status": "unsupported", "reason": "promotes to FP32"})
        self.assertEqual(declared["status"], "unsupported")
        self.assertIsNone(declared["value"])
        error = summarize(item, {"results": [{"status": "error", "reason": "unsupported compiler?"}]})
        self.assertEqual(error["status"], "error")
        self.assertIn("promotes to FP32", format_table([declared]))
        missing_path = summarize(item, {"status": "not_implemented", "reason": "no native path"})
        self.assertEqual(missing_path["status"], "not_implemented")
        self.assertIsNone(missing_path["value"])

    def test_nonfinite_rates_cannot_win_selection(self):
        result = summarize(metric("vector_fp32"), {"results": [
            {"peak_tflops": float("inf"), "tflops": 1},
            {"peak_tflops": 20, "tflops": 18},
            {"peak_tflops": 0, "tflops": 0}]})
        self.assertEqual(result["value"], 20)


class SuiteRunnerTests(unittest.TestCase):
    def args(self, root):
        return suite.build_parser().parse_args(["--output", str(root / "report.json"),
                                               "--quick", "--timing", "events"])

    def test_commands_use_current_python_and_explicit_precisions(self):
        args = self.args(Path("/tmp"))
        command = suite.metric_command(metric("tensor_tf32"), args, Path("out.json"))
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(Path(command[2]).is_absolute())
        self.assertEqual(command[command.index("--dtype") + 1], "fp32")
        self.assertEqual(command[command.index("--input-precision") + 1], "tf32")
        self.assertIn("--quick", command)

    def test_failed_child_cannot_reuse_stale_success_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            output = root / "vector_fp32.json"
            output.write_text(json.dumps({"results": [{"peak_tflops": 99, "tflops": 90}]}))
            with mock.patch.object(suite.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
                result, _ = suite.run_metric(metric("vector_fp32"), args, root)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["value"])
            self.assertFalse(output.exists())

    def test_timeout_is_error_and_declared_unsupported_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            with mock.patch.object(suite.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 900)):
                result, _ = suite.run_metric(metric("vector_fp8"), args, root)
            self.assertEqual(result["status"], "error")

            def unsupported(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"status": "unsupported", "reason": "no native FP8"}))
                return subprocess.CompletedProcess(command, 1)

            with mock.patch.object(suite.subprocess, "run", side_effect=unsupported):
                result, _ = suite.run_metric(metric("vector_fp8"), args, root)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(result["reason"], "no native FP8")

            def missing_path(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"status": "not_implemented", "reason": "no verified path"}))
                return subprocess.CompletedProcess(command, 1)

            with mock.patch.object(suite.subprocess, "run", side_effect=missing_path):
                result, _ = suite.run_metric(metric("vector_fp8"), args, root)
            self.assertEqual(result["status"], "not_implemented")
            self.assertEqual(result["reason"], "no verified path")

    def test_successful_quick_child_is_labeled_smoke_and_bad_json_is_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)

            def child(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"results": [{"peak_tflops": 20, "tflops": 19}]}))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(suite.subprocess, "run", side_effect=child):
                result, _ = suite.run_metric(metric("tensor_fp32"), args, root)
            self.assertTrue(result["smoke_only"])

            def malformed(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"device": ["not", "metadata"]}))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(suite.subprocess, "run", side_effect=malformed):
                result, _ = suite.run_metric(metric("memory_capacity"), args, root)
            self.assertEqual(result["status"], "error")
            self.assertIn("Invalid benchmark JSON", result["reason"])

    def test_suite_continues_after_errors_and_checkpoints_every_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            observed = []

            def child(item, args, directory):
                snapshot = json.loads(args.output.read_text())
                self.assertEqual(len(snapshot["results"]), 15)
                self.assertEqual(sum(row["status"] != "pending" for row in snapshot["results"]),
                                 len(observed))
                observed.append(item.key)
                status = "error" if len(observed) == 1 else "unsupported"
                return empty_metric(item, status, "test outcome"), None

            with mock.patch.object(suite, "run_metric", side_effect=child), redirect_stdout(io.StringIO()):
                report = suite.run_suite(args)
            self.assertEqual(len(observed), 15)
            self.assertEqual(report["status"], "complete_with_errors")
            self.assertNotIn("pending", {row["status"] for row in report["results"]})
            self.assertEqual(json.loads(args.output.read_text()), report)


if __name__ == "__main__":
    unittest.main()
