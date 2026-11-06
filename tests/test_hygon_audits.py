"""CPU regression checks for the Hygon DCU (AMDGCN-dialect) instruction audits.

Hygon is a separate FlagGems vendor that compiles through the AMDGCN path and
publishes final assembly under the same "amdgcn" key as AMD, so both the tensor
and vector audits treat it as part of that family. It spells its matrix
instruction v_mmac_* where AMD spells it v_mfma_*/v_wmma_*. These tests pin the
mnemonics observed on BW1101 and, just as importantly, pin that the AMD
spellings still pass so adding Hygon did not loosen the AMD audit.
"""

import unittest
from unittest import mock

import bench_tensor as tensor
import bench_vector as vector
from test_tensor_precision import compiled


# Mnemonics captured from FlagTree 0.7.0+hcu3.6 on BW1101 (gfx938), one per
# dtype the suite requests. Hygon omits the accumulator segment for its FP32
# matrix instruction, which is why v_mmac_16x16x8_f32 has no f32 accumulator
# field while the others do.
HYGON_MATRIX = {
    "fp16": "v_mmac_f32_16x16x16_f16",
    "bf16": "v_mmac_f32_16x16x16_bf16",
    "fp32": "v_mmac_16x16x8_f32",
    "int8": "v_mmac_i32_16x16x32_i8",
    "fp8": "v_mmac_f32_16x16x32_fp8_fp8",
}


def gcn(*instructions):
    """An AMDGCN listing with each instruction on its own indented line."""
    return "\n".join(f"\t{text} v[0:3], v[4:5], v[6:7], v[0:3]"
                     for text in instructions)


class HygonVendorFamilyTests(unittest.TestCase):
    def test_hygon_is_in_the_amdgcn_audit_family(self):
        self.assertIn("hygon", tensor._AMDGCN_VENDORS)
        self.assertIn("amd", tensor._AMDGCN_VENDORS)
        self.assertIn("hygon", vector.VECTOR_AMDGCN_VENDORS)
        self.assertIn("amd", vector.VECTOR_AMDGCN_VENDORS)

    def test_hygon_shares_the_amd_peak_tile_set(self):
        # A BW1101 sweep put the best shape inside the AMD set within 0.1% of
        # the best shape overall, so Hygon must not fall through to the generic
        # branch, whose tile list is tuned for whole-tile matrix units.
        for dtype in ("fp16", "bf16", "int8", "fp8"):
            with self.subTest(dtype=dtype):
                self.assertEqual(tensor._peak_configs("hygon", dtype),
                                 tensor._peak_configs("amd", dtype))

    def test_peak_configs_runs_without_module_globals(self):
        # tests/test_int8_peak_unroll.py extracts _peak_configs by AST and
        # execs it in a bare namespace, so the vendor branch must not reference
        # a module-level constant.
        import ast
        from pathlib import Path

        source = Path(tensor.__file__)
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_peak_configs")
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__",
                                 names=[ast.alias(name="annotations")], level=0),
                  function],
            type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        self.assertEqual(namespace["_peak_configs"]("hygon", "fp16"),
                         tensor._peak_configs("hygon", "fp16"))


class HygonMatrixFormatTests(unittest.TestCase):
    def test_hygon_mnemonics_match_their_requested_format(self):
        cases = [("fp16", "f32", "f16"), ("bf16", "f32", "bf16"),
                 ("fp32", "f32", "f32"), ("int8", "i32", "i8"),
                 ("fp8", "f32", "fp8")]
        for dtype, accumulator, operand in cases:
            with self.subTest(dtype=dtype):
                self.assertTrue(tensor._amd_matrix_format(
                    gcn(HYGON_MATRIX[dtype]), accumulator, operand))

    def test_amd_spellings_still_match(self):
        # Adding the Hygon mnemonic must not stop recognizing either AMD form.
        cases = [("v_mfma_f32_16x16x16f16", "f32", "f16"),       # legacy K+type
                 ("v_mfma_i32_16x16x32_i8", "i32", "i8"),        # normalized K_type
                 ("v_mfma_f64_16x16x4_f64", "f64", "f64"),
                 ("v_wmma_f32_16x16x16_bf16", "f32", "bf16")]
        for instruction, accumulator, operand in cases:
            with self.subTest(instruction=instruction):
                self.assertTrue(tensor._amd_matrix_format(
                    gcn(instruction), accumulator, operand))

    def test_a_mismatched_accumulator_or_operand_is_rejected(self):
        # The audit exists to reject a kernel the compiler lowered to another
        # dtype, so a wrong accumulator or operand must not match.
        self.assertFalse(tensor._amd_matrix_format(
            gcn(HYGON_MATRIX["fp16"]), "f64", "f64"))
        self.assertFalse(tensor._amd_matrix_format(
            gcn(HYGON_MATRIX["bf16"]), "f32", "f16"))
        self.assertFalse(tensor._amd_matrix_format(
            gcn(HYGON_MATRIX["int8"]), "f32", "f32"))

    def test_the_omitted_accumulator_segment_needs_matching_types(self):
        # v_mmac_16x16x8_f32 names only one type. Accepting it for f32/f32 must
        # not also accept it for a request with a different accumulator width,
        # which would let an FP32 instruction pass as an FP64 peak.
        self.assertTrue(tensor._amd_matrix_format(
            gcn("v_mmac_16x16x8_f32"), "f32", "f32"))
        self.assertFalse(tensor._amd_matrix_format(
            gcn("v_mmac_16x16x8_f32"), "f64", "f64"))
        self.assertFalse(tensor._amd_matrix_format(
            gcn("v_mmac_16x16x8_f32"), "i32", "i8"))

    def test_scalar_fma_is_not_a_matrix_instruction(self):
        for instruction in ("v_fma_f32", "v_fma_f64", "v_fmac_f32_e32"):
            with self.subTest(instruction=instruction):
                self.assertFalse(tensor._amd_matrix_format(
                    gcn(instruction), "f32", "f32"))


class HygonTensorAuditTests(unittest.TestCase):
    def info(self, dtype, assembly, **kwargs):
        return tensor._compiled_info(compiled(amdgcn=assembly), "hygon", dtype, **kwargs)

    def test_matrix_dtypes_audit_as_verified_tensor_execution(self):
        for dtype in ("fp16", "bf16", "int8", "fp8"):
            with self.subTest(dtype=dtype):
                result = self.info(dtype, gcn(HYGON_MATRIX[dtype]))
                self.assertEqual(result["assembly_audit"], "verified")
                self.assertEqual(result["execution_kind"], "tensor")
                self.assertTrue(result["native_tensor_verified"])
                self.assertTrue(result["instruction_precision_verified"])
                self.assertEqual(result["tensor_instruction_types"],
                                 [HYGON_MATRIX[dtype]])

    def test_ieee_fp32_matrix_instruction_is_verified(self):
        result = self.info("fp32", gcn(HYGON_MATRIX["fp32"]),
                           dot_input_precision="ieee")
        self.assertEqual(result["assembly_audit"], "verified")
        self.assertEqual(result["execution_kind"], "tensor")
        self.assertTrue(result["native_tensor_verified"])

    def test_fp64_is_verified_as_simt_not_tensor(self):
        # The DCU has no FP64 matrix instruction; FP64 reaches v_fma_f64. That
        # is a valid measurement but must never be labeled tensor execution.
        result = self.info("fp64", gcn("v_fma_f64"))
        self.assertEqual(result["assembly_audit"], "verified")
        self.assertEqual(result["execution_kind"], "simt")
        self.assertFalse(result["native_tensor_verified"])
        self.assertEqual(result["fp64_fma_instruction_types"], ["v_fma_f64"])

    def test_a_dtype_lowered_to_another_format_is_rejected(self):
        # A BF16 request answered by the FP16 matrix instruction must not be
        # reported as a BF16 peak.
        with self.assertRaises(tensor.TensorUnsupportedError):
            self.info("bf16", gcn(HYGON_MATRIX["fp16"]))

    def test_no_matrix_instruction_is_rejected(self):
        with self.assertRaises(tensor.TensorUnsupportedError):
            self.info("fp16", gcn("v_fma_f32"))


class HygonVectorAuditTests(unittest.TestCase):
    def inspect(self, dtype, assembly):
        return vector.inspect_assembly(
            compiled(amdgcn=assembly), "hygon", dtype, backend="hip")

    def test_native_float_fma_is_verified(self):
        for dtype, instruction in (("fp16", "v_fma_f16"), ("fp32", "v_fma_f32"),
                                   ("fp64", "v_fma_f64")):
            with self.subTest(dtype=dtype):
                result = self.inspect(dtype, gcn(instruction))
                self.assertTrue(result["native_arithmetic_verified"])
                self.assertNotIn("amdgcn_arithmetic_promoted", result)
                self.assertIn("passed", result["assembly_check"])

    def test_bf16_promoted_to_fp32_keeps_the_rate_but_is_unverified(self):
        # The DCU has no v_fma_bf16, so the compiler widens the arithmetic while
        # the recurrence state stays BF16. The measurement is kept, named as
        # promoted, and never marked native.
        result = self.inspect("bf16", gcn("v_fma_f32"))
        self.assertFalse(result["native_arithmetic_verified"])
        self.assertTrue(result["amdgcn_arithmetic_promoted"])
        self.assertEqual(result["amdgcn_promoted_instructions"], ["v_fma_f32"])
        self.assertIn("promoted", result["assembly_check"])
        self.assertGreater(result["amdgcn_arithmetic_instruction_sites"], 0)

    def test_int8_without_dp4a_is_retained_as_unverified(self):
        # 16-bit integer ops carry the declared INT8 values in wider registers,
        # so the rate is kept but native INT8 arithmetic is not established.
        result = self.inspect("int8", gcn("v_mul_lo_u16_e32", "v_mad_legacy_u16"))
        self.assertFalse(result["native_arithmetic_verified"])
        self.assertGreater(result["amdgcn_arithmetic_instruction_sites"], 0)
        self.assertIn("unverified", result["assembly_check"])

    def test_matrix_instructions_in_a_vector_kernel_are_an_error(self):
        for instruction in ("v_mmac_f32_16x16x16_f16", "v_mfma_f32_16x16x16f16"):
            with self.subTest(instruction=instruction):
                with self.assertRaises(RuntimeError):
                    self.inspect("fp16", gcn(instruction))

    def test_no_arithmetic_at_all_is_rejected(self):
        with self.assertRaises(vector.UnsupportedVectorDtype):
            self.inspect("fp32", gcn("v_mov_b32"))


class HygonUnsupportedPrecisionTests(unittest.TestCase):
    def arguments(self, *cli):
        parser = tensor.build_parser()
        args = parser.parse_args(cli)
        tensor._normalize_precision_args(parser, args)
        return args

    def test_a_precision_the_backend_does_not_declare_is_unsupported(self):
        # The hcu backend declares only ("ieee",). tl.dot asserts on anything
        # else inside the untimed probe, so the declaration is checked first and
        # the metric is reported unsupported instead of crashing the run.
        args = self.arguments("--dtype", "tf32")
        metadata = {"vendor_name": "hygon", "backend": "hip"}
        with mock.patch.object(
                tensor, "_backend_dot_input_precisions", return_value=("ieee",)):
            reason = tensor._capability_reason(args, metadata)
        self.assertIsNotNone(reason)
        self.assertIn("tf32", reason)
        self.assertIn("not a determination of chip support", reason)

    def test_a_declared_precision_still_reaches_the_probe(self):
        args = self.arguments("--dtype", "fp32")
        metadata = {"vendor_name": "hygon", "backend": "hip"}
        with mock.patch.object(
                tensor, "_backend_dot_input_precisions", return_value=("ieee",)):
            self.assertIsNone(tensor._capability_reason(args, metadata))

    def test_an_undeclared_list_is_not_treated_as_a_rejection(self):
        # A backend that publishes nothing is not evidence either way, so the
        # run must proceed to the probe rather than report unsupported.
        args = self.arguments("--dtype", "tf32")
        metadata = {"vendor_name": "hygon", "backend": "hip"}
        with mock.patch.object(
                tensor, "_backend_dot_input_precisions", return_value=None):
            self.assertIsNone(tensor._capability_reason(args, metadata))

    def test_the_assertion_text_is_classified_as_a_rejection(self):
        # Belt and braces for a backend that does not publish the list: the
        # raised assertion must still classify as unsupported, not as a crash.
        message = "input_precision must be one of ('ieee',). Got tf32"
        self.assertIsNotNone(tensor._unsupported_reason(AssertionError(message)))


if __name__ == "__main__":
    unittest.main()
