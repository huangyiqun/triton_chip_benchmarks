"""Offline native vector instruction checks, without accelerator execution."""

import os
import re
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import bench_vector as vector


@unittest.skipUnless("nvidia" in triton.backends.backends,
                     "Offline NVIDIA compiler is not installed")
class VectorCompilationTests(unittest.TestCase):
    def test_typed_int8_source_has_i8_arithmetic_without_overflow_assertions(self):
        # NVIDIA compilation supplies the shared Triton frontend in this CPU
        # test. Inspect TTIR only; this is not an XPU hardware/ISA validation.
        with TemporaryDirectory() as cache, mock.patch.dict(os.environ, TRITON_CACHE_DIR=cache):
            source = ASTSource(vector.vector_fma_kernel,
                               signature={name: "*i8" for name in ("Input", "Output", "Coefficients")},
                               constexprs={"ELEMENTS": 128, "ITERATIONS": 32, "BLOCK_SIZE": 128,
                                           "UNROLL": 1, "USE_EXPLICIT_FMA": False,
                                           "IS_INT8": True, "USE_DP4A": False})
            compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                                      options={"num_warps": 4, "sanitize_overflow": True})
            ttir = compiled.asm["ttir"]
            self.assertRegex(ttir, r"arith\.muli[^\n]*tensor<128xi8>")
            self.assertRegex(ttir, r"arith\.addi[^\n]*tensor<128xi8>")
            self.assertNotIn("tt.assert", ttir)
            self.assertNotIn("dp4a", ttir)

    def test_native_float_and_integer_operations_survive_loop_compilation(self):
        with TemporaryDirectory() as cache, mock.patch.dict(os.environ, TRITON_CACHE_DIR=cache):
            for dtype, storage in (("fp16", "fp16"), ("bf16", "bf16"),
                                   ("fp32", "fp32"), ("fp64", "fp64"), ("int8", "i32")):
                with self.subTest(dtype=dtype):
                    integer = dtype == "int8"
                    signature = {name: "*" + storage for name in ("Input", "Output", "Coefficients")}
                    constants = {"ELEMENTS": 128, "ITERATIONS": 32, "BLOCK_SIZE": 128,
                                 "UNROLL": 1, "USE_EXPLICIT_FMA": not integer,
                                 "IS_INT8": integer, "USE_DP4A": integer}
                    source = ASTSource(vector.vector_fma_kernel, signature=signature,
                                       constexprs=constants)
                    compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                                              options={"num_warps": 4})
                    info = vector.inspect_assembly(compiled, "nvidia", dtype)
                    self.assertTrue(info["native_arithmetic_verified"])
                    self.assertFalse(info["ptx_contains_tensor_instructions"])
                    self.assertGreaterEqual(info["ptx_arithmetic_instruction_sites"], vector.CHAINS)
                    ptx = compiled.asm["ptx"]
                    if not re.search(r"\bbra(?:\.uni)?\b", ptx):
                        # LLVM can fully unroll a short arithmetic loop. In
                        # that case all 32*8 updates must remain observable;
                        # packed half/bfloat16 instructions update two lanes.
                        opcodes = re.findall(
                            r"\b(?:fma\.(?:\w+\.)*(?:f16|bf16|f32|f64)(?:x2)?|dp4a\.s32\.s32)\b",
                            ptx)
                        scalar_updates = sum(2 if opcode.endswith("x2") else 1 for opcode in opcodes)
                        self.assertGreaterEqual(scalar_updates, 32 * vector.CHAINS)


if __name__ == "__main__":
    unittest.main()
