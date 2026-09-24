"""Offline SM90 compilation: verifies actual dot instructions without a GPU."""

import os
import tempfile
import unittest
from unittest import mock

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import bench_tensor as tensor


@unittest.skipUnless("nvidia" in triton.backends.backends,
                     "Offline NVIDIA compiler is not installed")
class TensorCompilationTests(unittest.TestCase):
    def compile_kernel(self, mode, dtype, precision=None):
        storage = {"fp16": "fp16", "bf16": "bf16", "fp8": "fp8e4nv", "fp32": "fp32",
                   "int8": "i8", "fp64": "fp64"}[dtype]
        output = "i32" if dtype == "int8" else "fp64" if dtype == "fp64" else "fp32"
        config = (tensor._peak_configs("nvidia", dtype, precision)[0] if mode == "peak"
                  else tensor._configs("nvidia", dtype)[0])
        constants = {"BLOCK_M": config["block_m"], "BLOCK_N": config["block_n"],
                     "BLOCK_K": config["block_k"], "DOT_INPUT_PRECISION": precision}
        if mode == "peak":
            function = tensor._tensor_peak_kernel
            constants.update(BLOCKS=2, ITERATIONS=8)
        else:
            function = tensor._gemm_kernel
            constants.update(M=65, N=67, K=69, GROUP_M=8)
        signature = {"A": "*" + storage, "B": "*" + storage, "C": "*" + output}
        source = ASTSource(function, signature=signature, constexprs=constants)
        return triton.compile(source, target=GPUTarget("cuda", 90, 32),
                              options={"num_warps": config["num_warps"], "num_stages": 1})

    def test_resident_kernels_emit_requested_native_instructions(self):
        expected = {
            "fp16": ".f32.f16.f16",
            "bf16": ".f32.bf16.bf16",
            "fp8": ".f32.e4m3.e4m3",
            "int8": ".s32.s8.s8",
            "fp64": "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64",
        }
        with tempfile.TemporaryDirectory() as cache, mock.patch.dict(os.environ, TRITON_CACHE_DIR=cache):
            for dtype, instruction in expected.items():
                with self.subTest(dtype=dtype):
                    compiled = self.compile_kernel("peak", dtype)
                    info = tensor._compiled_info(compiled, "nvidia", dtype)
                    self.assertTrue(info["native_tensor_verified"])
                    self.assertTrue(all(instruction in opcode for opcode in info["tensor_instruction_types"]))
                    self.assertIn("bra", compiled.asm["ptx"])

    def test_new_gemm_formats_compile_with_masked_tails(self):
        with tempfile.TemporaryDirectory() as cache, mock.patch.dict(os.environ, TRITON_CACHE_DIR=cache):
            for dtype in ("int8", "fp64"):
                with self.subTest(dtype=dtype):
                    compiled = self.compile_kernel("gemm", dtype)
                    info = tensor._compiled_info(compiled, "nvidia", dtype)
                    self.assertTrue(info["instruction_precision_verified"])


if __name__ == "__main__":
    unittest.main()
