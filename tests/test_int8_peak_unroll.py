"""Dependency-free regression checks for the small PPU peak patch."""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "bench_tensor.py"
TREE = ast.parse(SOURCE.read_text())


class Int8PeakUnrollTests(unittest.TestCase):
    def test_config_is_ppu_int8_only(self):
        function = next(n for n in TREE.body
                        if isinstance(n, ast.FunctionDef) and n.name == "_peak_configs")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), function], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        configs = namespace["_peak_configs"]
        for vendor in ("thead", "nvidia", "amd", "other"):
            for dtype in ("int8", "fp16", "bf16", "fp32", "fp64", "fp8"):
                candidates = configs(vendor, dtype)
                tuned = [c for c in candidates if c.get("unroll", 1) != 1]
                self.assertEqual(len(tuned), int((vendor, dtype) == ("thead", "int8")))
                if tuned:
                    self.assertEqual(tuned[0], dict(block_m=128, block_n=16,
                        block_k=128, num_warps=2, num_stages=2, chains=2,
                        unroll=4, b_transposed=True))

    def test_actual_loop_ast_preserves_both_chain_counts(self):
        kernel = next(n for n in TREE.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_tensor_peak_kernel")
        loops = [n for n in kernel.body if isinstance(n, ast.For)]

        class CountDots(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "static_range":
                    return ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=node.args, keywords=[])
                if isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                    return ast.BinOp(left=node.args[2], op=ast.Add(), right=ast.Constant(1))
                return self.generic_visit(node)

        module = CountDots().visit(ast.Module(body=loops, type_ignores=[]))
        code = compile(ast.fix_missing_locations(module), str(SOURCE), "exec")
        for unroll in (1, 4):
            for iterations in (1, 2, 3, 4, 5, 7, 64, 4095, 4096):
                scope = dict(ITERATIONS=iterations, UNROLL=unroll, acc0=0, acc1=0)
                exec(code, scope)
                self.assertEqual((scope["acc0"], scope["acc1"]), (iterations, iterations))


if __name__ == "__main__":
    unittest.main()
