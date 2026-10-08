"""Compare pre-refactor algorithm definitions, ignoring imports and docstrings only."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

class Normalize(ast.NodeTransformer):
    def visit_Import(self, node):
        return None
    def visit_ImportFrom(self, node):
        return None
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def canonical(node):
    if isinstance(node, ast.AST):
        return [type(node).__name__, {key: canonical(value) for key, value in ast.iter_fields(node) if key != 'type_params'}]
    if isinstance(node, list):
        return [canonical(value) for value in node]
    return node


def fingerprint(node):
    return hashlib.sha256(json.dumps(canonical(Normalize().visit(node)), sort_keys=True, default=repr).encode()).hexdigest()


class PreservationTests(unittest.TestCase):
    def test_original_algorithm_definitions(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / 'tests/algorithm_fingerprints.json').read_text())
        for path, expected in baseline.items():
            nodes = {n.name: n for n in ast.parse((root/path).read_text(encoding='utf-8')).body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name, digest in expected.items():
                with self.subTest(path=path, definition=name):
                    actual = fingerprint(nodes[name])
                    self.assertEqual(digest, actual)


if __name__ == '__main__':
    unittest.main()
