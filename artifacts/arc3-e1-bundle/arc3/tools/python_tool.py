"""Ephemeral, read-only Python analysis namespace (no network or file writes)."""
from __future__ import annotations
import ast
from typing import Any

class PythonAnalysisTool:
    def __init__(self, namespace: dict[str,Any]): self.namespace=dict(namespace)
    def run(self, expression: str) -> Any:
        tree=ast.parse(expression,mode="eval")
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom,ast.Call)):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"len","sum","min","max","sorted","set","tuple","list","dict","abs"}: continue
                raise ValueError("only read-only expressions and safe builtins are allowed")
        return eval(compile(tree,"<analysis>","eval"),{"__builtins__":{"len":len,"sum":sum,"min":min,"max":max,"sorted":sorted,"set":set,"tuple":tuple,"list":list,"dict":dict,"abs":abs}},self.namespace)
