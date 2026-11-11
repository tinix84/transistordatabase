"""Produce a read-only structural inventory of the legacy sc implementation.

This inventory distinguishes visible definitions/stubs from numerical parity
evidence. Matching names alone do not establish feature equivalence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def inventory(root: Path) -> dict:
    """Record every Python/MATLAB definition and notebook cell summary."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".py", ".m", ".ipynb"} or any(
            p in {".git", "__pycache__"} for p in path.parts
        ):
            continue
        content = path.read_bytes()
        text = content.decode("utf-8-sig", errors="replace")
        record = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "lines": len(text.splitlines()),
        }
        if path.suffix == ".py":
            try:
                tree = ast.parse(text)
                symbols = []

                def visit(
                    node: ast.AST, prefix: str = "", symbols: list[dict] = symbols
                ) -> None:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(
                            child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            name = prefix + child.name
                            if not isinstance(child, ast.ClassDef):
                                body = [
                                    s
                                    for s in child.body
                                    if not (
                                        isinstance(s, ast.Expr)
                                        and isinstance(s.value, ast.Constant)
                                        and isinstance(s.value.value, str)
                                    )
                                ]
                                stub = not body or all(
                                    isinstance(s, ast.Pass)
                                    or (
                                        isinstance(s, ast.Raise)
                                        and "NotImplemented" in ast.unparse(s)
                                    )
                                    for s in body
                                )
                                symbols.append(
                                    {
                                        "symbol": name,
                                        "line": child.lineno,
                                        "end_line": child.end_lineno,
                                        "stub": stub,
                                    }
                                )
                            visit(child, name + ".")
                        else:
                            visit(child, prefix)

                visit(tree)
                record["definitions"] = symbols
            except SyntaxError as exc:
                record["parse_error"] = f"{exc.msg}, line {exc.lineno}"
        elif path.suffix == ".m":
            record["definitions"] = [
                {"line": i, "signature": line.strip()}
                for i, line in enumerate(text.splitlines(), 1)
                if re.match(r"\s*function\b", line)
            ]
        else:
            nb = json.loads(text)
            record["cells"] = [
                {
                    "index": i,
                    "type": c["cell_type"],
                    "headings": [
                        line
                        for line in "".join(c.get("source", [])).splitlines()
                        if line.startswith("#")
                    ],
                    "output_count": len(c.get("outputs", [])),
                }
                for i, c in enumerate(nb.get("cells", []))
            ]
        files.append(record)
    return {
        "archive": str(root),
        "method": "Static inventory only; executable bodies and parity need numerical verification.",
        "files": files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inventory(args.archive)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(result["files"]),
                "definitions": sum(
                    len(f.get("definitions", [])) for f in result["files"]
                ),
                "stubs": sum(
                    d.get("stub", False)
                    for f in result["files"]
                    for d in f.get("definitions", [])
                ),
                "parse_errors": [
                    f["path"] for f in result["files"] if "parse_error" in f
                ],
            }
        )
    )
