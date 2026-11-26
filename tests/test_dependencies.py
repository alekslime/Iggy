"""
Guard against undeclared dependencies.

Every third-party package that seacode/ imports must be listed in
pyproject.toml's dependencies. (`rich` was once imported directly but
only installed by accident, as a dependency of textual.)

Run with:  python3 -m pytest tests/test_dependencies.py -v
       or: python3 -m unittest tests.test_dependencies -v
"""

import ast
import re
import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Import name -> distribution name, where they differ. Empty today:
# ollama, httpx, textual and rich all match their PyPI names.
IMPORT_TO_DIST = {}


def declared_dependencies() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    return {
        re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0].lower().replace("_", "-")
        for spec in data["project"]["dependencies"]
    }


def third_party_imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    for path in sorted((ROOT / "seacode").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                modules = [node.module]
            else:
                continue

            for module in modules:
                top = module.split(".")[0]

                if top in sys.stdlib_module_names or top == "seacode":
                    continue

                found.setdefault(top, set()).add(path.name)

    return found


class DependencyDeclarationTests(unittest.TestCase):

    def test_every_third_party_import_is_declared(self):
        declared = declared_dependencies()
        missing = {
            module: sorted(files)
            for module, files in third_party_imports().items()
            if IMPORT_TO_DIST.get(module, module).lower() not in declared
        }

        self.assertEqual(
            missing, {},
            f"imported but not declared in pyproject.toml: {missing}",
        )

    def test_the_known_runtime_dependencies_are_detected(self):
        # Sanity check that the scan itself works.
        self.assertTrue(
            {"ollama", "httpx", "textual", "rich"}
            <= set(third_party_imports())
        )


if __name__ == "__main__":
    unittest.main()
