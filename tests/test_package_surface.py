"""Package-surface tests for feature 001.core-fixes, step 001 (package surface).

Covers the install/packaging contract: pydantic is a declared runtime dependency (DoD-1)
and the package version is 0.2.0, identical in ``pyproject.toml`` and ``llm.__version__`` (DoD-3).
"""

import re
import tomllib
from pathlib import Path

import llm

# The project metadata is the contract under test, so it is read straight from the
# repository root (this file lives in ``tests/``, one level below it).
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Expected version fixed by the spec (DEC-001-4).
_EXPECTED_VERSION = "0.2.0"


def _project_table() -> dict:
    """Return the ``[project]`` table of the repository's ``pyproject.toml``."""
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _requirement_name(requirement: str) -> str:
    """Return the normalised distribution name of a PEP 508 requirement string."""
    name = re.split(r"[\s\[<>=!~;@(]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_pydantic_is_a_runtime_dependency__DoD1():
    """DoD-1: the declared runtime dependencies (not only an extra) include pydantic."""
    runtime_deps = _project_table().get("dependencies", [])
    names = {_requirement_name(req) for req in runtime_deps}
    assert "pydantic" in names


def test_pyproject_version_is_0_2_0__DoD3():
    """DoD-3: the version declared in ``pyproject.toml`` is 0.2.0."""
    assert _project_table()["version"] == _EXPECTED_VERSION


def test_package_version_is_0_2_0__DoD3():
    """DoD-3: ``llm.__version__`` is 0.2.0."""
    assert llm.__version__ == _EXPECTED_VERSION


def test_package_version_matches_pyproject__DoD3():
    """DoD-3: ``llm.__version__`` equals the version declared in ``pyproject.toml``."""
    assert llm.__version__ == _project_table()["version"]
