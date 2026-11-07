"""Package-surface tests: packaging contract and public exports.

Feature 001.core-fixes, step 001: pydantic is a declared runtime dependency (DoD-1 of that step).
Feature 003.responses-chat, step 001: the chat-completions classes and the ignored-setting warning
category are exported (DoD-1, DoD-2) and the package version is 0.3.0, identical in
``pyproject.toml`` and ``llm.__version__`` (DoD-10).
"""

import re
import tomllib
import warnings
from pathlib import Path

import pytest

import llm
from llm import (
    LlamaSwapAPIClient,
    LlamaSwapChatCompletionsClient,
    LLMClient,
    LLMSettingIgnoredWarning,
    OpenAIAPIClient,
    OpenAIChatCompletionsClient,
)

# The project metadata is the contract under test, so it is read straight from the
# repository root (this file lives in ``tests/``, one level below it).
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Expected version fixed by the spec (DEC-003-14).
_EXPECTED_VERSION = "0.3.0"


def _project_table() -> dict:
    """Return the ``[project]`` table of the repository's ``pyproject.toml``."""
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _requirement_name(requirement: str) -> str:
    """Return the normalised distribution name of a PEP 508 requirement string."""
    name = re.split(r"[\s\[<>=!~;@(]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_pydantic_is_a_runtime_dependency__DoD1():
    """001.core-fixes DoD-1: the declared runtime dependencies (not only an extra) include pydantic."""
    runtime_deps = _project_table().get("dependencies", [])
    names = {_requirement_name(req) for req in runtime_deps}
    assert "pydantic" in names


def test_pyproject_version_is_0_3_0__DoD10():
    """DoD-10: the version declared in ``pyproject.toml`` is 0.3.0."""
    assert _project_table()["version"] == _EXPECTED_VERSION


def test_package_version_is_0_3_0__DoD10():
    """DoD-10: ``llm.__version__`` is 0.3.0."""
    assert llm.__version__ == _EXPECTED_VERSION


def test_package_version_matches_pyproject__DoD10():
    """DoD-10: ``llm.__version__`` equals the version declared in ``pyproject.toml``."""
    assert llm.__version__ == _project_table()["version"]


def test_new_names_are_listed_in_all__DoD1():
    """DoD-1: both chat-completions classes and the warning category are listed in ``llm.__all__``."""
    for name in ("OpenAIChatCompletionsClient", "LlamaSwapChatCompletionsClient", "LLMSettingIgnoredWarning"):
        assert name in llm.__all__, name
        assert hasattr(llm, name), name


def test_responses_names_keep_their_exports__DoD1():
    """DoD-1 (Interface intent): ``OpenAIAPIClient`` / ``LlamaSwapAPIClient`` stay exported."""
    assert "OpenAIAPIClient" in llm.__all__
    assert "LlamaSwapAPIClient" in llm.__all__


def test_chat_completions_classes_are_distinct_clients__DoD1():
    """DoD-1: each chat-completions class is a client class distinct from its Responses counterpart."""
    assert isinstance(OpenAIChatCompletionsClient, type)
    assert isinstance(LlamaSwapChatCompletionsClient, type)
    assert issubclass(OpenAIChatCompletionsClient, LLMClient)
    assert issubclass(LlamaSwapChatCompletionsClient, LLMClient)
    assert OpenAIChatCompletionsClient is not OpenAIAPIClient
    assert LlamaSwapChatCompletionsClient is not LlamaSwapAPIClient
    assert OpenAIChatCompletionsClient is not LlamaSwapChatCompletionsClient


def test_no_subclassing_between_protocol_classes__DoD1():
    """DoD-1: neither class of a backend subclasses the other (DEC-arch-2, independent copies)."""
    assert not issubclass(OpenAIChatCompletionsClient, OpenAIAPIClient)
    assert not issubclass(OpenAIAPIClient, OpenAIChatCompletionsClient)
    assert not issubclass(LlamaSwapChatCompletionsClient, LlamaSwapAPIClient)
    assert not issubclass(LlamaSwapAPIClient, LlamaSwapChatCompletionsClient)


def test_warning_category_is_a_user_warning__DoD2():
    """DoD-2: ``LLMSettingIgnoredWarning`` is a subclass of ``UserWarning``."""
    assert issubclass(LLMSettingIgnoredWarning, UserWarning)


def test_warning_category_can_be_filtered_by_category__DoD2():
    """DoD-2: a filter on the category alone silences it, and a stricter one turns it into an error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=LLMSettingIgnoredWarning)
        warnings.warn("setting dropped", LLMSettingIgnoredWarning, stacklevel=1)
        warnings.warn("unrelated", UserWarning, stacklevel=1)
    assert [w.category for w in caught] == [UserWarning]

    with warnings.catch_warnings():
        warnings.simplefilter("error", LLMSettingIgnoredWarning)
        with pytest.raises(LLMSettingIgnoredWarning):
            warnings.warn("setting dropped", LLMSettingIgnoredWarning, stacklevel=1)
