"""Package-surface tests: packaging contract and public exports.

pydantic is a declared runtime dependency (DEC-001-3); the chat-completions classes and the
ignored-setting warning category are exported (DEC-003-7, DEC-003-5); the package version is 0.2.1,
identical in ``pyproject.toml`` and ``llm.__version__`` (DEC-001-4; feature 006 step 010 DoD-1, DEC-arch-23).

Feature 006 step 001: the tool-observer warning category is a ``UserWarning`` filterable apart from
the ignored-setting category (US-065.AC-9), and ``LLMErrorKind``, ``LLMToolObserverWarning``,
``ToolEvent`` and ``ToolErrorKind`` are exported alongside every earlier name (DEC-006-3).
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
    LLMToolObserverWarning,
    OpenAIAPIClient,
    OpenAIChatCompletionsClient,
)

# The project metadata is the contract under test, so it is read straight from the
# repository root (this file lives in ``tests/``, one level below it).
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Expected version fixed by the spec: feature 006 step 010 DoD-1 (DEC-arch-23, the 0.2.1 release).
_EXPECTED_VERSION = "0.2.1"


def _project_table() -> dict:
    """Return the ``[project]`` table of the repository's ``pyproject.toml``."""
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _requirement_name(requirement: str) -> str:
    """Return the normalised distribution name of a PEP 508 requirement string."""
    name = re.split(r"[\s\[<>=!~;@(]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_pydantic_is_a_runtime_dependency__DoD1():
    """DEC-001-3: the declared runtime dependencies (not only an extra) include pydantic."""
    runtime_deps = _project_table().get("dependencies", [])
    names = {_requirement_name(req) for req in runtime_deps}
    assert "pydantic" in names


def test_pyproject_version_is_0_2_1__s010_DoD1():
    """Feature 006 step 010 DoD-1 (DEC-arch-23): the version declared in ``pyproject.toml`` is 0.2.1."""
    assert _project_table()["version"] == _EXPECTED_VERSION


def test_package_version_is_0_2_1__s010_DoD1():
    """Feature 006 step 010 DoD-1 (DEC-arch-23): ``llm.__version__`` is 0.2.1."""
    assert llm.__version__ == _EXPECTED_VERSION


def test_package_version_matches_pyproject__s010_DoD1():
    """Feature 006 step 010 DoD-1 (DEC-001-4): ``llm.__version__`` equals the version declared in ``pyproject.toml``."""
    assert llm.__version__ == _project_table()["version"]


def test_new_names_are_listed_in_all__DoD1():
    """DEC-003-7, DEC-003-5: both chat-completions classes and the warning category are listed in ``llm.__all__``."""
    for name in ("OpenAIChatCompletionsClient", "LlamaSwapChatCompletionsClient", "LLMSettingIgnoredWarning"):
        assert name in llm.__all__, name
        assert hasattr(llm, name), name


def test_responses_names_keep_their_exports__DoD1():
    """DEC-003-7: ``OpenAIAPIClient`` / ``LlamaSwapAPIClient`` stay exported."""
    assert "OpenAIAPIClient" in llm.__all__
    assert "LlamaSwapAPIClient" in llm.__all__


def test_chat_completions_classes_are_distinct_clients__DoD1():
    """DEC-003-7: each chat-completions class is a client class distinct from its Responses counterpart."""
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


# --- Feature 006, step 001 -------------------------------------------------------------------

# Names exported before feature 006 (step 001 context), which must stay exported.
_PREVIOUS_EXPORTS = (
    "__version__",
    "LLMError",
    "LLMSettingIgnoredWarning",
    "LLMClient",
    "OllamaAPIClient",
    "OpenAIAPIClient",
    "LlamaSwapAPIClient",
    "OpenAIChatCompletionsClient",
    "LlamaSwapChatCompletionsClient",
    "get_llm_client",
    "substitute_placeholders",
    "pydantic_to_openai_tool",
    "LLMTracer",
)

# Names feature 006 step 001 adds to the package surface (DoD-6, DEC-006-3).
_NEW_006_EXPORTS = ("LLMErrorKind", "LLMToolObserverWarning", "ToolEvent", "ToolErrorKind")


def test_tool_observer_warning_is_a_user_warning__DoD5():
    """Feature 006 step 001 DoD-5: ``LLMToolObserverWarning`` is a subclass of ``UserWarning``."""
    assert issubclass(LLMToolObserverWarning, UserWarning)


def test_tool_observer_warning_is_distinct_from_setting_warning__DoD5():
    """Feature 006 step 001 DoD-5: the observer warning and the ignored-setting warning are separate
    categories, neither a subclass of the other."""
    assert LLMToolObserverWarning is not LLMSettingIgnoredWarning
    assert not issubclass(LLMToolObserverWarning, LLMSettingIgnoredWarning)
    assert not issubclass(LLMSettingIgnoredWarning, LLMToolObserverWarning)


def test_tool_observer_warning_is_filtered_on_its_own__DoD5():
    """Feature 006 step 001 DoD-5 (US-065.AC-9): a filter on either category silences only that one,
    and a stricter filter turns only the observer warning into an error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=LLMToolObserverWarning)
        warnings.warn("observer failed", LLMToolObserverWarning, stacklevel=1)
        warnings.warn("setting dropped", LLMSettingIgnoredWarning, stacklevel=1)
    assert [w.category for w in caught] == [LLMSettingIgnoredWarning]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=LLMSettingIgnoredWarning)
        warnings.warn("observer failed", LLMToolObserverWarning, stacklevel=1)
        warnings.warn("setting dropped", LLMSettingIgnoredWarning, stacklevel=1)
    assert [w.category for w in caught] == [LLMToolObserverWarning]

    with warnings.catch_warnings():
        warnings.simplefilter("error", LLMToolObserverWarning)
        with pytest.raises(LLMToolObserverWarning):
            warnings.warn("observer failed", LLMToolObserverWarning, stacklevel=1)


def test_new_006_names_are_importable_and_listed_in_all__DoD6():
    """Feature 006 step 001 DoD-6: the four new names are attributes of ``llm`` and listed in ``llm.__all__``."""
    for name in _NEW_006_EXPORTS:
        assert name in llm.__all__, name
        assert hasattr(llm, name), name


def test_new_006_names_import_from_package_top_level__DoD6():
    """Feature 006 step 001 DoD-6: ``from llm import ...`` works for each of the four new names."""
    from llm import LLMErrorKind, LLMToolObserverWarning, ToolErrorKind, ToolEvent  # noqa: F401


def test_previous_exports_are_kept__DoD6():
    """Feature 006 step 001 DoD-6: every name exported before the feature stays importable and in ``llm.__all__``."""
    for name in _PREVIOUS_EXPORTS:
        assert name in llm.__all__, name
        assert hasattr(llm, name), name
