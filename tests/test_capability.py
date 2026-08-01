"""Tests for SubAgentCapability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.test import TestModel

from subagents_pydantic_ai import (
    DynamicAgentRegistry,
    SubAgentCapability,
    SubAgentConfig,
    create_subagent_toolset,
)

_CAPABILITY_OWNED_ARGUMENTS = frozenset({"id"})
"""Toolset arguments the capability sets itself rather than exposing.

`id` is fixed to `"subagents"` so the capability's toolset is addressable by a
stable name.
"""

_MODEL = TestModel()


@dataclass
class MockDeps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> MockDeps:
        return MockDeps(subagents={} if max_depth <= 0 else self.subagents.copy())


def _cap(**kwargs):
    """Create SubAgentCapability with TestModel default."""
    kwargs.setdefault("default_model", _MODEL)
    return SubAgentCapability(**kwargs)


class TestSubAgentCapability:
    """Tests for SubAgentCapability construction and configuration."""

    def test_default_creates_toolset(self):
        """Default capability creates toolset with general-purpose subagent."""
        cap = _cap()
        assert cap.include_general_purpose is True
        assert cap.get_toolset() is not None

    def test_custom_subagents(self):
        """Custom subagent configs are accepted."""
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
            ),
        ]
        cap = _cap(subagents=configs)
        assert cap.subagents == configs
        assert cap.get_toolset() is not None

    def test_without_general_purpose(self):
        """Can disable general-purpose subagent."""
        configs = [
            SubAgentConfig(
                name="writer",
                description="Writes content",
                instructions="You are a writer.",
            ),
        ]
        cap = _cap(subagents=configs, include_general_purpose=False)
        assert cap.include_general_purpose is False

    def test_serialization_name(self):
        """Serialization name for AgentSpec."""
        assert SubAgentCapability.get_serialization_name() == "SubAgentCapability"

    def test_get_instructions_returns_callable(self):
        """get_instructions returns a callable."""
        configs = [
            SubAgentConfig(
                name="helper",
                description="Helps with tasks",
                instructions="You help.",
            ),
        ]
        cap = _cap(subagents=configs)
        instructions = cap.get_instructions()
        assert callable(instructions)

    def test_instructions_contain_subagent_names(self):
        """Dynamic instructions list available subagents."""
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="Research.",
            ),
        ]
        cap = _cap(subagents=configs)
        instructions_fn = cap.get_instructions()
        ctx = type("FakeCtx", (), {"deps": None})()
        result = instructions_fn(ctx)
        assert "researcher" in result

    def test_task_manager_property(self):
        """task_manager property is accessible."""
        cap = _cap()
        assert hasattr(cap, "task_manager")

    def test_nesting_depth(self):
        """Max nesting depth is forwarded."""
        cap = _cap(max_nesting_depth=2)
        assert cap.max_nesting_depth == 2

    def test_usage_limits_forwarded_to_toolset(self):
        """usage_limits is forwarded to the underlying toolset factory."""
        usage_limits = UsageLimits(request_limit=3, total_tokens_limit=1000)

        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap(usage_limits=usage_limits)

        assert cap.usage_limits is usage_limits
        assert create_toolset.call_args.kwargs["usage_limits"] is usage_limits

    def test_delegation_configuration_forwarded_to_toolset(self):
        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap(
                delegation_configuration="persisted_and_oneshot",
                allowed_models=["openai:gpt-4.1"],
            )

        assert cap.delegation_configuration == "persisted_and_oneshot"
        assert (
            create_toolset.call_args.kwargs["delegation_configuration"] == "persisted_and_oneshot"
        )
        assert create_toolset.call_args.kwargs["allowed_models"] == ["openai:gpt-4.1"]

    def test_delegate_tool_present_in_combined_mode(self):
        cap = _cap(
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        toolset = cap.get_toolset()
        assert toolset is not None
        assert "delegate" in toolset.tools

    def test_oneshot_only_instructions_reference_delegate(self):
        cap = _cap(
            delegation_configuration="oneshot_only",
            include_general_purpose=False,
        )
        instructions_fn = cap.get_instructions()
        ctx = type("FakeCtx", (), {"deps": None})()
        result = instructions_fn(ctx)
        assert "delegate" in result
        assert "`task`" not in result

    def test_oneshot_only_rejects_configured_subagents(self):
        with pytest.raises(ValueError, match="cannot be combined with"):
            _cap(
                delegation_configuration="oneshot_only",
                subagents=[
                    SubAgentConfig(
                        name="researcher",
                        description="Researches topics",
                        instructions="Research.",
                    )
                ],
            )

    def test_oneshot_only_rejects_registry(self):
        with pytest.raises(ValueError, match="cannot be combined with a registry"):
            _cap(
                delegation_configuration="oneshot_only",
                registry=DynamicAgentRegistry(),
            )

    def test_default_mode_rejects_dynamic_agent_config(self):
        """The capability is the entry point most users configure, so it must reject too."""
        with pytest.raises(ValueError, match="exposes no dynamic-agent tool"):
            _cap(allowed_models=["openai:gpt-4.1"])

    def test_max_result_chars_defaults_to_2000(self):
        """The result preview budget defaults to 2000 characters."""
        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap()

        assert cap.max_result_chars == 2000
        assert create_toolset.call_args.kwargs["max_result_chars"] == 2000

    def test_max_result_chars_forwarded_to_toolset(self):
        """max_result_chars is forwarded to the underlying toolset factory."""
        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap(max_result_chars=None)

        assert cap.max_result_chars is None
        assert create_toolset.call_args.kwargs["max_result_chars"] is None


class TestSubAgentCapabilityIntegration:
    """Integration tests with real Agent."""

    @pytest.mark.anyio
    async def test_agent_with_capability(self):
        """Agent with SubAgentCapability can run successfully."""
        cap = _cap()
        agent = Agent(_MODEL, deps_type=MockDeps, capabilities=[cap])
        result = await agent.run("Delegate a task", deps=MockDeps())
        assert result.output is not None

    @pytest.mark.anyio
    async def test_agent_with_custom_subagents(self):
        """Agent with custom subagents runs successfully."""
        configs = [
            SubAgentConfig(
                name="analyst",
                description="Analyzes data",
                instructions="You analyze data.",
            ),
        ]
        cap = _cap(subagents=configs)
        agent = Agent(_MODEL, deps_type=MockDeps, capabilities=[cap])
        result = await agent.run("Analyze something", deps=MockDeps())
        assert result.output is not None

    @pytest.mark.anyio
    async def test_toolset_has_expected_tools(self):
        """Toolset has core subagent management tools."""
        cap = _cap()
        toolset = cap.get_toolset()
        assert toolset is not None
        tool_names = set(toolset.tools.keys())
        assert "task" in tool_names
        assert "check_task" in tool_names
        assert "list_active_tasks" in tool_names


class TestCapabilityTracksToolset:
    """`SubAgentCapability` must forward every `create_subagent_toolset` argument.

    It used to omit three. `ask_user` was the one that bit: it is the only channel
    a sync-mode subagent has for `ask_parent`, and sync is the default mode, so the
    documented "a subagent can ask its parent" feature was off by construction on
    the primary entry point -- and the error text told the user to pass an argument
    the capability did not accept.
    """

    def test_every_toolset_argument_is_reachable(self):
        import inspect

        toolset_args = set(inspect.signature(create_subagent_toolset).parameters)
        capability_fields = {
            name for name in SubAgentCapability.__dataclass_fields__ if not name.startswith("_")
        }

        assert toolset_args - capability_fields - _CAPABILITY_OWNED_ARGUMENTS == set(), (
            "create_subagent_toolset gained an argument SubAgentCapability cannot "
            "reach. Add a field and forward it in __post_init__, or list it in "
            "_CAPABILITY_OWNED_ARGUMENTS if the capability must own its value."
        )

    def test_ask_user_is_forwarded(self):
        """A sync-mode subagent's only route to its parent has to survive the hop."""

        async def ask(question: str) -> str:
            return "42"

        cap = _cap(ask_user=ask)

        assert cap.get_toolset()._ask_user is ask

    def test_memory_bounds_are_forwarded(self):
        cap = _cap(max_chat_traces=7, max_task_handles=9)
        toolset = cap.get_toolset()

        assert toolset._chat_traces._max_traces == 7
        assert toolset._max_task_handles == 9
