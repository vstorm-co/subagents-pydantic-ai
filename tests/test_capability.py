"""Tests for SubAgentCapability."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

_MODEL = TestModel()


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


class TestSubAgentCapabilityIntegration:
    """Integration tests with real Agent."""

    @pytest.mark.anyio
    async def test_agent_with_capability(self):
        """Agent with SubAgentCapability can run successfully."""
        cap = _cap()
        agent = Agent(_MODEL, capabilities=[cap])
        result = await agent.run("Delegate a task")
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
        agent = Agent(_MODEL, capabilities=[cap])
        result = await agent.run("Analyze something")
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
