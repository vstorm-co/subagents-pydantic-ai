"""Tests for dynamic agent helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai.dynamic_agent import (
    build_dynamic_agent,
    validate_agent_name,
    validate_capabilities,
    validate_capabilities_with_factory,
    validate_model,
)


@dataclass
class MockDeps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> MockDeps:
        return MockDeps(subagents={} if max_depth <= 0 else self.subagents.copy())


@dataclass
class MockRunContext:
    deps: MockDeps


class TestDynamicAgentValidation:
    def test_validate_agent_name(self):
        assert validate_agent_name("valid-agent") is None
        assert validate_agent_name("") is not None
        assert validate_agent_name("bad name") is not None

    @pytest.mark.parametrize("name", ["агент", "１２３", "café", "agent⁵"])
    def test_validate_agent_name_is_ascii_only(self, name: str):
        """The rule the `create_agent` tool descriptions state, actually enforced.

        `str.isalnum` is Unicode-aware, so Cyrillic and fullwidth-digit names sailed
        through an allow-list the model was told read "letters, numbers, and
        hyphens", while `café` was rejected only over a combining accent.
        """
        assert validate_agent_name(name) is not None

    def test_validate_model(self):
        assert validate_model("openai:gpt-4", ["openai:gpt-4"]) is None
        assert validate_model("anthropic:claude", ["openai:gpt-4"]) is not None

    def test_validate_capabilities(self):
        caps_map: dict[str, Any] = {"filesystem": lambda deps: []}
        assert validate_capabilities(["filesystem"], caps_map) is None
        assert validate_capabilities(["missing"], caps_map) is not None

    def test_validate_capabilities_with_factory(self):
        assert validate_capabilities_with_factory(["filesystem"], MagicMock()) is not None
        assert validate_capabilities_with_factory(None, MagicMock()) is None


class TestBuildDynamicAgent:
    @pytest.mark.asyncio
    async def test_build_success(self):
        ctx = MockRunContext(deps=MockDeps())
        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            result = build_dynamic_agent(
                ctx,
                name="test-agent",
                description="Test agent",
                instructions="Do testing",
                model="openai:gpt-4.1",
            )

        assert not isinstance(result, str)
        agent, config = result
        assert agent is not None
        assert config["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_build_disallowed_model(self):
        ctx = MockRunContext(deps=MockDeps())
        result = build_dynamic_agent(
            ctx,
            name="test-agent",
            description="Test agent",
            instructions="Do testing",
            model="anthropic:claude",
            allowed_models=["openai:gpt-4.1"],
        )

        assert isinstance(result, str)
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_build_with_capabilities(self):
        ctx = MockRunContext(deps=MockDeps())

        def mock_capability_factory(deps: MockDeps) -> list[FunctionToolset[Any]]:
            toolset: FunctionToolset[Any] = FunctionToolset(id="mock_cap")
            return [toolset]

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            result = build_dynamic_agent(
                ctx,
                name="test-agent",
                description="Test agent",
                instructions="Do testing",
                model="openai:gpt-4.1",
                capabilities=["filesystem"],
                capabilities_map={"filesystem": mock_capability_factory},
            )

        assert not isinstance(result, str)
        agent, _config = result
        assert agent is not None
        mock_agent_class.assert_called_once()
        assert mock_agent_class.call_args.kwargs.get("toolsets") is not None

    @pytest.mark.asyncio
    async def test_build_rejects_capabilities_with_custom_factory(self):
        ctx = MockRunContext(deps=MockDeps())
        factory = MagicMock(return_value=MagicMock())

        result = build_dynamic_agent(
            ctx,
            name="test-agent",
            description="Test agent",
            instructions="Do testing",
            model="openai:gpt-4.1",
            capabilities=["filesystem"],
            capabilities_map={"filesystem": lambda deps: []},
            default_agent_factory=factory,
        )

        assert isinstance(result, str)
        assert "not supported" in result
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_value_error(self):
        ctx = MockRunContext(deps=MockDeps())
        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.side_effect = ValueError("Invalid configuration")
            result = build_dynamic_agent(
                ctx,
                name="test-agent",
                description="Test agent",
                instructions="Do testing",
                model="openai:gpt-4.1",
            )

        assert isinstance(result, str)
        assert "Invalid configuration" in result

    @pytest.mark.asyncio
    async def test_build_generic_exception(self):
        ctx = MockRunContext(deps=MockDeps())
        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.side_effect = RuntimeError("Something went wrong")
            result = build_dynamic_agent(
                ctx,
                name="test-agent",
                description="Test agent",
                instructions="Do testing",
                model="openai:gpt-4.1",
            )

        assert isinstance(result, str)
        assert "Error creating agent" in result
