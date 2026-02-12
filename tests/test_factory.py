"""Tests for factory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai import DynamicAgentRegistry, create_agent_factory_toolset


@dataclass
class MockDeps:
    """Mock dependencies for testing."""

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> MockDeps:
        return MockDeps(subagents={} if max_depth <= 0 else self.subagents.copy())


@dataclass
class MockRunContext:
    """Mock run context for testing."""

    deps: MockDeps


class TestCreateAgentFactoryToolset:
    """Tests for create_agent_factory_toolset."""

    def test_creates_toolset(self, registry: DynamicAgentRegistry):
        """Test toolset creation."""
        toolset = create_agent_factory_toolset(registry=registry)

        tool_names = list(toolset.tools.keys())
        assert "create_agent" in tool_names
        assert "list_agents" in tool_names
        assert "remove_agent" in tool_names
        assert "get_agent_info" in tool_names

    def test_custom_id(self, registry: DynamicAgentRegistry):
        """Test toolset with custom ID."""
        toolset = create_agent_factory_toolset(
            registry=registry,
            id="custom_factory",
        )
        assert toolset.id == "custom_factory"

    def test_updates_registry_max_agents(self, registry: DynamicAgentRegistry):
        """Test that max_agents is updated in registry."""
        create_agent_factory_toolset(
            registry=registry,
            max_agents=5,
        )
        assert registry.max_agents == 5

    @pytest.mark.asyncio
    async def test_create_agent_success(self, registry: DynamicAgentRegistry):
        """Test successful agent creation."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            result = await create_tool.function(
                ctx,
                name="test-agent",
                description="A test agent",
                instructions="Do testing",
            )

        assert "created successfully" in result
        assert registry.exists("test-agent")

    @pytest.mark.asyncio
    async def test_create_agent_invalid_name(self, registry: DynamicAgentRegistry):
        """Test agent creation with invalid name."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        # Empty name
        result = await create_tool.function(
            ctx,
            name="",
            description="Test",
            instructions="Test",
        )
        assert "Error" in result

        # Name with spaces
        result = await create_tool.function(
            ctx,
            name="test agent",
            description="Test",
            instructions="Test",
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_create_agent_duplicate(self, registry: DynamicAgentRegistry):
        """Test creating duplicate agent fails."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            # Create first agent
            await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
            )

            # Try to create duplicate
            result = await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
            )

        assert "Error" in result
        assert "already exists" in result

    @pytest.mark.asyncio
    async def test_create_agent_disallowed_model(self, registry: DynamicAgentRegistry):
        """Test creating agent with disallowed model."""
        toolset = create_agent_factory_toolset(
            registry=registry,
            allowed_models=["openai:gpt-4", "openai:gpt-3.5-turbo"],
        )
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())
        result = await create_tool.function(
            ctx,
            name="test-agent",
            description="Test",
            instructions="Test",
            model="anthropic:claude-3",
        )

        assert "Error" in result
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_create_agent_with_capabilities(self, registry: DynamicAgentRegistry):
        """Test creating agent with capabilities."""

        def mock_capability_factory(deps: MockDeps) -> list[FunctionToolset[Any]]:
            toolset: FunctionToolset[Any] = FunctionToolset(id="mock_cap")

            @toolset.tool
            async def mock_tool(x: str) -> str:
                return x

            return [toolset]

        toolset = create_agent_factory_toolset(
            registry=registry,
            capabilities_map={
                "filesystem": mock_capability_factory,
            },
        )
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            result = await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
                capabilities=["filesystem"],
            )

        assert "created successfully" in result
        assert "filesystem" in result
        # Verify toolsets were passed to Agent constructor
        call_kwargs = mock_agent_class.call_args
        passed_toolsets = call_kwargs.kwargs.get("toolsets")
        assert passed_toolsets is not None

    @pytest.mark.asyncio
    async def test_create_agent_invalid_capability(self, registry: DynamicAgentRegistry):
        """Test creating agent with invalid capability."""
        toolset = create_agent_factory_toolset(
            registry=registry,
            capabilities_map={
                "filesystem": lambda deps: [],
            },
        )
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())
        result = await create_tool.function(
            ctx,
            name="test-agent",
            description="Test",
            instructions="Test",
            capabilities=["invalid_cap"],
        )

        assert "Error" in result
        assert "Unknown capabilities" in result

    @pytest.mark.asyncio
    async def test_create_agent_with_toolsets_factory(self, registry: DynamicAgentRegistry):
        """Test creating agent with toolsets_factory."""

        def mock_factory(deps: MockDeps) -> list[FunctionToolset[Any]]:
            toolset: FunctionToolset[Any] = FunctionToolset(id="mock")
            return [toolset]

        toolset = create_agent_factory_toolset(
            registry=registry,
            toolsets_factory=mock_factory,
        )
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            result = await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
            )

        assert "created successfully" in result
        # Verify toolsets were passed to Agent constructor
        call_kwargs = mock_agent_class.call_args
        passed_toolsets = call_kwargs.kwargs.get("toolsets")
        assert passed_toolsets is not None

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, registry: DynamicAgentRegistry):
        """Test listing agents when empty."""
        toolset = create_agent_factory_toolset(registry=registry)
        list_tool = toolset.tools["list_agents"]

        ctx = MockRunContext(deps=MockDeps())
        result = await list_tool.function(ctx)

        assert "No dynamically created agents" in result

    @pytest.mark.asyncio
    async def test_list_agents_with_agents(self, registry: DynamicAgentRegistry):
        """Test listing agents when some exist."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]
        list_tool = toolset.tools["list_agents"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            await create_tool.function(
                ctx,
                name="agent-1",
                description="First agent",
                instructions="Do stuff",
            )
            await create_tool.function(
                ctx,
                name="agent-2",
                description="Second agent",
                instructions="Do more stuff",
            )

        result = await list_tool.function(ctx)

        assert "agent-1" in result
        assert "agent-2" in result
        assert "First agent" in result

    @pytest.mark.asyncio
    async def test_remove_agent_success(self, registry: DynamicAgentRegistry):
        """Test removing agent successfully."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]
        remove_tool = toolset.tools["remove_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            # Create agent
            await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
            )

        # Remove agent
        result = await remove_tool.function(ctx, "test-agent")

        assert "removed" in result
        assert not registry.exists("test-agent")

    @pytest.mark.asyncio
    async def test_remove_agent_not_found(self, registry: DynamicAgentRegistry):
        """Test removing non-existent agent."""
        toolset = create_agent_factory_toolset(registry=registry)
        remove_tool = toolset.tools["remove_agent"]

        ctx = MockRunContext(deps=MockDeps())
        result = await remove_tool.function(ctx, "nonexistent")

        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_get_agent_info_success(self, registry: DynamicAgentRegistry):
        """Test getting agent info successfully."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]
        info_tool = toolset.tools["get_agent_info"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            # Create agent
            await create_tool.function(
                ctx,
                name="test-agent",
                description="A test agent for testing",
                instructions="You are a test agent that tests things.",
                can_ask_questions=False,
            )

        # Get info
        result = await info_tool.function(ctx, "test-agent")

        assert "test-agent" in result
        assert "A test agent for testing" in result
        assert "Can ask questions: False" in result
        assert "You are a test agent" in result

    @pytest.mark.asyncio
    async def test_get_agent_info_not_found(self, registry: DynamicAgentRegistry):
        """Test getting info for non-existent agent."""
        toolset = create_agent_factory_toolset(registry=registry)
        info_tool = toolset.tools["get_agent_info"]

        ctx = MockRunContext(deps=MockDeps())
        result = await info_tool.function(ctx, "nonexistent")

        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_get_agent_info_long_instructions(self, registry: DynamicAgentRegistry):
        """Test getting info for agent with long instructions."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]
        info_tool = toolset.tools["get_agent_info"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            # Create agent with long instructions
            long_instructions = "A" * 1000
            await create_tool.function(
                ctx,
                name="verbose-agent",
                description="Verbose agent",
                instructions=long_instructions,
            )

        # Get info
        result = await info_tool.function(ctx, "verbose-agent")

        # Should truncate instructions
        assert "..." in result
        assert len(result) < len(long_instructions) + 200

    @pytest.mark.asyncio
    async def test_create_agent_max_reached(self):
        """Test creating agent when max limit reached."""
        registry = DynamicAgentRegistry(max_agents=1)
        toolset = create_agent_factory_toolset(registry=registry, max_agents=1)
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()

            # Create first agent
            await create_tool.function(
                ctx,
                name="agent-1",
                description="First",
                instructions="First",
            )

            # Try to create second agent
            result = await create_tool.function(
                ctx,
                name="agent-2",
                description="Second",
                instructions="Second",
            )

        assert "Error" in result
        assert "Maximum" in result

    @pytest.mark.asyncio
    async def test_create_agent_value_error(self, registry: DynamicAgentRegistry):
        """Test handling of ValueError during agent creation."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.side_effect = ValueError("Invalid configuration")

            result = await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
            )

        assert "Error" in result
        assert "Invalid configuration" in result

    @pytest.mark.asyncio
    async def test_create_agent_generic_exception(self, registry: DynamicAgentRegistry):
        """Test handling of generic exception during agent creation."""
        toolset = create_agent_factory_toolset(registry=registry)
        create_tool = toolset.tools["create_agent"]

        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.factory.Agent") as mock_agent_class:
            mock_agent_class.side_effect = RuntimeError("Something went wrong")

            result = await create_tool.function(
                ctx,
                name="test-agent",
                description="Test",
                instructions="Test",
            )

        assert "Error creating agent" in result
        assert "Something went wrong" in result
