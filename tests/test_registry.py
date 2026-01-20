"""Tests for registry module."""

from __future__ import annotations

import pytest

from subagents_pydantic_ai import DynamicAgentRegistry, SubAgentConfig


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str):
        self.name = name


class TestDynamicAgentRegistry:
    """Tests for DynamicAgentRegistry."""

    def test_register_agent(self, registry: DynamicAgentRegistry):
        """Test registering an agent."""
        config = SubAgentConfig(
            name="test-agent",
            description="A test agent",
            instructions="Do tests",
        )
        agent = MockAgent("test-agent")

        registry.register(config, agent)

        assert registry.exists("test-agent")
        assert registry.get("test-agent") == agent
        assert registry.get_config("test-agent") == config

    def test_register_duplicate_raises(self, registry: DynamicAgentRegistry):
        """Test that registering duplicate agent raises error."""
        config = SubAgentConfig(
            name="test-agent",
            description="A test agent",
            instructions="Do tests",
        )
        agent = MockAgent("test-agent")

        registry.register(config, agent)

        with pytest.raises(ValueError, match="already exists"):
            registry.register(config, agent)

    def test_register_max_agents_reached(self):
        """Test that max_agents limit is enforced."""
        registry = DynamicAgentRegistry(max_agents=2)

        for i in range(2):
            config = SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            registry.register(config, MockAgent(f"agent-{i}"))

        config = SubAgentConfig(
            name="agent-3",
            description="Agent 3",
            instructions="Do stuff",
        )
        with pytest.raises(ValueError, match="Maximum number of agents"):
            registry.register(config, MockAgent("agent-3"))

    def test_get_nonexistent(self, registry: DynamicAgentRegistry):
        """Test getting non-existent agent returns None."""
        assert registry.get("nonexistent") is None

    def test_get_config_nonexistent(self, registry: DynamicAgentRegistry):
        """Test getting config for non-existent agent returns None."""
        assert registry.get_config("nonexistent") is None

    def test_get_compiled(self, registry: DynamicAgentRegistry):
        """Test getting compiled agent."""
        config = SubAgentConfig(
            name="test-agent",
            description="A test agent",
            instructions="Do tests",
        )
        agent = MockAgent("test-agent")

        registry.register(config, agent)
        compiled = registry.get_compiled("test-agent")

        assert compiled is not None
        assert compiled.name == "test-agent"
        assert compiled.description == "A test agent"
        assert compiled.agent == agent
        assert compiled.config == config

    def test_get_compiled_nonexistent(self, registry: DynamicAgentRegistry):
        """Test getting compiled for non-existent agent returns None."""
        assert registry.get_compiled("nonexistent") is None

    def test_remove_agent(self, registry: DynamicAgentRegistry):
        """Test removing an agent."""
        config = SubAgentConfig(
            name="test-agent",
            description="A test agent",
            instructions="Do tests",
        )
        registry.register(config, MockAgent("test-agent"))

        result = registry.remove("test-agent")

        assert result is True
        assert not registry.exists("test-agent")
        assert registry.get("test-agent") is None
        assert registry.get_config("test-agent") is None
        assert registry.get_compiled("test-agent") is None

    def test_remove_nonexistent(self, registry: DynamicAgentRegistry):
        """Test removing non-existent agent returns False."""
        result = registry.remove("nonexistent")
        assert result is False

    def test_list_agents(self, registry: DynamicAgentRegistry):
        """Test listing all agent names."""
        for i in range(3):
            config = SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            registry.register(config, MockAgent(f"agent-{i}"))

        names = registry.list_agents()
        assert set(names) == {"agent-0", "agent-1", "agent-2"}

    def test_list_configs(self, registry: DynamicAgentRegistry):
        """Test listing all configurations."""
        configs_to_add = [
            SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            for i in range(2)
        ]

        for config in configs_to_add:
            registry.register(config, MockAgent(config["name"]))

        configs = registry.list_configs()
        assert len(configs) == 2
        assert all(c["name"].startswith("agent-") for c in configs)

    def test_list_compiled(self, registry: DynamicAgentRegistry):
        """Test listing all compiled agents."""
        for i in range(2):
            config = SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            registry.register(config, MockAgent(f"agent-{i}"))

        compiled = registry.list_compiled()
        assert len(compiled) == 2
        assert all(c.name.startswith("agent-") for c in compiled)

    def test_exists(self, registry: DynamicAgentRegistry):
        """Test checking if agent exists."""
        config = SubAgentConfig(
            name="test-agent",
            description="A test agent",
            instructions="Do tests",
        )
        registry.register(config, MockAgent("test-agent"))

        assert registry.exists("test-agent") is True
        assert registry.exists("nonexistent") is False

    def test_count(self, registry: DynamicAgentRegistry):
        """Test counting registered agents."""
        assert registry.count() == 0

        for i in range(3):
            config = SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            registry.register(config, MockAgent(f"agent-{i}"))

        assert registry.count() == 3

    def test_clear(self, registry: DynamicAgentRegistry):
        """Test clearing all agents."""
        for i in range(3):
            config = SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            registry.register(config, MockAgent(f"agent-{i}"))

        registry.clear()

        assert registry.count() == 0
        assert registry.list_agents() == []

    def test_get_summary_empty(self, registry: DynamicAgentRegistry):
        """Test summary for empty registry."""
        summary = registry.get_summary()
        assert summary == "No dynamically created agents."

    def test_get_summary_with_agents(self, registry: DynamicAgentRegistry):
        """Test summary with registered agents."""
        config1 = SubAgentConfig(
            name="researcher",
            description="Researches topics",
            instructions="Do research",
            model="gpt-4",
        )
        config2 = SubAgentConfig(
            name="writer",
            description="Writes content",
            instructions="Write things",
            # No model specified
        )

        registry.register(config1, MockAgent("researcher"))
        registry.register(config2, MockAgent("writer"))

        summary = registry.get_summary()

        assert "Dynamic Agents (2):" in summary
        assert "researcher [gpt-4]: Researches topics" in summary
        assert "writer [default]: Writes content" in summary

    def test_no_max_agents_limit(self):
        """Test registry with no max_agents limit."""
        registry = DynamicAgentRegistry()  # No max_agents

        # Should be able to add many agents
        for i in range(100):
            config = SubAgentConfig(
                name=f"agent-{i}",
                description=f"Agent {i}",
                instructions="Do stuff",
            )
            registry.register(config, MockAgent(f"agent-{i}"))

        assert registry.count() == 100
