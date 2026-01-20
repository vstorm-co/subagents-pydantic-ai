"""Dynamic agent registry for runtime agent management.

This module provides a registry for dynamically created agents,
allowing agents to be added, removed, and queried at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from subagents_pydantic_ai.types import CompiledSubAgent, SubAgentConfig


@dataclass
class DynamicAgentRegistry:
    """Registry for dynamically created agents.

    Provides storage and management for agents created at runtime.
    Used by the agent factory toolset to track created agents.

    Attributes:
        agents: Dictionary mapping agent names to Agent instances.
        configs: Dictionary mapping agent names to their configurations.
        max_agents: Maximum number of agents allowed (optional limit).

    Example:
        ```python
        registry = DynamicAgentRegistry(max_agents=10)

        # Register a new agent
        registry.register(config, agent)

        # Get an agent
        agent = registry.get("my-agent")

        # List all agents
        names = registry.list_agents()

        # Remove an agent
        registry.remove("my-agent")
        ```
    """

    agents: dict[str, Any] = field(default_factory=dict)
    configs: dict[str, SubAgentConfig] = field(default_factory=dict)
    _compiled: dict[str, CompiledSubAgent] = field(default_factory=dict)
    max_agents: int | None = None

    def register(
        self,
        config: SubAgentConfig,
        agent: Any,
    ) -> None:
        """Register a new agent.

        Args:
            config: The agent's configuration.
            agent: The Agent instance.

        Raises:
            ValueError: If agent name already exists or max_agents reached.
        """
        name = config["name"]

        if name in self.agents:
            raise ValueError(f"Agent '{name}' already exists")

        if self.max_agents and len(self.agents) >= self.max_agents:
            raise ValueError(
                f"Maximum number of agents ({self.max_agents}) reached. "
                f"Remove an agent before creating a new one."
            )

        self.agents[name] = agent
        self.configs[name] = config
        self._compiled[name] = CompiledSubAgent(
            name=name,
            description=config["description"],
            agent=agent,
            config=config,
        )

    def get(self, name: str) -> Any | None:
        """Get an agent by name.

        Args:
            name: The agent name.

        Returns:
            The Agent instance, or None if not found.
        """
        return self.agents.get(name)

    def get_config(self, name: str) -> SubAgentConfig | None:
        """Get an agent's configuration by name.

        Args:
            name: The agent name.

        Returns:
            The SubAgentConfig, or None if not found.
        """
        return self.configs.get(name)

    def get_compiled(self, name: str) -> CompiledSubAgent | None:
        """Get a compiled agent by name.

        Args:
            name: The agent name.

        Returns:
            The CompiledSubAgent, or None if not found.
        """
        return self._compiled.get(name)

    def remove(self, name: str) -> bool:
        """Remove an agent from the registry.

        Args:
            name: The agent name to remove.

        Returns:
            True if agent was removed, False if not found.
        """
        if name not in self.agents:
            return False

        del self.agents[name]
        del self.configs[name]
        del self._compiled[name]
        return True

    def list_agents(self) -> list[str]:
        """Get list of all registered agent names.

        Returns:
            List of agent names.
        """
        return list(self.agents.keys())

    def list_configs(self) -> list[SubAgentConfig]:
        """Get list of all agent configurations.

        Returns:
            List of SubAgentConfig for all registered agents.
        """
        return list(self.configs.values())

    def list_compiled(self) -> list[CompiledSubAgent]:
        """Get list of all compiled agents.

        Returns:
            List of CompiledSubAgent for all registered agents.
        """
        return list(self._compiled.values())

    def exists(self, name: str) -> bool:
        """Check if an agent exists.

        Args:
            name: The agent name.

        Returns:
            True if agent exists, False otherwise.
        """
        return name in self.agents

    def count(self) -> int:
        """Get the number of registered agents.

        Returns:
            Number of agents in the registry.
        """
        return len(self.agents)

    def clear(self) -> None:
        """Remove all agents from the registry."""
        self.agents.clear()
        self.configs.clear()
        self._compiled.clear()

    def get_summary(self) -> str:
        """Get a formatted summary of all registered agents.

        Returns:
            Multi-line string describing all agents.
        """
        if not self.agents:
            return "No dynamically created agents."

        lines = [f"Dynamic Agents ({len(self.agents)}):"]
        for name, config in self.configs.items():
            model = config.get("model", "default")
            lines.append(f"- {name} [{model}]: {config['description']}")

        return "\n".join(lines)
