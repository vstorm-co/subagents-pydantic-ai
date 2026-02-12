"""Agent factory toolset for dynamic agent creation.

This module provides a toolset that allows agents to create
new specialized agents at runtime.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.registry import DynamicAgentRegistry
from subagents_pydantic_ai.types import SubAgentConfig, ToolsetFactory

# Type alias for capability factory functions
CapabilityFactory = ToolsetFactory


def create_agent_factory_toolset(
    registry: DynamicAgentRegistry,
    allowed_models: list[str] | None = None,
    default_model: str = "openai:gpt-4.1",
    max_agents: int = 10,
    toolsets_factory: ToolsetFactory | None = None,
    capabilities_map: dict[str, CapabilityFactory] | None = None,
    id: str | None = None,
) -> FunctionToolset[Any]:
    """Create a toolset for dynamic agent creation.

    This toolset provides tools for creating, listing, and removing
    agents at runtime. Created agents are stored in the provided
    registry and can be used with the main subagent toolset.

    Args:
        registry: Registry to store created agents.
        allowed_models: List of allowed model names. If None, any model
            is allowed.
        default_model: Default model to use when not specified.
        max_agents: Maximum number of dynamic agents allowed.
        toolsets_factory: Factory to create toolsets for new agents.
            Takes priority over capabilities if both are provided.
        capabilities_map: Mapping of capability names to factory functions.
            E.g., {"filesystem": create_fs_toolset, "todo": create_todo_toolset}.
            Used when capabilities are specified in create_agent.
        id: Optional toolset ID. Defaults to "agent_factory".

    Returns:
        FunctionToolset with agent management tools.

    Example:
        ```python
        from pydantic_ai import Agent
        from subagents_pydantic_ai import (
            create_agent_factory_toolset,
            DynamicAgentRegistry,
        )

        registry = DynamicAgentRegistry()

        # With capabilities map
        factory_toolset = create_agent_factory_toolset(
            registry=registry,
            allowed_models=["openai:gpt-4.1", "openai:gpt-4o-mini"],
            max_agents=5,
            capabilities_map={
                "filesystem": lambda deps: [create_fs_toolset(deps.backend)],
                "todo": lambda deps: [create_todo_toolset()],
            },
        )

        agent = Agent("openai:gpt-4.1", toolsets=[factory_toolset])
        ```
    """
    # Update registry max_agents
    registry.max_agents = max_agents

    # Format allowed models for docstring
    models_desc = (
        f"Allowed models: {', '.join(allowed_models)}" if allowed_models else "Any model is allowed"
    )

    # Format available capabilities for docstring
    caps_desc = (
        f"Available capabilities: {', '.join(capabilities_map.keys())}"
        if capabilities_map
        else "No predefined capabilities available"
    )

    toolset: FunctionToolset[Any] = FunctionToolset(id=id or "agent_factory")

    @toolset.tool
    async def create_agent(
        ctx: RunContext[SubAgentDepsProtocol],
        name: str,
        description: str,
        instructions: str,
        model: str | None = None,
        capabilities: list[str] | None = None,
        can_ask_questions: bool = True,
    ) -> str:
        f"""Create a new specialized agent at runtime.

        Creates a new agent with the specified configuration. The agent
        will be available for delegation via the task tool.

        {models_desc}
        {caps_desc}

        Args:
            ctx: The run context.
            name: Unique name for the agent (letters, numbers, hyphens only).
            description: Brief description of what the agent does.
            instructions: System prompt / instructions for the agent.
            model: Model to use (optional, defaults to {default_model}).
            capabilities: List of capability names to enable (e.g., ["filesystem", "todo"]).
            can_ask_questions: Whether agent can ask parent questions.

        Returns:
            Confirmation message or error.
        """
        # Validate name
        if not name or not all(c.isalnum() or c == "-" for c in name):
            return "Error: Name must contain only letters, numbers, and hyphens"

        if registry.exists(name):
            return f"Error: Agent '{name}' already exists"

        # Validate model
        actual_model = model or default_model
        if allowed_models and actual_model not in allowed_models:
            allowed = ", ".join(allowed_models)
            return f"Error: Model '{actual_model}' is not allowed. Use one of: {allowed}"

        # Validate capabilities
        if capabilities and capabilities_map:
            invalid_caps = [c for c in capabilities if c not in capabilities_map]
            if invalid_caps:
                available = ", ".join(capabilities_map.keys())
                invalid = ", ".join(invalid_caps)
                return f"Error: Unknown capabilities: {invalid}. Available: {available}"

        # Create config
        config = SubAgentConfig(
            name=name,
            description=description,
            instructions=instructions,
            model=actual_model,
            can_ask_questions=can_ask_questions,
        )

        # Create agent
        try:
            # Collect toolsets from factory or capabilities
            agent_toolsets: list[Any] = []
            if toolsets_factory:
                agent_toolsets.extend(toolsets_factory(ctx.deps))
            elif capabilities and capabilities_map:
                for cap_name in capabilities:
                    cap_factory = capabilities_map[cap_name]
                    agent_toolsets.extend(cap_factory(ctx.deps))

            agent: Agent[Any, str] = Agent(
                actual_model,
                system_prompt=instructions,
                toolsets=agent_toolsets or None,
            )

            registry.register(config, agent)

            caps_info = f"\nCapabilities: {', '.join(capabilities)}" if capabilities else ""
            return (
                f"Agent '{name}' created successfully.\n"
                f"Model: {actual_model}\n"
                f"Description: {description}{caps_info}\n"
                f"Use task(description, '{name}') to delegate tasks."
            )

        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating agent: {e}"

    @toolset.tool
    async def list_agents(
        ctx: RunContext[SubAgentDepsProtocol],
    ) -> str:
        """List all dynamically created agents.

        Returns:
            List of agent names and descriptions.
        """
        return registry.get_summary()

    @toolset.tool
    async def remove_agent(
        ctx: RunContext[SubAgentDepsProtocol],
        name: str,
    ) -> str:
        """Remove a dynamically created agent.

        The agent will no longer be available for task delegation.

        Args:
            ctx: The run context.
            name: Name of the agent to remove.

        Returns:
            Confirmation or error message.
        """
        if registry.remove(name):
            return f"Agent '{name}' has been removed."
        return f"Error: Agent '{name}' not found."

    @toolset.tool
    async def get_agent_info(
        ctx: RunContext[SubAgentDepsProtocol],
        name: str,
    ) -> str:
        """Get detailed information about a dynamic agent.

        Args:
            ctx: The run context.
            name: Name of the agent.

        Returns:
            Agent details or error message.
        """
        config = registry.get_config(name)
        if config is None:
            return f"Error: Agent '{name}' not found."

        info = [
            f"Agent: {name}",
            f"Description: {config['description']}",
            f"Model: {config.get('model', default_model)}",
            f"Can ask questions: {config.get('can_ask_questions', True)}",
            "",
            "Instructions:",
            config["instructions"][:500] + ("..." if len(config["instructions"]) > 500 else ""),
        ]

        return "\n".join(info)

    return toolset
