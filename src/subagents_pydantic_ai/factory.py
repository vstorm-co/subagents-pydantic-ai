"""Agent factory toolset for dynamic agent creation.

This module provides a toolset that allows agents to create
new specialized agents at runtime.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai.dynamic_agent import (
    AgentFactory,
    CapabilityFactory,
    build_dynamic_agent,
)
from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.registry import DynamicAgentRegistry
from subagents_pydantic_ai.types import ToolsetFactory


def create_agent_factory_toolset(
    registry: DynamicAgentRegistry,
    allowed_models: list[str] | None = None,
    default_model: str | Model | None = None,
    max_agents: int = 10,
    toolsets_factory: ToolsetFactory | None = None,
    capabilities_map: dict[str, CapabilityFactory] | None = None,
    id: str | None = None,
    default_agent_factory: AgentFactory | None = None,
) -> FunctionToolset[Any]:
    """Create a toolset for dynamic agent creation.

    This toolset provides tools for creating, listing, and removing
    agents at runtime. Created agents are stored in the provided
    registry and can be used with the main subagent toolset.

    Args:
        registry: Registry to store created agents.
        allowed_models: List of allowed model names. If None, any model
            is allowed.
        default_model: Model to use for a `create_agent` call that names none.
            There is no implicit default: leave it unset and such a call is
            refused, rather than creating an agent on a model the library picked
            and therefore on whatever provider credential the process environment
            happens to hold.
        max_agents: Maximum number of dynamic agents allowed. This is written
            onto `registry.max_agents`, so it wins over whatever cap the
            registry was constructed with — pass it explicitly when the limit
            matters.
        toolsets_factory: Factory to create toolsets for new agents.
            Takes priority over capabilities if both are provided.
        capabilities_map: Mapping of capability names to factory functions.
            E.g., {"filesystem": create_fs_toolset, "todo": create_todo_toolset}.
            Used when capabilities are specified in create_agent.
        id: Optional toolset ID. Defaults to "agent_factory".
        default_agent_factory: Optional builder for created agents, replacing the
            default plain `pydantic_ai.Agent`. When set, `create_agent` rejects
            requested `capabilities`, since the factory owns the agent's toolsets.

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

    # Tool description passed to the model. This MUST be supplied via the
    # decorator: an `f"""..."""` as the first statement of the function body is
    # NOT a docstring (`__doc__` stays `None`) — it would be evaluated and
    # discarded on every call, throwing away the computed models/capabilities.
    # A model told there is a default will happily omit one, so only say so when
    # the consumer named it.
    default_desc = (
        f"Default model when none is given: {default_model}."
        if default_model is not None
        else "There is no default model: name the model to use in every call."
    )
    create_agent_description = (
        "Create a new specialized agent at runtime.\n\n"
        "Creates a new agent with the specified configuration. The agent "
        "will be available for delegation via the task tool.\n\n"
        f"{models_desc}\n{caps_desc}\n\n{default_desc}"
    )

    @toolset.tool(description=create_agent_description)
    async def create_agent(
        ctx: RunContext[SubAgentDepsProtocol],
        name: str,
        description: str,
        instructions: str,
        model: str | None = None,
        capabilities: list[str] | None = None,
        can_ask_questions: bool = True,
    ) -> str:
        """Create a new specialized agent at runtime.

        The model-facing description (with the allowed models / capabilities /
        default model interpolated) is supplied via the `@toolset.tool`
        decorator above, not this docstring.

        Args:
            ctx: The run context.
            name: Unique name for the agent (letters, numbers, hyphens only).
            description: Brief description of what the agent does.
            instructions: System prompt / instructions for the agent.
            model: Model to use (optional, defaults to the factory default).
            capabilities: List of capability names to enable (e.g., ["filesystem", "todo"]).
            can_ask_questions: Whether agent can ask parent questions.

        Returns:
            Confirmation message or error.
        """
        if registry.exists(name):
            return f"Error: Agent '{name}' already exists"

        actual_model = model or default_model
        if actual_model is None:
            # A tool result rather than an exception: the model named nothing and
            # can name something on its next turn. The fallback this replaces put
            # the agent on a model of the library's choosing, and so on whichever
            # provider credential the environment held.
            allowed = f" Allowed models: {', '.join(allowed_models)}." if allowed_models else ""
            return (
                "Error: no model was given and there is no default model. "
                f"Name the model to use and try again.{allowed}"
            )

        result = build_dynamic_agent(
            ctx,
            name=name,
            description=description,
            instructions=instructions,
            model=actual_model,
            can_ask_questions=can_ask_questions,
            capabilities=capabilities,
            allowed_models=allowed_models,
            toolsets_factory=toolsets_factory,
            capabilities_map=capabilities_map,
            default_agent_factory=default_agent_factory,
        )
        if isinstance(result, str):
            return result
        agent, config = result

        try:
            registry.register(config, agent)
        except ValueError as e:
            return f"Error: {e}"

        caps_info = f"\nCapabilities: {', '.join(capabilities)}" if capabilities else ""
        return (
            f"Agent '{name}' created successfully.\n"
            f"Model: {actual_model}\n"
            f"Description: {description}{caps_info}\n"
            f"Use task(description, '{name}') to delegate tasks."
        )

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
            f"Model: {config.get('model') or default_model or 'not configured'}",
            f"Can ask questions: {config.get('can_ask_questions', True)}",
            "",
            "Instructions:",
            config["instructions"][:500] + ("..." if len(config["instructions"]) > 500 else ""),
        ]

        return "\n".join(info)

    return toolset
