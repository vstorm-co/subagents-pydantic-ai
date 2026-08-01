"""Shared helpers for building dynamic subagents at runtime."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.types import SubAgentConfig, ToolsetFactory

CapabilityFactory = ToolsetFactory

AgentFactory = Callable[[SubAgentConfig], Any]
"""Builds the agent instance for a dynamically created subagent.

Receives the freshly built `SubAgentConfig` and returns an agent instance. The
return type is `Any` because the agent need not be a `pydantic_ai.Agent` —
consumers such as pydantic-deep return their own agent objects.

A factory must not attach an `ask_parent` toolset; the subagent toolset injects
one at run time for registry-backed and one-shot agents, and a second copy would
collide on the tool name.
"""


_AGENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9-]+")
"""ASCII only, matching what the `create_agent` tool descriptions promise the model.

`str.isalnum` is Unicode-aware, so it admitted Cyrillic and fullwidth digits while
rejecting `café` — an allow-list a model can silently exceed is not one.
"""


def validate_agent_name(name: str) -> str | None:
    """Return an error message when the agent name is invalid."""
    if not _AGENT_NAME_PATTERN.fullmatch(name):
        return "Error: Name must contain only letters, numbers, and hyphens"
    return None


def validate_model(
    model: str | Model,
    allowed_models: list[str] | None,
) -> str | None:
    """Return an error message when the model is not allowed."""
    if allowed_models and model not in allowed_models:
        allowed = ", ".join(allowed_models)
        return f"Error: Model '{model}' is not allowed. Use one of: {allowed}"
    return None


def validate_capabilities(
    capabilities: list[str] | None,
    capabilities_map: dict[str, CapabilityFactory] | None,
) -> str | None:
    """Return an error message when capabilities are unknown."""
    if capabilities and capabilities_map:
        invalid_caps = [cap for cap in capabilities if cap not in capabilities_map]
        if invalid_caps:
            available = ", ".join(capabilities_map.keys())
            invalid = ", ".join(invalid_caps)
            return f"Error: Unknown capabilities: {invalid}. Available: {available}"
    return None


def validate_capabilities_with_factory(
    capabilities: list[str] | None,
    default_agent_factory: AgentFactory | None,
) -> str | None:
    """Return an error when capabilities cannot be injected with a custom factory."""
    if default_agent_factory is not None and capabilities:
        return (
            "Error: capabilities are not supported when a custom "
            "default_agent_factory is configured. The factory builds the "
            "agent itself; create the agent without capabilities or "
            "configure the factory to attach the required toolsets."
        )
    return None


def build_subagent_config(
    *,
    name: str,
    description: str,
    instructions: str,
    model: str | Model,
    can_ask_questions: bool = True,
) -> SubAgentConfig:
    """Build a runtime subagent configuration."""
    return SubAgentConfig(
        name=name,
        description=description,
        instructions=instructions,
        model=model,
        can_ask_questions=can_ask_questions,
    )


def collect_agent_toolsets(
    ctx: RunContext[SubAgentDepsProtocol],
    capabilities: list[str] | None,
    *,
    toolsets_factory: ToolsetFactory | None,
    capabilities_map: dict[str, CapabilityFactory] | None,
) -> list[Any]:
    """Collect toolsets to attach to a dynamically built agent."""
    agent_toolsets: list[Any] = []
    if toolsets_factory:
        agent_toolsets.extend(toolsets_factory(ctx.deps))
    elif capabilities and capabilities_map:
        for cap_name in capabilities:
            cap_factory = capabilities_map[cap_name]
            agent_toolsets.extend(cap_factory(ctx.deps))
    return agent_toolsets


def build_agent_instance(
    ctx: RunContext[SubAgentDepsProtocol],
    config: SubAgentConfig,
    *,
    model: str | Model,
    capabilities: list[str] | None,
    toolsets_factory: ToolsetFactory | None,
    capabilities_map: dict[str, CapabilityFactory] | None,
    default_agent_factory: AgentFactory | None,
) -> Any:
    """Build a pydantic-ai Agent instance for a dynamic subagent."""
    if default_agent_factory is not None:
        return default_agent_factory(config)

    agent_toolsets = collect_agent_toolsets(
        ctx,
        capabilities,
        toolsets_factory=toolsets_factory,
        capabilities_map=capabilities_map,
    )
    return Agent(
        model,
        system_prompt=config["instructions"],
        toolsets=agent_toolsets or None,
    )


def build_dynamic_agent(
    ctx: RunContext[SubAgentDepsProtocol],
    *,
    name: str,
    description: str,
    instructions: str,
    model: str | Model,
    can_ask_questions: bool = True,
    capabilities: list[str] | None = None,
    allowed_models: list[str] | None = None,
    toolsets_factory: ToolsetFactory | None = None,
    capabilities_map: dict[str, CapabilityFactory] | None = None,
    default_agent_factory: AgentFactory | None = None,
) -> str | tuple[Any, SubAgentConfig]:
    """Validate inputs and build a dynamic agent.

    Returns:
        An error string on failure, or ``(agent, config)`` on success.
    """
    name_error = validate_agent_name(name)
    if name_error is not None:
        return name_error

    model_error = validate_model(model, allowed_models)
    if model_error is not None:
        return model_error

    caps_error = validate_capabilities(capabilities, capabilities_map)
    if caps_error is not None:
        return caps_error

    factory_caps_error = validate_capabilities_with_factory(
        capabilities,
        default_agent_factory,
    )
    if factory_caps_error is not None:
        return factory_caps_error

    config = build_subagent_config(
        name=name,
        description=description,
        instructions=instructions,
        model=model,
        can_ask_questions=can_ask_questions,
    )

    try:
        agent = build_agent_instance(
            ctx,
            config,
            model=model,
            capabilities=capabilities,
            toolsets_factory=toolsets_factory,
            capabilities_map=capabilities_map,
            default_agent_factory=default_agent_factory,
        )
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error creating agent: {exc}"

    return agent, config
