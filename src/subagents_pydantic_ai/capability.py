"""Subagent capability for pydantic-ai agents.

Provides a ``SubAgentCapability`` that integrates the subagent toolset +
dynamic instructions via the pydantic-ai capabilities API.

Example:
    ```python
    from pydantic_ai import Agent
    from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[SubAgentCapability(
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches topics",
                    instructions="You are a research assistant.",
                ),
            ],
        )],
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from subagents_pydantic_ai.prompts import get_subagent_system_prompt
from subagents_pydantic_ai.toolset import create_subagent_toolset
from subagents_pydantic_ai.types import SubAgentConfig, ToolsetFactory


@dataclass
class SubAgentCapability(AbstractCapability[Any]):
    """Capability that provides subagent delegation tools and dynamic instructions.

    Combines the subagent toolset (task, check_task, answer_subagent, etc.)
    with dynamic system prompt injection listing available subagents.

    Example:
        ```python
        from pydantic_ai import Agent
        from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

        cap = SubAgentCapability(
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches topics",
                    instructions="You are a research assistant.",
                ),
            ],
        )
        agent = Agent("openai:gpt-4.1", capabilities=[cap])
        ```

    Attributes:
        subagents: List of subagent configurations.
        default_model: Default model for subagents.
        include_general_purpose: Include general-purpose subagent.
        max_nesting_depth: Max depth for nested subagents.
        toolsets_factory: Factory for subagent toolsets.
        registry: Dynamic agent registry.
        descriptions: Custom tool descriptions override.
    """

    subagents: list[SubAgentConfig] | None = None
    default_model: Any = "openai:gpt-4.1"
    include_general_purpose: bool = True
    max_nesting_depth: int = 0
    toolsets_factory: ToolsetFactory | None = None
    registry: Any = None
    descriptions: dict[str, str] | None = None
    _toolset: AbstractToolset[Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the underlying subagent toolset."""
        self._toolset = create_subagent_toolset(
            subagents=self.subagents,
            default_model=self.default_model,
            toolsets_factory=self.toolsets_factory,
            include_general_purpose=self.include_general_purpose,
            max_nesting_depth=self.max_nesting_depth,
            id="subagents",
            registry=self.registry,
            descriptions=self.descriptions,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return name for AgentSpec YAML/JSON serialization."""
        return "SubAgentCapability"

    @property
    def task_manager(self) -> Any:
        """Access the underlying task manager for observability."""
        return getattr(self._toolset, "task_manager", None)

    def get_toolset(self) -> AbstractToolset[Any] | None:
        """Return the subagent toolset with all registered tools."""
        return self._toolset

    def get_instructions(self) -> Any:
        """Return dynamic instructions listing available subagents."""
        configs = list(self.subagents) if self.subagents else []

        def _instructions(ctx: RunContext[Any]) -> str:
            return get_subagent_system_prompt(configs)

        return _instructions
