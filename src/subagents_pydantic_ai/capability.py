"""Subagent capability for pydantic-ai agents.

Provides a `SubAgentCapability` that integrates the subagent toolset +
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

from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.toolsets import AbstractToolset

from subagents_pydantic_ai._execution import DEFAULT_ASK_TIMEOUT_SECONDS
from subagents_pydantic_ai.dynamic_agent import AgentFactory, CapabilityFactory
from subagents_pydantic_ai.message_bus import TaskManager
from subagents_pydantic_ai.prompts import get_subagent_system_prompt
from subagents_pydantic_ai.registry import DynamicAgentRegistry
from subagents_pydantic_ai.toolset import SubAgentToolset, create_subagent_toolset
from subagents_pydantic_ai.types import (
    AskUserCallback,
    DelegationConfiguration,
    SubAgentConfig,
    ToolsetFactory,
    UsageLimitsFactory,
)


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
        usage_limits: Optional static or per-task usage limits for delegated
            subagent runs.
        delegation_configuration: Select default, persisted, combined, or
            one-shot-only delegation entry points. A mode that hides a tool
            rejects that tool's configuration rather than ignoring it — see
            `Raises`.
        allowed_models: Optional model allow-list for dynamic specialists.
        capabilities_map: Optional capability factories for dynamic specialists.
        default_agent_factory: Optional custom agent factory for dynamic specialists.
            Do not attach an `ask_parent` toolset in the factory; the toolset
            injects it at run time when needed.
        max_agents: Maximum number of persistent dynamic agents, applied to the
            registry the toolset creates for `create_agent`. Ignored when
            `registry` is set — that registry keeps its own `max_agents`.
        max_chat_traces: Maximum number of subagent conversations kept for
            `chat_trace_id` continuation, least-recently-used evicted first.
        max_task_handles: Maximum number of finished task handles retained for
            status queries. Evicted usage still counts toward `get_total_usage()`.
        max_result_chars: Character budget for a completed task's result in the
            `wait_tasks` listing. Truncated results carry an explicit marker
            pointing at `check_task`, which always returns the full text. Pass
            `None` to never truncate.
        ask_user: Callback backing `ask_parent`. Required for a sync-mode subagent
            to ask its parent anything: the parent's run loop is blocked inside the
            delegation, so `answer_subagent` cannot be reached until it returns.

    Raises:
        ValueError: From `create_subagent_toolset` when the configuration is
            self-contradictory — an invalid `delegation_configuration`, or one
            whose hidden tools make `subagents`, `registry`, `allowed_models`,
            `capabilities_map`, or `default_agent_factory` unreachable.
    """

    subagents: list[SubAgentConfig] | None = None
    default_model: Any = "openai:gpt-4.1"
    include_general_purpose: bool = True
    max_nesting_depth: int = 0
    toolsets_factory: ToolsetFactory | None = None
    registry: DynamicAgentRegistry | None = None
    descriptions: dict[str, str] | None = None
    usage_limits: UsageLimits | UsageLimitsFactory | None = None
    delegation_configuration: DelegationConfiguration = "default"
    allowed_models: list[str] | None = None
    capabilities_map: dict[str, CapabilityFactory] | None = None
    default_agent_factory: AgentFactory | None = None
    max_agents: int = 10
    max_chat_traces: int = 100
    max_task_handles: int = 500
    max_result_chars: int | None = 2000
    ask_user: AskUserCallback | None = None
    ask_timeout_seconds: float = DEFAULT_ASK_TIMEOUT_SECONDS
    contain_errors: bool = True
    _toolset: SubAgentToolset = field(init=False, repr=False)

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
            usage_limits=self.usage_limits,
            delegation_configuration=self.delegation_configuration,
            allowed_models=self.allowed_models,
            capabilities_map=self.capabilities_map,
            default_agent_factory=self.default_agent_factory,
            max_agents=self.max_agents,
            max_chat_traces=self.max_chat_traces,
            max_task_handles=self.max_task_handles,
            max_result_chars=self.max_result_chars,
            ask_user=self.ask_user,
            ask_timeout_seconds=self.ask_timeout_seconds,
            contain_errors=self.contain_errors,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return name for AgentSpec YAML/JSON serialization."""
        return "SubAgentCapability"

    @property
    def task_manager(self) -> TaskManager:
        """The task manager behind the toolset, for observability."""
        return self._toolset.task_manager

    def get_toolset(self) -> AbstractToolset[Any] | None:
        """Return the subagent toolset with all registered tools."""
        return self._toolset

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Run the agent, then stop every background task this run started.

        A background delegation is an `asyncio.Task` the parent run does not await.
        Without this finalizer it keeps executing after the run returns, against
        deps the application has already torn down, and one blocked in `ask_parent`
        waits for an answer that can never arrive.
        """
        try:
            return await handler()
        finally:
            await self._toolset.cancel_run_tasks(ctx.run_id)

    def get_instructions(self) -> Any:
        """Return dynamic instructions listing available subagents."""
        configs = list(self.subagents) if self.subagents else []

        def _instructions(ctx: RunContext[Any]) -> str:
            if self.delegation_configuration == "oneshot_only":
                return (
                    "## One-Shot Delegation\n\n"
                    "Use the `delegate` tool to create an ephemeral specialist "
                    "and run a task in one call."
                )
            return get_subagent_system_prompt(configs)

        return _instructions
