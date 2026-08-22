"""The subagent toolset: delegate tasks to child agents, synchronously or in the background.

`SubAgentToolset` is a `FunctionToolset` subclass that owns the delegation tools and
the state behind them -- the task manager, the chat-trace store, and the dynamic
agent registry. `create_subagent_toolset` builds one and is the entry point most
applications use.

The heavy lifting lives in sibling modules: `_execution` runs a delegation,
`_observability` captures its telemetry, `_chat_trace` stores conversations a
`chat_trace_id` can resume, and `_state` carries the channel `ask_parent` uses to
reach the parent.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from typing import Any, Literal

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.models import Model
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from subagents_pydantic_ai._chat_trace import ChatTraceKey, ChatTraceStore
from subagents_pydantic_ai._execution import (
    DEFAULT_ASK_TIMEOUT_SECONDS,
    drain_steering_messages,
    error_for_model,
)
from subagents_pydantic_ai._execution import _run_async as _run_async
from subagents_pydantic_ai._execution import _run_sync as _run_sync
from subagents_pydantic_ai._observability import (
    capture_message_history,
    capture_result_observability,
    model_responses,
    result_traceparent,
    serialize_output,
)
from subagents_pydantic_ai._state import current_subagent_state
from subagents_pydantic_ai.dynamic_agent import (
    AgentFactory,
    CapabilityFactory,
    build_dynamic_agent,
)
from subagents_pydantic_ai.message_bus import (
    DEFAULT_CANCEL_GRACE_SECONDS,
    InMemoryMessageBus,
    TaskManager,
)
from subagents_pydantic_ai.prompts import (
    ANSWER_SUBAGENT_DESCRIPTION,
    CHECK_TASK_DESCRIPTION,
    CREATE_AGENT_TEXT,
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DELEGATE_TEXT,
    HARD_CANCEL_TASK_DESCRIPTION,
    LIST_ACTIVE_TASKS_DESCRIPTION,
    SEND_MESSAGE_TO_SUBAGENT_DESCRIPTION,
    SOFT_CANCEL_TASK_DESCRIPTION,
    SUBAGENT_SYSTEM_PROMPT,
    TASK_TEXT,
    WAIT_TASKS_DESCRIPTION,
)
from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.registry import DynamicAgentRegistry
from subagents_pydantic_ai.types import (
    AgentMessage,
    AskUserCallback,
    CompiledSubAgent,
    DelegationConfiguration,
    EventStreamHandlerFactory,
    ExecutionMode,
    MessageType,
    SubAgentConfig,
    TaskCharacteristics,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    ToolsetFactory,
    UsageLimitsFactory,
    decide_execution_mode,
    utcnow,
)

# Aliases under the historical private names. The implementations moved into
# `_execution` and `_observability`, but `_run_sync`, `_run_async`, and the capture
# helpers are imported from this module by the test suite, and `_compile_subagent`
# by pydantic-deep's team toolset.
_capture_message_history = capture_message_history
_capture_result_observability = capture_result_observability
_drain_steering_messages = drain_steering_messages
_get_result_traceparent = result_traceparent
_iter_model_responses = model_responses
_serialize_output = serialize_output

_VALID_DELEGATION_CONFIGURATIONS = frozenset(
    {"default", "persisted", "persisted_and_oneshot", "oneshot_only"}
)

_DEFAULT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "check_task": CHECK_TASK_DESCRIPTION,
    "answer_subagent": ANSWER_SUBAGENT_DESCRIPTION,
    "send_message_to_subagent": SEND_MESSAGE_TO_SUBAGENT_DESCRIPTION,
    "list_active_tasks": LIST_ACTIVE_TASKS_DESCRIPTION,
    "wait_tasks": WAIT_TASKS_DESCRIPTION,
    "soft_cancel_task": SOFT_CANCEL_TASK_DESCRIPTION,
    "hard_cancel_task": HARD_CANCEL_TASK_DESCRIPTION,
}
"""Model-facing description per background-task tool, overridable via `descriptions`."""


def _format_chat_trace_result(output: str, chat_trace_id: str) -> str:
    """Append a compact chat trace identifier to a subagent result."""
    return f"{output}\n\nChat Trace ID: {chat_trace_id}"


def _preview_result(result: str, task_id: str, max_chars: int | None) -> str:
    """Render a task result for `wait_tasks`, marking truncation explicitly.

    A silent cut reads to the orchestrator like a subagent that stopped
    mid-sentence, so it "recovers" by re-delegating the same work. The marker
    states that the cut is ours, that the stored answer is complete, and which
    tool returns the untruncated text.

    Args:
        result: The subagent's full result text.
        task_id: Task ID the orchestrator passes to `check_task` for the rest.
        max_chars: Preview budget in characters; `None` disables truncation.

    Returns:
        The result unchanged when it fits the budget, otherwise the first
        `max_chars` characters followed by the truncation marker.
    """
    if max_chars is None or len(result) <= max_chars:
        return result
    return (
        f"{result[:max_chars]}\n\n"
        f"[Result truncated for display: showing {max_chars} of {len(result)} characters. "
        f"The subagent's answer is complete and stored in full. "
        f"Call check_task('{task_id}') to read all of it.]"
    )


def _already_finished(task_id: str, handle: TaskHandle | None) -> str:
    """Explain that a cancel arrived after the task was already over.

    Reporting "not found" for a task whose result the orchestrator can still read
    with `check_task` invites it to conclude the work was lost.
    """
    status = handle.status.value if handle is not None else "unknown"
    return (
        f"Task '{task_id}' is no longer running (status: {status}); nothing to cancel. "
        f"Call check_task('{task_id}') for its outcome."
    )


def _create_general_purpose_config(
    default_model: str | Model | None,
    agent_factory: AgentFactory | None,
) -> SubAgentConfig:
    """Create the default general-purpose subagent config.

    Both arguments are the toolset's, carried onto the config so that this
    delegate is built the same way every other one is -- through
    `_compile_subagent`, from the consumer's own factory when there is one. It
    used to carry neither, so it was always compiled from `default_model`: a
    deployment that resolves a model per tenant (its own credential, its own
    budget) has no single `default_model` to give, and the library's own fallback
    put this delegate on whatever provider key happened to be in the process
    environment. The toolset refuses to build it when neither is available, so at
    least one of the two is always set here.

    Args:
        default_model: The toolset's default model, written onto the config so a
            factory reading `config["model"]` sees the same value the library
            would have compiled from. `None` when the consumer supplied none.
        agent_factory: The toolset's `default_agent_factory`, if any. Takes
            priority over `model` in `_compile_subagent`.
    """
    config = SubAgentConfig(
        name="general-purpose",
        description=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
        instructions=SUBAGENT_SYSTEM_PROMPT,
        can_ask_questions=True,
    )
    if default_model is not None:
        config["model"] = default_model
    if agent_factory is not None:
        config["agent_factory"] = agent_factory
    return config


def _compile_subagent(
    config: SubAgentConfig,
    default_model: str | Model | None,
) -> CompiledSubAgent:
    """Compile a subagent configuration into a ready-to-use agent.

    Agent resolution priority:
    1. `config["agent"]` — pre-built agent instance, used as-is
    2. `config["agent_factory"]` — callable(config) -> agent
    3. Default — creates `pydantic_ai.Agent` from config fields

    Args:
        config: The subagent configuration.
        default_model: Default model to use if not specified in config. `None`
            means the consumer named none, and a config that supplies no model,
            agent or factory of its own is then refused rather than compiled
            against a model the library picked.

    Returns:
        CompiledSubAgent with agent instance.

    Raises:
        ValueError: If nothing in the config or `default_model` says which model
            this subagent runs on.
    """
    prebuilt = config.get("agent")
    if prebuilt is not None:
        return CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            agent=prebuilt,
            config=config,
        )

    factory = config.get("agent_factory")
    if factory is not None:
        return CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            agent=factory(config),
            config=config,
        )

    model = config.get("model", default_model)
    if model is None:
        raise ValueError(
            f"Subagent '{config['name']}' says nothing about which model it runs on: "
            "its config sets no 'model', 'agent' or 'agent_factory', and the toolset "
            "was given no 'default_model'. Name a model, or build the agent yourself. "
            "The library used to fall back on a model of its own choosing, which "
            "resolves whatever provider credential is in the process environment -- "
            "in a multi-tenant deployment, not the caller's."
        )

    # `can_ask_questions=False` has to remove the tool, not just tell the subagent
    # not to use it. `_execute` already honours the flag when it injects
    # `ask_parent` at run time; leaving it out here meant a configured subagent
    # could still park a background task in WAITING_FOR_ANSWER for the full
    # `ask_timeout_seconds` while the parent's own instructions said it never asks.
    toolsets: list[AbstractToolset[Any]] = []
    if config.get("can_ask_questions", True):
        toolsets.append(_create_ask_parent_toolset())
    toolsets.extend(config.get("toolsets") or [])

    agent: Agent[Any, str] = Agent(
        model,
        system_prompt=config["instructions"],
        toolsets=toolsets,
        **config.get("agent_kwargs", {}),
    )

    return CompiledSubAgent(
        name=config["name"],
        description=config["description"],
        agent=agent,
        config=config,
    )


def _agent_supplied_by_caller(config: SubAgentConfig) -> bool:
    """Whether the caller handed in a ready agent rather than letting us build one.

    Both the `agent` and `agent_factory` branches of `_compile_subagent` return the
    caller's agent as-is, so neither reaches the point where an `ask_parent` toolset
    is appended to a freshly built `Agent`. Only the default branch does that. A
    caller-supplied agent that honours `can_ask_questions` therefore has to have the
    tool injected at run time instead -- see the configured branch of `task`.
    """
    return config.get("agent") is not None or config.get("agent_factory") is not None


def _create_ask_parent_toolset() -> FunctionToolset[Any]:
    """Create a toolset with the `ask_parent` tool a subagent uses to reach its parent."""
    toolset: FunctionToolset[Any] = FunctionToolset(id="ask_parent")

    @toolset.tool
    async def ask_parent(ctx: RunContext[Any], question: str) -> str:
        """Ask the parent agent a question and wait for the answer.

        Use this when you need clarification or additional information
        to complete your task. Keep questions specific and actionable.

        Args:
            ctx: The run context.
            question: The question to ask the parent.

        Returns:
            The parent's answer.
        """
        # The state bound for this delegation, or -- for a caller that injects it
        # onto its own deps -- the legacy attribute. The library only ever binds
        # the context variable; writing to the caller's deps object breaks a
        # `frozen=True` or `slots=True` deps class.
        state = current_subagent_state()
        legacy = getattr(ctx.deps, "_subagent_state", None)
        ask_callback = state.ask_callback if state is not None else None
        task_manager = state.task_manager if state is not None else None
        task_id = state.task_id if state is not None else None
        timeout = state.ask_timeout_seconds if state is not None else DEFAULT_ASK_TIMEOUT_SECONDS
        if isinstance(legacy, dict):
            ask_callback = ask_callback or legacy.get("ask_callback")
            task_manager = task_manager or legacy.get("task_manager")
            task_id = task_id or legacy.get("task_id")

        budget = state.questions if state is not None else None
        if budget is not None and not budget.consume():
            # Spent before dispatch, so the limit also caps the no-channel case --
            # a subagent looping on a configuration error is the loop that
            # `max_questions` is documented to prevent.
            return (
                f"Error: question limit reached ({budget.limit} for this task). "
                "Finish with the information you already have."
            )

        if ask_callback is not None:
            return str(await ask_callback(question))

        if task_manager is not None and task_id is not None:
            handle = task_manager.get_handle(task_id)
            if handle is not None:
                handle.pending_question = question
                handle.status = TaskStatus.WAITING_FOR_ANSWER

                answer_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                task_manager.set_answer_future(task_id, answer_future)
                try:
                    return await asyncio.wait_for(answer_future, timeout=timeout)
                except asyncio.TimeoutError:
                    return "Error: Parent did not respond in time"
                finally:
                    handle.status = TaskStatus.RUNNING
                    handle.pending_question = None
                    task_manager.clear_answer_future(task_id)

        ask_user = getattr(ctx.deps, "ask_user", None)
        if ask_user:
            return str(await ask_user(question, []))

        return (
            "Error: Cannot ask parent - no communication channel configured. "
            "In sync mode, pass `ask_user=...` to create_subagent_toolset() or "
            "SubAgentCapability(), "
            "or use mode='async' so the parent can respond via answer_subagent()."
        )

    return toolset


class SubAgentToolset(FunctionToolset[Any]):
    """Delegation tools plus the state that backs them.

    Registers `task` (and, depending on `delegation_configuration`, `create_agent`
    and `delegate`) alongside the background-task lifecycle tools `check_task`,
    `answer_subagent`, `send_message_to_subagent`, `list_active_tasks`,
    `wait_tasks`, `soft_cancel_task`, and `hard_cancel_task`.

    `task_manager` and `get_total_usage()` are the supported observability surface;
    they were previously attributes attached to a plain `FunctionToolset` after
    construction.

    Example:
        ```python
        from pydantic_ai import Agent
        from subagents_pydantic_ai import SubAgentConfig, SubAgentToolset

        toolset = SubAgentToolset(
            default_model="openai:gpt-4.1",
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches topics",
                    instructions="You are a research assistant.",
                ),
            ],
        )
        agent = Agent("openai:gpt-4.1", toolsets=[toolset])
        ```
    """

    def __init__(
        self,
        subagents: list[SubAgentConfig] | None = None,
        default_model: str | Model | None = None,
        toolsets_factory: ToolsetFactory | None = None,
        include_general_purpose: bool = True,
        max_nesting_depth: int = 0,
        id: str | None = None,
        registry: DynamicAgentRegistry | None = None,
        descriptions: dict[str, str] | None = None,
        ask_user: AskUserCallback | None = None,
        usage_limits: UsageLimits | UsageLimitsFactory | None = None,
        delegation_configuration: DelegationConfiguration = "default",
        allowed_models: list[str] | None = None,
        capabilities_map: dict[str, CapabilityFactory] | None = None,
        default_agent_factory: AgentFactory | None = None,
        max_agents: int = 10,
        max_chat_traces: int = 100,
        max_task_handles: int = 500,
        max_result_chars: int | None = 2000,
        ask_timeout_seconds: float = DEFAULT_ASK_TIMEOUT_SECONDS,
        contain_errors: bool = True,
        event_stream_handler: EventStreamHandler[Any] | None = None,
        event_stream_handler_factory: EventStreamHandlerFactory | None = None,
        cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS,
    ) -> None:
        """Build the toolset. See `create_subagent_toolset` for the argument reference."""
        super().__init__(id=id or "subagents")
        self._validate(
            delegation_configuration=delegation_configuration,
            subagents=subagents,
            registry=registry,
            include_general_purpose=include_general_purpose,
            default_model=default_model,
            allowed_models=allowed_models,
            capabilities_map=capabilities_map,
            default_agent_factory=default_agent_factory,
            max_result_chars=max_result_chars,
            ask_timeout_seconds=ask_timeout_seconds,
            max_agents=max_agents,
            max_chat_traces=max_chat_traces,
            max_task_handles=max_task_handles,
            event_stream_handler=event_stream_handler,
            event_stream_handler_factory=event_stream_handler_factory,
            cancel_grace_seconds=cancel_grace_seconds,
        )

        self._descriptions = descriptions or {}
        self._default_model = default_model
        self._toolsets_factory = toolsets_factory
        self._max_nesting_depth = max_nesting_depth
        self._ask_user = ask_user
        self._usage_limits = usage_limits
        self._allowed_models = allowed_models
        self._capabilities_map = capabilities_map
        self._default_agent_factory = default_agent_factory
        self._max_task_handles = max_task_handles
        self._max_result_chars = max_result_chars
        self._ask_timeout_seconds = ask_timeout_seconds
        self._contain_errors = contain_errors
        self._event_stream_handler = event_stream_handler
        self._event_stream_handler_factory = event_stream_handler_factory

        self.registry = (
            registry if registry is not None else DynamicAgentRegistry(max_agents=max_agents)
        )
        self.task_manager = TaskManager(
            message_bus=InMemoryMessageBus(), cancel_grace_seconds=cancel_grace_seconds
        )
        self._chat_traces = ChatTraceStore(max_traces=max_chat_traces)
        # Usage from evicted handles, so `get_total_usage` survives eviction.
        self._evicted_usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

        configs: list[SubAgentConfig] = list(subagents) if subagents else []
        if self._general_purpose:
            configs.append(_create_general_purpose_config(default_model, default_agent_factory))
        self._compiled: dict[str, CompiledSubAgent] = {
            config["name"]: _compile_subagent(config, default_model) for config in configs
        }

        self._register_tools()

    # -- construction ----------------------------------------------------------

    def _validate(
        self,
        *,
        delegation_configuration: DelegationConfiguration,
        subagents: list[SubAgentConfig] | None,
        registry: DynamicAgentRegistry | None,
        include_general_purpose: bool,
        default_model: str | Model | None,
        allowed_models: list[str] | None,
        capabilities_map: dict[str, CapabilityFactory] | None,
        default_agent_factory: AgentFactory | None,
        max_result_chars: int | None,
        ask_timeout_seconds: float,
        max_agents: int,
        max_chat_traces: int,
        max_task_handles: int,
        event_stream_handler: EventStreamHandler[Any] | None,
        event_stream_handler_factory: EventStreamHandlerFactory | None,
        cancel_grace_seconds: float,
    ) -> None:
        """Reject a configuration that contradicts itself.

        Hiding a tool also hides everything only that tool reads, so configuration
        for a hidden tool can never take effect. Raising here surfaces the
        contradiction where the caller can still see their own arguments; dropping
        it silently only shows up later as a subagent that ignores an allow-list,
        or a capability the model is never offered.
        """
        if delegation_configuration not in _VALID_DELEGATION_CONFIGURATIONS:
            valid = ", ".join(sorted(_VALID_DELEGATION_CONFIGURATIONS))
            raise ValueError(
                f"Invalid delegation_configuration '{delegation_configuration}'. "
                f"Expected one of: {valid}"
            )

        self._delegation_configuration = delegation_configuration
        # Asking for the delegate without the tool that reaches it is not a
        # contradiction worth raising over -- `task` is what names subagents, so
        # a mode that hides it hides this one too, and always has.
        self._general_purpose = include_general_purpose and self._expose_task

        if self._general_purpose and default_model is None and default_agent_factory is None:
            raise ValueError(
                "include_general_purpose=True needs something to build the general-purpose "
                "subagent from, and neither 'default_model' nor 'default_agent_factory' was "
                "given. The library used to compile it from a model of its own choosing, "
                "which resolves whatever provider credential is in the process environment: "
                "on a deployment holding no such key the build raises, and on one that has "
                "it in its environment a caller's work runs on a credential that is not "
                "theirs. Pass 'default_model' to say which model this delegate runs on, pass "
                "'default_agent_factory' to build it yourself, or set "
                "include_general_purpose=False."
            )

        self._reject_unreachable_configuration(
            delegation_configuration=delegation_configuration,
            subagents=subagents,
            registry=registry,
            allowed_models=allowed_models,
            capabilities_map=capabilities_map,
            default_agent_factory=default_agent_factory,
        )

        if max_result_chars is not None and max_result_chars < 0:
            raise ValueError(f"max_result_chars must be >= 0 or None, got {max_result_chars}")

        if ask_timeout_seconds <= 0:
            raise ValueError(f"ask_timeout_seconds must be > 0, got {ask_timeout_seconds}")

        # A store that cannot hold one entry is not a small store, it is a broken
        # one: every save is evicted before it can be read back, so continuing a
        # chat trace or checking a finished task always fails.
        for name, bound in (
            ("max_chat_traces", max_chat_traces),
            ("max_task_handles", max_task_handles),
        ):
            if bound < 1:
                raise ValueError(f"{name} must be >= 1, got {bound}")

        if max_agents < 0:
            raise ValueError(f"max_agents must be >= 0, got {max_agents}")

        # Both are callables, so nothing downstream could tell them apart and
        # one would silently win. Which one is not something a caller should
        # have to discover from the source.
        if event_stream_handler is not None and event_stream_handler_factory is not None:
            raise ValueError(
                "event_stream_handler and event_stream_handler_factory are mutually "
                "exclusive. Pass the factory when the handler depends on the task, "
                "the handler when it does not."
            )

        if cancel_grace_seconds <= 0:
            raise ValueError(f"cancel_grace_seconds must be > 0, got {cancel_grace_seconds}")

    def _reject_unreachable_configuration(
        self,
        *,
        delegation_configuration: DelegationConfiguration,
        subagents: list[SubAgentConfig] | None,
        registry: DynamicAgentRegistry | None,
        allowed_models: list[str] | None,
        capabilities_map: dict[str, CapabilityFactory] | None,
        default_agent_factory: AgentFactory | None,
    ) -> None:
        """Reject configuration meant for a tool the chosen mode does not expose.

        Reads the `_expose_*` and `_general_purpose` flags `_validate` has already
        set. A mode that hides `task` hides the subagents and registry it reaches;
        a mode that exposes no dynamic-agent tool leaves the arguments only those
        tools read with nowhere to take effect.
        """
        if not self._expose_task:
            if subagents:
                raise ValueError(
                    f"delegation_configuration={delegation_configuration!r} cannot be combined "
                    "with non-empty subagents; configured subagents would be unreachable "
                    "without the task tool. Omit subagents, or use a mode that exposes task."
                )
            if registry is not None:
                raise ValueError(
                    f"delegation_configuration={delegation_configuration!r} cannot be combined "
                    "with a registry; registry-backed agents are only reachable through the "
                    "task tool. Omit registry, or use a mode that exposes task."
                )

        if not self._expose_create_agent and not self._expose_delegate:
            candidates: list[tuple[str, Any]] = [
                ("allowed_models", allowed_models),
                ("capabilities_map", capabilities_map),
            ]
            # `default_agent_factory` also builds the general-purpose delegate, so
            # it is only unread when that delegate is not being built either.
            if not self._general_purpose:
                candidates.append(("default_agent_factory", default_agent_factory))
            unreachable = [name for name, value in candidates if value is not None]
            if unreachable:
                raise ValueError(
                    f"delegation_configuration={delegation_configuration!r} exposes no "
                    f"dynamic-agent tool, so {', '.join(unreachable)} would be ignored. "
                    "Use 'persisted', 'persisted_and_oneshot', or 'oneshot_only'."
                )

    @property
    def _expose_create_agent(self) -> bool:
        return self._delegation_configuration in {"persisted", "persisted_and_oneshot"}

    @property
    def _expose_task(self) -> bool:
        return self._delegation_configuration != "oneshot_only"

    @property
    def _expose_delegate(self) -> bool:
        return self._delegation_configuration in {"persisted_and_oneshot", "oneshot_only"}

    def _register_tools(self) -> None:
        models_desc = (
            f"Allowed models: {', '.join(self._allowed_models)}"
            if self._allowed_models
            else "Any model is allowed"
        )
        caps_desc = (
            f"Available capabilities: {', '.join(self._capabilities_map.keys())}"
            if self._capabilities_map
            else "No predefined capabilities available"
        )
        # A model that is told there is a default will happily omit one. There is
        # no default to omit unless the consumer named it, so say which it is.
        default_desc = (
            f"Default model when none is given: {self._default_model}."
            if self._default_model is not None
            else "There is no default model: name the model to use in every call."
        )
        dynamic_agent_desc = f"{models_desc}\n{caps_desc}\n\n{default_desc}"

        if self._expose_create_agent:
            self.add_function(
                self.create_agent,
                description=self._descriptions.get(
                    "create_agent",
                    CREATE_AGENT_TEXT.render(dynamic_agent_desc),
                ),
            )

        if self._expose_task:
            subagent_list = "\n".join(
                f"- {name}: {compiled.description}" for name, compiled in self._compiled.items()
            )
            self.add_function(
                self.task,
                description=self._descriptions.get(
                    "task",
                    # Inside the summary, not after it: the subagent list is
                    # prose about when to call this, and appending it past the
                    # closing tag would leave the tags bracketing half the text.
                    TASK_TEXT.render(f"Available subagent types:\n{subagent_list}"),
                ),
            )

        if self._expose_delegate:
            self.add_function(
                self.delegate,
                description=self._descriptions.get(
                    "delegate",
                    DELEGATE_TEXT.render(dynamic_agent_desc),
                ),
            )

        self.add_function(self.check_task, description=self._describe("check_task"))
        self.add_function(self.answer_subagent, description=self._describe("answer_subagent"))
        self.add_function(
            self.send_message_to_subagent,
            description=self._describe("send_message_to_subagent"),
        )
        self.add_function(self.list_active_tasks, description=self._describe("list_active_tasks"))
        self.add_function(self.wait_tasks, description=self._describe("wait_tasks"))
        self.add_function(self.soft_cancel_task, description=self._describe("soft_cancel_task"))
        self.add_function(self.hard_cancel_task, description=self._describe("hard_cancel_task"))

    def _describe(self, tool_name: str) -> str:
        """The caller's description override for a tool, or the built-in default."""
        return self._descriptions.get(tool_name, _DEFAULT_TOOL_DESCRIPTIONS[tool_name])

    def _refuse_without_model(self) -> str:
        """Refuse a dynamic-agent call that named no model when there is no default.

        A tool result rather than an exception, like every other refusal these two
        tools make: the model named nothing, and it can name something and call
        again. What it replaces is worse than a refusal -- the library filled the
        gap with a model of its own choosing, so a specialist nobody chose a
        provider for ran on whichever provider credential the process environment
        happened to hold.
        """
        allowed = (
            f" Allowed models: {', '.join(self._allowed_models)}." if self._allowed_models else ""
        )
        return (
            "Error: no model was given and there is no default model. "
            f"Name the model to use and try again.{allowed}"
        )

    # -- observability surface -------------------------------------------------

    @property
    def message_history_store(self) -> OrderedDict[ChatTraceKey, list[Any]]:
        """Stored chat-trace histories, keyed by `(subagent_name, chat_trace_id)`."""
        return self._chat_traces.history

    def get_total_usage(self) -> dict[str, int]:
        """Aggregate token usage across every subagent task this toolset has run.

        Usage from evicted handles is folded in, so the totals do not shrink when
        `max_task_handles` evicts old tasks.

        Returns:
            `input_tokens`, `output_tokens`, `total_tokens`, and `requests`.
        """
        totals = {
            "input_tokens": self._evicted_usage["input_tokens"],
            "output_tokens": self._evicted_usage["output_tokens"],
            "total_tokens": 0,
            "requests": self._evicted_usage["requests"],
        }
        for handle in self.task_manager.list_handles():
            if handle.usage is not None:
                totals["input_tokens"] += getattr(handle.usage, "input_tokens", 0)
                totals["output_tokens"] += getattr(handle.usage, "output_tokens", 0)
                totals["requests"] += getattr(handle.usage, "requests", 0)
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
        return totals

    def _evict_finished_handles(self) -> None:
        """Drop the oldest finished handles past `max_task_handles`.

        Running and waiting tasks are never evicted. Evicted usage is accumulated
        so `get_total_usage` stays correct.
        """
        finished = [h for h in self.task_manager.handles.values() if h.is_finished]
        overflow = len(finished) - self._max_task_handles
        if overflow <= 0:
            return
        finished.sort(key=lambda h: h.completed_at or h.created_at)
        for old in finished[:overflow]:
            if old.usage is not None:
                self._evicted_usage["input_tokens"] += getattr(old.usage, "input_tokens", 0)
                self._evicted_usage["output_tokens"] += getattr(old.usage, "output_tokens", 0)
                self._evicted_usage["requests"] += getattr(old.usage, "requests", 0)
            self.task_manager.handles.pop(old.task_id, None)

    def _handle_for(self, ctx: RunContext[SubAgentDepsProtocol], task_id: str) -> TaskHandle | None:
        """The handle for `task_id`, if this run is allowed to see it.

        One toolset instance is typically built per agent and shared by every run
        that agent serves, so an unfiltered lookup would let one run inspect,
        answer, or cancel another run's task. Handles created without a
        `parent_run_id` (constructed directly, or by an older caller) stay visible
        to everyone.
        """
        handle = self.task_manager.get_handle(task_id)
        if handle is None:
            return None
        if handle.parent_run_id is not None and handle.parent_run_id != ctx.run_id:
            return None
        return handle

    def _cancel_reads_as_missing(self, task_id: str, handle: TaskHandle | None) -> bool:
        """Whether a cancel for `task_id` must read as "not found" to this run.

        `handle` is the run-scoped lookup, so `None` means the id is either unknown
        or owned by another run -- and the two have to be indistinguishable. Task
        ids are short and appear in tool output, so admitting a foreign id here
        lets one run kill another run's work. A task with no handle at all has no
        owner to compare against and stays cancellable.
        """
        if handle is not None:
            return False
        return (
            self.task_manager.get_handle(task_id) is not None
            or task_id not in self.task_manager.tasks
        )

    async def cancel_run_tasks(self, run_id: str | None) -> None:
        """Cancel every background task started by `run_id` and await its cleanup."""
        await self.task_manager.cancel_all(run_id)

    def answer_task(self, task_id: str, answer: str) -> bool:
        """Answer a background task blocked in `ask_parent`, from Python.

        The programmatic half of the `answer_subagent` tool, for an application
        that drives delegation itself rather than letting a model call the tools.
        Unlike the tool, it performs no run scoping: the caller already knows which
        task it owns.

        Args:
            task_id: The task waiting for an answer.
            answer: The answer to deliver.

        Returns:
            Whether a waiting `ask_parent` call was resolved. `False` means the
            task was not waiting -- it may have finished, or never asked.
        """
        return self.task_manager.resolve_answer(task_id, answer)

    async def steer_task(self, task_id: str, message: str) -> bool:
        """Steer a running background task, from Python.

        The programmatic half of the `send_message_to_subagent` tool. The message
        is folded into the subagent's next model request, so it adapts without
        losing partial progress.

        Args:
            task_id: The running task to steer.
            message: The steering instruction.

        Returns:
            Whether the message was queued. `False` means the task is not running,
            so there is no next model request to deliver into.
        """
        agent_id = f"subagent-{task_id}"
        if not self.task_manager.message_bus.is_registered(agent_id):
            return False
        await self.task_manager.message_bus.send(
            AgentMessage(
                type=MessageType.TASK_UPDATE,
                sender="parent",
                receiver=agent_id,
                payload={"message": message},
                task_id=task_id,
            )
        )
        return True

    # -- delegation ------------------------------------------------------------

    def _resolve_event_stream_handler(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        config: SubAgentConfig,
        task_id: str,
    ) -> EventStreamHandler[Any] | None:
        """The toolset's handler for one delegation, from the factory or the static one.

        Resolved per delegation rather than baked onto an agent at construction,
        which is what lets a dynamically created specialist stream: the library
        builds that agent itself, so there is no instance for the application to
        attach a handler to.
        """
        if self._event_stream_handler_factory is not None:
            return self._event_stream_handler_factory(ctx, config, task_id)
        return self._event_stream_handler

    async def _execute(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        subagent: CompiledSubAgent,
        description: str,
        *,
        mode: ExecutionMode,
        priority: TaskPriority,
        complexity: Literal["simple", "moderate", "complex"] | None,
        requires_user_context: bool,
        may_need_clarification: bool,
        inject_ask_parent: bool = False,
        task_id: str | None = None,
        chat_trace_id: str | None = None,
        persist_chat_trace: bool = True,
    ) -> str:
        """Run a compiled subagent with chat-trace and observability support.

        `persist_chat_trace` must be `False` for a subagent the orchestrator cannot
        address by name later -- a one-shot specialist. A chat trace is only worth
        handing out if `task` can resolve the subagent it belongs to, and storing
        one costs a slot in the chat-trace LRU that a continuable conversation
        would otherwise keep.
        """
        config = subagent.config
        agent = subagent.agent
        if agent is None:
            return f"Error: Subagent '{subagent.name}' is not properly initialized"

        resolved_usage_limits = (
            self._usage_limits(ctx, config) if callable(self._usage_limits) else self._usage_limits
        )

        subagent_deps = ctx.deps.clone_for_subagent(self._max_nesting_depth - 1)

        runtime_toolsets: list[AbstractToolset[Any]] | None = None
        if self._toolsets_factory or inject_ask_parent:
            runtime_toolsets = []
            if inject_ask_parent and config.get("can_ask_questions", True):
                runtime_toolsets.append(_create_ask_parent_toolset())
            if self._toolsets_factory:
                runtime_toolsets.extend(self._toolsets_factory(subagent_deps))

        actual_task_id = task_id or uuid.uuid4().hex[:8]
        # After the task id exists, because that is the argument that makes a
        # fan-out readable: the events themselves carry nothing to tell three
        # concurrent specialists apart.
        resolved_event_stream_handler = self._resolve_event_stream_handler(
            ctx, config, actual_task_id
        )
        effective_chat_trace_id = chat_trace_id or uuid.uuid4().hex
        trace_key: ChatTraceKey = (config["name"], effective_chat_trace_id)

        unknown_trace = (
            f"Error: no saved conversation for chat_trace_id '{chat_trace_id}' "
            f"with subagent '{config['name']}' (unknown, evicted, or its first "
            f"run failed). Omit chat_trace_id to start a new conversation."
        )
        # A trace owned by another run has to read exactly like an unknown one, and
        # be refused before the "already running" branch below -- that branch would
        # otherwise confirm the id exists.
        if not self._chat_traces.owned_by(trace_key, ctx.run_id):
            return unknown_trace
        if self._chat_traces.is_active(trace_key):
            return (
                f"Error: chat trace '{effective_chat_trace_id}' already has a running "
                f"task on subagent '{config['name']}'. Wait for it to finish "
                f"(check_task/wait_tasks) before continuing this conversation."
            )
        message_history = self._chat_traces.history_for(trace_key)
        if message_history is None and chat_trace_id is not None:
            return unknown_trace

        def save_message_history(messages: list[Any]) -> None:
            self._chat_traces.save(trace_key, messages)

        # A one-shot run exposes no chat trace, so it neither stores history nor
        # reports an id: `handle.chat_trace_id` stays `None` and check_task /
        # wait_tasks skip it for the same reason the returned text does.
        on_history = save_message_history if persist_chat_trace else None
        reported_chat_trace_id = effective_chat_trace_id if persist_chat_trace else None

        self._evict_finished_handles()

        if mode == "auto":
            characteristics = TaskCharacteristics(
                estimated_complexity=complexity or config.get("typical_complexity", "moderate"),
                requires_user_context=requires_user_context
                or config.get("typically_needs_context", False),
                may_need_clarification=may_need_clarification,
            )
            resolved_mode = decide_execution_mode(characteristics, config)
        else:
            resolved_mode = mode

        if resolved_mode == "sync":
            handle = TaskHandle(
                task_id=actual_task_id,
                subagent_name=config["name"],
                description=description,
                status=TaskStatus.RUNNING,
                priority=priority,
                chat_trace_id=reported_chat_trace_id,
                started_at=utcnow(),
                parent_run_id=ctx.run_id,
            )
            self.task_manager.handles[actual_task_id] = handle
            self._chat_traces.mark_active(trace_key, ctx.run_id)
            try:
                result = await _run_sync(
                    agent=agent,
                    config=config,
                    description=description,
                    deps=subagent_deps,
                    task_id=actual_task_id,
                    extra_toolsets=runtime_toolsets,
                    ask_user=self._ask_user,
                    usage_limits=resolved_usage_limits,
                    handle=handle,
                    message_history=message_history,
                    on_message_history=on_history,
                    ask_timeout_seconds=self._ask_timeout_seconds,
                    contain_errors=config.get("contain_errors", self._contain_errors),
                    event_stream_handler=resolved_event_stream_handler,
                )
            finally:
                self._chat_traces.release(trace_key)
            # Don't advertise continuation when the run failed and nothing was ever
            # saved for this trace -- the chat_trace_id would resume nothing.
            if persist_chat_trace and (
                handle.status != TaskStatus.FAILED or trace_key in self._chat_traces
            ):
                return _format_chat_trace_result(result, effective_chat_trace_id)
            return result

        self._chat_traces.mark_active(trace_key, ctx.run_id)
        try:
            return await _run_async(
                agent=agent,
                config=config,
                description=description,
                deps=subagent_deps,
                task_id=actual_task_id,
                task_manager=self.task_manager,
                message_bus=self.task_manager.message_bus,
                extra_toolsets=runtime_toolsets,
                priority=priority,
                usage_limits=resolved_usage_limits,
                chat_trace_id=reported_chat_trace_id,
                message_history=message_history,
                on_message_history=on_history,
                on_run_finished=lambda: self._chat_traces.release(trace_key),
                ask_timeout_seconds=self._ask_timeout_seconds,
                parent_run_id=ctx.run_id,
                event_stream_handler=resolved_event_stream_handler,
            )
        except BaseException:
            # `_run_async` failed before the background task took ownership.
            self._chat_traces.release(trace_key)
            raise

    # -- tools -----------------------------------------------------------------

    async def create_agent(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        name: str,
        description: str,
        instructions: str,
        model: str | None = None,
        capabilities: list[str] | None = None,
        can_ask_questions: bool = True,
    ) -> str:
        """Create and register a reusable specialized agent.

        Args:
            ctx: The run context.
            name: Unique name for the agent (letters, numbers, hyphens only).
            description: Brief description of what the agent does.
            instructions: System prompt for the agent.
            model: Model to use. Defaults to the toolset's default model.
            capabilities: Capability names to enable for the agent.
            can_ask_questions: Whether the agent can ask the parent questions.
        """
        if self.registry.exists(name):
            return f"Error: Agent '{name}' already exists"

        actual_model = model or self._default_model
        if actual_model is None:
            return self._refuse_without_model()

        result = build_dynamic_agent(
            ctx,
            name=name,
            description=description,
            instructions=instructions,
            model=actual_model,
            can_ask_questions=can_ask_questions,
            capabilities=capabilities,
            allowed_models=self._allowed_models,
            toolsets_factory=None,
            capabilities_map=self._capabilities_map,
            default_agent_factory=self._default_agent_factory,
        )
        if isinstance(result, str):
            return result
        agent, config = result

        try:
            self.registry.register(config, agent)
        except ValueError as exc:
            return f"Error: {exc}"

        caps_info = f"\nCapabilities: {', '.join(capabilities)}" if capabilities else ""
        return (
            f"Agent '{name}' created successfully.\n"
            f"Model: {actual_model}\n"
            f"Description: {description}{caps_info}\n"
            f"Use task(description, '{name}') to delegate tasks."
        )

    async def task(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        description: str,
        subagent_type: str,
        mode: ExecutionMode = "sync",
        priority: TaskPriority = TaskPriority.NORMAL,
        complexity: Literal["simple", "moderate", "complex"] | None = None,
        requires_user_context: bool = False,
        may_need_clarification: bool = False,
        chat_trace_id: str | None = None,
    ) -> str:
        """Delegate a task to a specialized subagent.

        Args:
            ctx: The run context with dependencies.
            description: Detailed description of the task to perform.
            subagent_type: Name of the subagent to use.
            mode: Execution mode - "sync" (blocking), "async" (background), or "auto".
            priority: Task priority level (for async tasks).
            complexity: Override complexity estimate ("simple", "moderate", "complex").
            requires_user_context: Whether task needs ongoing user interaction.
            may_need_clarification: Whether task might need clarifying questions.
            chat_trace_id: Optional explicit chat trace ID. When omitted, a new subagent
                conversation is created. When provided, this subagent resumes from
                the previous successful task with the same chat trace.
        """
        if subagent_type in self._compiled:
            subagent = self._compiled[subagent_type]
            # A configured subagent whose agent we built already has `ask_parent`
            # compiled in when `can_ask_questions` allows it. One that supplied its
            # own agent (`agent` or `agent_factory`) skipped that step, so it is
            # injected at run time instead -- `_execute` still gates on
            # `can_ask_questions`, so a caller-supplied agent asks only when its
            # config opts in. Without this such a subagent could never ask, whatever
            # its `can_ask_questions` said.
            inject_ask_parent = _agent_supplied_by_caller(subagent.config)
        elif (registry_subagent := self.registry.get_compiled(subagent_type)) is not None:
            subagent = registry_subagent
            inject_ask_parent = True
        else:
            # Only the configured subagents are named. The registry is shared by
            # every run of this agent, and `create_agent` names are model-authored
            # and describe the work ("invoice-parser-acme"), so enumerating them
            # told one tenant what the others were doing.
            available = ", ".join(self._compiled) or "none"
            hint = (
                " Agents created with create_agent are also addressable by their name."
                if self.registry.count()
                else ""
            )
            return f"Error: Unknown subagent '{subagent_type}'. Available: {available}.{hint}"

        return await self._execute(
            ctx,
            subagent,
            description,
            mode=mode,
            priority=priority,
            complexity=complexity,
            requires_user_context=requires_user_context,
            may_need_clarification=may_need_clarification,
            inject_ask_parent=inject_ask_parent,
            chat_trace_id=chat_trace_id,
        )

    async def delegate(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        description: str,
        instructions: str,
        name: str,
        model: str | None = None,
        capabilities: list[str] | None = None,
        can_ask_questions: bool = True,
        mode: ExecutionMode = "sync",
        priority: TaskPriority = TaskPriority.NORMAL,
        complexity: Literal["simple", "moderate", "complex"] | None = None,
        requires_user_context: bool = False,
        may_need_clarification: bool = False,
    ) -> str:
        """Create an ephemeral specialist and delegate a task to it in one call.

        Args:
            ctx: The run context.
            description: The task for the specialist to execute.
            instructions: The specialist's system prompt.
            name: Label for the specialist (letters, numbers, hyphens), used in
                logs and as `TaskHandle.subagent_name`. Naming it does not
                register it: it still cannot be reused via `task`.
            model: Model to use. Defaults to the toolset's default model.
            capabilities: Capability names to attach to the specialist.
            can_ask_questions: Whether the specialist can ask the parent questions.
            mode: Execution mode - "sync" (blocking), "async" (background), or "auto".
            priority: Task priority level (for async tasks).
            complexity: Override complexity estimate.
            requires_user_context: Whether task needs ongoing user interaction.
            may_need_clarification: Whether task might need clarifying questions.
        """
        chosen_model = model or self._default_model
        if chosen_model is None:
            return self._refuse_without_model()

        task_id = uuid.uuid4().hex[:8]
        agent_description = description[:120] or "Ephemeral specialist"

        result = build_dynamic_agent(
            ctx,
            name=name,
            description=agent_description,
            instructions=instructions,
            model=chosen_model,
            can_ask_questions=can_ask_questions,
            capabilities=capabilities,
            allowed_models=self._allowed_models,
            toolsets_factory=None,
            capabilities_map=self._capabilities_map,
            default_agent_factory=self._default_agent_factory,
        )
        if isinstance(result, str):
            return result
        agent, config = result

        return await self._execute(
            ctx,
            CompiledSubAgent(
                name=name,
                description=agent_description,
                agent=agent,
                config=config,
            ),
            description,
            mode=mode,
            priority=priority,
            complexity=complexity,
            requires_user_context=requires_user_context,
            may_need_clarification=may_need_clarification,
            inject_ask_parent=True,
            task_id=task_id,
            persist_chat_trace=False,
        )

    async def check_task(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Check the status of a background task.

        Args:
            ctx: The run context.
            task_id: The task ID returned when the task was started.
        """
        handle = self._handle_for(ctx, task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        status_info = [
            f"Task: {task_id}",
            f"Subagent: {handle.subagent_name}",
            f"Status: {handle.status}",
            f"Description: {handle.description}",
        ]
        # Only advertise continuation for completed tasks -- a failed or still
        # running task has not saved this run's history yet (matches wait_tasks).
        if handle.chat_trace_id is not None and handle.status == TaskStatus.COMPLETED:
            status_info.append(f"Chat Trace ID: {handle.chat_trace_id}")

        if handle.status == TaskStatus.COMPLETED:
            status_info.append(f"Result: {handle.result}")
        elif handle.status == TaskStatus.FAILED:
            status_info.append(f"Error: {error_for_model(handle)}")
        elif handle.status == TaskStatus.WAITING_FOR_ANSWER:
            status_info.append(f"Question: {handle.pending_question}")
        elif handle.is_finished:
            # CANCELLED. Every terminal status has to report its outcome here;
            # falling through to the elapsed-time line would tell the model a
            # finished task is still running, and hide why it stopped.
            status_info.append(f"Outcome: {error_for_model(handle)}")
        elif handle.status == TaskStatus.RETRYING:
            status_info.append(f"Retry {handle.retry_count}: {error_for_model(handle)}")
        elif handle.started_at:
            elapsed = (utcnow() - handle.started_at).total_seconds()
            status_info.append(f"Running for: {elapsed:.1f}s")

        return "\n".join(status_info)

    async def answer_subagent(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
        answer: str,
    ) -> str:
        """Answer a question from a subagent.

        Args:
            ctx: The run context.
            task_id: The task ID of the waiting subagent.
            answer: Your answer to the subagent's question.
        """
        handle = self._handle_for(ctx, task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        if handle.status != TaskStatus.WAITING_FOR_ANSWER:
            return f"Error: Task '{task_id}' is not waiting for an answer (status: {handle.status})"

        if self.answer_task(task_id, answer):
            return f"Answer sent to task '{task_id}'"

        return "Error: Could not send answer - subagent is no longer waiting"

    async def send_message_to_subagent(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
        message: str,
    ) -> str:
        """Steer a running async subagent with an unprompted message.

        The message is queued for the subagent and folded into its next model
        request as an extra user instruction, so it adapts without losing
        partial progress. Works only while the task is still running.

        Args:
            ctx: The run context.
            task_id: The task ID of the running async subagent.
            message: The steering instruction to deliver.
        """
        handle = self._handle_for(ctx, task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        if not await self.steer_task(task_id, message):
            return (
                f"Error: Task '{task_id}' is not accepting messages "
                f"(status: {handle.status}). Steering only works for running "
                "async tasks."
            )

        return (
            f"Message delivered to task '{task_id}'; "
            "it will be applied on the subagent's next step."
        )

    async def list_active_tasks(self, ctx: RunContext[SubAgentDepsProtocol]) -> str:
        """List all active background tasks."""
        lines = ["Active background tasks:"]
        for tid in self.task_manager.list_active_tasks():
            handle = self._handle_for(ctx, tid)
            if handle is None:
                continue
            desc = handle.description[:50]
            lines.append(f"- {tid}: {handle.subagent_name} ({handle.status}) - {desc}...")

        if len(lines) == 1:
            return "No active background tasks."
        return "\n".join(lines)

    async def wait_tasks(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        task_ids: list[str],
        timeout: float = 300.0,
        mode: Literal["all", "any"] = "all",
    ) -> str:
        """Wait for multiple background tasks to complete.

        Args:
            ctx: The run context.
            task_ids: List of task IDs to wait for.
            timeout: Maximum seconds to wait (default 300s / 5 minutes).
            mode: `"all"` (default) waits for every task to finish.
                `"any"` returns as soon as one task reaches a terminal
                state (completed, failed, or cancelled), so the orchestrator
                can react to the first finisher without stalling on the
                slowest one.
        """
        # Scoped the same way the reporting below is. An unscoped await let one run
        # block for the full `timeout` on another run's task -- and the difference
        # between that and an id that does not exist is an existence oracle, since
        # both render as "not found".
        pending = [
            task
            for tid in task_ids
            if self._handle_for(ctx, tid) is not None
            and (task := self.task_manager.tasks.get(tid)) is not None
            and not task.done()
        ]
        if pending:
            # Both modes route through `asyncio.wait`. Unlike
            # `asyncio.wait_for(asyncio.gather(...))`, `asyncio.wait` does *not*
            # cascade cancellation to its constituent tasks -- neither on timeout
            # nor when its caller is cancelled (e.g. pydantic-ai's `_call_tools`
            # sibling-cancel hitting this tool call). Workers keep owning their
            # lifecycle, which is what an orchestrator expects.
            return_when = asyncio.FIRST_COMPLETED if mode == "any" else asyncio.ALL_COMPLETED
            await asyncio.wait(pending, timeout=timeout, return_when=return_when)

        lines: list[str] = []
        finished_count = 0
        missing_count = 0
        for tid in task_ids:
            handle = self._handle_for(ctx, tid)
            if handle is None:
                missing_count += 1
                lines.append(f"- {tid}: not found")
                continue
            if handle.status == TaskStatus.COMPLETED:
                finished_count += 1
                preview = _preview_result(handle.result or "", tid, self._max_result_chars)
                trace_line = (
                    f"Chat Trace ID: {handle.chat_trace_id}\n"
                    if handle.chat_trace_id is not None
                    else ""
                )
                lines.append(f"- {tid} ({handle.subagent_name}): COMPLETED\n{trace_line}{preview}")
            elif handle.is_finished:
                finished_count += 1
                lines.append(
                    f"- {tid} ({handle.subagent_name}): "
                    f"{handle.status.value.upper()} - {error_for_model(handle)}"
                )
            else:
                lines.append(f"- {tid} ({handle.subagent_name}): {handle.status}")

        total = len(task_ids)
        header_parts = [f"mode={mode}", f"{finished_count}/{total} finished"]
        # A missing id is neither finished nor running. Folding it into the running
        # count told the orchestrator, in the same message that said "not found",
        # that the task was still going -- so it kept polling an id that never
        # resolves.
        running = total - finished_count - missing_count
        if running > 0:
            header_parts.append(f"{running} still running")
        if missing_count > 0:
            header_parts.append(f"{missing_count} not found")

        return f"Task results ({', '.join(header_parts)}):\n" + "\n\n".join(lines)

    async def soft_cancel_task(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Request cooperative cancellation of a background task.

        Args:
            ctx: The run context.
            task_id: The task to cancel.
        """
        handle = self._handle_for(ctx, task_id)
        if self._cancel_reads_as_missing(task_id, handle):
            return f"Error: Task '{task_id}' not found"
        if await self.task_manager.soft_cancel(task_id):
            return f"Cancellation requested for task '{task_id}'"
        return _already_finished(task_id, handle)

    async def hard_cancel_task(
        self,
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Immediately cancel a background task.

        Args:
            ctx: The run context.
            task_id: The task to cancel.
        """
        handle = self._handle_for(ctx, task_id)
        if self._cancel_reads_as_missing(task_id, handle):
            return f"Error: Task '{task_id}' not found"
        if await self.task_manager.hard_cancel(task_id):
            return f"Task '{task_id}' has been cancelled"
        return _already_finished(task_id, handle)


def create_subagent_toolset(
    subagents: list[SubAgentConfig] | None = None,
    default_model: str | Model | None = None,
    toolsets_factory: ToolsetFactory | None = None,
    include_general_purpose: bool = True,
    max_nesting_depth: int = 0,
    id: str | None = None,
    registry: DynamicAgentRegistry | None = None,
    descriptions: dict[str, str] | None = None,
    ask_user: AskUserCallback | None = None,
    usage_limits: UsageLimits | UsageLimitsFactory | None = None,
    delegation_configuration: DelegationConfiguration = "default",
    allowed_models: list[str] | None = None,
    capabilities_map: dict[str, CapabilityFactory] | None = None,
    default_agent_factory: AgentFactory | None = None,
    max_agents: int = 10,
    max_chat_traces: int = 100,
    max_task_handles: int = 500,
    max_result_chars: int | None = 2000,
    ask_timeout_seconds: float = DEFAULT_ASK_TIMEOUT_SECONDS,
    contain_errors: bool = True,
    event_stream_handler: EventStreamHandler[Any] | None = None,
    event_stream_handler_factory: EventStreamHandlerFactory | None = None,
    cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS,
) -> SubAgentToolset:
    """Create a toolset for delegating tasks to subagents.

    This is the main entry point for using the subagent system. It creates
    a toolset with tools for:

    - `create_agent`: Create a reusable, registry-backed specialist (opt-in modes)
    - `task`: Delegate a task to a configured or registry-backed specialist
    - `delegate`: Create and run an ephemeral specialist in one call (opt-in modes)
    - `check_task`: Check status of an async task
    - `answer_subagent`: Answer a question from a subagent
    - `send_message_to_subagent`: Steer a running async subagent
    - `list_active_tasks`: List all running background tasks
    - `wait_tasks`: Wait for one or more background tasks to finish
    - `soft_cancel_task`: Request cooperative cancellation
    - `hard_cancel_task`: Immediately cancel a task

    Args:
        subagents: List of subagent configurations. If None, only
            general-purpose subagent will be available.
        default_model: Default model for subagents that don't specify one, for the
            general-purpose subagent, and for a `create_agent` or `delegate` call
            that names no model. There is **no** implicit default: leave it unset
            and anything that would have relied on one is refused instead, rather
            than running on a model the library picked and therefore on whatever
            provider credential the process environment happens to hold.
        toolsets_factory: Factory function that creates toolsets for subagents.
            Called with deps when running a task.
        include_general_purpose: Whether to include the default general-purpose
            subagent. Set to False if you want only specialized subagents.
            Needs `default_model` or `default_agent_factory` to build it from --
            see `Raises`. When a `default_agent_factory` is given it is what
            builds this delegate, so a consumer resolving a model, a credential
            and a budget per caller gets one it can account for.
        max_nesting_depth: Depth budget handed to `deps.clone_for_subagent`, which
            decides what a subagent may delegate to in turn. This library does not
            itself stop a nested toolset from delegating further -- the gate is
            whatever `toolsets_factory` gives the child.
        id: Optional toolset ID. Defaults to "subagents".
        descriptions: Optional mapping of tool name to custom description.
            Keys are tool names (`task`, `create_agent`, `delegate`, `check_task`,
            `answer_subagent`, `send_message_to_subagent`, `list_active_tasks`,
            `wait_tasks`, `soft_cancel_task`, `hard_cancel_task`).
            When provided, the custom description replaces the built-in default.
        ask_user: Optional callback invoked when a subagent calls `ask_parent`
            in sync mode. Receives the question and must return the answer.
            Required for sync-mode subagents with `can_ask_questions=True`;
            without it the subagent gets a configuration error. In async mode
            the parent answers via `answer_subagent` instead.
        usage_limits: Optional pydantic-ai usage limits for delegated subagent
            runs. Pass a `UsageLimits` instance to reuse the same limits for
            every task, or a factory called once per task with the parent run
            context and selected subagent config. A factory may return `None`
            to run that task without explicit limits. Limits are honoured on
            every retry attempt as well.
        delegation_configuration: Controls the delegation entry points:
            `"default"` exposes `task` only (backward-compatible);
            `"persisted"` exposes `create_agent` and `task`;
            `"persisted_and_oneshot"` also exposes `delegate`;
            `"oneshot_only"` exposes only `delegate`.
            Async task lifecycle tools remain available in every mode.
            Custom factories that already attach `ask_parent` should not do so;
            the toolset injects it at run time for registry-backed agents.
            A mode that hides a tool rejects that tool's configuration rather
            than ignoring it — see `Raises`.
            `"persisted"` and `"persisted_and_oneshot"` expose a `create_agent`
            tool of their own, so they cannot be combined with
            `create_agent_factory_toolset` on one agent (pydantic-ai rejects the
            duplicate tool name). Use `"default"` and let the factory toolset own
            agent creation when you also want its `remove_agent`.
        allowed_models: Optional model allow-list for dynamically created specialists.
        capabilities_map: Optional capability factories for dynamically created
            specialists.
        default_agent_factory: Optional custom agent factory for dynamically
            created specialists, and for the general-purpose subagent when
            `include_general_purpose` is on. When set, requested `capabilities`
            are rejected. Do not attach an `ask_parent` toolset in the factory;
            the toolset injects it at run time when needed.
        max_agents: Maximum number of persistent dynamic agents, applied to the
            registry this toolset creates for `create_agent`. `0` rejects every
            `create_agent` call. Ignored when `registry` is passed — that registry
            keeps its own `max_agents`.
        max_chat_traces: Maximum number of chat traces (subagent conversations)
            whose message history is kept in memory for continuation via
            `chat_trace_id`. Least-recently-used traces are evicted past this
            limit; continuing an evicted trace returns an error. Bounds memory
            in long-lived sessions.
        max_task_handles: Maximum number of finished (completed/failed/cancelled)
            task handles retained for status queries and observability. The
            oldest finished handles are evicted past this limit; their token
            usage is folded into `get_total_usage()` totals so aggregates stay
            correct. Bounds memory in long-lived sessions.
        max_result_chars: Character budget for a completed task's result in the
            `wait_tasks` listing, keeping a fan-out of verbose subagents from
            flooding the orchestrator's context. Results past the budget are cut
            and carry an explicit marker pointing at `check_task`, which always
            returns the full text. Pass `None` to never truncate.
        ask_timeout_seconds: How long `ask_parent` waits for the parent's answer
            before telling the subagent to proceed on its own.
        contain_errors: Whether an unexpected subagent crash is converted into a
            `ModelRetry` for the parent instead of aborting the parent run.
            Defaults to `True`. Control-flow signals (`CallDeferred`,
            `ApprovalRequired`, `Skip*`), `UserError`, and `UsageLimitExceeded`
            always propagate. Individual subagents can
            override this with `SubAgentConfig["contain_errors"]`.
        event_stream_handler: Streams every delegation's events -- model text,
            thinking, tool calls and their results -- as they happen, so an
            application can show what a specialist is doing rather than a
            spinner. Applies to dynamically created specialists too, which the
            library builds itself and which therefore cannot carry a handler of
            their own. An agent passed in as `SubAgentConfig["agent"]` with its
            own `event_stream_handler` keeps it: the specific choice wins and
            this is the default for everything else.
        event_stream_handler_factory: The same, resolved per delegation from the
            parent run context, the subagent config and the task id. Use it when
            the handler has to label its events -- a fan-out of three specialists
            streaming into one callback is otherwise indistinguishable. Mutually
            exclusive with `event_stream_handler`.
        cancel_grace_seconds: How long `cancel_all` waits for a cancelled
            background task to unwind before logging it and moving on. Bounded
            because the wait happens in the finalizer of the parent run: a
            subagent that swallows `CancelledError` would otherwise hold the
            whole run's teardown open.

    Returns:
        A `SubAgentToolset` configured with the subagent management tools.

    Raises:
        ValueError: If `include_general_purpose` is on (and `task` exposed) with
            neither `default_model` nor `default_agent_factory` to build that
            delegate from; if a `subagents` entry names no `model` and supplies no
            `agent` or `agent_factory` while `default_model` is unset; if
            `max_result_chars` or `max_agents` is negative; if
            `ask_timeout_seconds` or `cancel_grace_seconds` is not positive; if
            `max_chat_traces` or
            `max_task_handles` is below 1; if `delegation_configuration` is invalid; if
            `"oneshot_only"` is combined with `subagents` or a `registry`, neither
            of which is reachable without `task`; if a mode exposing neither
            `create_agent` nor `delegate` is given `allowed_models`,
            `capabilities_map`, or `default_agent_factory`, which only those
            tools consult; or if both `event_stream_handler` and
            `event_stream_handler_factory` are given.

    Example:
        ```python
        from pydantic_ai import Agent
        from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

        subagents = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
            ),
        ]

        toolset = create_subagent_toolset(
            subagents=subagents,
            default_model="openai:gpt-4.1",
        )

        agent = Agent("openai:gpt-4.1", toolsets=[toolset])
        ```
    """
    return SubAgentToolset(
        subagents=subagents,
        default_model=default_model,
        toolsets_factory=toolsets_factory,
        include_general_purpose=include_general_purpose,
        max_nesting_depth=max_nesting_depth,
        id=id,
        registry=registry,
        descriptions=descriptions,
        ask_user=ask_user,
        usage_limits=usage_limits,
        delegation_configuration=delegation_configuration,
        allowed_models=allowed_models,
        capabilities_map=capabilities_map,
        default_agent_factory=default_agent_factory,
        max_agents=max_agents,
        max_chat_traces=max_chat_traces,
        max_task_handles=max_task_handles,
        max_result_chars=max_result_chars,
        ask_timeout_seconds=ask_timeout_seconds,
        contain_errors=contain_errors,
        event_stream_handler=event_stream_handler,
        event_stream_handler_factory=event_stream_handler_factory,
        cancel_grace_seconds=cancel_grace_seconds,
    )
