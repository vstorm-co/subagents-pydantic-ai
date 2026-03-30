"""Main subagent toolset with dual-mode execution support.

This module provides the core toolset for delegating tasks to subagents.
It supports both synchronous (blocking) and asynchronous (background)
execution modes, with automatic mode selection based on task characteristics.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
from subagents_pydantic_ai.prompts import (
    ANSWER_SUBAGENT_DESCRIPTION,
    CHECK_TASK_DESCRIPTION,
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    HARD_CANCEL_TASK_DESCRIPTION,
    LIST_ACTIVE_TASKS_DESCRIPTION,
    SOFT_CANCEL_TASK_DESCRIPTION,
    SUBAGENT_SYSTEM_PROMPT,
    TASK_TOOL_DESCRIPTION,
    WAIT_TASKS_DESCRIPTION,
    get_task_instructions_prompt,
)
from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.types import (
    CompiledSubAgent,
    ExecutionMode,
    SubAgentConfig,
    TaskCharacteristics,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    ToolsetFactory,
    decide_execution_mode,
)


def _serialize_output(output: Any) -> str:
    """Serialize subagent output preserving structure for Pydantic models.

    For Pydantic models (BaseModel), returns JSON via ``model_dump_json()``.
    For dataclasses with ``__dataclass_fields__``, returns JSON via ``json.dumps``.
    For everything else, returns ``str(output)``.
    """
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()
    if hasattr(output, "__dataclass_fields__"):
        import dataclasses
        import json

        return json.dumps(dataclasses.asdict(output), default=str)
    return str(output)


def _create_general_purpose_config() -> SubAgentConfig:
    """Create the default general-purpose subagent config."""
    return SubAgentConfig(
        name="general-purpose",
        description=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
        instructions=SUBAGENT_SYSTEM_PROMPT,
        can_ask_questions=True,
    )


def _compile_subagent(
    config: SubAgentConfig,
    default_model: str | Model,
) -> CompiledSubAgent:
    """Compile a subagent configuration into a ready-to-use agent.

    Agent resolution priority:
    1. ``config["agent"]`` — pre-built agent instance, used as-is
    2. ``config["agent_factory"]`` — callable(config) -> agent
    3. Default — creates ``pydantic_ai.Agent`` from config fields

    Args:
        config: The subagent configuration.
        default_model: Default model to use if not specified in config.

    Returns:
        CompiledSubAgent with agent instance.
    """
    # 1. Pre-built agent — use as-is
    if config.get("agent") is not None:
        return CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            agent=config["agent"],
            config=config,
        )

    # 2. Agent factory — call it
    factory = config.get("agent_factory")
    if factory is not None:
        agent = factory(config)
        return CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            agent=agent,
            config=config,
        )

    # 3. Default: create plain pydantic-ai Agent
    model = config.get("model", default_model)

    toolsets: list[Any] = []
    ask_parent_toolset = _create_ask_parent_toolset()
    toolsets.append(ask_parent_toolset)

    if config.get("toolsets"):
        toolsets.extend(config["toolsets"])

    agent_kwargs = config.get("agent_kwargs", {})

    agent: Agent[Any, str] = Agent(
        model,
        system_prompt=config["instructions"],
        toolsets=toolsets,
        **agent_kwargs,
    )

    return CompiledSubAgent(
        name=config["name"],
        description=config["description"],
        agent=agent,
        config=config,
    )


def _create_ask_parent_toolset() -> FunctionToolset[Any]:
    """Create toolset with ask_parent tool for subagent communication."""
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
        # Try _subagent_state on deps (async mode)
        state = getattr(ctx.deps, "_subagent_state", None)
        if state is not None:
            ask_callback = state.get("ask_callback")
            if ask_callback:
                result: str = await ask_callback(question)
                return result

            _task_manager = state.get("task_manager")
            _task_id = state.get("task_id")

            if _task_manager and _task_id:
                handle = _task_manager.get_handle(_task_id)
                if handle is not None:
                    # Set question on handle so parent can see it via check_task
                    handle.pending_question = question
                    handle.status = TaskStatus.WAITING_FOR_ANSWER

                    # Create a future and wait for answer_subagent to resolve it
                    loop = asyncio.get_running_loop()
                    answer_future: asyncio.Future[str] = loop.create_future()
                    _task_manager.set_answer_future(_task_id, answer_future)

                    try:
                        answer = await asyncio.wait_for(answer_future, timeout=300.0)
                        handle.status = TaskStatus.RUNNING
                        handle.pending_question = None
                        return answer
                    except asyncio.TimeoutError:
                        handle.status = TaskStatus.RUNNING
                        handle.pending_question = None
                        _task_manager.clear_answer_future(_task_id)
                        return "Error: Parent did not respond in time"

        # Fallback: use deps.ask_user callback (sync mode / plan toolset)
        ask_user = getattr(ctx.deps, "ask_user", None)
        if ask_user:
            return str(await ask_user(question, []))

        return "Error: Cannot ask parent - no communication channel configured"

    return toolset


def create_subagent_toolset(  # noqa: C901
    subagents: list[SubAgentConfig] | None = None,
    default_model: str | Model = "openai:gpt-4.1",
    toolsets_factory: ToolsetFactory | None = None,
    include_general_purpose: bool = True,
    max_nesting_depth: int = 0,
    id: str | None = None,
    registry: Any | None = None,
    descriptions: dict[str, str] | None = None,
) -> FunctionToolset[Any]:
    """Create a toolset for delegating tasks to subagents.

    This is the main entry point for using the subagent system. It creates
    a toolset with tools for:
    - `task`: Delegate a task to a subagent (sync or async)
    - `check_task`: Check status of an async task
    - `answer_subagent`: Answer a question from a subagent
    - `list_active_tasks`: List all running background tasks
    - `soft_cancel_task`: Request cooperative cancellation
    - `hard_cancel_task`: Immediately cancel a task

    Args:
        subagents: List of subagent configurations. If None, only
            general-purpose subagent will be available.
        default_model: Default model for subagents that don't specify one.
        toolsets_factory: Factory function that creates toolsets for subagents.
            Called with deps when running a task.
        include_general_purpose: Whether to include the default general-purpose
            subagent. Set to False if you want only specialized subagents.
        max_nesting_depth: Maximum depth for nested subagents. 0 means
            subagents cannot spawn their own subagents.
        id: Optional toolset ID. Defaults to "subagents".
        descriptions: Optional mapping of tool name to custom description.
            Keys are tool names (task, check_task, answer_subagent,
            list_active_tasks, wait_tasks, soft_cancel_task, hard_cancel_task).
            When provided, the custom description replaces the built-in default.

    Returns:
        FunctionToolset configured with subagent management tools.

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
    _descs = descriptions or {}

    # Build subagent configs
    configs: list[SubAgentConfig] = list(subagents) if subagents else []
    if include_general_purpose:
        configs.append(_create_general_purpose_config())

    # Compile subagents
    compiled: dict[str, CompiledSubAgent] = {}
    for config in configs:
        compiled[config["name"]] = _compile_subagent(config, default_model)

    # Create shared state
    message_bus = InMemoryMessageBus()
    task_manager = TaskManager(message_bus=message_bus)

    # Build dynamic task description with available subagents
    subagent_list = "\n".join(f"- {name}: {c.description}" for name, c in compiled.items())
    task_description = _descs.get(
        "task",
        TASK_TOOL_DESCRIPTION.rstrip() + f"\n\nAvailable subagent types:\n{subagent_list}",
    )

    toolset: FunctionToolset[Any] = FunctionToolset(id=id or "subagents")

    @toolset.tool(description=task_description)
    async def task(
        ctx: RunContext[SubAgentDepsProtocol],
        description: str,
        subagent_type: str,
        mode: ExecutionMode = "sync",
        priority: TaskPriority = TaskPriority.NORMAL,
        complexity: Literal["simple", "moderate", "complex"] | None = None,
        requires_user_context: bool = False,
        may_need_clarification: bool = False,
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
        """
        # Validate subagent_type — check static compiled dict first, then dynamic registry
        if subagent_type in compiled:
            subagent = compiled[subagent_type]
        elif (
            registry is not None
            and hasattr(registry, "get_compiled")
            and registry.get_compiled(subagent_type)
        ):
            subagent = registry.get_compiled(subagent_type)
        else:
            # Build available list from both sources
            available_names = list(compiled.keys())
            if registry is not None and hasattr(registry, "list_agents"):
                available_names.extend(registry.list_agents())
            available = ", ".join(available_names)
            return f"Error: Unknown subagent '{subagent_type}'. Available: {available}"

        config = subagent.config
        agent = subagent.agent

        if agent is None:
            return f"Error: Subagent '{subagent_type}' is not properly initialized"

        # Create deps for subagent
        parent_deps = ctx.deps
        subagent_deps = parent_deps.clone_for_subagent(max_nesting_depth - 1)

        # Build runtime toolsets from factory if provided
        runtime_toolsets = toolsets_factory(subagent_deps) if toolsets_factory else None

        # Generate task ID
        task_id = str(uuid.uuid4())[:8]

        # Resolve mode if "auto"
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
            return await _run_sync(
                agent=agent,
                config=config,
                description=description,
                deps=subagent_deps,
                task_id=task_id,
                extra_toolsets=runtime_toolsets,
            )
        else:
            return await _run_async(
                agent=agent,
                config=config,
                description=description,
                deps=subagent_deps,
                task_id=task_id,
                task_manager=task_manager,
                message_bus=message_bus,
                extra_toolsets=runtime_toolsets,
                priority=priority,
            )

    @toolset.tool(description=_descs.get("check_task", CHECK_TASK_DESCRIPTION))
    async def check_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Check the status of a background task.

        Args:
            ctx: The run context.
            task_id: The task ID returned when the task was started.
        """
        handle = task_manager.get_handle(task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        status_info = [
            f"Task: {task_id}",
            f"Subagent: {handle.subagent_name}",
            f"Status: {handle.status}",
            f"Description: {handle.description}",
        ]

        if handle.status == TaskStatus.COMPLETED:
            status_info.append(f"Result: {handle.result}")
            if handle.usage is not None:
                u = handle.usage
                tokens = getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)
                status_info.append(f"Usage: {tokens} tokens ({getattr(u, 'input_tokens', 0)} in / {getattr(u, 'output_tokens', 0)} out)")
        elif handle.status == TaskStatus.FAILED:
            status_info.append(f"Error: {handle.error}")
        elif handle.status == TaskStatus.WAITING_FOR_ANSWER:
            status_info.append(f"Question: {handle.pending_question}")
        elif handle.started_at:
            elapsed = (datetime.now() - handle.started_at).total_seconds()
            status_info.append(f"Running for: {elapsed:.1f}s")

        return "\n".join(status_info)

    @toolset.tool(description=_descs.get("answer_subagent", ANSWER_SUBAGENT_DESCRIPTION))
    async def answer_subagent(
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
        handle = task_manager.get_handle(task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        if handle.status != TaskStatus.WAITING_FOR_ANSWER:
            return f"Error: Task '{task_id}' is not waiting for an answer (status: {handle.status})"

        # Resolve the answer future that ask_parent is waiting on
        future = task_manager.get_answer_future(task_id)
        if future is not None and not future.done():
            future.set_result(answer)
            return f"Answer sent to task '{task_id}'"

        return "Error: Could not send answer - subagent is no longer waiting"

    @toolset.tool(description=_descs.get("list_active_tasks", LIST_ACTIVE_TASKS_DESCRIPTION))
    async def list_active_tasks(
        ctx: RunContext[SubAgentDepsProtocol],
    ) -> str:
        """List all active background tasks."""
        active_ids = task_manager.list_active_tasks()

        if not active_ids:
            return "No active background tasks."

        lines = ["Active background tasks:"]
        for tid in active_ids:
            handle = task_manager.get_handle(tid)
            if handle:  # pragma: no branch
                desc = handle.description[:50]
                lines.append(f"- {tid}: {handle.subagent_name} ({handle.status}) - {desc}...")

        return "\n".join(lines)

    @toolset.tool(description=_descs.get("wait_tasks", WAIT_TASKS_DESCRIPTION))
    async def wait_tasks(
        ctx: RunContext[SubAgentDepsProtocol],
        task_ids: list[str],
        timeout: float = 300.0,
    ) -> str:
        """Wait for multiple background tasks to complete.

        Args:
            ctx: The run context.
            task_ids: List of task IDs to wait for.
            timeout: Maximum seconds to wait (default 300s / 5 minutes).
        """
        # Collect asyncio.Task objects for the requested task_ids
        tasks_to_await: list[tuple[str, asyncio.Task[Any]]] = []
        for tid in task_ids:
            t = task_manager.tasks.get(tid)
            if t is not None and not t.done():
                tasks_to_await.append((tid, t))

        # Wait for all with timeout
        if tasks_to_await:
            aws = [t for _, t in tasks_to_await]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*aws, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                pass  # Report what we have so far

        # Collect results
        lines: list[str] = []
        for tid in task_ids:
            handle = task_manager.get_handle(tid)
            if handle is None:
                lines.append(f"- {tid}: not found")
                continue
            status = handle.status
            if status == "completed":
                result_preview = (handle.result or "")[:2000]
                lines.append(f"- {tid} ({handle.subagent_name}): COMPLETED\n{result_preview}")
            elif status == "failed":
                lines.append(f"- {tid} ({handle.subagent_name}): FAILED - {handle.error}")
            else:
                lines.append(f"- {tid} ({handle.subagent_name}): {status}")

        return "Task results:\n" + "\n\n".join(lines)

    @toolset.tool(description=_descs.get("soft_cancel_task", SOFT_CANCEL_TASK_DESCRIPTION))
    async def soft_cancel_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Request cooperative cancellation of a background task.

        Args:
            ctx: The run context.
            task_id: The task to cancel.
        """
        success = await task_manager.soft_cancel(task_id)
        if success:
            return f"Cancellation requested for task '{task_id}'"
        return f"Error: Task '{task_id}' not found"

    @toolset.tool(description=_descs.get("hard_cancel_task", HARD_CANCEL_TASK_DESCRIPTION))
    async def hard_cancel_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Immediately cancel a background task.

        Args:
            ctx: The run context.
            task_id: The task to cancel.
        """
        success = await task_manager.hard_cancel(task_id)
        if success:
            return f"Task '{task_id}' has been cancelled"
        return f"Error: Task '{task_id}' not found"

    # Expose task_manager for external monitoring (e.g., push notifications)
    toolset.task_manager = task_manager  # type: ignore[attr-defined]

    def get_total_usage() -> dict[str, int]:
        """Get aggregate token usage across all completed subagent tasks.

        Returns dict with ``input_tokens``, ``output_tokens``, ``total_tokens``, ``requests``.
        """
        totals: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}
        for handle in task_manager.list_handles():
            if handle.usage is not None:
                totals["input_tokens"] += getattr(handle.usage, "input_tokens", 0)
                totals["output_tokens"] += getattr(handle.usage, "output_tokens", 0)
                totals["requests"] += getattr(handle.usage, "requests", 0)
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
        return totals

    toolset.get_total_usage = get_total_usage  # type: ignore[attr-defined]

    return toolset


async def _run_sync(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    extra_toolsets: list[Any] | None = None,
) -> str:
    """Run a subagent task synchronously (blocking).

    Args:
        agent: The pydantic-ai Agent instance.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        extra_toolsets: Additional toolsets to pass to agent.run().

    Returns:
        The subagent's response.
    """
    can_ask = config.get("can_ask_questions", True)
    max_questions = config.get("max_questions")

    prompt = get_task_instructions_prompt(
        description,
        can_ask_questions=can_ask,
        max_questions=max_questions,
    )

    run_kwargs: dict[str, Any] = {"deps": deps}
    if extra_toolsets:
        run_kwargs["toolsets"] = extra_toolsets

    try:
        result = await agent.run(prompt, **run_kwargs)
        return _serialize_output(result.output)
    except Exception as e:
        return f"Error executing task: {e}"


async def _run_async(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    task_manager: TaskManager,
    message_bus: InMemoryMessageBus,
    priority: TaskPriority = TaskPriority.NORMAL,
    extra_toolsets: list[Any] | None = None,
) -> str:
    """Run a subagent task asynchronously (background).

    Args:
        agent: The pydantic-ai Agent instance.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        task_manager: Task manager for tracking.
        message_bus: Message bus for communication.
        priority: Task priority level.
        extra_toolsets: Additional toolsets to pass to agent.run().

    Returns:
        Task handle information as string.
    """
    # Create task handle
    handle = TaskHandle(
        task_id=task_id,
        subagent_name=config["name"],
        description=description,
        status=TaskStatus.PENDING,
        priority=priority,
    )

    # Register subagent for messaging
    agent_id = f"subagent-{task_id}"
    try:
        message_bus.register_agent(agent_id)
    except ValueError:
        pass  # Already registered

    # Inject _subagent_state on deps so ask_parent can communicate with parent
    deps._subagent_state = {
        "task_manager": task_manager,
        "task_id": task_id,
    }

    async def run_task() -> None:
        """Execute the task and update handle."""
        can_ask = config.get("can_ask_questions", True)
        max_questions = config.get("max_questions")

        prompt = get_task_instructions_prompt(
            description,
            can_ask_questions=can_ask,
            max_questions=max_questions,
        )

        run_kwargs: dict[str, Any] = {"deps": deps}
        if extra_toolsets:
            run_kwargs["toolsets"] = extra_toolsets

        try:
            result = await agent.run(prompt, **run_kwargs)
            handle.result = _serialize_output(result.output)
            if hasattr(result, "usage"):
                handle.usage = result.usage()
            handle.status = TaskStatus.COMPLETED
        except asyncio.CancelledError:
            handle.status = TaskStatus.CANCELLED
            handle.error = "Task was cancelled"
        except Exception as e:
            handle.status = TaskStatus.FAILED
            handle.error = str(e)
        finally:
            handle.completed_at = datetime.now()
            message_bus.unregister_agent(agent_id)
            task_manager.clear_answer_future(task_id)
            task_manager.cleanup_task(task_id)

    # Create the background task
    task_manager.create_task(task_id, run_task(), handle)

    return (
        f"Task started in background.\n"
        f"Task ID: {task_id}\n"
        f"Subagent: {config['name']}\n"
        f"Use check_task('{task_id}') to check status."
    )


# Alias for backwards compatibility
SubAgentToolset = create_subagent_toolset
