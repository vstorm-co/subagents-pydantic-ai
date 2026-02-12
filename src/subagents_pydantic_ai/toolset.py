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
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
from subagents_pydantic_ai.prompts import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    SUBAGENT_SYSTEM_PROMPT,
    get_task_instructions_prompt,
)
from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.types import (
    AgentMessage,
    CompiledSubAgent,
    ExecutionMode,
    MessageType,
    SubAgentConfig,
    TaskCharacteristics,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    ToolsetFactory,
    decide_execution_mode,
)


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
    default_model: str,
) -> CompiledSubAgent:
    """Compile a subagent configuration into a ready-to-use agent.

    Args:
        config: The subagent configuration.
        default_model: Default model to use if not specified in config.

    Returns:
        CompiledSubAgent with agent instance.
    """
    model = config.get("model", default_model)

    # Build toolsets list
    toolsets: list[Any] = []

    # Add ask_parent tool for sync mode communication
    ask_parent_toolset = _create_ask_parent_toolset()
    toolsets.append(ask_parent_toolset)

    # Add custom toolsets from config
    if config.get("toolsets"):
        toolsets.extend(config["toolsets"])

    # Note: toolsets_factory will be called at runtime with deps

    # Get additional agent kwargs (e.g., builtin_tools)
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
        # This tool is a placeholder - actual implementation depends on
        # whether we're in sync or async mode
        # In sync mode: will be replaced with direct callback
        # In async mode: will use message bus
        state = getattr(ctx, "_subagent_state", None)
        if state is None:
            return "Error: Cannot ask parent - no communication channel available"

        ask_callback = state.get("ask_callback")
        if ask_callback:
            result: str = await ask_callback(question)
            return result

        message_bus = state.get("message_bus")
        task_id = state.get("task_id")
        parent_id = state.get("parent_id")
        agent_id = state.get("agent_id")

        if message_bus and task_id and parent_id:
            try:
                response = await message_bus.ask(
                    sender=agent_id,
                    receiver=parent_id,
                    question=question,
                    task_id=task_id,
                    timeout=300.0,  # 5 minute timeout for async questions
                )
                return str(response.payload)
            except asyncio.TimeoutError:
                return "Error: Parent did not respond in time"
            except KeyError:
                return "Error: Parent is not available"

        return "Error: Cannot ask parent - no communication channel configured"

    return toolset


def create_subagent_toolset(
    subagents: list[SubAgentConfig] | None = None,
    default_model: str = "openai:gpt-4.1",
    toolsets_factory: ToolsetFactory | None = None,
    include_general_purpose: bool = True,
    max_nesting_depth: int = 0,
    id: str | None = None,
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

    # Build available subagents description for tool docstring
    subagent_list = "\n".join(f"- {name}: {c.description}" for name, c in compiled.items())

    toolset: FunctionToolset[Any] = FunctionToolset(id=id or "subagents")

    @toolset.tool
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
        f"""Delegate a task to a specialized subagent.

        Choose the appropriate subagent_type based on the task requirements.
        The task will be executed according to the specified mode.

        Available subagents:
        {subagent_list}

        Args:
            ctx: The run context with dependencies.
            description: Detailed description of the task to perform.
            subagent_type: Name of the subagent to use.
            mode: Execution mode - "sync" (blocking), "async" (background), or "auto".
            priority: Task priority level (for async tasks).
            complexity: Override complexity estimate ("simple", "moderate", "complex").
            requires_user_context: Whether task needs ongoing user interaction.
            may_need_clarification: Whether task might need clarifying questions.

        Returns:
            In sync mode: The subagent's response.
            In async mode: Task handle information with task_id.
        """
        # Validate subagent_type
        if subagent_type not in compiled:
            available = ", ".join(compiled.keys())
            return f"Error: Unknown subagent '{subagent_type}'. Available: {available}"

        subagent = compiled[subagent_type]
        config = subagent.config
        agent = subagent.agent

        if agent is None:
            return f"Error: Subagent '{subagent_type}' is not properly initialized"

        # Create deps for subagent
        parent_deps = ctx.deps
        subagent_deps = parent_deps.clone_for_subagent(max_nesting_depth - 1)

        # Collect toolsets from factory if provided
        additional_toolsets = None
        if toolsets_factory:
            additional_toolsets = toolsets_factory(subagent_deps)

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
                additional_toolsets=additional_toolsets,
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
                priority=priority,
                additional_toolsets=additional_toolsets,
            )

    @toolset.tool
    async def check_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Check the status of a background task.

        Use this to check if an async task has completed and get its result.

        Args:
            ctx: The run context.
            task_id: The task ID returned when the task was started.

        Returns:
            Status information and result if completed.
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
        elif handle.status == TaskStatus.FAILED:
            status_info.append(f"Error: {handle.error}")
        elif handle.status == TaskStatus.WAITING_FOR_ANSWER:
            status_info.append(f"Question: {handle.pending_question}")
        elif handle.started_at:
            elapsed = (datetime.now() - handle.started_at).total_seconds()
            status_info.append(f"Running for: {elapsed:.1f}s")

        return "\n".join(status_info)

    @toolset.tool
    async def answer_subagent(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
        answer: str,
    ) -> str:
        """Answer a question from a subagent.

        When a background task is waiting for an answer (status: WAITING_FOR_ANSWER),
        use this tool to provide the requested information.

        Args:
            ctx: The run context.
            task_id: The task ID of the waiting subagent.
            answer: Your answer to the subagent's question.

        Returns:
            Confirmation that the answer was sent.
        """
        handle = task_manager.get_handle(task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        if handle.status != TaskStatus.WAITING_FOR_ANSWER:
            return f"Error: Task '{task_id}' is not waiting for an answer (status: {handle.status})"

        # Find the pending question message and answer it
        try:
            # Create answer message
            answer_msg = AgentMessage(
                type=MessageType.ANSWER,
                sender="parent",
                receiver=handle.subagent_name,
                payload=answer,
                task_id=task_id,
            )
            await message_bus.send(answer_msg)

            # Update handle status
            handle.status = TaskStatus.RUNNING
            handle.pending_question = None

            return f"Answer sent to task '{task_id}'"
        except KeyError:
            return "Error: Could not send answer - subagent not available"

    @toolset.tool
    async def list_active_tasks(
        ctx: RunContext[SubAgentDepsProtocol],
    ) -> str:
        """List all active background tasks.

        Returns:
            List of active task IDs and their status.
        """
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

    @toolset.tool
    async def soft_cancel_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Request cooperative cancellation of a background task.

        The subagent will be notified and can clean up before stopping.
        Use this for graceful cancellation.

        Args:
            ctx: The run context.
            task_id: The task to cancel.

        Returns:
            Confirmation or error message.
        """
        success = await task_manager.soft_cancel(task_id)
        if success:
            return f"Cancellation requested for task '{task_id}'"
        return f"Error: Task '{task_id}' not found"

    @toolset.tool
    async def hard_cancel_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Immediately cancel a background task.

        The task will be forcefully stopped. Use this only when soft
        cancellation doesn't work or immediate stopping is required.

        Args:
            ctx: The run context.
            task_id: The task to cancel.

        Returns:
            Confirmation or error message.
        """
        success = await task_manager.hard_cancel(task_id)
        if success:
            return f"Task '{task_id}' has been cancelled"
        return f"Error: Task '{task_id}' not found"

    return toolset


async def _run_sync(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    additional_toolsets: list[Any] | None = None,
) -> str:
    """Run a subagent task synchronously (blocking).

    Args:
        agent: The pydantic-ai Agent instance.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        additional_toolsets: Optional runtime toolsets to pass to agent.run().

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

    try:
        result = await agent.run(prompt, deps=deps, toolsets=additional_toolsets)
        return str(result.output)
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
    additional_toolsets: list[Any] | None = None,
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
        additional_toolsets: Optional runtime toolsets to pass to agent.run().

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

    async def run_task() -> None:
        """Execute the task and update handle."""
        can_ask = config.get("can_ask_questions", True)
        max_questions = config.get("max_questions")

        prompt = get_task_instructions_prompt(
            description,
            can_ask_questions=can_ask,
            max_questions=max_questions,
        )

        try:
            result = await agent.run(prompt, deps=deps, toolsets=additional_toolsets)
            handle.result = str(result.output)
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
