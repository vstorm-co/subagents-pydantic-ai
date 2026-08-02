"""Message bus and background-task bookkeeping for subagent delegation.

`TaskManager` owns the lifecycle of background (async-mode) delegations:
the `asyncio.Task` running each one, its `TaskHandle`, its cancellation event,
and the future a blocked `ask_parent` is waiting on.

`InMemoryMessageBus` carries parent-to-child steering messages. Its
request-response half (`ask`, `answer`, handlers) is not used by the toolset; it
exists as an extension point for applications that want to drive the bus
themselves, or to back a different transport behind `MessageBusProtocol`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from subagents_pydantic_ai.types import AgentMessage, MessageType, TaskHandle, TaskStatus, utcnow

logger = logging.getLogger(__name__)

DEFAULT_CANCEL_GRACE_SECONDS = 5.0
"""How long a cancelled background task is given to unwind before it is left behind.

Long enough for a real `finally` -- closing a client, flushing a span -- and short
enough that a subagent which ignores cancellation cannot hold a parent run's
teardown open. See `TaskManager.cancel_all`.
"""


@dataclass
class InMemoryMessageBus:
    """In-memory message bus using asyncio queues.

    This is the default message bus implementation, suitable for
    single-process applications. For distributed systems, consider
    implementing a Redis-based bus using the MessageBusProtocol.

    Example:
        ```python
        bus = InMemoryMessageBus()

        # Register agents
        parent_queue = bus.register_agent("parent")
        worker_queue = bus.register_agent("worker-1")

        # Send a message
        await bus.send(AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="parent",
            receiver="worker-1",
            payload={"task": "analyze data"},
            task_id="task-123",
        ))

        # Worker receives message
        msg = await worker_queue.get()
        ```
    """

    _queues: dict[str, asyncio.Queue[AgentMessage]] = field(default_factory=dict)
    _pending_questions: dict[str, asyncio.Future[AgentMessage]] = field(default_factory=dict)
    _handlers: list[Callable[[AgentMessage], Awaitable[None]]] = field(default_factory=list)

    async def send(self, message: AgentMessage) -> None:
        """Send a message to a specific agent.

        Args:
            message: The message to send.

        Raises:
            KeyError: If the receiver is not registered.
        """
        if message.receiver not in self._queues:
            raise KeyError(f"Agent '{message.receiver}' is not registered")

        await self._queues[message.receiver].put(message)

        for handler in self._handlers:
            # Handlers are observers (logging, tracing). One that raises must not
            # stop delivery to the others or fail the send, but it is logged
            # rather than discarded.
            try:
                await handler(message)
            except Exception as e:
                logger.warning("Message bus handler %r failed: %s", handler, e)

    async def ask(
        self,
        sender: str,
        receiver: str,
        question: Any,
        task_id: str,
        timeout: float = 30.0,
    ) -> AgentMessage:
        """Send a question and wait for a response.

        Args:
            sender: ID of the asking agent.
            receiver: ID of the agent to ask.
            question: The question payload.
            task_id: Task ID for correlation.
            timeout: Maximum time to wait in seconds.

        Returns:
            The response message.

        Raises:
            asyncio.TimeoutError: If no response within timeout.
            KeyError: If the receiver is not registered.
        """
        if receiver not in self._queues:
            raise KeyError(f"Agent '{receiver}' is not registered")

        correlation_id = str(uuid.uuid4())

        response_future: asyncio.Future[AgentMessage] = asyncio.get_running_loop().create_future()
        self._pending_questions[correlation_id] = response_future

        try:
            message = AgentMessage(
                type=MessageType.QUESTION,
                sender=sender,
                receiver=receiver,
                payload=question,
                task_id=task_id,
                correlation_id=correlation_id,
            )
            await self.send(message)

            return await asyncio.wait_for(response_future, timeout=timeout)
        finally:
            self._pending_questions.pop(correlation_id, None)

    async def answer(self, original: AgentMessage, answer: Any) -> None:
        """Answer a previously received question.

        Args:
            original: The original question message.
            answer: The answer payload.

        Raises:
            KeyError: If the original sender is not registered or
                     if there's no pending question with the correlation_id.
        """
        if original.sender not in self._queues:
            raise KeyError(f"Agent '{original.sender}' is not registered")

        response = AgentMessage(
            type=MessageType.ANSWER,
            sender=original.receiver,  # We are the original receiver
            receiver=original.sender,  # Send back to original sender
            payload=answer,
            task_id=original.task_id,
            correlation_id=original.correlation_id,
        )

        if original.correlation_id and original.correlation_id in self._pending_questions:
            future = self._pending_questions[original.correlation_id]
            if not future.done():
                future.set_result(response)
        else:
            await self.send(response)

    def register_agent(self, agent_id: str) -> asyncio.Queue[AgentMessage]:
        """Register an agent to receive messages.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            A queue where messages for this agent will be delivered.

        Raises:
            ValueError: If agent_id is already registered.
        """
        if agent_id in self._queues:
            raise ValueError(f"Agent '{agent_id}' is already registered")

        queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._queues[agent_id] = queue
        return queue

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message bus.

        Args:
            agent_id: The agent to unregister.
        """
        self._queues.pop(agent_id, None)

    def add_handler(self, handler: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """Add a message handler for debugging/logging.

        Handlers are called for every message sent through the bus.

        Args:
            handler: Async function that receives messages.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """Remove a previously added handler.

        Args:
            handler: The handler to remove.
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    def is_registered(self, agent_id: str) -> bool:
        """Check if an agent is registered.

        Args:
            agent_id: The agent ID to check.

        Returns:
            True if the agent is registered, False otherwise.
        """
        return agent_id in self._queues

    def registered_agents(self) -> list[str]:
        """Get list of registered agent IDs.

        Returns:
            List of registered agent IDs.
        """
        return list(self._queues.keys())

    async def get_messages(
        self,
        agent_id: str,
        timeout: float = 0.0,
    ) -> list[AgentMessage]:
        """Get pending messages for an agent.

        Non-blocking retrieval of all pending messages in the agent's queue.
        Optionally waits up to `timeout` seconds for at least one message.

        Args:
            agent_id: The agent to get messages for.
            timeout: Maximum time to wait for a message (0 = no wait).

        Returns:
            List of pending messages (may be empty).

        Raises:
            KeyError: If the agent is not registered.
        """
        if agent_id not in self._queues:
            raise KeyError(f"Agent '{agent_id}' is not registered")

        queue = self._queues[agent_id]
        messages: list[AgentMessage] = []

        if timeout > 0 and queue.empty():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=timeout)
                messages.append(msg)
            except asyncio.TimeoutError:
                return messages

        while not queue.empty():
            try:
                msg = queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:  # pragma: no cover
                break  # Race condition - queue was emptied between empty() check and get_nowait()

        return messages


def create_message_bus(backend: str = "memory", **kwargs: Any) -> InMemoryMessageBus:
    """Create a message bus instance.

    Factory function for creating message bus implementations.
    Currently only supports in-memory backend.

    Args:
        backend: The backend type ("memory" is currently supported).
        **kwargs: Backend-specific configuration.

    Returns:
        A message bus instance.

    Raises:
        ValueError: If the backend is not supported.

    Example:
        ```python
        # Create in-memory bus (default)
        bus = create_message_bus()

        # Future: Redis bus
        # bus = create_message_bus("redis", url="redis://localhost")
        ```
    """
    if backend == "memory":
        return InMemoryMessageBus()

    raise ValueError(f"Unknown message bus backend: {backend}")


@dataclass
class TaskManager:
    """Manages background tasks and their lifecycle.

    Tracks running tasks, handles cancellation, and provides
    status querying capabilities.

    Attributes:
        tasks: Live `asyncio.Task` per task id. An entry is removed by
            `cleanup_task` when the task finishes.
        handles: `TaskHandle` per task id, kept after the task finishes so its
            result and telemetry stay queryable.
        message_bus: Message bus used to deliver steering and cancel messages.
        cancel_grace_seconds: How long `cancel_all` waits for a cancelled task to
            unwind before logging it and moving on.
    """

    tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    handles: dict[str, TaskHandle] = field(default_factory=dict)
    message_bus: InMemoryMessageBus = field(default_factory=InMemoryMessageBus)
    cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS
    _cancel_events: dict[str, asyncio.Event] = field(default_factory=dict)
    _answer_futures: dict[str, asyncio.Future[str]] = field(default_factory=dict)
    _strong_refs: set[asyncio.Task[None]] = field(default_factory=set)

    def create_task(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, None],
        handle: TaskHandle,
    ) -> asyncio.Task[None]:
        """Create and track a new background task.

        Args:
            task_id: Unique identifier for the task.
            coro: The coroutine to run.
            handle: TaskHandle for status tracking.

        Returns:
            The created asyncio.Task.
        """
        task = asyncio.create_task(coro, name=f"subagent-{task_id}")
        self.tasks[task_id] = task
        self.handles[task_id] = handle
        self._cancel_events[task_id] = asyncio.Event()
        # `cleanup_task` drops `tasks[task_id]` from inside the task's own
        # `finally`, and the event loop only holds a weak reference, so a second
        # strong reference keeps the task alive until it is truly done.
        self._strong_refs.add(task)
        task.add_done_callback(self._strong_refs.discard)

        handle.status = TaskStatus.RUNNING
        handle.started_at = utcnow()

        return task

    def get_handle(self, task_id: str) -> TaskHandle | None:
        """Get the handle for a task.

        Args:
            task_id: The task ID.

        Returns:
            The TaskHandle if found, None otherwise.
        """
        return self.handles.get(task_id)

    def get_cancel_event(self, task_id: str) -> asyncio.Event | None:
        """Get the cancellation event for a task.

        Args:
            task_id: The task ID.

        Returns:
            The cancellation event if found, None otherwise.
        """
        return self._cancel_events.get(task_id)

    def set_answer_future(self, task_id: str, future: asyncio.Future[str]) -> None:
        """Set an answer future for a task waiting for parent response.

        Args:
            task_id: The task ID.
            future: The future to resolve when the answer arrives.
        """
        self._answer_futures[task_id] = future

    def get_answer_future(self, task_id: str) -> asyncio.Future[str] | None:
        """Get the pending answer future for a task.

        Args:
            task_id: The task ID.

        Returns:
            The answer future if set, None otherwise.
        """
        return self._answer_futures.get(task_id)

    def clear_answer_future(self, task_id: str) -> None:
        """Remove the answer future for a task.

        Args:
            task_id: The task ID.
        """
        self._answer_futures.pop(task_id, None)

    def resolve_answer(self, task_id: str, answer: str) -> bool:
        """Deliver an answer to a task blocked in `ask_parent`.

        Args:
            task_id: The task ID.
            answer: The answer to deliver.

        Returns:
            Whether a waiting `ask_parent` call was resolved.
        """
        future = self._answer_futures.get(task_id)
        if future is not None and not future.done():
            future.set_result(answer)
            return True
        return False

    async def soft_cancel(self, task_id: str) -> bool:
        """Request cooperative cancellation of a task.

        Sets a cancellation event that the run loop checks between graph nodes,
        so the subagent stops at a clean boundary with its partial progress
        intact. A task blocked in `ask_parent` sits inside a tool call rather
        than at a node boundary, so its pending question is resolved too --
        otherwise the cancel would only take effect after the ask timeout.

        Args:
            task_id: The task to cancel.

        Returns:
            True if cancellation was requested, False if task not found.
        """
        if task_id not in self._cancel_events:
            return False

        self._cancel_events[task_id].set()
        self.resolve_answer(
            task_id,
            "Your parent agent cancelled this task. Stop working and wrap up immediately.",
        )

        if task_id in self.handles:
            with contextlib.suppress(KeyError):
                # The running subagent registers on the bus as
                # `subagent-{task_id}`, not under its subagent name.
                await self.message_bus.send(
                    AgentMessage(
                        type=MessageType.CANCEL_REQUEST,
                        sender="task_manager",
                        receiver=f"subagent-{task_id}",
                        payload={"reason": "soft_cancel"},
                        task_id=task_id,
                    )
                )

        return True

    async def hard_cancel(self, task_id: str) -> bool:
        """Immediately cancel a task.

        Calls `cancel()` on the `asyncio.Task`. The task's own
        `except asyncio.CancelledError` branch records the terminal status; the
        handle is only marked here for a bare task registered without that
        wrapper. `TaskHandle.finish` makes the first terminal transition win, so
        a cancel arriving while the task is already completing cannot overwrite
        the real outcome.

        Args:
            task_id: The task to cancel.

        Returns:
            True if task was cancelled, False if task not found.
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if not task.done():
            task.cancel()
            handle = self.handles.get(task_id)
            if handle is not None:
                handle.finish(TaskStatus.CANCELLED, error="Task was cancelled")

        return True

    async def cancel_all(self, parent_run_id: str | None = None) -> None:
        """Cancel every live task and wait, up to the grace period, for it to unwind.

        Called when a parent run ends. A background delegation that outlives its
        parent keeps working against deps the application has already torn down,
        and one blocked in `ask_parent` waits for an answer that can never come.

        The wait is bounded because the caller is a `finally` block. A
        `CancelledError` can be caught, and a subagent's toolset is arbitrary
        consumer code -- an HTTP client with its own shield, a slow cleanup, a
        `contextlib.suppress` a little too wide. An unbounded wait on one of
        those never returns, so the parent run's teardown never returns either,
        and the application hangs with nothing logged. A leaked task that is
        reported is recoverable; a `finally` that never finishes is not.

        The guarantee is therefore: every matching task is cancelled, and its
        cleanup is awaited for at most `cancel_grace_seconds`.

        Args:
            parent_run_id: Only cancel tasks started by this parent run. `None`
                cancels every live task.
        """
        live: dict[asyncio.Task[None], str] = {}
        for task_id, task in list(self.tasks.items()):
            handle = self.handles.get(task_id)
            if parent_run_id is not None and (
                handle is None or handle.parent_run_id != parent_run_id
            ):
                continue
            if not task.done():
                # `cancel()` raises into whatever the task is awaiting, including a
                # future held by `ask_parent`, so there is nothing to unblock first.
                task.cancel()
                live[task] = task_id
                # A task cancelled before its coroutine started never reaches its
                # own `except asyncio.CancelledError`, so record the outcome here.
                # `finish` is idempotent, so a task that does get there wins.
                if handle is not None:
                    handle.finish(TaskStatus.CANCELLED, error="Task was cancelled")

        if not live:
            return

        # `asyncio.wait` rather than awaiting each task: it returns on timeout
        # instead of raising, and it never re-raises a task's own exception, so
        # one subagent's failure cannot stop the others from being waited on.
        cancelled = set(live)
        pending: set[asyncio.Task[None]] = cancelled
        with contextlib.suppress(asyncio.CancelledError):
            # Suppressed because this runs during the parent's own cancellation,
            # where a `CancelledError` delivered here belongs to that outer
            # cancel. Swallowing it is safe only because we stop waiting
            # immediately after; the outer cancellation continues to propagate
            # from the `finally` this was called in.
            _, pending = await asyncio.wait(cancelled, timeout=self.cancel_grace_seconds)

        for task in pending:
            logger.warning(
                "Subagent task %s did not unwind within %.1fs of being cancelled; "
                "leaving it to the event loop",
                live[task],
                self.cancel_grace_seconds,
            )

    def cleanup_task(self, task_id: str) -> None:
        """Clean up resources for a completed task.

        The handle is kept so status and telemetry stay queryable.

        Args:
            task_id: The task to clean up.
        """
        self.tasks.pop(task_id, None)
        self._cancel_events.pop(task_id, None)

    def list_active_tasks(self) -> list[str]:
        """Get list of active (non-completed) task IDs.

        Returns:
            List of task IDs for tasks that haven't completed.
        """
        return [task_id for task_id, task in self.tasks.items() if not task.done()]

    def list_handles(self) -> list[TaskHandle]:
        """Get all task handles (completed and active).

        Returns:
            List of TaskHandle objects.
        """
        return list(self.handles.values())
