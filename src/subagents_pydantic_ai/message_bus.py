"""Message bus implementations for inter-agent communication.

This module provides message bus implementations that enable communication
between parent agents and subagents. The default implementation uses
asyncio queues for in-memory communication.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from subagents_pydantic_ai.types import AgentMessage, MessageType


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

        # Notify handlers
        for handler in self._handlers:
            try:
                await handler(message)
            except Exception:  # pragma: no cover
                pass  # Don't let handler errors break message delivery

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

        # Create future for the response
        loop = asyncio.get_event_loop()
        response_future: asyncio.Future[AgentMessage] = loop.create_future()
        self._pending_questions[correlation_id] = response_future

        try:
            # Send the question
            message = AgentMessage(
                type=MessageType.QUESTION,
                sender=sender,
                receiver=receiver,
                payload=question,
                task_id=task_id,
                correlation_id=correlation_id,
            )
            await self.send(message)

            # Wait for response
            return await asyncio.wait_for(response_future, timeout=timeout)
        finally:
            # Clean up
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

        # If there's a pending future for this correlation_id, resolve it
        if original.correlation_id and original.correlation_id in self._pending_questions:
            future = self._pending_questions[original.correlation_id]
            if not future.done():
                future.set_result(response)
        else:
            # Otherwise, put in queue
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

        # If timeout > 0 and queue is empty, wait for first message
        if timeout > 0 and queue.empty():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=timeout)
                messages.append(msg)
            except asyncio.TimeoutError:
                return messages

        # Drain all available messages
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break

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
        tasks: Dictionary of task_id -> asyncio.Task
        handles: Dictionary of task_id -> TaskHandle
        message_bus: Message bus for communication
    """

    tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    handles: dict[str, Any] = field(default_factory=dict)  # TaskHandle
    message_bus: InMemoryMessageBus = field(default_factory=InMemoryMessageBus)
    _cancel_events: dict[str, asyncio.Event] = field(default_factory=dict)

    def create_task(
        self,
        task_id: str,
        coro: Any,  # Coroutine
        handle: Any,  # TaskHandle
    ) -> asyncio.Task[Any]:
        """Create and track a new background task.

        Args:
            task_id: Unique identifier for the task.
            coro: The coroutine to run.
            handle: TaskHandle for status tracking.

        Returns:
            The created asyncio.Task.
        """
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        self.handles[task_id] = handle
        self._cancel_events[task_id] = asyncio.Event()

        # Update handle when task starts
        handle.status = "running"
        handle.started_at = datetime.now()

        return task

    def get_handle(self, task_id: str) -> Any | None:
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

    async def soft_cancel(self, task_id: str) -> bool:
        """Request cooperative cancellation of a task.

        Sets a cancellation event that the task can check periodically.
        The task is expected to clean up and exit gracefully.

        Args:
            task_id: The task to cancel.

        Returns:
            True if cancellation was requested, False if task not found.
        """
        if task_id not in self._cancel_events:
            return False

        self._cancel_events[task_id].set()

        # Send cancel request message
        if task_id in self.handles:
            handle = self.handles[task_id]
            try:
                await self.message_bus.send(
                    AgentMessage(
                        type=MessageType.CANCEL_REQUEST,
                        sender="task_manager",
                        receiver=handle.subagent_name,
                        payload={"reason": "soft_cancel"},
                        task_id=task_id,
                    )
                )
            except KeyError:
                pass  # Agent not registered, that's OK

        return True

    async def hard_cancel(self, task_id: str) -> bool:
        """Immediately cancel a task.

        Calls cancel() on the asyncio.Task, causing CancelledError
        to be raised in the task.

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

        # Update handle
        if task_id in self.handles:
            handle = self.handles[task_id]
            handle.status = "cancelled"
            handle.completed_at = datetime.now()

        return True

    def cleanup_task(self, task_id: str) -> None:
        """Clean up resources for a completed task.

        Args:
            task_id: The task to clean up.
        """
        self.tasks.pop(task_id, None)
        self._cancel_events.pop(task_id, None)
        # Keep handle for status queries

    def list_active_tasks(self) -> list[str]:
        """Get list of active (non-completed) task IDs.

        Returns:
            List of task IDs for tasks that haven't completed.
        """
        return [task_id for task_id, task in self.tasks.items() if not task.done()]
