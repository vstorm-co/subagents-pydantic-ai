"""Protocol definitions for subagent system.

This module defines the core protocols (interfaces) that enable
pluggable implementations for dependencies and message passing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from subagents_pydantic_ai.types import AgentMessage


@runtime_checkable
class SubAgentDepsProtocol(Protocol):
    """Protocol for dependencies that support subagent management.

    Any deps class that wants to use the subagent toolset must implement
    this protocol. The key requirement is a `subagents` dict for storing
    compiled agent instances and a `clone_for_subagent` method for creating
    isolated deps for nested subagents.

    Example:
        ```python
        @dataclass
        class MyDeps:
            subagents: dict[str, Any] = field(default_factory=dict)

            def clone_for_subagent(self, max_depth: int = 0) -> "MyDeps":
                return MyDeps(
                    subagents={} if max_depth <= 0 else self.subagents,
                )
        ```
    """

    subagents: dict[str, Any]

    def clone_for_subagent(self, max_depth: int = 0) -> SubAgentDepsProtocol:
        """Create a new deps instance for a subagent.

        Subagents typically get:
        - Shared resources (backend, files, etc.)
        - Empty or limited subagents dict (based on max_depth)
        - Fresh state for task-specific data

        Args:
            max_depth: Maximum nesting depth for the subagent.
                If 0, the subagent cannot spawn further subagents.

        Returns:
            A new deps instance configured for the subagent.
        """
        ...


@runtime_checkable
class MessageBusProtocol(Protocol):
    """Protocol for message bus implementations.

    The message bus enables communication between agents, supporting
    both fire-and-forget messages and request-response patterns.

    Implementations can use different backends:
    - In-memory (default): Uses asyncio queues
    - Redis: For distributed multi-process setups
    - Custom: Any backend implementing this protocol

    Example:
        ```python
        bus = InMemoryMessageBus()

        # Register an agent
        queue = bus.register_agent("worker-1")

        # Send a message
        await bus.send(AgentMessage(
            type=MessageType.TASK_UPDATE,
            sender="parent",
            receiver="worker-1",
            payload={"status": "starting"},
        ))

        # Request-response pattern
        response = await bus.ask(
            sender="parent",
            receiver="worker-1",
            question="What is your status?",
            task_id="task-123",
            timeout=30.0,
        )
        ```
    """

    async def send(self, message: AgentMessage) -> None:
        """Send a message to a specific agent.

        Args:
            message: The message to send. Must have a valid receiver.

        Raises:
            KeyError: If the receiver is not registered.
        """
        ...

    async def ask(
        self,
        sender: str,
        receiver: str,
        question: Any,
        task_id: str,
        timeout: float = 30.0,
    ) -> AgentMessage:
        """Send a question and wait for a response.

        This implements a request-response pattern where the sender
        blocks until the receiver answers or the timeout expires.

        Args:
            sender: ID of the asking agent.
            receiver: ID of the agent to ask.
            question: The question payload.
            task_id: Task ID for correlation.
            timeout: Maximum time to wait for response in seconds.

        Returns:
            The response message from the receiver.

        Raises:
            asyncio.TimeoutError: If no response within timeout.
            KeyError: If the receiver is not registered.
        """
        ...

    async def answer(self, original: AgentMessage, answer: Any) -> None:
        """Answer a previously received question.

        Args:
            original: The original question message.
            answer: The answer payload.
        """
        ...

    def register_agent(self, agent_id: str) -> asyncio.Queue[AgentMessage]:
        """Register an agent to receive messages.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            A queue where messages for this agent will be delivered.

        Raises:
            ValueError: If agent_id is already registered.
        """
        ...

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message bus.

        After unregistration, messages sent to this agent will raise errors.

        Args:
            agent_id: The agent to unregister.
        """
        ...

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
        ...
