"""Tests for message_bus module."""

from __future__ import annotations

import asyncio

import pytest

from subagents_pydantic_ai import InMemoryMessageBus, TaskManager, create_message_bus
from subagents_pydantic_ai.types import AgentMessage, MessageType, TaskHandle


class TestInMemoryMessageBus:
    """Tests for InMemoryMessageBus."""

    def test_register_agent(self, message_bus: InMemoryMessageBus):
        """Test agent registration."""
        queue = message_bus.register_agent("agent-1")
        assert queue is not None
        assert message_bus.is_registered("agent-1")
        assert "agent-1" in message_bus.registered_agents()

    def test_register_agent_duplicate_raises(self, message_bus: InMemoryMessageBus):
        """Test that registering duplicate agent raises error."""
        message_bus.register_agent("agent-1")
        with pytest.raises(ValueError, match="already registered"):
            message_bus.register_agent("agent-1")

    def test_unregister_agent(self, message_bus: InMemoryMessageBus):
        """Test agent unregistration."""
        message_bus.register_agent("agent-1")
        message_bus.unregister_agent("agent-1")
        assert not message_bus.is_registered("agent-1")

    def test_unregister_nonexistent_agent(self, message_bus: InMemoryMessageBus):
        """Test unregistering non-existent agent doesn't raise."""
        message_bus.unregister_agent("nonexistent")  # Should not raise

    def test_registered_agents(self, message_bus: InMemoryMessageBus):
        """Test listing registered agents."""
        message_bus.register_agent("agent-1")
        message_bus.register_agent("agent-2")
        agents = message_bus.registered_agents()
        assert "agent-1" in agents
        assert "agent-2" in agents

    @pytest.mark.asyncio
    async def test_send_message(self, message_bus: InMemoryMessageBus):
        """Test sending a message."""
        queue = message_bus.register_agent("receiver")
        message = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="sender",
            receiver="receiver",
            payload={"data": "test"},
            task_id="task-1",
        )
        await message_bus.send(message)
        received = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert received.payload == {"data": "test"}

    @pytest.mark.asyncio
    async def test_send_to_unregistered_raises(self, message_bus: InMemoryMessageBus):
        """Test sending to unregistered agent raises error."""
        message = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="sender",
            receiver="nonexistent",
            payload={},
            task_id="task-1",
        )
        with pytest.raises(KeyError, match="not registered"):
            await message_bus.send(message)

    @pytest.mark.asyncio
    async def test_ask_and_answer(self, message_bus: InMemoryMessageBus):
        """Test ask/answer flow."""
        message_bus.register_agent("parent")
        receiver_queue = message_bus.register_agent("worker")

        async def answerer():
            msg = await asyncio.wait_for(receiver_queue.get(), timeout=2.0)
            await message_bus.answer(msg, "the answer")

        answer_task = asyncio.create_task(answerer())

        response = await message_bus.ask(
            sender="parent",
            receiver="worker",
            question="what is the answer?",
            task_id="task-1",
            timeout=5.0,
        )

        await answer_task
        assert response.payload == "the answer"
        assert response.type == MessageType.ANSWER

    @pytest.mark.asyncio
    async def test_ask_timeout(self, message_bus: InMemoryMessageBus):
        """Test ask timeout."""
        message_bus.register_agent("parent")
        message_bus.register_agent("worker")

        with pytest.raises(asyncio.TimeoutError):
            await message_bus.ask(
                sender="parent",
                receiver="worker",
                question="question",
                task_id="task-1",
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_ask_unregistered_receiver_raises(self, message_bus: InMemoryMessageBus):
        """Test asking unregistered receiver raises error."""
        message_bus.register_agent("parent")
        with pytest.raises(KeyError, match="not registered"):
            await message_bus.ask(
                sender="parent",
                receiver="nonexistent",
                question="question",
                task_id="task-1",
            )

    @pytest.mark.asyncio
    async def test_answer_unregistered_sender_raises(self, message_bus: InMemoryMessageBus):
        """Test answering to unregistered sender raises error."""
        original = AgentMessage(
            type=MessageType.QUESTION,
            sender="nonexistent",
            receiver="worker",
            payload="question",
            task_id="task-1",
        )
        with pytest.raises(KeyError, match="not registered"):
            await message_bus.answer(original, "answer")

    @pytest.mark.asyncio
    async def test_answer_without_correlation_sends_to_queue(self, message_bus: InMemoryMessageBus):
        """Test answer without pending future sends to queue."""
        sender_queue = message_bus.register_agent("sender")
        message_bus.register_agent("receiver")

        original = AgentMessage(
            type=MessageType.QUESTION,
            sender="sender",
            receiver="receiver",
            payload="question",
            task_id="task-1",
            correlation_id=None,  # No correlation
        )
        await message_bus.answer(original, "answer")

        # Should be in sender's queue
        msg = await asyncio.wait_for(sender_queue.get(), timeout=1.0)
        assert msg.payload == "answer"

    @pytest.mark.asyncio
    async def test_handler_called_on_send(self, message_bus: InMemoryMessageBus):
        """Test message handlers are called."""
        message_bus.register_agent("receiver")
        received_messages: list[AgentMessage] = []

        async def handler(msg: AgentMessage):
            received_messages.append(msg)

        message_bus.add_handler(handler)

        message = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="sender",
            receiver="receiver",
            payload={"data": "test"},
            task_id="task-1",
        )
        await message_bus.send(message)

        assert len(received_messages) == 1
        assert received_messages[0].payload == {"data": "test"}

    def test_remove_handler(self, message_bus: InMemoryMessageBus):
        """Test removing handlers."""

        async def handler(msg: AgentMessage):
            pass

        message_bus.add_handler(handler)
        assert handler in message_bus._handlers

        message_bus.remove_handler(handler)
        assert handler not in message_bus._handlers

    def test_remove_nonexistent_handler(self, message_bus: InMemoryMessageBus):
        """Test removing non-existent handler doesn't raise."""

        async def handler(msg: AgentMessage):
            pass

        message_bus.remove_handler(handler)  # Should not raise

    @pytest.mark.asyncio
    async def test_get_messages_empty(self, message_bus: InMemoryMessageBus):
        """Test get_messages returns empty list when no messages."""
        message_bus.register_agent("agent")
        messages = await message_bus.get_messages("agent", timeout=0.0)
        assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_with_pending(self, message_bus: InMemoryMessageBus):
        """Test get_messages returns pending messages."""
        message_bus.register_agent("agent")
        message_bus.register_agent("sender")

        msg1 = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="sender",
            receiver="agent",
            payload={"id": 1},
            task_id="task-1",
        )
        msg2 = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="sender",
            receiver="agent",
            payload={"id": 2},
            task_id="task-2",
        )

        await message_bus.send(msg1)
        await message_bus.send(msg2)

        messages = await message_bus.get_messages("agent", timeout=0.0)
        assert len(messages) == 2
        assert messages[0].payload == {"id": 1}
        assert messages[1].payload == {"id": 2}

    @pytest.mark.asyncio
    async def test_get_messages_with_timeout(self, message_bus: InMemoryMessageBus):
        """Test get_messages waits for message with timeout."""
        message_bus.register_agent("agent")
        message_bus.register_agent("sender")

        async def delayed_send():
            await asyncio.sleep(0.05)
            await message_bus.send(
                AgentMessage(
                    type=MessageType.TASK_ASSIGNED,
                    sender="sender",
                    receiver="agent",
                    payload={"delayed": True},
                    task_id="task-1",
                )
            )

        send_task = asyncio.create_task(delayed_send())
        messages = await message_bus.get_messages("agent", timeout=1.0)
        await send_task

        assert len(messages) == 1
        assert messages[0].payload == {"delayed": True}

    @pytest.mark.asyncio
    async def test_get_messages_timeout_expires(self, message_bus: InMemoryMessageBus):
        """Test get_messages returns empty on timeout."""
        message_bus.register_agent("agent")
        messages = await message_bus.get_messages("agent", timeout=0.05)
        assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_unregistered_raises(self, message_bus: InMemoryMessageBus):
        """Test get_messages for unregistered agent raises error."""
        with pytest.raises(KeyError, match="not registered"):
            await message_bus.get_messages("nonexistent")


class TestCreateMessageBus:
    """Tests for create_message_bus factory function."""

    def test_create_memory_bus(self):
        """Test creating in-memory bus."""
        bus = create_message_bus("memory")
        assert isinstance(bus, InMemoryMessageBus)

    def test_create_default_bus(self):
        """Test default bus is in-memory."""
        bus = create_message_bus()
        assert isinstance(bus, InMemoryMessageBus)

    def test_unknown_backend_raises(self):
        """Test unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown message bus backend"):
            create_message_bus("redis")


class TestTaskManager:
    """Tests for TaskManager."""

    @pytest.mark.asyncio
    async def test_create_task(self, task_manager: TaskManager):
        """Test creating a task."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test task",
            status="pending",
        )

        async def dummy_coro():
            await asyncio.sleep(0.1)
            return "done"

        task = task_manager.create_task("task-1", dummy_coro(), handle)

        assert "task-1" in task_manager.tasks
        assert handle.status == "running"
        assert handle.started_at is not None

        await task

    def test_get_handle(self, task_manager: TaskManager):
        """Test getting task handle."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="pending",
        )
        task_manager.handles["task-1"] = handle

        assert task_manager.get_handle("task-1") == handle
        assert task_manager.get_handle("nonexistent") is None

    def test_get_cancel_event(self, task_manager: TaskManager):
        """Test getting cancel event."""
        event = asyncio.Event()
        task_manager._cancel_events["task-1"] = event

        assert task_manager.get_cancel_event("task-1") == event
        assert task_manager.get_cancel_event("nonexistent") is None

    @pytest.mark.asyncio
    async def test_soft_cancel(self, task_manager: TaskManager):
        """Test soft cancellation."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )
        task_manager.message_bus.register_agent("worker")

        async def long_task():
            cancel_event = task_manager.get_cancel_event("task-1")
            while cancel_event and not cancel_event.is_set():
                await asyncio.sleep(0.01)
            return "cancelled"

        task_manager.create_task("task-1", long_task(), handle)

        result = await task_manager.soft_cancel("task-1")
        assert result is True
        assert task_manager._cancel_events["task-1"].is_set()

    @pytest.mark.asyncio
    async def test_soft_cancel_nonexistent(self, task_manager: TaskManager):
        """Test soft cancel for non-existent task."""
        result = await task_manager.soft_cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_soft_cancel_unregistered_agent(self, task_manager: TaskManager):
        """Test soft cancel when agent not registered."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="unregistered",
            description="test",
            status="running",
        )

        async def dummy():
            await asyncio.sleep(1)

        task_manager.create_task("task-1", dummy(), handle)

        # Should not raise even if agent not registered
        result = await task_manager.soft_cancel("task-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_hard_cancel(self, task_manager: TaskManager):
        """Test hard cancellation."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )

        async def long_task():
            await asyncio.sleep(10)
            return "done"

        task_manager.create_task("task-1", long_task(), handle)

        result = await task_manager.hard_cancel("task-1")
        assert result is True
        assert handle.status == "cancelled"
        assert handle.completed_at is not None

        # Wait for task to be cancelled
        await asyncio.sleep(0.1)
        assert task_manager.tasks["task-1"].done()

    @pytest.mark.asyncio
    async def test_hard_cancel_nonexistent(self, task_manager: TaskManager):
        """Test hard cancel for non-existent task."""
        result = await task_manager.hard_cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_hard_cancel_already_done(self, task_manager: TaskManager):
        """Test hard cancel for already completed task."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )

        async def quick_task():
            return "done"

        task = task_manager.create_task("task-1", quick_task(), handle)
        await task  # Wait for completion

        result = await task_manager.hard_cancel("task-1")
        assert result is True  # Returns True but doesn't cancel

    @pytest.mark.asyncio
    async def test_cleanup_task(self, task_manager: TaskManager):
        """Test task cleanup."""
        task_manager.tasks["task-1"] = asyncio.create_task(asyncio.sleep(0))
        task_manager._cancel_events["task-1"] = asyncio.Event()
        task_manager.handles["task-1"] = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="completed",
        )

        task_manager.cleanup_task("task-1")

        assert "task-1" not in task_manager.tasks
        assert "task-1" not in task_manager._cancel_events
        # Handle is kept for status queries
        assert "task-1" in task_manager.handles

    @pytest.mark.asyncio
    async def test_list_active_tasks(self, task_manager: TaskManager):
        """Test listing active tasks."""
        handle1 = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test1",
            status="pending",
        )
        handle2 = TaskHandle(
            task_id="task-2",
            subagent_name="worker",
            description="test2",
            status="pending",
        )

        async def long_task():
            await asyncio.sleep(10)

        async def quick_task():
            return "done"

        task_manager.create_task("task-1", long_task(), handle1)
        task2 = task_manager.create_task("task-2", quick_task(), handle2)
        await task2  # Complete task-2

        active = task_manager.list_active_tasks()
        assert "task-1" in active
        assert "task-2" not in active

    @pytest.mark.asyncio
    async def test_soft_cancel_sends_message(self, task_manager: TaskManager):
        """Test soft cancel sends cancel message to registered agent."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )

        # Register the agent to receive cancel message
        task_manager.message_bus.register_agent("worker")

        async def long_task():
            cancel_event = task_manager.get_cancel_event("task-1")
            while cancel_event and not cancel_event.is_set():
                await asyncio.sleep(0.01)
            return "cancelled"

        task_manager.create_task("task-1", long_task(), handle)

        result = await task_manager.soft_cancel("task-1")
        assert result is True

        # Check that a cancel message was sent to the agent's queue
        queue = task_manager.message_bus._queues["worker"]
        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert msg.type == MessageType.CANCEL_REQUEST

    @pytest.mark.asyncio
    async def test_hard_cancel_updates_handle(self, task_manager: TaskManager):
        """Test hard cancel updates handle status and completed_at."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )

        async def long_task():
            await asyncio.sleep(10)
            return "done"

        task_manager.create_task("task-1", long_task(), handle)

        # Hard cancel
        result = await task_manager.hard_cancel("task-1")
        assert result is True
        assert handle.status == "cancelled"
        assert handle.completed_at is not None


class TestInMemoryMessageBusAnswerEdgeCases:
    """Edge case tests for message bus answer functionality."""

    @pytest.mark.asyncio
    async def test_answer_future_already_done(self, message_bus: InMemoryMessageBus):
        """Test answering when future is already completed."""
        message_bus.register_agent("parent")
        receiver_queue = message_bus.register_agent("worker")

        # Start the ask call
        async def do_ask():
            return await message_bus.ask(
                sender="parent",
                receiver="worker",
                question="question",
                task_id="task-1",
                timeout=5.0,
            )

        ask_task = asyncio.create_task(do_ask())

        # Wait for the question to arrive
        await asyncio.sleep(0.05)
        msg = await receiver_queue.get()

        # Answer the question
        await message_bus.answer(msg, "first answer")

        # Get the result
        response = await ask_task
        assert response.payload == "first answer"

        # Try to answer again (future is already done) - should not raise
        await message_bus.answer(msg, "second answer")

    @pytest.mark.asyncio
    async def test_get_messages_queue_empty_exception(self, message_bus: InMemoryMessageBus):
        """Test get_messages handles QueueEmpty during drain."""
        message_bus.register_agent("agent")
        message_bus.register_agent("sender")

        # Send multiple messages
        for i in range(3):
            await message_bus.send(
                AgentMessage(
                    type=MessageType.TASK_ASSIGNED,
                    sender="sender",
                    receiver="agent",
                    payload={"id": i},
                    task_id=f"task-{i}",
                )
            )

        # Get all messages - this will drain the queue
        messages = await message_bus.get_messages("agent", timeout=0.0)
        assert len(messages) == 3

        # Get again - should be empty now
        messages = await message_bus.get_messages("agent", timeout=0.0)
        assert messages == []


class TestTaskManagerBranchCoverage:
    """Tests for TaskManager branch coverage."""

    @pytest.mark.asyncio
    async def test_soft_cancel_without_handle(self, task_manager: TaskManager):
        """Test soft_cancel when task exists but has no handle."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )

        async def task_func():
            cancel_event = task_manager.get_cancel_event("task-1")
            while cancel_event and not cancel_event.is_set():
                await asyncio.sleep(0.01)
            return "done"

        task_manager.create_task("task-1", task_func(), handle)

        # Remove handle but keep task
        del task_manager.handles["task-1"]

        # Soft cancel should still work (sets event) but skip message send
        result = await task_manager.soft_cancel("task-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_hard_cancel_without_handle(self, task_manager: TaskManager):
        """Test hard_cancel when task exists but has no handle."""
        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test",
            status="running",
        )

        async def task_func():
            await asyncio.sleep(10)
            return "done"

        task_manager.create_task("task-1", task_func(), handle)

        # Remove handle but keep task
        del task_manager.handles["task-1"]

        # Hard cancel should still work (cancels task) but skip handle update
        result = await task_manager.hard_cancel("task-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_answer_when_future_already_done(self, message_bus: InMemoryMessageBus):
        """Test answer when future has already been resolved."""
        message_bus.register_agent("parent")
        receiver_queue = message_bus.register_agent("worker")

        # Start an ask
        ask_task = asyncio.create_task(
            message_bus.ask(
                sender="parent",
                receiver="worker",
                question="question",
                task_id="task-1",
                timeout=5.0,
            )
        )

        # Wait for question to arrive
        await asyncio.sleep(0.05)
        msg = await receiver_queue.get()

        # Get the future for this correlation_id
        future = message_bus._pending_questions.get(msg.correlation_id)

        # Manually resolve the future (simulating race condition)
        if future and not future.done():
            future.set_result(
                AgentMessage(
                    type=MessageType.ANSWER,
                    sender="worker",
                    receiver="parent",
                    payload="first answer",
                    task_id="task-1",
                    correlation_id=msg.correlation_id,
                )
            )

        # Now try to answer through normal path - future is already done
        await message_bus.answer(msg, "second answer")

        # The ask should complete with the first answer
        response = await ask_task
        assert response.payload == "first answer"
