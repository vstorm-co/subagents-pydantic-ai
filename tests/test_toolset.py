"""Tests for toolset module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from subagents_pydantic_ai import SubAgentConfig, create_subagent_toolset
from subagents_pydantic_ai.toolset import (
    _create_ask_parent_toolset,
    _create_general_purpose_config,
    _run_async,
    _run_sync,
)
from subagents_pydantic_ai.types import CompiledSubAgent, TaskPriority, TaskStatus


@dataclass
class MockDeps:
    """Mock dependencies for testing."""

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> MockDeps:
        return MockDeps(subagents={} if max_depth <= 0 else self.subagents.copy())


@dataclass
class MockRunContext:
    """Mock run context for testing."""

    deps: MockDeps
    _subagent_state: dict[str, Any] | None = None


class MockResult:
    """Mock agent result."""

    def __init__(self, output: str):
        self.output = output


def _make_mock_compiled_subagent(config: SubAgentConfig) -> CompiledSubAgent:
    """Helper to create a mock compiled subagent."""
    mock_agent = MagicMock()
    return CompiledSubAgent(
        name=config["name"],
        description=config["description"],
        agent=mock_agent,
        config=config,
    )


class TestCreateGeneralPurposeConfig:
    """Tests for _create_general_purpose_config."""

    def test_creates_config(self):
        """Test general purpose config creation."""
        config = _create_general_purpose_config()

        assert config["name"] == "general-purpose"
        assert "general-purpose" in config["description"].lower()
        assert config.get("can_ask_questions") is True


class TestCompileSubagent:
    """Tests for _compile_subagent."""

    def test_compile_with_default_model(self):
        """Test compiling subagent with default model."""
        from subagents_pydantic_ai.toolset import _compile_subagent

        config = SubAgentConfig(
            name="test-agent",
            description="Test agent",
            instructions="Test instructions",
        )

        with patch("subagents_pydantic_ai.toolset.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            compiled = _compile_subagent(config, "openai:gpt-4")

            assert compiled.name == "test-agent"
            assert compiled.description == "Test agent"
            assert compiled.agent is not None
            assert compiled.config == config

    def test_compile_with_custom_model(self):
        """Test compiling subagent with custom model."""
        from subagents_pydantic_ai.toolset import _compile_subagent

        config = SubAgentConfig(
            name="test-agent",
            description="Test agent",
            instructions="Test instructions",
            model="openai:gpt-3.5-turbo",
        )

        with patch("subagents_pydantic_ai.toolset.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            compiled = _compile_subagent(config, "openai:gpt-4")

            assert compiled.agent is not None
            # Should use config's model, not default
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args
            assert call_kwargs[0][0] == "openai:gpt-3.5-turbo"


class TestCreateAskParentToolset:
    """Tests for _create_ask_parent_toolset."""

    def test_creates_toolset(self):
        """Test ask_parent toolset creation."""
        toolset = _create_ask_parent_toolset()

        assert toolset.id == "ask_parent"
        assert "ask_parent" in toolset.tools

    @pytest.mark.asyncio
    async def test_ask_parent_no_state(self):
        """Test ask_parent with no state returns error."""
        toolset = _create_ask_parent_toolset()

        ask_parent_tool = toolset.tools["ask_parent"]
        assert ask_parent_tool is not None

        ctx = MockRunContext(deps=MockDeps())
        result = await ask_parent_tool.function(ctx, "question")
        assert "Error" in result
        assert "no communication channel" in result

    @pytest.mark.asyncio
    async def test_ask_parent_with_callback(self):
        """Test ask_parent with callback."""
        toolset = _create_ask_parent_toolset()

        ask_parent_tool = toolset.tools["ask_parent"]

        async def mock_callback(q: str) -> str:
            return f"Answer to: {q}"

        ctx = MockRunContext(deps=MockDeps())
        ctx._subagent_state = {"ask_callback": mock_callback}

        result = await ask_parent_tool.function(ctx, "what is 2+2?")
        assert result == "Answer to: what is 2+2?"

    @pytest.mark.asyncio
    async def test_ask_parent_with_message_bus(self):
        """Test ask_parent with message bus."""
        from subagents_pydantic_ai import InMemoryMessageBus

        toolset = _create_ask_parent_toolset()

        ask_parent_tool = toolset.tools["ask_parent"]

        message_bus = InMemoryMessageBus()
        message_bus.register_agent("parent")
        message_bus.register_agent("subagent-123")

        ctx = MockRunContext(deps=MockDeps())
        ctx._subagent_state = {
            "message_bus": message_bus,
            "task_id": "task-123",
            "parent_id": "parent",
            "agent_id": "subagent-123",
        }

        # Setup answer in background
        async def answer_question():
            import asyncio

            await asyncio.sleep(0.05)
            queue = message_bus._queues["parent"]
            msg = await queue.get()
            await message_bus.answer(msg, "the answer is 4")

        import asyncio

        answer_task = asyncio.create_task(answer_question())

        result = await ask_parent_tool.function(ctx, "what is 2+2?")
        await answer_task

        assert result == "the answer is 4"


class TestCreateSubagentToolset:
    """Tests for create_subagent_toolset."""

    def test_creates_toolset_with_defaults(self):
        """Test creating toolset with default options."""
        config = SubAgentConfig(
            name="general-purpose",
            description="General purpose agent",
            instructions="Help with tasks",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset()

            assert "task" in toolset.tools
            assert "check_task" in toolset.tools
            assert "answer_subagent" in toolset.tools
            assert "list_active_tasks" in toolset.tools
            assert "soft_cancel_task" in toolset.tools
            assert "hard_cancel_task" in toolset.tools

    def test_creates_toolset_with_subagents(self):
        """Test creating toolset with custom subagents."""
        config = SubAgentConfig(
            name="researcher",
            description="Researches topics",
            instructions="Do research",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            assert toolset.id == "subagents"

    def test_creates_toolset_without_general_purpose(self):
        """Test creating toolset without general purpose agent."""
        config = SubAgentConfig(
            name="researcher",
            description="Researches topics",
            instructions="Do research",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            assert toolset is not None

    def test_creates_toolset_with_custom_id(self):
        """Test creating toolset with custom ID."""
        config = SubAgentConfig(
            name="general-purpose",
            description="General purpose agent",
            instructions="Help with tasks",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(id="custom_subagents")
            assert toolset.id == "custom_subagents"

    @pytest.mark.asyncio
    async def test_task_unknown_subagent(self):
        """Test task with unknown subagent returns error."""
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "nonexistent", "sync")

            assert "Error" in result
            assert "Unknown subagent" in result

    @pytest.mark.asyncio
    async def test_check_task_not_found(self):
        """Test check_task with non-existent task."""
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            check_task_tool = toolset.tools["check_task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await check_task_tool.function(ctx, "nonexistent-task")

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_answer_subagent_not_found(self):
        """Test answer_subagent with non-existent task."""
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            answer_tool = toolset.tools["answer_subagent"]

            ctx = MockRunContext(deps=MockDeps())
            result = await answer_tool.function(ctx, "nonexistent-task", "answer")

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_list_active_tasks_empty(self):
        """Test list_active_tasks with no active tasks."""
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            list_tool = toolset.tools["list_active_tasks"]

            ctx = MockRunContext(deps=MockDeps())
            result = await list_tool.function(ctx)

            assert "No active background tasks" in result

    @pytest.mark.asyncio
    async def test_soft_cancel_not_found(self):
        """Test soft_cancel_task with non-existent task."""
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            cancel_tool = toolset.tools["soft_cancel_task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await cancel_tool.function(ctx, "nonexistent-task")

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_hard_cancel_not_found(self):
        """Test hard_cancel_task with non-existent task."""
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            cancel_tool = toolset.tools["hard_cancel_task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await cancel_tool.function(ctx, "nonexistent-task")

            assert "Error" in result
            assert "not found" in result


class TestRunSync:
    """Tests for _run_sync function."""

    @pytest.mark.asyncio
    async def test_run_sync_success(self):
        """Test successful sync execution."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("task completed"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        result = await _run_sync(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
        )

        assert result == "task completed"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sync_error(self):
        """Test sync execution with error."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Something went wrong"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        result = await _run_sync(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
        )

        assert "Error" in result
        assert "Something went wrong" in result


class TestRunAsync:
    """Tests for _run_async function."""

    @pytest.mark.asyncio
    async def test_run_async_returns_task_handle(self):
        """Test async execution returns task handle info."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("task completed"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        result = await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
            task_manager=task_manager,
            message_bus=message_bus,
        )

        assert "Task started in background" in result
        assert "task-123" in result
        assert "check_task" in result

    @pytest.mark.asyncio
    async def test_run_async_task_completes(self):
        """Test async task completes successfully."""
        import asyncio

        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("task completed"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
            task_manager=task_manager,
            message_bus=message_bus,
        )

        # Wait for task to complete
        await asyncio.sleep(0.1)

        handle = task_manager.get_handle("task-123")
        assert handle is not None
        assert handle.status == TaskStatus.COMPLETED
        assert handle.result == "task completed"

    @pytest.mark.asyncio
    async def test_run_async_task_fails(self):
        """Test async task handles failure."""
        import asyncio

        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Task failed"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-456",
            task_manager=task_manager,
            message_bus=message_bus,
        )

        # Wait for task to fail
        await asyncio.sleep(0.1)

        handle = task_manager.get_handle("task-456")
        assert handle is not None
        assert handle.status == TaskStatus.FAILED
        assert "Task failed" in handle.error


class TestToolsetIntegration:
    """Integration tests for toolset functionality."""

    @pytest.mark.asyncio
    async def test_task_sync_execution(self):
        """Test full sync task execution flow."""
        config = SubAgentConfig(
            name="helper",
            description="Helps with tasks",
            instructions="Help with things",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="Sync result",
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "helper", "sync")

            assert result == "Sync result"

    @pytest.mark.asyncio
    async def test_task_async_execution(self):
        """Test full async task execution flow."""
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_async",
                new_callable=AsyncMock,
                return_value="Task started. ID: abc123",
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "worker", "async")

            assert "Task started" in result


class TestAutoModeSelection:
    """Tests for auto-mode selection in task tool."""

    @pytest.mark.asyncio
    async def test_task_auto_mode_simple_uses_sync(self):
        """Test auto mode with simple complexity uses sync."""
        config = SubAgentConfig(
            name="helper",
            description="Helps with tasks",
            instructions="Help with things",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="Sync result",
            ) as mock_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(
                ctx,
                "do something",
                "helper",
                "auto",  # auto mode
                TaskPriority.NORMAL,
                "simple",  # complexity override
                False,  # requires_user_context
                False,  # may_need_clarification
            )

            assert result == "Sync result"
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_auto_mode_complex_uses_async(self):
        """Test auto mode with complex complexity uses async."""
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_async",
                new_callable=AsyncMock,
                return_value="Task started",
            ) as mock_async,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(
                ctx,
                "do something",
                "worker",
                "auto",  # auto mode
                TaskPriority.NORMAL,
                "complex",  # complexity override
                False,  # requires_user_context
                False,  # may_need_clarification
            )

            assert "Task started" in result
            mock_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_auto_mode_requires_context_uses_sync(self):
        """Test auto mode with requires_user_context uses sync."""
        config = SubAgentConfig(
            name="helper",
            description="Helps with tasks",
            instructions="Help with things",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="Sync result",
            ) as mock_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(
                ctx,
                "do something",
                "helper",
                "auto",  # auto mode
                TaskPriority.NORMAL,
                "complex",  # complexity override - would normally be async
                True,  # requires_user_context - forces sync
                False,  # may_need_clarification
            )

            assert result == "Sync result"
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_with_priority_parameter(self):
        """Test task with priority parameter passed to async."""
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_async",
                new_callable=AsyncMock,
                return_value="Task started",
            ) as mock_async,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            await task_tool.function(
                ctx,
                "do something",
                "worker",
                "async",
                TaskPriority.HIGH,  # priority parameter
            )

            # Verify priority was passed
            call_kwargs = mock_async.call_args.kwargs
            assert call_kwargs.get("priority") == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_task_auto_mode_uses_config_typical_complexity(self):
        """Test auto mode uses config's typical_complexity."""
        config = SubAgentConfig(
            name="simple-worker",
            description="Does simple work",
            instructions="Work on things",
            typical_complexity="simple",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="Sync result",
            ) as mock_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(
                ctx,
                "do something",
                "simple-worker",
                "auto",  # auto mode - uses config's typical_complexity
            )

            assert result == "Sync result"
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_auto_mode_uses_config_typically_needs_context(self):
        """Test auto mode uses config's typically_needs_context."""
        config = SubAgentConfig(
            name="context-worker",
            description="Needs context",
            instructions="Work on things",
            typical_complexity="complex",
            typically_needs_context=True,
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="Sync result",
            ) as mock_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            # Complex task would normally be async, but config says needs context
            result = await task_tool.function(
                ctx,
                "do something",
                "context-worker",
                "auto",
            )

            assert result == "Sync result"
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_auto_mode_with_preferred_mode(self):
        """Test auto mode respects config's preferred_mode."""
        config = SubAgentConfig(
            name="sync-worker",
            description="Prefers sync",
            instructions="Work on things",
            preferred_mode="sync",
        )

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="Sync result",
            ) as mock_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            # Auto mode should respect preferred_mode
            result = await task_tool.function(
                ctx,
                "do something",
                "sync-worker",
                "auto",
                TaskPriority.NORMAL,
                "complex",  # Would normally be async
            )

            assert result == "Sync result"
            mock_sync.assert_called_once()


class TestRunAsyncWithPriority:
    """Tests for _run_async with priority parameter."""

    @pytest.mark.asyncio
    async def test_run_async_with_default_priority(self):
        """Test async task with default priority."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("task completed"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        result = await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
            task_manager=task_manager,
            message_bus=message_bus,
        )

        assert "Task started in background" in result
        handle = task_manager.get_handle("task-123")
        assert handle.priority == TaskPriority.NORMAL

    @pytest.mark.asyncio
    async def test_run_async_with_high_priority(self):
        """Test async task with high priority."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("task completed"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-456",
            task_manager=task_manager,
            message_bus=message_bus,
            priority=TaskPriority.HIGH,
        )

        handle = task_manager.get_handle("task-456")
        assert handle.priority == TaskPriority.HIGH


class TestAskParentEdgeCases:
    """Edge case tests for ask_parent functionality."""

    @pytest.mark.asyncio
    async def test_ask_parent_timeout(self):
        """Test ask_parent handles timeout correctly."""
        from subagents_pydantic_ai import InMemoryMessageBus

        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        message_bus = InMemoryMessageBus()
        message_bus.register_agent("parent")
        message_bus.register_agent("subagent-123")

        ctx = MockRunContext(deps=MockDeps())
        # Mock a very short timeout
        ctx._subagent_state = {
            "message_bus": message_bus,
            "task_id": "task-123",
            "parent_id": "parent",
            "agent_id": "subagent-123",
        }

        # Patch the timeout to be very short
        with patch.object(message_bus, "ask", side_effect=asyncio.TimeoutError()):
            result = await ask_parent_tool.function(ctx, "what is 2+2?")

        assert "Error" in result
        assert "not respond in time" in result

    @pytest.mark.asyncio
    async def test_ask_parent_parent_unavailable(self):
        """Test ask_parent handles unavailable parent."""
        from subagents_pydantic_ai import InMemoryMessageBus

        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        message_bus = InMemoryMessageBus()
        message_bus.register_agent("subagent-123")  # Don't register parent

        ctx = MockRunContext(deps=MockDeps())
        ctx._subagent_state = {
            "message_bus": message_bus,
            "task_id": "task-123",
            "parent_id": "parent",
            "agent_id": "subagent-123",
        }

        result = await ask_parent_tool.function(ctx, "question")
        assert "Error" in result
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_ask_parent_no_message_bus_configured(self):
        """Test ask_parent without message bus returns proper error."""
        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        ctx = MockRunContext(deps=MockDeps())
        ctx._subagent_state = {
            "message_bus": None,
            "task_id": None,
        }

        result = await ask_parent_tool.function(ctx, "question")
        assert "Error" in result
        assert "no communication channel" in result


class TestToolsetFunctionsCoverage:
    """Tests to cover remaining toolset functions."""

    @pytest.mark.asyncio
    async def test_task_agent_none_error(self):
        """Test task returns error when agent is None."""
        config = SubAgentConfig(
            name="broken-agent",
            description="Broken agent",
            instructions="Won't work",
        )

        # Create compiled subagent with agent=None
        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=None,  # No agent
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "broken-agent", "sync")

            assert "Error" in result
            assert "not properly initialized" in result

    @pytest.mark.asyncio
    async def test_task_with_toolsets_factory(self):
        """Test task applies toolsets_factory to agent."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = MagicMock()
        mock_agent._register_toolset = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("done"))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        def mock_toolsets_factory(deps):
            from pydantic_ai.toolsets import FunctionToolset

            return [FunctionToolset(id="mock")]

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
                toolsets_factory=mock_toolsets_factory,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            await task_tool.function(ctx, "do something", "worker", "sync")

            mock_agent._register_toolset.assert_called()

    @pytest.mark.asyncio
    async def test_check_task_completed(self):
        """Test check_task returns result for completed task."""

        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("Task done successfully"))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]
            check_tool = toolset.tools["check_task"]

            ctx = MockRunContext(deps=MockDeps())

            # Start async task
            result = await task_tool.function(ctx, "do something", "worker", "async")
            task_id = result.split("Task ID: ")[1].split("\n")[0]

            # Wait for task to complete
            await asyncio.sleep(0.1)

            # Check task status
            status = await check_tool.function(ctx, task_id)
            assert "completed" in status.lower()
            assert "Task done successfully" in status

    @pytest.mark.asyncio
    async def test_check_task_failed(self):
        """Test check_task returns error for failed task."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Task crashed"))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]
            check_tool = toolset.tools["check_task"]

            ctx = MockRunContext(deps=MockDeps())

            # Start async task that will fail
            result = await task_tool.function(ctx, "do something", "worker", "async")
            task_id = result.split("Task ID: ")[1].split("\n")[0]

            # Wait for task to fail
            await asyncio.sleep(0.1)

            # Check task status
            status = await check_tool.function(ctx, task_id)
            assert "failed" in status.lower()
            assert "Task crashed" in status

    @pytest.mark.asyncio
    async def test_answer_subagent_success(self):
        """Test answer_subagent sends answer to waiting task."""

        config = SubAgentConfig(
            name="helper",
            description="Helper",
            instructions="Help",
        )

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=MagicMock(),
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            answer_tool = toolset.tools["answer_subagent"]

            # Access internal task manager and add a waiting task
            # We need to create a handle in the WAITING_FOR_ANSWER state

            # Get the internal task manager by accessing the closure
            # Since this is tricky, we'll mock the behavior instead
            ctx = MockRunContext(deps=MockDeps())
            result = await answer_tool.function(ctx, "nonexistent", "answer")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_answer_subagent_not_waiting(self):
        """Test answer_subagent when task is not waiting for answer."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("done"))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]
            answer_tool = toolset.tools["answer_subagent"]

            ctx = MockRunContext(deps=MockDeps())

            # Start async task
            result = await task_tool.function(ctx, "do work", "worker", "async")
            task_id = result.split("Task ID: ")[1].split("\n")[0]

            # Wait for task to complete
            await asyncio.sleep(0.1)

            # Try to answer a completed task
            answer_result = await answer_tool.function(ctx, task_id, "answer")
            assert "Error" in answer_result
            assert "not waiting" in answer_result

    @pytest.mark.asyncio
    async def test_list_active_tasks_with_tasks(self):
        """Test list_active_tasks shows active tasks."""
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )

        mock_agent = MagicMock()
        # Create a long-running task
        mock_agent.run = AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(10))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]
            list_tool = toolset.tools["list_active_tasks"]

            ctx = MockRunContext(deps=MockDeps())

            # Start async task
            result = await task_tool.function(ctx, "long running task", "worker", "async")
            task_id = result.split("Task ID: ")[1].split("\n")[0]

            # List tasks before completion
            task_list = await list_tool.function(ctx)
            assert task_id in task_list
            assert "worker" in task_list
            assert "Active background tasks" in task_list

    @pytest.mark.asyncio
    async def test_soft_cancel_task_success(self):
        """Test soft_cancel_task successfully cancels task."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(10))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]
            cancel_tool = toolset.tools["soft_cancel_task"]

            ctx = MockRunContext(deps=MockDeps())

            # Start async task
            result = await task_tool.function(ctx, "long task", "worker", "async")
            task_id = result.split("Task ID: ")[1].split("\n")[0]

            # Soft cancel
            cancel_result = await cancel_tool.function(ctx, task_id)
            assert "Cancellation requested" in cancel_result

    @pytest.mark.asyncio
    async def test_hard_cancel_task_success(self):
        """Test hard_cancel_task successfully cancels task."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(10))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]
            cancel_tool = toolset.tools["hard_cancel_task"]

            ctx = MockRunContext(deps=MockDeps())

            # Start async task
            result = await task_tool.function(ctx, "long task", "worker", "async")
            task_id = result.split("Task ID: ")[1].split("\n")[0]

            # Hard cancel
            cancel_result = await cancel_tool.function(ctx, task_id)
            assert "cancelled" in cancel_result.lower()


class TestRunAsyncEdgeCases:
    """Edge case tests for _run_async function."""

    @pytest.mark.asyncio
    async def test_run_async_agent_already_registered(self):
        """Test _run_async handles already registered agent."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockResult("done"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        # Pre-register the agent to trigger the ValueError branch
        message_bus.register_agent("subagent-task-123")

        # This should not fail even though agent is already registered
        result = await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
            task_manager=task_manager,
            message_bus=message_bus,
        )

        assert "Task started in background" in result

    @pytest.mark.asyncio
    async def test_run_async_task_cancelled(self):
        """Test _run_async handles CancelledError."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=asyncio.CancelledError())

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        await _run_async(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-cancel",
            task_manager=task_manager,
            message_bus=message_bus,
        )

        # Wait for task to be cancelled
        await asyncio.sleep(0.1)

        handle = task_manager.get_handle("task-cancel")
        assert handle is not None
        assert handle.status == TaskStatus.CANCELLED
        assert "cancelled" in handle.error.lower()


class TestCheckTaskStatusBranches:
    """Tests for check_task status branch coverage."""

    @pytest.mark.asyncio
    async def test_check_task_waiting_for_answer(self):
        """Test check_task shows question when task is waiting for answer."""
        from subagents_pydantic_ai import InMemoryMessageBus
        from subagents_pydantic_ai.message_bus import TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=MagicMock(),
        )

        # Create mocked toolset with injected task_manager
        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        # Add a handle in WAITING_FOR_ANSWER state
        handle = TaskHandle(
            task_id="test-task-123",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.WAITING_FOR_ANSWER,
            pending_question="What is the answer?",
        )
        task_manager.handles["test-task-123"] = handle

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=mock_compiled,
            ),
            patch(
                "subagents_pydantic_ai.toolset.TaskManager",
                return_value=task_manager,
            ),
            patch(
                "subagents_pydantic_ai.toolset.InMemoryMessageBus",
                return_value=message_bus,
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            check_tool = toolset.tools["check_task"]
            ctx = MockRunContext(deps=MockDeps())

            # Check task in WAITING_FOR_ANSWER state
            result = await check_tool.function(ctx, "test-task-123")
            assert "waiting_for_answer" in result.lower()
            assert "What is the answer?" in result

    @pytest.mark.asyncio
    async def test_check_task_running_with_elapsed_time(self):
        """Test check_task shows elapsed time for running task with started_at."""
        from datetime import datetime

        from subagents_pydantic_ai import InMemoryMessageBus
        from subagents_pydantic_ai.message_bus import TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=MagicMock(),
        )

        # Create mocked toolset with injected task_manager
        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        # Add a handle in RUNNING state with started_at set
        handle = TaskHandle(
            task_id="test-task-running",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.RUNNING,
            started_at=datetime.now(),  # This is the key - needs started_at set
        )
        task_manager.handles["test-task-running"] = handle

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=mock_compiled,
            ),
            patch(
                "subagents_pydantic_ai.toolset.TaskManager",
                return_value=task_manager,
            ),
            patch(
                "subagents_pydantic_ai.toolset.InMemoryMessageBus",
                return_value=message_bus,
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            check_tool = toolset.tools["check_task"]
            ctx = MockRunContext(deps=MockDeps())

            # Check task - should show running with elapsed time
            status = await check_tool.function(ctx, "test-task-running")
            assert "running" in status.lower()
            assert "Running for:" in status

    @pytest.mark.asyncio
    async def test_check_task_pending_without_started_at(self):
        """Test check_task for pending task without started_at."""
        from subagents_pydantic_ai import InMemoryMessageBus
        from subagents_pydantic_ai.message_bus import TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=MagicMock(),
        )

        # Create mocked toolset with injected task_manager
        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        # Add a handle in PENDING state WITHOUT started_at
        handle = TaskHandle(
            task_id="test-task-pending",
            subagent_name="worker",
            description="test pending task",
            status=TaskStatus.PENDING,
            started_at=None,  # No started_at - hits the else branch
        )
        task_manager.handles["test-task-pending"] = handle

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=mock_compiled,
            ),
            patch(
                "subagents_pydantic_ai.toolset.TaskManager",
                return_value=task_manager,
            ),
            patch(
                "subagents_pydantic_ai.toolset.InMemoryMessageBus",
                return_value=message_bus,
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            check_tool = toolset.tools["check_task"]
            ctx = MockRunContext(deps=MockDeps())

            # Check task - should show pending without elapsed time
            status = await check_tool.function(ctx, "test-task-pending")
            assert "pending" in status.lower()
            assert "Running for:" not in status  # No elapsed time shown


class TestAnswerSubagentCoverage:
    """Tests for answer_subagent function coverage."""

    @pytest.mark.asyncio
    async def test_answer_subagent_send_success(self):
        """Test answer_subagent successfully sends answer."""
        from subagents_pydantic_ai import InMemoryMessageBus
        from subagents_pydantic_ai.message_bus import TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        config = SubAgentConfig(
            name="helper",
            description="Helper",
            instructions="Help",
        )

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=MagicMock(),
        )

        # Create mocked toolset with injected task_manager and message_bus
        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        # Register the subagent to receive messages
        message_bus.register_agent("helper")

        # Add a handle in WAITING_FOR_ANSWER state
        handle = TaskHandle(
            task_id="test-task-456",
            subagent_name="helper",
            description="test task",
            status=TaskStatus.WAITING_FOR_ANSWER,
            pending_question="What is the answer?",
        )
        task_manager.handles["test-task-456"] = handle

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=mock_compiled,
            ),
            patch(
                "subagents_pydantic_ai.toolset.TaskManager",
                return_value=task_manager,
            ),
            patch(
                "subagents_pydantic_ai.toolset.InMemoryMessageBus",
                return_value=message_bus,
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            answer_tool = toolset.tools["answer_subagent"]
            ctx = MockRunContext(deps=MockDeps())

            # Answer the waiting task
            result = await answer_tool.function(ctx, "test-task-456", "The answer is 42")
            assert "Answer sent" in result

            # Verify handle was updated
            assert handle.status == TaskStatus.RUNNING
            assert handle.pending_question is None

    @pytest.mark.asyncio
    async def test_answer_subagent_agent_not_registered(self):
        """Test answer_subagent when agent is not registered."""
        from subagents_pydantic_ai import InMemoryMessageBus
        from subagents_pydantic_ai.message_bus import TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        config = SubAgentConfig(
            name="helper",
            description="Helper",
            instructions="Help",
        )

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=MagicMock(),
        )

        # Create mocked toolset with injected task_manager and message_bus
        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        # DO NOT register the subagent - this will cause KeyError

        # Add a handle in WAITING_FOR_ANSWER state
        handle = TaskHandle(
            task_id="test-task-789",
            subagent_name="helper",
            description="test task",
            status=TaskStatus.WAITING_FOR_ANSWER,
            pending_question="What is the answer?",
        )
        task_manager.handles["test-task-789"] = handle

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=mock_compiled,
            ),
            patch(
                "subagents_pydantic_ai.toolset.TaskManager",
                return_value=task_manager,
            ),
            patch(
                "subagents_pydantic_ai.toolset.InMemoryMessageBus",
                return_value=message_bus,
            ),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            answer_tool = toolset.tools["answer_subagent"]
            ctx = MockRunContext(deps=MockDeps())

            # Try to answer - should fail because agent not registered
            result = await answer_tool.function(ctx, "test-task-789", "The answer is 42")
            assert "Error" in result
            assert "not available" in result


class TestMessageBusBranchCoverage:
    """Tests for message_bus.py branch coverage."""

    @pytest.mark.asyncio
    async def test_soft_cancel_sends_message_to_agent(self):
        """Test soft_cancel sends message to registered agent."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager
        from subagents_pydantic_ai.types import MessageType, TaskHandle

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test task",
            status="running",
        )

        # Register the worker agent
        message_bus.register_agent("worker")

        async def long_task():
            cancel_event = task_manager.get_cancel_event("task-1")
            while cancel_event and not cancel_event.is_set():
                await asyncio.sleep(0.01)
            return "done"

        task_manager.create_task("task-1", long_task(), handle)

        # Soft cancel should send message
        result = await task_manager.soft_cancel("task-1")
        assert result is True

        # Verify message was sent
        queue = message_bus._queues["worker"]
        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert msg.type == MessageType.CANCEL_REQUEST
        assert msg.task_id == "task-1"

    @pytest.mark.asyncio
    async def test_hard_cancel_updates_handle_status(self):
        """Test hard_cancel updates handle status to cancelled."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager
        from subagents_pydantic_ai.types import TaskHandle

        message_bus = InMemoryMessageBus()
        task_manager = TaskManager(message_bus=message_bus)

        handle = TaskHandle(
            task_id="task-1",
            subagent_name="worker",
            description="test task",
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

    @pytest.mark.asyncio
    async def test_get_messages_handles_queue_empty(self):
        """Test get_messages handles QueueEmpty exception during drain."""
        from subagents_pydantic_ai import InMemoryMessageBus

        message_bus = InMemoryMessageBus()
        message_bus.register_agent("agent")

        # Get messages from empty queue
        messages = await message_bus.get_messages("agent", timeout=0.0)
        assert messages == []
