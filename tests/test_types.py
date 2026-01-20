"""Tests for types module."""

from __future__ import annotations

from datetime import datetime

from subagents_pydantic_ai.types import (
    AgentMessage,
    CompiledSubAgent,
    MessageType,
    SubAgentConfig,
    TaskCharacteristics,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    decide_execution_mode,
)


class TestMessageType:
    """Tests for MessageType enum."""

    def test_all_message_types_exist(self):
        """Verify all expected message types are defined."""
        assert MessageType.TASK_ASSIGNED == "task_assigned"
        assert MessageType.TASK_UPDATE == "task_update"
        assert MessageType.TASK_COMPLETED == "task_completed"
        assert MessageType.TASK_FAILED == "task_failed"
        assert MessageType.QUESTION == "question"
        assert MessageType.ANSWER == "answer"
        assert MessageType.CANCEL_REQUEST == "cancel_request"
        assert MessageType.CANCEL_FORCED == "cancel_forced"

    def test_message_type_is_string(self):
        """MessageType should be usable as a string."""
        assert isinstance(MessageType.TASK_ASSIGNED.value, str)
        assert str(MessageType.TASK_ASSIGNED) == "MessageType.TASK_ASSIGNED"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all task statuses are defined."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.WAITING_FOR_ANSWER == "waiting_for_answer"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"


class TestSubAgentConfig:
    """Tests for SubAgentConfig TypedDict."""

    def test_required_fields(self):
        """SubAgentConfig requires name, description, instructions."""
        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do things",
        )
        assert config["name"] == "test"
        assert config["description"] == "Test agent"
        assert config["instructions"] == "Do things"

    def test_optional_fields(self):
        """SubAgentConfig can include optional fields."""
        config = SubAgentConfig(
            name="test",
            description="Test",
            instructions="Do things",
            model="gpt-4",
            can_ask_questions=False,
            max_questions=5,
        )
        assert config["model"] == "gpt-4"
        assert config["can_ask_questions"] is False
        assert config["max_questions"] == 5


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_message_creation(self):
        """Test creating an AgentMessage."""
        msg = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="parent",
            receiver="worker",
            payload={"task": "do work"},
            task_id="task-123",
        )
        assert msg.type == MessageType.TASK_ASSIGNED
        assert msg.sender == "parent"
        assert msg.receiver == "worker"
        assert msg.payload == {"task": "do work"}
        assert msg.task_id == "task-123"
        assert isinstance(msg.timestamp, datetime)
        assert msg.correlation_id is None
        # New: id field should be auto-generated
        assert isinstance(msg.id, str)
        assert len(msg.id) > 0

    def test_message_with_correlation(self):
        """Test message with correlation ID."""
        msg = AgentMessage(
            type=MessageType.QUESTION,
            sender="worker",
            receiver="parent",
            payload="What should I do?",
            task_id="task-123",
            correlation_id="corr-456",
        )
        assert msg.correlation_id == "corr-456"

    def test_message_unique_ids(self):
        """Test that each message gets a unique ID."""
        msg1 = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="parent",
            receiver="worker",
            payload={},
            task_id="task-1",
        )
        msg2 = AgentMessage(
            type=MessageType.TASK_ASSIGNED,
            sender="parent",
            receiver="worker",
            payload={},
            task_id="task-2",
        )
        assert msg1.id != msg2.id


class TestTaskHandle:
    """Tests for TaskHandle dataclass."""

    def test_handle_creation(self):
        """Test creating a TaskHandle."""
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="researcher",
            description="Research Python",
        )
        assert handle.task_id == "task-123"
        assert handle.subagent_name == "researcher"
        assert handle.description == "Research Python"
        assert handle.status == TaskStatus.PENDING
        assert isinstance(handle.created_at, datetime)
        assert handle.started_at is None
        assert handle.completed_at is None
        assert handle.result is None
        assert handle.error is None
        assert handle.pending_question is None

    def test_handle_with_result(self):
        """Test handle with completed result."""
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="researcher",
            description="Research Python",
            status=TaskStatus.COMPLETED,
            result="Python is great!",
        )
        assert handle.status == TaskStatus.COMPLETED
        assert handle.result == "Python is great!"

    def test_handle_with_error(self):
        """Test handle with error."""
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="researcher",
            description="Research Python",
            status=TaskStatus.FAILED,
            error="Network error",
        )
        assert handle.status == TaskStatus.FAILED
        assert handle.error == "Network error"


class TestCompiledSubAgent:
    """Tests for CompiledSubAgent TypedDict."""

    def test_compiled_subagent_creation(self):
        """Test creating a CompiledSubAgent."""
        config = SubAgentConfig(
            name="test",
            description="Test",
            instructions="Do things",
        )
        compiled = CompiledSubAgent(
            name="test",
            description="Test",
            config=config,
        )
        assert compiled["name"] == "test"
        assert compiled["description"] == "Test"
        assert compiled["config"] == config

    def test_compiled_subagent_with_agent(self):
        """Test CompiledSubAgent with agent instance."""
        config = SubAgentConfig(
            name="test",
            description="Test",
            instructions="Do things",
        )
        mock_agent = object()
        compiled = CompiledSubAgent(
            name="test",
            description="Test",
            agent=mock_agent,
            config=config,
        )
        assert compiled["agent"] is mock_agent


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_all_priorities_exist(self):
        """Verify all priority levels are defined."""
        assert TaskPriority.LOW == "low"
        assert TaskPriority.NORMAL == "normal"
        assert TaskPriority.HIGH == "high"
        assert TaskPriority.CRITICAL == "critical"

    def test_priority_is_string(self):
        """TaskPriority should be usable as a string."""
        assert isinstance(TaskPriority.NORMAL.value, str)
        assert str(TaskPriority.NORMAL) == "TaskPriority.NORMAL"


class TestTaskCharacteristics:
    """Tests for TaskCharacteristics dataclass."""

    def test_default_values(self):
        """Test default values for TaskCharacteristics."""
        chars = TaskCharacteristics()
        assert chars.estimated_complexity == "moderate"
        assert chars.requires_user_context is False
        assert chars.is_time_sensitive is False
        assert chars.can_run_independently is True
        assert chars.may_need_clarification is False

    def test_custom_values(self):
        """Test TaskCharacteristics with custom values."""
        chars = TaskCharacteristics(
            estimated_complexity="complex",
            requires_user_context=True,
            is_time_sensitive=True,
            can_run_independently=False,
            may_need_clarification=True,
        )
        assert chars.estimated_complexity == "complex"
        assert chars.requires_user_context is True
        assert chars.is_time_sensitive is True
        assert chars.can_run_independently is False
        assert chars.may_need_clarification is True

    def test_simple_complexity(self):
        """Test with simple complexity."""
        chars = TaskCharacteristics(estimated_complexity="simple")
        assert chars.estimated_complexity == "simple"


class TestDecideExecutionMode:
    """Tests for decide_execution_mode function."""

    def test_force_mode_sync(self):
        """Test forced sync mode."""
        chars = TaskCharacteristics()
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config, force_mode="sync")
        assert result == "sync"

    def test_force_mode_async(self):
        """Test forced async mode."""
        chars = TaskCharacteristics()
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config, force_mode="async")
        assert result == "async"

    def test_force_mode_auto_is_ignored(self):
        """Test that force_mode='auto' doesn't override logic."""
        chars = TaskCharacteristics(estimated_complexity="simple")
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config, force_mode="auto")
        assert result == "sync"  # Simple = sync

    def test_config_preferred_mode_sync(self):
        """Test config-level sync preference."""
        chars = TaskCharacteristics(estimated_complexity="complex")
        config = SubAgentConfig(
            name="test", description="", instructions="", preferred_mode="sync"
        )
        result = decide_execution_mode(chars, config)
        assert result == "sync"

    def test_config_preferred_mode_async(self):
        """Test config-level async preference."""
        chars = TaskCharacteristics(estimated_complexity="simple")
        config = SubAgentConfig(
            name="test", description="", instructions="", preferred_mode="async"
        )
        result = decide_execution_mode(chars, config)
        assert result == "async"

    def test_requires_user_context_forces_sync(self):
        """Test that requiring user context forces sync."""
        chars = TaskCharacteristics(
            estimated_complexity="complex",
            requires_user_context=True,
            can_run_independently=True,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "sync"

    def test_may_need_clarification_time_sensitive_forces_sync(self):
        """Test that needing clarification with time sensitivity forces sync."""
        chars = TaskCharacteristics(
            may_need_clarification=True,
            is_time_sensitive=True,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "sync"

    def test_complex_independent_prefers_async(self):
        """Test that complex independent tasks prefer async."""
        chars = TaskCharacteristics(
            estimated_complexity="complex",
            can_run_independently=True,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "async"

    def test_simple_tasks_use_sync(self):
        """Test that simple tasks use sync."""
        chars = TaskCharacteristics(
            estimated_complexity="simple",
            can_run_independently=True,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "sync"

    def test_moderate_independent_uses_async(self):
        """Test that moderate complexity independent tasks use async."""
        chars = TaskCharacteristics(
            estimated_complexity="moderate",
            can_run_independently=True,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "async"

    def test_moderate_dependent_uses_sync(self):
        """Test that moderate complexity dependent tasks use sync."""
        chars = TaskCharacteristics(
            estimated_complexity="moderate",
            can_run_independently=False,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "sync"

    def test_complex_dependent_uses_sync(self):
        """Test that complex dependent tasks use sync."""
        chars = TaskCharacteristics(
            estimated_complexity="complex",
            can_run_independently=False,
        )
        config = SubAgentConfig(name="test", description="", instructions="")
        result = decide_execution_mode(chars, config)
        assert result == "sync"

    def test_force_mode_overrides_config_preference(self):
        """Test that force_mode overrides config preference."""
        chars = TaskCharacteristics()
        config = SubAgentConfig(
            name="test", description="", instructions="", preferred_mode="sync"
        )
        result = decide_execution_mode(chars, config, force_mode="async")
        assert result == "async"


class TestSubAgentConfigExtended:
    """Tests for extended SubAgentConfig fields."""

    def test_preferred_mode_field(self):
        """Test preferred_mode field in SubAgentConfig."""
        config = SubAgentConfig(
            name="test",
            description="Test",
            instructions="Do things",
            preferred_mode="async",
        )
        assert config["preferred_mode"] == "async"

    def test_typical_complexity_field(self):
        """Test typical_complexity field in SubAgentConfig."""
        config = SubAgentConfig(
            name="test",
            description="Test",
            instructions="Do things",
            typical_complexity="complex",
        )
        assert config["typical_complexity"] == "complex"

    def test_typically_needs_context_field(self):
        """Test typically_needs_context field in SubAgentConfig."""
        config = SubAgentConfig(
            name="test",
            description="Test",
            instructions="Do things",
            typically_needs_context=True,
        )
        assert config["typically_needs_context"] is True


class TestTaskHandleExtended:
    """Tests for extended TaskHandle with priority."""

    def test_handle_default_priority(self):
        """Test TaskHandle has default NORMAL priority."""
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="researcher",
            description="Research Python",
        )
        assert handle.priority == TaskPriority.NORMAL

    def test_handle_with_custom_priority(self):
        """Test TaskHandle with custom priority."""
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="researcher",
            description="Research Python",
            priority=TaskPriority.HIGH,
        )
        assert handle.priority == TaskPriority.HIGH
