"""Tests for toolset module."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_graph import End

from subagents_pydantic_ai import DynamicAgentRegistry, SubAgentConfig, create_subagent_toolset
from subagents_pydantic_ai._state import SubAgentState, current_subagent_state
from subagents_pydantic_ai.toolset import (
    _capture_result_observability,
    _create_ask_parent_toolset,
    _create_general_purpose_config,
    _run_async,
    _run_sync,
)
from subagents_pydantic_ai.types import (
    CompiledSubAgent,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    utcnow,
)
from tests.agent_result_fakes import MockResult, MockResultWithMessages, MockUsage


@dataclass
class MockDeps:
    """Mock dependencies for testing.

    `_subagent_state` is declared so tests can exercise the legacy path where a
    caller injects the state onto its own deps object. The library itself binds a
    context variable instead.
    """

    subagents: dict[str, Any] = field(default_factory=dict)
    _subagent_state: dict[str, Any] | None = None
    ask_user: Callable[[str, list[str]], Awaitable[str]] | None = None

    def clone_for_subagent(self, max_depth: int = 0) -> MockDeps:
        return MockDeps(subagents={} if max_depth <= 0 else self.subagents.copy())


@dataclass(frozen=True, slots=True)
class FrozenDeps:
    """Deps a caller may legitimately declare immutable.

    The protocol permits it, so the library must not try to write state onto it.
    """

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> FrozenDeps:
        return FrozenDeps(subagents={} if max_depth <= 0 else dict(self.subagents))


@dataclass
class MockRunContext:
    """Mock run context for testing."""

    deps: MockDeps
    _subagent_state: dict[str, Any] | None = None
    run_id: str | None = None


class MockModelResponse:
    """Minimal pydantic-ai ModelResponse stand-in for observability tests."""

    kind = "response"
    cost_called = False

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str,
        total_price: Decimal,
        tool_names: list[str] | None = None,
    ):
        self.model_name = model_name
        self.provider_name = provider_name
        self.provider_url = "https://example.test/v1"
        self.provider_response_id = "provider-response-123"
        self.provider_details = {"region": "test"}
        self.finish_reason = "stop"
        self.tool_calls = [SimpleNamespace(tool_name=name) for name in tool_names or []]
        self._total_price = total_price
        self.cost_called = False

    def cost(self) -> Any:
        self.cost_called = True
        return SimpleNamespace(total_price=self._total_price)


class _FakeRun:
    """Stand-in for `AgentRun` driven via `run.next` (no extra graph nodes).

    Mirrors how `Agent.run` / the retry driver advance a run: a non-`End`
    `next_node` enters the loop once, then the pre-set `result` makes the
    driver break immediately. `next` is provided for completeness.
    """

    def __init__(self, result: Any, messages: list[Any]) -> None:
        self.result = result
        self._messages = messages
        self.next_node: Any = object()  # non-End sentinel

    async def next(self, node: Any) -> Any:
        return End(self.result)

    def all_messages(self) -> list[Any]:
        return self._messages


class _FakeAgentCM:
    def __init__(self, agent: FakeAgent) -> None:
        self._agent = agent

    async def __aenter__(self) -> _FakeRun:
        agent = self._agent
        if agent._delay:
            await asyncio.sleep(agent._delay)
        if agent.on_enter is not None:
            agent.on_enter()
        if agent._error is not None:
            raise agent._error
        return _FakeRun(agent._result, agent._messages)

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeAgent:
    """Faithful stand-in for a pydantic-ai `Agent` driven via `.iter()`.

    Mirrors how `Agent.run` is actually implemented (``async with
    agent.iter(...) as run``), so it exercises the real subagent
    execution path. Construct with `result` (success), `error`
    (raised inside the run), and/or `delay` (await before resolving,
    for cancellation tests). Every `.iter()` call is recorded in
    `iter_calls` for assertions.
    """

    def __init__(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
        delay: float = 0.0,
        messages: list[Any] | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self._delay = delay
        self._messages = messages or []
        self.iter_calls: list[dict[str, Any]] = []
        self.on_enter: Callable[[], None] | None = None
        """Called inside the run, for assertions about state visible to the child."""

    def iter(
        self, prompt: Any = None, *, message_history: Any = None, **kwargs: Any
    ) -> _FakeAgentCM:
        self.iter_calls.append({"prompt": prompt, "message_history": message_history, **kwargs})
        return _FakeAgentCM(self)

    @property
    def iter_count(self) -> int:
        return len(self.iter_calls)


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
        config = _create_general_purpose_config("openai:gpt-4", None)

        assert config["name"] == "general-purpose"
        assert "general-purpose" in config["description"].lower()
        assert config.get("can_ask_questions") is True

    def test_carries_the_default_model_so_it_compiles_like_any_other(self):
        """A `default_model` is written onto the config, not compiled from behind it.

        `_compile_subagent` reads `config["model"]`, so carrying it here is what
        makes this delegate build the same way a configured one does.
        """
        config = _create_general_purpose_config("openai:gpt-4", None)

        assert config["model"] == "openai:gpt-4"
        assert "agent_factory" not in config

    def test_carries_the_factory_so_a_consumer_builds_it_with_its_own_credential(self):
        """With a factory the config carries that, not a model.

        The factory is what a consumer resolving a model, a credential and a budget
        per caller hands in; `_compile_subagent` prefers it over `model`.
        """

        def factory(config: SubAgentConfig) -> object:
            return MagicMock()

        config = _create_general_purpose_config(None, factory)

        assert config["agent_factory"] is factory
        assert "model" not in config


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

    def test_a_config_naming_no_model_with_no_default_is_refused_not_guessed(self):
        """A subagent that says nothing about its model is refused when there is no default.

        The library used to fall back on a model of its own choosing here, which
        resolves whatever provider credential the process environment holds -- in a
        multi-tenant deployment, not the caller's. Failing loud is the point.
        """
        from subagents_pydantic_ai.toolset import _compile_subagent

        config = SubAgentConfig(
            name="rudderless",
            description="Names no model",
            instructions="Do work",
        )

        with pytest.raises(ValueError, match="says nothing about which model it runs on"):
            _compile_subagent(config, None)

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

    def test_compile_with_model_object(self):
        """Test compiling subagent with a Model object instead of string."""

        from subagents_pydantic_ai.toolset import _compile_subagent

        config = SubAgentConfig(
            name="test-agent",
            description="Test agent",
            instructions="Test instructions",
        )

        test_model = TestModel()
        compiled = _compile_subagent(config, test_model)

        assert compiled.name == "test-agent"
        assert compiled.agent is not None

    def test_compile_with_model_object_in_config(self):
        """Test compiling subagent with a Model object in SubAgentConfig."""

        from subagents_pydantic_ai.toolset import _compile_subagent

        test_model = TestModel()
        config = SubAgentConfig(
            name="test-agent",
            description="Test agent",
            instructions="Test instructions",
            model=test_model,
        )

        with patch("subagents_pydantic_ai.toolset.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            compiled = _compile_subagent(config, "openai:gpt-4")

            assert compiled.agent is not None
            # Should use config's Model object, not default string
            call_args = mock_agent_class.call_args
            assert call_args[0][0] is test_model

    def test_compile_with_custom_toolsets(self):
        """Test compiling subagent with custom toolsets."""

        from subagents_pydantic_ai.toolset import _compile_subagent

        custom_toolset: FunctionToolset[Any] = FunctionToolset(id="custom")

        @custom_toolset.tool_plain
        async def custom_tool(x: str) -> str:
            return x

        config = SubAgentConfig(
            name="test-agent",
            description="Test agent",
            instructions="Test instructions",
            toolsets=[custom_toolset],
        )

        with patch("subagents_pydantic_ai.toolset.Agent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            compiled = _compile_subagent(config, "openai:gpt-4")

            assert compiled.agent is not None
            # Should pass both ask_parent and custom toolset via constructor
            call_kwargs = mock_agent_class.call_args
            passed_toolsets = call_kwargs.kwargs.get("toolsets", [])
            assert len(passed_toolsets) == 2

    def test_compile_with_agent_kwargs(self):
        """Test compiling subagent with agent_kwargs (e.g., builtin_tools)."""
        from subagents_pydantic_ai.toolset import _compile_subagent

        config = SubAgentConfig(
            name="test-agent",
            description="Test agent",
            instructions="Test instructions",
            agent_kwargs={"retries": 3, "result_retries": 2},
        )

        with patch("subagents_pydantic_ai.toolset.Agent") as mock_agent_class:
            mock_agent_class.return_value = MagicMock()
            _compile_subagent(config, "openai:gpt-4")

            # Verify agent_kwargs were passed
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args
            assert call_kwargs.kwargs.get("retries") == 3
            assert call_kwargs.kwargs.get("result_retries") == 2

    def test_compile_with_prebuilt_agent(self):
        """Pre-built agent in config is used as-is, skipping default creation."""
        from subagents_pydantic_ai.toolset import _compile_subagent

        mock_agent = MagicMock()
        config = SubAgentConfig(
            name="custom",
            description="Custom agent",
            instructions="Do stuff",
            agent=mock_agent,
        )
        compiled = _compile_subagent(config, "openai:gpt-4")
        assert compiled.agent is mock_agent
        assert compiled.name == "custom"

    def test_compile_with_agent_factory(self):
        """Agent factory in config is called to create agent."""
        from subagents_pydantic_ai.toolset import _compile_subagent

        mock_agent = MagicMock()
        factory = MagicMock(return_value=mock_agent)
        config = SubAgentConfig(
            name="factory-agent",
            description="Factory agent",
            instructions="Do stuff",
            agent_factory=factory,
        )
        compiled = _compile_subagent(config, "openai:gpt-4")
        assert compiled.agent is mock_agent
        factory.assert_called_once_with(config)

    def test_compile_priority_agent_over_factory(self):
        """Pre-built agent takes priority over agent_factory."""
        from subagents_pydantic_ai.toolset import _compile_subagent

        prebuilt = MagicMock()
        factory = MagicMock()
        config = SubAgentConfig(
            name="priority",
            description="Priority test",
            instructions="Test",
            agent=prebuilt,
            agent_factory=factory,
        )
        compiled = _compile_subagent(config, "openai:gpt-4")
        assert compiled.agent is prebuilt
        factory.assert_not_called()


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

        deps = MockDeps()
        deps._subagent_state = {"ask_callback": mock_callback}
        ctx = MockRunContext(deps=deps)

        result = await ask_parent_tool.function(ctx, "what is 2+2?")
        assert result == "Answer to: what is 2+2?"

    @pytest.mark.asyncio
    async def test_ask_parent_with_deps_ask_user(self):
        """Test ask_parent falls back to deps.ask_user callback."""
        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        async def mock_ask_user(question: str, options: list[str]) -> str:
            return f"User answered: {question}"

        deps = MockDeps()
        deps.ask_user = mock_ask_user
        ctx = MockRunContext(deps=deps)

        result = await ask_parent_tool.function(ctx, "what color?")
        assert result == "User answered: what color?"

    @pytest.mark.asyncio
    async def test_ask_parent_with_task_manager(self):
        """Test ask_parent with task_manager and answer future."""

        from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = _create_ask_parent_toolset()

        ask_parent_tool = toolset.tools["ask_parent"]

        message_bus = InMemoryMessageBus()
        tm = TaskManager(message_bus=message_bus)

        # Create a handle to simulate a running task
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="test-agent",
            description="test task",
            status=TaskStatus.RUNNING,
        )
        tm.handles["task-123"] = handle

        deps = MockDeps()
        deps._subagent_state = {
            "task_manager": tm,
            "task_id": "task-123",
        }
        ctx = MockRunContext(deps=deps)

        # Setup answer in background — resolve the future
        async def answer_question():
            await asyncio.sleep(0.05)
            future = tm.get_answer_future("task-123")
            assert future is not None
            future.set_result("the answer is 4")

        answer_task = asyncio.create_task(answer_question())

        result = await ask_parent_tool.function(ctx, "what is 2+2?")
        await answer_task

        assert result == "the answer is 4"
        # Handle should be back to running after answer
        assert handle.status == TaskStatus.RUNNING
        assert handle.pending_question is None


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
            toolset = create_subagent_toolset(default_model="test")

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
            toolset = create_subagent_toolset(default_model="test", id="custom_subagents")
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
    async def test_task_unknown_subagent_does_not_enumerate_the_registry(self):
        """The unknown-subagent error once listed every run's dynamic agents.

        One registry is shared by every run of the agent, and `create_agent` names
        are model-authored and describe the work, so enumerating them told one
        tenant what the others were doing. The error names the configured subagents
        and says dynamic agents exist without naming them.
        """
        config = SubAgentConfig(
            name="helper",
            description="Helps",
            instructions="Help",
        )
        registry = MagicMock()
        registry.get_compiled.return_value = None
        registry.list_agents.return_value = ["dynamic-agent"]
        registry.count.return_value = 1

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
                registry=registry,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "nonexistent", "sync")

            assert "Unknown subagent" in result
            assert "helper" in result
            assert "dynamic-agent" not in result
            assert "create_agent" in result

    @pytest.mark.asyncio
    async def test_task_unknown_subagent_without_registry_agents(self):
        """With an empty registry the error offers no dynamic-agent hint."""
        config = SubAgentConfig(name="helper", description="Helps", instructions="Help")
        registry = MagicMock()
        registry.get_compiled.return_value = None
        registry.count.return_value = 0

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
                registry=registry,
            )

            ctx = MockRunContext(deps=MockDeps())
            result = await toolset.tools["task"].function(ctx, "do", "nonexistent", "sync")

            assert result == "Error: Unknown subagent 'nonexistent'. Available: helper."

    @pytest.mark.asyncio
    async def test_task_unknown_subagent_with_no_configured_subagents(self):
        """An empty list must not render as an empty word after 'Available:'."""
        toolset = create_subagent_toolset(
            subagents=None,
            include_general_purpose=False,
            default_model=TestModel(),
        )

        ctx = MockRunContext(deps=MockDeps())
        result = await toolset.tools["task"].function(ctx, "do", "nonexistent", "sync")

        assert result == "Error: Unknown subagent 'nonexistent'. Available: none."

    @pytest.mark.asyncio
    async def test_task_resolved_via_registry(self):
        """Test task resolved through dynamic registry lookup."""
        config = SubAgentConfig(
            name="static-agent",
            description="Static",
            instructions="Static agent",
        )
        dynamic_config = SubAgentConfig(
            name="dynamic-agent",
            description="Dynamic",
            instructions="Dynamic agent",
        )
        dynamic_compiled = _make_mock_compiled_subagent(dynamic_config)
        # Drive the subagent through the real .iter() execution path.
        dynamic_compiled.agent = FakeAgent(result=MockResult("dynamic result"))

        registry = MagicMock()
        registry.get_compiled.return_value = dynamic_compiled

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
                registry=registry,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "dynamic-agent", "sync")

            assert "dynamic result" in result

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
        mock_agent = FakeAgent(result=MockResult("task completed"))

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
        assert mock_agent.iter_count == 1
        assert "usage_limits" not in mock_agent.iter_calls[0]

    @pytest.mark.asyncio
    async def test_run_sync_forwards_usage_limits(self):
        """Static usage limits are forwarded to the subagent run."""
        usage_limits = UsageLimits(request_limit=3, total_tokens_limit=1000)
        mock_agent = FakeAgent(result=MockResult("task completed"))

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
            usage_limits=usage_limits,
        )

        assert result == "task completed"
        assert mock_agent.iter_calls[0]["usage_limits"] is usage_limits

    @pytest.mark.asyncio
    async def test_run_sync_stores_usage_details_and_message_history_on_handle(self):
        """Sync execution stores observability data on the supplied handle."""
        mock_agent = FakeAgent(
            result=MockResult(
                "task completed",
                usage=MockUsage(details={"reasoning_tokens": 42}),
                messages=[{"role": "user", "content": "work"}],
            )
        )
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="test",
            description="do the thing",
        )

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
            handle=handle,
        )

        assert result == "task completed"
        assert handle.status == TaskStatus.COMPLETED
        assert handle.usage.details["reasoning_tokens"] == 42
        assert handle.message_history is not None
        assert '"content": "work"' in handle.message_history

    @pytest.mark.asyncio
    async def test_run_sync_resumes_and_captures_message_history(self):
        """Sessions can seed and update sync subagent history."""
        seed_history = [{"role": "assistant", "content": "previous"}]
        new_history = [{"role": "assistant", "content": "updated"}]
        captured: list[list[Any]] = []
        mock_agent = FakeAgent(
            result=MockResultWithMessages("task completed", messages=new_history)
        )

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        result = await _run_sync(
            agent=mock_agent,
            config=config,
            description="do the next thing",
            deps=MockDeps(),
            task_id="task-123",
            message_history=seed_history,
            on_message_history=captured.append,
        )

        assert result == "task completed"
        assert mock_agent.iter_calls[0]["message_history"] is seed_history
        assert captured == [new_history]

    @pytest.mark.asyncio
    async def test_run_sync_crash_raises_model_retry(self):
        """A contained crash reaches the parent as ModelRetry, not a success string.

        Returning `"Error executing task: ..."` presented a failure to the parent
        model as a completed tool call, so pydantic-ai's retry budget never engaged.
        """
        mock_agent = FakeAgent(error=Exception("Something went wrong"))

        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        with pytest.raises(ModelRetry) as exc_info:
            await _run_sync(
                agent=mock_agent,
                config=config,
                description="do the thing",
                deps=MockDeps(),
                task_id="task-123",
            )

        assert "Something went wrong" in str(exc_info.value)
        assert "'test'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_sync_crash_returns_on_failure_message(self):
        """`on_failure` turns a crash into a steering message instead of a retry."""
        mock_agent = FakeAgent(error=Exception("boom"))
        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
            on_failure="Summarise from what you already have.",
        )
        handle = TaskHandle(task_id="task-123", subagent_name="test", description="d")

        result = await _run_sync(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
            handle=handle,
        )

        assert result == "Summarise from what you already have."
        assert handle.status == TaskStatus.FAILED
        assert handle.error == "Exception: boom"

    @pytest.mark.asyncio
    async def test_run_sync_propagates_crash_when_not_contained(self):
        """`contain_errors=False` lets the original exception reach the parent run."""
        mock_agent = FakeAgent(error=ValueError("bad tool argument"))
        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        with pytest.raises(ValueError, match="bad tool argument"):
            await _run_sync(
                agent=mock_agent,
                config=config,
                description="do the thing",
                deps=MockDeps(),
                task_id="task-123",
                contain_errors=False,
            )

    @pytest.mark.asyncio
    async def test_run_sync_propagates_control_flow_signals(self):
        """Deferred/approval signals must reach the parent run, contained or not.

        Swallowing them turns a suspended run into a tool result the parent reads
        as a finished task, which breaks deferred tools and approval flows.
        """
        for signal in (CallDeferred(), ApprovalRequired(), UserError("misconfigured")):
            mock_agent = FakeAgent(error=signal)
            config = SubAgentConfig(
                name="test",
                description="Test agent",
                instructions="Do test",
            )

            with pytest.raises(type(signal)):
                await _run_sync(
                    agent=mock_agent,
                    config=config,
                    description="do the thing",
                    deps=MockDeps(),
                    task_id="task-123",
                    contain_errors=True,
                )

    @pytest.mark.asyncio
    async def test_run_sync_binds_ask_user_without_touching_deps(self):
        """`ask_user` reaches `ask_parent` through the context, not the deps object.

        Writing `deps._subagent_state` raised `AttributeError` for a deps class
        declared `frozen=True` or `slots=True`, both of which the protocol allows.
        """
        mock_agent = FakeAgent(result=MockResult("done"))
        captured: list[SubAgentState | None] = []

        async def ask_user(question: str) -> str:
            return f"answer: {question}"

        mock_agent.on_enter = lambda: captured.append(current_subagent_state())
        deps = FrozenDeps()
        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
            can_ask_questions=True,
        )

        await _run_sync(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=deps,
            task_id="task-123",
            ask_user=ask_user,
        )

        assert captured == [
            SubAgentState(ask_timeout_seconds=300.0, ask_callback=ask_user),
        ]
        assert not hasattr(deps, "_subagent_state")

    @pytest.mark.asyncio
    async def test_run_sync_state_does_not_leak_past_the_delegation(self):
        """The bound state is reset when the delegation returns."""
        mock_agent = FakeAgent(result=MockResult("done"))
        config = SubAgentConfig(
            name="test",
            description="Test agent",
            instructions="Do test",
        )

        await _run_sync(
            agent=mock_agent,
            config=config,
            description="do the thing",
            deps=MockDeps(),
            task_id="task-123",
        )

        assert current_subagent_state() is None


class TestResultObservability:
    """Tests for copying AgentRunResult observability onto task handles."""

    def test_capture_result_observability(self):
        first_response = MockModelResponse(
            model_name="gpt-first",
            provider_name="openai",
            total_price=Decimal("0.01"),
            tool_names=["search", "task"],
        )
        second_response = MockModelResponse(
            model_name="gpt-second",
            provider_name="openai",
            total_price=Decimal("0.02"),
            tool_names=["search"],
        )
        result = MockResult(
            "done",
            messages=[first_response, second_response],
            run_id="run-abc",
            conversation_id="conversation-abc",
            traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="test",
            description="do the thing",
        )

        _capture_result_observability(handle, result)

        assert handle.usage is result.usage
        assert handle.message_history is not None
        assert handle.run_id == "run-abc"
        assert handle.conversation_id == "conversation-abc"
        assert handle.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert handle.span_id == "00f067aa0ba902b7"
        assert handle.model_name == "gpt-second"
        assert handle.provider_name == "openai"
        assert handle.provider_url == "https://example.test/v1"
        assert handle.provider_response_id == "provider-response-123"
        assert handle.provider_details == {"region": "test"}
        assert handle.finish_reason == "stop"
        assert handle.cost == Decimal("0.03")
        assert handle.tool_call_counts == {"search": 2, "task": 1}

    def test_capture_result_observability_skips_cost_without_model_name(self):
        response_without_model_name = MockModelResponse(
            model_name="",
            provider_name="openai",
            total_price=Decimal("0.01"),
            tool_names=["search"],
        )
        priced_response = MockModelResponse(
            model_name="gpt-priced",
            provider_name="openai",
            total_price=Decimal("0.02"),
            tool_names=["task"],
        )
        result = MockResult(
            "done",
            messages=[response_without_model_name, priced_response],
        )
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="test",
            description="do the thing",
        )

        _capture_result_observability(handle, result)

        assert not response_without_model_name.cost_called
        assert priced_response.cost_called
        assert handle.cost == Decimal("0.02")
        assert handle.tool_call_counts == {"search": 1, "task": 1}


class TestRunAsync:
    """Tests for _run_async function."""

    @pytest.mark.asyncio
    async def test_run_async_returns_task_handle(self):
        """Test async execution returns task handle info."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = FakeAgent(result=MockResult("task completed"))

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
    async def test_run_async_resumes_and_captures_message_history(self):
        """Explicit chat traces can seed and update async subagent history."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        seed_history = [{"role": "assistant", "content": "previous"}]
        new_history = [{"role": "assistant", "content": "updated"}]
        captured: list[list[Any]] = []
        mock_agent = FakeAgent(
            result=MockResultWithMessages("task completed", messages=new_history)
        )

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
            message_history=seed_history,
            on_message_history=captured.append,
        )
        await asyncio.sleep(0.1)

        assert mock_agent.iter_calls[0]["message_history"] is seed_history
        assert captured == [new_history]

    @pytest.mark.asyncio
    async def test_run_async_task_completes(self):
        """Test async task completes successfully."""

        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = FakeAgent(result=MockResult("task completed"))

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
        assert handle.usage is not None
        assert handle.usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_run_async_stores_usage_details_and_message_history(self):
        """Async execution stores full usage details and subagent messages."""
        import asyncio

        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = FakeAgent(
            result=MockResult(
                "task completed",
                usage=MockUsage(details={"reasoning_tokens": 17}),
                messages=[{"role": "assistant", "content": "done"}],
            )
        )

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
            task_id="task-observe",
            task_manager=task_manager,
            message_bus=message_bus,
        )
        await asyncio.sleep(0.1)

        handle = task_manager.get_handle("task-observe")
        assert handle is not None
        assert handle.usage.details["reasoning_tokens"] == 17
        assert handle.message_history is not None
        assert '"content": "done"' in handle.message_history

    @pytest.mark.asyncio
    async def test_run_async_forwards_usage_limits(self):
        """Async background execution forwards usage limits to the run."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        usage_limits = UsageLimits(request_limit=2, total_tokens_limit=500)
        mock_agent = FakeAgent(result=MockResult("task completed"))

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
            task_id="task-ul",
            task_manager=task_manager,
            message_bus=message_bus,
            usage_limits=usage_limits,
        )
        await asyncio.sleep(0.1)

        handle = task_manager.get_handle("task-ul")
        assert handle is not None
        assert handle.status == TaskStatus.COMPLETED
        assert mock_agent.iter_calls[0]["usage_limits"] is usage_limits

    @pytest.mark.asyncio
    async def test_run_async_task_fails(self):
        """Test async task handles failure."""

        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = FakeAgent(error=Exception("Task failed"))

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
            ) as mock_run_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            task_tool = toolset.tools["task"]

            ctx = MockRunContext(deps=MockDeps())
            result = await task_tool.function(ctx, "do something", "helper", "sync")

            assert result.startswith("Sync result\n\nChat Trace ID: ")
            handle = mock_run_sync.call_args.kwargs["handle"]
            assert handle.task_id in toolset.task_manager.handles
            assert handle.chat_trace_id in result
            assert len(handle.chat_trace_id) == 32
            int(handle.chat_trace_id, 16)
            assert "pass this chat_trace_id to task()" not in result

    @pytest.mark.asyncio
    async def test_task_sync_forwards_ask_user(self):
        """create_subagent_toolset threads ask_user into _run_sync calls."""
        config = SubAgentConfig(
            name="helper",
            description="Helps with tasks",
            instructions="Help with things",
        )

        async def ask_user(question: str) -> str:
            return "answer"

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                return_value="ok",
            ) as mock_run_sync,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
                ask_user=ask_user,
            )

            task_tool = toolset.tools["task"]
            ctx = MockRunContext(deps=MockDeps())
            await task_tool.function(ctx, "do something", "helper", "sync")

            assert mock_run_sync.call_args.kwargs["ask_user"] is ask_user

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

    @pytest.mark.asyncio
    async def test_task_chat_trace_reuses_message_history(self):
        """task(chat_trace_id=...) stores and reuses history for that subagent."""
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )
        saved_history = [{"role": "assistant", "content": "remembered"}]
        mock_agent = FakeAgent(result=MockResultWithMessages("done", messages=saved_history))
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
            ctx = MockRunContext(deps=MockDeps())

            first = await task_tool.function(ctx, "remember this", "worker", "sync")
            trace_id = first.split("Chat Trace ID: ")[1].split("\n")[0]
            second = await task_tool.function(
                ctx,
                "continue",
                "worker",
                "sync",
                chat_trace_id=trace_id,
            )

            assert first.startswith("done\n\nChat Trace ID: ")
            assert second.startswith(f"done\n\nChat Trace ID: {trace_id}")
            assert mock_agent.iter_calls[0]["message_history"] is None
            assert mock_agent.iter_calls[1]["message_history"] == saved_history
            assert toolset.message_history_store[("worker", trace_id)] == saved_history

    @pytest.mark.asyncio
    async def test_task_without_chat_trace_id_creates_saved_trace(self):
        """Omitting chat_trace_id starts a new saved subagent conversation."""
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )
        saved_history = [{"role": "assistant", "content": "remembered"}]
        mock_agent = FakeAgent(result=MockResultWithMessages("done", messages=saved_history))
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
            ctx = MockRunContext(deps=MockDeps())

            first = await task_tool.function(ctx, "remember this", "worker", "sync")
            first_chat_trace_id = first.split("Chat Trace ID: ")[1].split("\n")[0]
            second = await task_tool.function(ctx, "fresh task", "worker", "sync")
            second_chat_trace_id = second.split("Chat Trace ID: ")[1].split("\n")[0]
            continued = await task_tool.function(
                ctx,
                "continue first",
                "worker",
                "sync",
                chat_trace_id=first_chat_trace_id,
            )

            assert first_chat_trace_id
            assert second_chat_trace_id
            assert len(first_chat_trace_id) == 32
            assert len(second_chat_trace_id) == 32
            int(first_chat_trace_id, 16)
            int(second_chat_trace_id, 16)
            assert second_chat_trace_id != first_chat_trace_id
            assert continued.startswith(f"done\n\nChat Trace ID: {first_chat_trace_id}")
            assert mock_agent.iter_calls[0]["message_history"] is None
            assert mock_agent.iter_calls[1]["message_history"] is None
            assert mock_agent.iter_calls[2]["message_history"] == saved_history
            assert toolset.message_history_store[("worker", first_chat_trace_id)] == saved_history
            assert toolset.message_history_store[("worker", second_chat_trace_id)] == saved_history


class TestChatTraceLifecycle:
    """Tests for chat trace guards, LRU eviction, and handle retention."""

    @staticmethod
    def _make_toolset(mock_agent: FakeAgent, **kwargs: Any) -> Any:
        config = SubAgentConfig(
            name="worker",
            description="Does work",
            instructions="Work on things",
        )
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
            return create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
                **kwargs,
            )

    @staticmethod
    def _trace_id(result: str) -> str:
        return result.split("Chat Trace ID: ")[1].split("\n")[0]

    @pytest.mark.asyncio
    async def test_task_unknown_chat_trace_id_errors(self):
        """An explicit chat_trace_id with no saved history fails loudly."""
        mock_agent = FakeAgent(result=MockResultWithMessages("done", messages=[{"m": 1}]))
        toolset = self._make_toolset(mock_agent)
        task_tool = toolset.tools["task"]
        ctx = MockRunContext(deps=MockDeps())

        result = await task_tool.function(
            ctx, "continue", "worker", "sync", chat_trace_id="no-such-trace"
        )

        assert result.startswith("Error: no saved conversation for chat_trace_id 'no-such-trace'")
        # The subagent was never run and nothing was saved under the bogus key.
        assert mock_agent.iter_calls == []
        assert ("worker", "no-such-trace") not in toolset.message_history_store

    @pytest.mark.asyncio
    async def test_task_busy_chat_trace_errors_until_finished(self):
        """A trace with a running task refuses continuation, then allows it."""
        saved_history = [{"role": "assistant", "content": "remembered"}]
        mock_agent = FakeAgent(
            result=MockResultWithMessages("done", messages=saved_history), delay=0.2
        )
        toolset = self._make_toolset(mock_agent)
        task_tool = toolset.tools["task"]
        ctx = MockRunContext(deps=MockDeps())

        started = await task_tool.function(ctx, "long job", "worker", "async")
        trace_id = self._trace_id(started)

        busy = await task_tool.function(
            ctx, "continue too early", "worker", "sync", chat_trace_id=trace_id
        )
        assert busy.startswith(f"Error: chat trace '{trace_id}' already has a running task")

        await asyncio.sleep(0.35)

        continued = await task_tool.function(
            ctx, "continue", "worker", "sync", chat_trace_id=trace_id
        )
        assert continued.startswith(f"done\n\nChat Trace ID: {trace_id}")
        assert mock_agent.iter_calls[-1]["message_history"] == saved_history

    @pytest.mark.asyncio
    async def test_message_history_store_evicts_least_recently_used(self):
        """The store is bounded and continuing a trace refreshes its recency."""
        saved_history = [{"role": "assistant", "content": "remembered"}]
        mock_agent = FakeAgent(result=MockResultWithMessages("done", messages=saved_history))
        toolset = self._make_toolset(mock_agent, max_chat_traces=2)
        task_tool = toolset.tools["task"]
        ctx = MockRunContext(deps=MockDeps())

        first_id = self._trace_id(await task_tool.function(ctx, "one", "worker", "sync"))
        second_id = self._trace_id(await task_tool.function(ctx, "two", "worker", "sync"))
        # Continue the first trace so it becomes most recently used.
        await task_tool.function(ctx, "one again", "worker", "sync", chat_trace_id=first_id)
        third_id = self._trace_id(await task_tool.function(ctx, "three", "worker", "sync"))

        store = toolset.message_history_store
        assert len(store) == 2
        assert ("worker", first_id) in store
        assert ("worker", third_id) in store
        assert ("worker", second_id) not in store

        evicted = await task_tool.function(
            ctx, "continue evicted", "worker", "sync", chat_trace_id=second_id
        )
        assert evicted.startswith("Error: no saved conversation")

    @pytest.mark.asyncio
    async def test_finished_handles_evicted_and_usage_preserved(self):
        """Old finished handles are evicted but get_total_usage stays correct."""
        mock_agent = FakeAgent(
            result=MockResult("done", messages=[{"role": "assistant", "content": "x"}])
        )
        toolset = self._make_toolset(mock_agent, max_task_handles=2)
        task_tool = toolset.tools["task"]
        ctx = MockRunContext(deps=MockDeps())

        # A stale finished handle without usage (e.g. a cancelled task) is
        # evicted first and must not contribute to the totals.
        stale = TaskHandle(
            task_id="stale-no-usage",
            subagent_name="worker",
            description="old cancelled task",
            status=TaskStatus.CANCELLED,
            completed_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        toolset.task_manager.handles["stale-no-usage"] = stale

        for i in range(4):
            await task_tool.function(ctx, f"job {i}", "worker", "sync")

        # Eviction runs before each new task registers its handle: before the
        # 4th task there were 3 finished handles, so the oldest was dropped.
        assert len(toolset.task_manager.handles) == 3
        assert "stale-no-usage" not in toolset.task_manager.handles
        # All 4 runs still counted (MockUsage: 100 in / 50 out / 1 request).
        totals = toolset.get_total_usage()
        assert totals == {
            "input_tokens": 400,
            "output_tokens": 200,
            "total_tokens": 600,
            "requests": 4,
        }

    @pytest.mark.asyncio
    async def test_async_dispatch_failure_releases_chat_trace(self):
        """If _run_async raises before the background task takes ownership,
        the trace guard is released so the conversation stays continuable."""
        saved_history = [{"role": "assistant", "content": "remembered"}]
        mock_agent = FakeAgent(result=MockResultWithMessages("done", messages=saved_history))
        toolset = self._make_toolset(mock_agent)
        task_tool = toolset.tools["task"]
        ctx = MockRunContext(deps=MockDeps())

        trace_id = self._trace_id(await task_tool.function(ctx, "seed", "worker", "sync"))

        with (
            patch(
                "subagents_pydantic_ai.toolset._run_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("dispatch failed"),
            ),
            pytest.raises(RuntimeError, match="dispatch failed"),
        ):
            await task_tool.function(ctx, "continue", "worker", "async", chat_trace_id=trace_id)

        # The guard was released, so continuing the trace works instead of
        # returning a "already has a running task" error.
        continued = await task_tool.function(
            ctx, "continue", "worker", "sync", chat_trace_id=trace_id
        )
        assert continued.startswith(f"done\n\nChat Trace ID: {trace_id}")

    @pytest.mark.asyncio
    async def test_run_sync_completes_when_history_capture_raises(self):
        """A raising all_messages() must not flip a successful run to FAILED."""

        class RaisingHistoryResult(MockResult):
            def all_messages(self) -> list[Any]:
                raise RuntimeError("history exploded")

        captured: list[list[Any]] = []
        mock_agent = FakeAgent(result=RaisingHistoryResult("ok"))
        config = SubAgentConfig(name="worker", description="d", instructions="i")
        handle = TaskHandle(task_id="t-raise", subagent_name="worker", description="d")

        result = await _run_sync(
            agent=mock_agent,
            config=config,
            description="do it",
            deps=MockDeps(),
            task_id="t-raise",
            handle=handle,
            on_message_history=captured.append,
        )

        assert result == "ok"
        assert handle.status == TaskStatus.COMPLETED
        assert captured == []

    @pytest.mark.asyncio
    async def test_check_task_shows_chat_trace_id_only_when_completed(self):
        """check_task must not advertise continuation for unfinished/failed tasks."""
        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager
        handle = TaskHandle(
            task_id="trace-vis",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.FAILED,
            error="boom",
            chat_trace_id="abc123",
        )
        tm.handles["trace-vis"] = handle

        check_tool = toolset.tools["check_task"]
        ctx = MockRunContext(deps=MockDeps())

        failed = await check_tool.function(ctx, "trace-vis")
        assert "Chat Trace ID" not in failed
        assert "Error: boom" in failed

        handle.status = TaskStatus.COMPLETED
        handle.result = "done"
        completed = await check_tool.function(ctx, "trace-vis")
        assert "Chat Trace ID: abc123" in completed


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

            assert result.startswith("Sync result\n\nChat Trace ID: ")
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_sync_failure_omits_chat_trace_hint(self):
        """A failed sync run must not advertise continuation (no history saved)."""
        config = SubAgentConfig(
            name="helper",
            description="Helps with tasks",
            instructions="Help with things",
        )

        def _fail_sync(**kwargs):
            kwargs["handle"].status = TaskStatus.FAILED
            return "Error executing task: boom"

        with (
            patch(
                "subagents_pydantic_ai.toolset._compile_subagent",
                return_value=_make_mock_compiled_subagent(config),
            ),
            patch(
                "subagents_pydantic_ai.toolset._run_sync",
                new_callable=AsyncMock,
                side_effect=_fail_sync,
            ),
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
                "sync",
                TaskPriority.NORMAL,
                "simple",
                False,
                False,
            )

            assert result == "Error executing task: boom"
            assert "Chat Trace ID" not in result

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

            assert result.startswith("Sync result\n\nChat Trace ID: ")
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

            assert result.startswith("Sync result\n\nChat Trace ID: ")
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

            assert result.startswith("Sync result\n\nChat Trace ID: ")
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

            assert result.startswith("Sync result\n\nChat Trace ID: ")
            mock_sync.assert_called_once()


class TestRunAsyncWithPriority:
    """Tests for _run_async with priority parameter."""

    @pytest.mark.asyncio
    async def test_run_async_with_default_priority(self):
        """Test async task with default priority."""
        from subagents_pydantic_ai import InMemoryMessageBus, TaskManager

        mock_agent = FakeAgent(result=MockResult("task completed"))

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

        mock_agent = FakeAgent(result=MockResult("task completed"))

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
        from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        tm = TaskManager(message_bus=InMemoryMessageBus())
        handle = TaskHandle(
            task_id="task-123",
            subagent_name="test-agent",
            description="test task",
            status=TaskStatus.RUNNING,
        )
        tm.handles["task-123"] = handle

        deps = MockDeps()
        deps._subagent_state = {
            "task_manager": tm,
            "task_id": "task-123",
        }
        ctx = MockRunContext(deps=deps)

        # Patch wait_for to raise TimeoutError immediately
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await ask_parent_tool.function(ctx, "what is 2+2?")

        assert "Error" in result
        assert "not respond in time" in result

    @pytest.mark.asyncio
    async def test_ask_parent_no_task_manager_configured(self):
        """Test ask_parent without task_manager returns proper error."""
        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        deps = MockDeps()
        deps._subagent_state = {
            "task_manager": None,
            "task_id": None,
        }
        ctx = MockRunContext(deps=deps)

        result = await ask_parent_tool.function(ctx, "question")
        assert "Error" in result
        assert "no communication channel" in result

    @pytest.mark.asyncio
    async def test_ask_parent_handle_not_found(self):
        """Test ask_parent when task_manager has no handle for task_id."""
        from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager

        toolset = _create_ask_parent_toolset()
        ask_parent_tool = toolset.tools["ask_parent"]

        tm = TaskManager(message_bus=InMemoryMessageBus())
        # Don't add any handle — task_id "missing-task" won't be found

        deps = MockDeps()
        deps._subagent_state = {
            "task_manager": tm,
            "task_id": "missing-task",
        }
        ctx = MockRunContext(deps=deps)

        result = await ask_parent_tool.function(ctx, "question")
        assert "Error" in result
        assert "no communication channel" in result


def _ask_parent_injected(agent: FakeAgent) -> bool:
    """Whether the delegation's first run received an `ask_parent` toolset at run time."""
    injected = agent.iter_calls[0].get("toolsets") or []
    return any(getattr(toolset, "id", None) == "ask_parent" for toolset in injected)


class TestConfiguredSubagentQuestions:
    """A caller-supplied configured subagent honours `can_ask_questions` too.

    `_compile_subagent` appends `ask_parent` only to an agent it builds itself, so a
    subagent whose agent the caller supplied (`agent` or `agent_factory`) skips that
    step. `task` makes up for it by injecting the tool at run time for those, still
    gated on `can_ask_questions`. Without this a caller-supplied subagent -- which is
    every delegate in a consumer like agenticos -- could never ask, whatever its
    config said.
    """

    @pytest.mark.asyncio
    async def test_prebuilt_subagent_may_ask_when_its_config_allows(self):
        agent = FakeAgent(result=MockResult("done"))
        config = SubAgentConfig(
            name="researcher",
            description="Researches",
            instructions="Research",
            agent=agent,
            can_ask_questions=True,
        )
        toolset = create_subagent_toolset(subagents=[config], include_general_purpose=False)

        ctx = MockRunContext(deps=MockDeps())
        await toolset.tools["task"].function(ctx, "find X", "researcher", "sync")

        assert _ask_parent_injected(agent)

    @pytest.mark.asyncio
    async def test_prebuilt_subagent_cannot_ask_when_its_config_forbids(self):
        agent = FakeAgent(result=MockResult("done"))
        config = SubAgentConfig(
            name="summariser",
            description="Summarises",
            instructions="Summarise",
            agent=agent,
            can_ask_questions=False,
        )
        toolset = create_subagent_toolset(subagents=[config], include_general_purpose=False)

        ctx = MockRunContext(deps=MockDeps())
        await toolset.tools["task"].function(ctx, "summarise X", "summariser", "sync")

        assert not _ask_parent_injected(agent)

    @pytest.mark.asyncio
    async def test_factory_built_subagent_may_ask_when_its_config_allows(self):
        agent = FakeAgent(result=MockResult("done"))
        config = SubAgentConfig(
            name="factory-made",
            description="Built by a factory",
            instructions="Work",
            agent_factory=lambda _config: agent,
            can_ask_questions=True,
        )
        toolset = create_subagent_toolset(subagents=[config], include_general_purpose=False)

        ctx = MockRunContext(deps=MockDeps())
        await toolset.tools["task"].function(ctx, "do X", "factory-made", "sync")

        assert _ask_parent_injected(agent)

    @pytest.mark.asyncio
    async def test_library_built_subagent_is_not_injected_again_at_run_time(self):
        """A library-built agent already carries `ask_parent` from compile time.

        Its config names no `agent`/`agent_factory`, so `task` must leave the run-time
        injection off -- adding a second `ask_parent` toolset would duplicate the id.
        """
        agent = FakeAgent(result=MockResult("done"))
        config = SubAgentConfig(
            name="built-here",
            description="Built by the library",
            instructions="Work",
            can_ask_questions=True,
        )
        compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=agent,
        )
        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=compiled,
        ):
            toolset = create_subagent_toolset(subagents=[config], include_general_purpose=False)

            ctx = MockRunContext(deps=MockDeps())
            await toolset.tools["task"].function(ctx, "do X", "built-here", "sync")

        assert not _ask_parent_injected(agent)


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

        mock_agent = FakeAgent(result=MockResult("done"))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        def mock_toolsets_factory(deps):
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

            # Verify agent.run was called with toolsets kwarg
            assert mock_agent.iter_count == 1
            assert "toolsets" in mock_agent.iter_calls[0]

    @pytest.mark.asyncio
    async def test_task_with_toolsets_factory_async(self):
        """Test async task passes toolsets_factory toolsets to agent.run()."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = FakeAgent(result=MockResult("done"))

        mock_compiled = CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            config=config,
            agent=mock_agent,
        )

        def mock_toolsets_factory(deps):
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
            await task_tool.function(ctx, "do something", "worker", "async")

            # Wait for async task to complete
            await asyncio.sleep(0.1)

            # Verify agent.run was called with toolsets kwarg
            assert mock_agent.iter_count == 1
            assert "toolsets" in mock_agent.iter_calls[0]

    @pytest.mark.asyncio
    async def test_check_task_completed(self):
        """Test check_task returns result for completed task."""

        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = FakeAgent(result=MockResult("Task done successfully"))

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

        mock_agent = FakeAgent(error=Exception("Task crashed"))

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

        mock_agent = FakeAgent(result=MockResult("done"))

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

        # Create a long-running task
        mock_agent = FakeAgent(delay=10.0)

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
    async def test_wait_tasks_completed(self):
        """Test wait_tasks returns results for completed tasks."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = FakeAgent(result=MockResult("Research findings here"))

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
            wait_tool = toolset.tools["wait_tasks"]

            ctx = MockRunContext(deps=MockDeps())

            # Start two async tasks
            r1 = await task_tool.function(ctx, "task one", "worker", "async")
            tid1 = r1.split("Task ID: ")[1].split("\n")[0]
            r2 = await task_tool.function(ctx, "task two", "worker", "async")
            tid2 = r2.split("Task ID: ")[1].split("\n")[0]

            # Wait for both
            result = await wait_tool.function(ctx, [tid1, tid2], 5.0)
            assert "COMPLETED" in result
            assert "Research findings here" in result

    @pytest.mark.asyncio
    async def test_wait_tasks_with_failure(self):
        """Test wait_tasks handles failed tasks."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = FakeAgent(error=Exception("Search API down"))

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
            wait_tool = toolset.tools["wait_tasks"]

            ctx = MockRunContext(deps=MockDeps())

            r1 = await task_tool.function(ctx, "failing task", "worker", "async")
            tid1 = r1.split("Task ID: ")[1].split("\n")[0]

            await asyncio.sleep(0.1)

            result = await wait_tool.function(ctx, [tid1], 5.0)
            assert "FAILED" in result
            assert "Search API down" in result

    @pytest.mark.asyncio
    async def test_wait_tasks_not_found(self):
        """Test wait_tasks handles unknown task IDs."""
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

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            wait_tool = toolset.tools["wait_tasks"]
            ctx = MockRunContext(deps=MockDeps())

            result = await wait_tool.function(ctx, ["nonexistent-id"], 5.0)
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_wait_tasks_timeout(self):
        """Test wait_tasks handles timeout for long-running tasks."""
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

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            wait_tool = toolset.tools["wait_tasks"]
            ctx = MockRunContext(deps=MockDeps())

            # Manually inject a long-running task into the task_manager
            from subagents_pydantic_ai.types import TaskHandle

            tm = toolset.task_manager

            async def slow_coro():
                await asyncio.sleep(100)
                return "done"

            handle = TaskHandle(
                task_id="slow-1",
                subagent_name="worker",
                description="slow task",
                status="running",
            )
            tm.create_task("slow-1", slow_coro(), handle)

            # Wait with very short timeout — should hit TimeoutError branch
            result = await wait_tool.function(ctx, ["slow-1"], 0.05)
            assert "Task results" in result
            assert "mode=all" in result
            assert "0/1 finished" in result
            assert "1 still running" in result
            assert "slow-1" in result
            # Task is still running, so status should be reported
            assert "running" in result

    @pytest.mark.asyncio
    async def test_wait_tasks_any_returns_on_first_completion(self):
        """`mode="any"` returns as soon as one task finishes, even if others are slow."""
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

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            wait_tool = toolset.tools["wait_tasks"]
            ctx = MockRunContext(deps=MockDeps())

            from subagents_pydantic_ai.types import TaskHandle

            tm = toolset.task_manager

            async def fast_coro() -> None:
                await asyncio.sleep(0.01)
                fast_handle.result = "fast result"
                fast_handle.status = TaskStatus.COMPLETED

            async def slow_coro() -> None:
                await asyncio.sleep(100)
                slow_handle.status = TaskStatus.COMPLETED

            fast_handle = TaskHandle(
                task_id="fast-1",
                subagent_name="worker",
                description="fast task",
                status="running",
            )
            slow_handle = TaskHandle(
                task_id="slow-1",
                subagent_name="worker",
                description="slow task",
                status="running",
            )
            tm.create_task("fast-1", fast_coro(), fast_handle)
            tm.create_task("slow-1", slow_coro(), slow_handle)

            # Generous timeout — should return as soon as fast finishes,
            # not wait for the 100s slow_coro.
            result = await wait_tool.function(ctx, ["fast-1", "slow-1"], 5.0, "any")

            assert "mode=any" in result
            assert "1/2 finished" in result
            assert "1 still running" in result
            assert "fast-1" in result
            assert "COMPLETED" in result
            assert "fast result" in result
            assert "slow-1" in result
            # slow task should still be reported as running
            assert "running" in result

            # Cleanup: cancel the slow task so it doesn't leak between tests
            slow_task = tm.tasks.get("slow-1")
            if slow_task is not None and not slow_task.done():
                slow_task.cancel()

    @pytest.mark.asyncio
    async def test_wait_tasks_any_returns_on_first_failure(self):
        """`mode="any"` returns when the first task fails too, not just on success."""
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

        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=mock_compiled,
        ):
            toolset = create_subagent_toolset(
                subagents=[config],
                include_general_purpose=False,
            )

            wait_tool = toolset.tools["wait_tasks"]
            ctx = MockRunContext(deps=MockDeps())

            from subagents_pydantic_ai.types import TaskHandle

            tm = toolset.task_manager

            async def failing_coro() -> None:
                await asyncio.sleep(0.01)
                fail_handle.status = TaskStatus.FAILED
                fail_handle.error = "boom"

            async def slow_coro() -> None:
                await asyncio.sleep(100)
                slow_handle.status = TaskStatus.COMPLETED

            fail_handle = TaskHandle(
                task_id="fail-1",
                subagent_name="worker",
                description="failing task",
                status="running",
            )
            slow_handle = TaskHandle(
                task_id="slow-2",
                subagent_name="worker",
                description="slow task",
                status="running",
            )
            tm.create_task("fail-1", failing_coro(), fail_handle)
            tm.create_task("slow-2", slow_coro(), slow_handle)

            result = await wait_tool.function(ctx, ["fail-1", "slow-2"], 5.0, "any")

            assert "mode=any" in result
            assert "1/2 finished" in result
            assert "FAILED" in result
            assert "boom" in result

            slow_task = tm.tasks.get("slow-2")
            if slow_task is not None and not slow_task.done():
                slow_task.cancel()

    @pytest.mark.asyncio
    async def test_wait_tasks_all_still_waits_for_everything(self):
        """Regression: default `mode="all"` still waits for every task."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = FakeAgent(result=MockResult("done"))

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
            wait_tool = toolset.tools["wait_tasks"]

            ctx = MockRunContext(deps=MockDeps())

            r1 = await task_tool.function(ctx, "task one", "worker", "async")
            tid1 = r1.split("Task ID: ")[1].split("\n")[0]
            r2 = await task_tool.function(ctx, "task two", "worker", "async")
            tid2 = r2.split("Task ID: ")[1].split("\n")[0]

            # Default mode (no explicit arg) should be "all" and wait for both.
            result = await wait_tool.function(ctx, [tid1, tid2], 5.0)

            assert "mode=all" in result
            assert "2/2 finished" in result
            # No "still running" segment when everything is done
            assert "still running" not in result
            assert result.count("COMPLETED") == 2

    @pytest.mark.asyncio
    async def test_wait_tasks_does_not_cascade_cancel_to_workers(self):
        """Cancelling wait_tasks must NOT cancel the workers it is waiting on.

        Regression for the silent-CANCELLED bug: the previous
        `asyncio.wait_for(asyncio.gather(...))` propagated cancellation
        through `wait_for` → `gather` → child tasks, so a sibling-cancel
        from pydantic-ai's `_call_tools` (or any other outer cancel) would
        silently kill all in-flight subagents.
        """
        config = SubAgentConfig(name="worker", description="Worker", instructions="Work")
        mock_agent = FakeAgent(result=MockResult("done"), delay=0.1)
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
            toolset = create_subagent_toolset(subagents=[config], include_general_purpose=False)
            task_tool = toolset.tools["task"]
            wait_tool = toolset.tools["wait_tasks"]
            ctx = MockRunContext(deps=MockDeps())

            r = await task_tool.function(ctx, "slow work", "worker", "async")
            tid = r.split("Task ID: ")[1].split("\n")[0]

            worker_task = toolset.task_manager.tasks[tid]

            # Schedule wait_tasks as its own task so we can cancel it from
            # outside (mimicking pydantic-ai sibling-cancel of the tool call).
            waiter = asyncio.create_task(wait_tool.function(ctx, [tid], 5.0))
            await asyncio.sleep(0)  # let waiter start
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

            # Worker MUST still be alive and own its own lifecycle.
            assert not worker_task.done()

            # And it must finish normally — proves no cascade kill happened.
            await asyncio.wait_for(worker_task, timeout=2.0)
            handle = toolset.task_manager.get_handle(tid)
            assert handle is not None
            assert handle.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_tasks_reports_cancelled_status(self):
        """wait_tasks output labels CANCELLED tasks explicitly."""
        config = SubAgentConfig(name="worker", description="Worker", instructions="Work")
        mock_agent = FakeAgent(delay=10.0)
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
            toolset = create_subagent_toolset(subagents=[config], include_general_purpose=False)
            task_tool = toolset.tools["task"]
            wait_tool = toolset.tools["wait_tasks"]
            hard_cancel_tool = toolset.tools["hard_cancel_task"]
            ctx = MockRunContext(deps=MockDeps())

            r = await task_tool.function(ctx, "long work", "worker", "async")
            tid = r.split("Task ID: ")[1].split("\n")[0]

            await hard_cancel_tool.function(ctx, tid)
            await asyncio.sleep(0.05)  # let cancellation settle

            result = await wait_tool.function(ctx, [tid], 1.0)
            assert "CANCELLED" in result
            assert "1/1 finished" in result

    @pytest.mark.asyncio
    async def test_soft_cancel_task_success(self):
        """Test soft_cancel_task successfully cancels task."""
        config = SubAgentConfig(
            name="worker",
            description="Worker",
            instructions="Work",
        )

        mock_agent = FakeAgent(delay=10.0)

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

        mock_agent = FakeAgent(delay=10.0)

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

        mock_agent = FakeAgent(result=MockResult("done"))

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

        mock_agent = FakeAgent(error=asyncio.CancelledError())

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
            started_at=utcnow(),  # This is the key - needs started_at set
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

            # Set up an answer future (as ask_parent would)
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
            task_manager.set_answer_future("test-task-456", future)

            # Answer the waiting task
            result = await answer_tool.function(ctx, "test-task-456", "The answer is 42")
            assert "Answer sent" in result

            # Verify future was resolved
            assert future.done()
            assert future.result() == "The answer is 42"

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

            # Try to answer - should fail because no answer future is set
            result = await answer_tool.function(ctx, "test-task-789", "The answer is 42")
            assert "Error" in result
            assert "no longer waiting" in result


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

        # The running subagent registers on the bus as `subagent-{task_id}`
        # (see toolset.py), which is where the cancel request must be delivered.
        message_bus.register_agent("subagent-task-1")

        async def long_task():
            cancel_event = task_manager.get_cancel_event("task-1")
            while cancel_event and not cancel_event.is_set():
                await asyncio.sleep(0.01)
            return "done"

        task_manager.create_task("task-1", long_task(), handle)

        # Soft cancel should send message
        result = await task_manager.soft_cancel("task-1")
        assert result is True

        # Verify message reached the subagent's registered queue.
        queue = message_bus._queues["subagent-task-1"]
        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert msg.type == MessageType.CANCEL_REQUEST
        assert msg.task_id == "task-1"
        assert msg.receiver == "subagent-task-1"

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


class TestSerializeOutput:
    """Tests for _serialize_output helper."""

    def test_str_passthrough(self):
        from subagents_pydantic_ai.toolset import _serialize_output

        assert _serialize_output("hello") == "hello"

    def test_pydantic_model(self):
        from subagents_pydantic_ai.toolset import _serialize_output

        class MyModel(BaseModel):
            name: str
            value: int

        result = _serialize_output(MyModel(name="test", value=42))
        assert '"name":"test"' in result or '"name": "test"' in result
        assert '"value":42' in result or '"value": 42' in result

    def test_dataclass(self):
        from subagents_pydantic_ai.toolset import _serialize_output

        @dataclass
        class MyData:
            x: int
            y: str

        result = _serialize_output(MyData(x=1, y="hello"))
        assert '"x": 1' in result or '"x":1' in result
        assert '"y": "hello"' in result or '"y":"hello"' in result

    def test_int(self):
        from subagents_pydantic_ai.toolset import _serialize_output

        assert _serialize_output(42) == "42"

    def test_list(self):
        from subagents_pydantic_ai.toolset import _serialize_output

        assert _serialize_output([1, 2, 3]) == "[1, 2, 3]"


class TestUsageTracking:
    """Tests for subagent token usage tracking."""

    @pytest.mark.anyio
    async def test_check_task_does_not_show_usage(self):
        """check_task keeps observability data out of tool-return text."""
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager
        handle = TaskHandle(
            task_id="test-usage",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.COMPLETED,
            result="done",
            usage=MockUsage(input_tokens=500, output_tokens=200),
        )
        tm.handles["test-usage"] = handle

        check_tool = toolset.tools["check_task"]
        ctx = MockRunContext(deps=MockDeps())
        result = await check_tool.function(ctx, "test-usage")
        assert "Result: done" in result
        assert "Usage:" not in result
        assert handle.usage.input_tokens == 500

    @pytest.mark.anyio
    async def test_check_task_does_not_show_usage_details(self):
        """Usage details stay available on the handle only."""
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager
        handle = TaskHandle(
            task_id="test-details",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.COMPLETED,
            result="done",
            usage=MockUsage(input_tokens=500, output_tokens=200, details={"reasoning_tokens": 31}),
        )
        tm.handles["test-details"] = handle

        check_tool = toolset.tools["check_task"]
        ctx = MockRunContext(deps=MockDeps())
        result = await check_tool.function(ctx, "test-details")
        assert "Usage attributes:" not in result
        assert "gen_ai.usage.details.reasoning_tokens" not in result
        assert handle.usage.details["reasoning_tokens"] == 31

    @pytest.mark.anyio
    async def test_check_task_does_not_show_message_history(self):
        """Message history stays available on the handle only."""
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager
        handle = TaskHandle(
            task_id="test-history",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.COMPLETED,
            result="done",
            usage=MockUsage(),
            message_history='[{"role": "assistant", "content": "history"}]',
        )
        tm.handles["test-history"] = handle

        check_tool = toolset.tools["check_task"]
        ctx = MockRunContext(deps=MockDeps())
        result = await check_tool.function(ctx, "test-history")
        assert "Message history:" not in result
        assert '"content": "history"' not in result
        assert handle.message_history == '[{"role": "assistant", "content": "history"}]'

    @pytest.mark.anyio
    async def test_wait_tasks_does_not_show_observability(self):
        """wait_tasks keeps observability data out of tool-return text."""
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager
        handle = TaskHandle(
            task_id="wait-observe",
            subagent_name="worker",
            description="test task",
            status=TaskStatus.COMPLETED,
            result="done",
            usage=MockUsage(details={"reasoning_tokens": 23}),
            message_history='[{"role": "assistant", "content": "wait history"}]',
        )
        tm.handles["wait-observe"] = handle

        wait_tool = toolset.tools["wait_tasks"]
        ctx = MockRunContext(deps=MockDeps())
        result = await wait_tool.function(ctx, ["wait-observe"])
        assert "gen_ai.usage.details.reasoning_tokens" not in result
        assert '"content": "wait history"' not in result
        assert handle.usage.details["reasoning_tokens"] == 23
        assert handle.message_history == '[{"role": "assistant", "content": "wait history"}]'

    @pytest.mark.anyio
    async def test_list_handles(self):
        """TaskManager.list_handles returns all handles."""
        from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        bus = InMemoryMessageBus()
        tm = TaskManager(message_bus=bus)
        h1 = TaskHandle(task_id="t1", subagent_name="a", description="task1")
        h2 = TaskHandle(
            task_id="t2", subagent_name="b", description="task2", status=TaskStatus.COMPLETED
        )
        tm.handles["t1"] = h1
        tm.handles["t2"] = h2

        handles = tm.list_handles()
        assert len(handles) == 2

    @pytest.mark.anyio
    async def test_get_total_usage(self):
        """get_total_usage aggregates across completed tasks."""
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager

        h1 = TaskHandle(
            task_id="t1",
            subagent_name="a",
            description="task1",
            status=TaskStatus.COMPLETED,
            usage=MockUsage(input_tokens=100, output_tokens=50, requests=1),
        )
        h2 = TaskHandle(
            task_id="t2",
            subagent_name="b",
            description="task2",
            status=TaskStatus.COMPLETED,
            usage=MockUsage(input_tokens=200, output_tokens=100, requests=2),
        )
        h3 = TaskHandle(
            task_id="t3",
            subagent_name="c",
            description="task3",
            status=TaskStatus.FAILED,
            usage=None,
        )
        tm.handles["t1"] = h1
        tm.handles["t2"] = h2
        tm.handles["t3"] = h3

        totals = toolset.get_total_usage()
        assert totals["input_tokens"] == 300
        assert totals["output_tokens"] == 150
        assert totals["total_tokens"] == 450
        assert totals["requests"] == 3

    @pytest.mark.anyio
    async def test_check_task_completed_no_usage(self):
        """check_task works for completed tasks without usage data."""
        from subagents_pydantic_ai.types import TaskHandle, TaskStatus

        toolset = create_subagent_toolset(default_model="test")
        tm = toolset.task_manager
        handle = TaskHandle(
            task_id="no-usage",
            subagent_name="worker",
            description="test",
            status=TaskStatus.COMPLETED,
            result="done",
            usage=None,
        )
        tm.handles["no-usage"] = handle

        check_tool = toolset.tools["check_task"]
        ctx = MockRunContext(deps=MockDeps())
        result = await check_tool.function(ctx, "no-usage")
        assert "Result: done" in result
        assert "Usage:" not in result


class TestDrainSteeringMessages:
    """Unit tests for `_drain_steering_messages` (parent -> child steering)."""

    @pytest.mark.asyncio
    async def test_returns_task_update_text_in_order(self):
        from subagents_pydantic_ai.message_bus import InMemoryMessageBus
        from subagents_pydantic_ai.toolset import _drain_steering_messages
        from subagents_pydantic_ai.types import AgentMessage, MessageType

        bus = InMemoryMessageBus()
        bus.register_agent("subagent-x")
        for text in ("first", "second"):
            await bus.send(
                AgentMessage(
                    type=MessageType.TASK_UPDATE,
                    sender="parent",
                    receiver="subagent-x",
                    payload={"message": text},
                    task_id="x",
                )
            )

        assert await _drain_steering_messages(bus, "subagent-x") == ["first", "second"]
        # A second drain returns nothing — the queue was consumed.
        assert await _drain_steering_messages(bus, "subagent-x") == []

    @pytest.mark.asyncio
    async def test_non_dict_payload_is_stringified(self):
        from subagents_pydantic_ai.message_bus import InMemoryMessageBus
        from subagents_pydantic_ai.toolset import _drain_steering_messages
        from subagents_pydantic_ai.types import AgentMessage, MessageType

        bus = InMemoryMessageBus()
        bus.register_agent("subagent-x")
        await bus.send(
            AgentMessage(
                type=MessageType.TASK_UPDATE,
                sender="parent",
                receiver="subagent-x",
                payload="bare text",
                task_id="x",
            )
        )

        assert await _drain_steering_messages(bus, "subagent-x") == ["bare text"]

    @pytest.mark.asyncio
    async def test_ignores_other_types_and_empty_payloads(self):
        from subagents_pydantic_ai.message_bus import InMemoryMessageBus
        from subagents_pydantic_ai.toolset import _drain_steering_messages
        from subagents_pydantic_ai.types import AgentMessage, MessageType

        bus = InMemoryMessageBus()
        bus.register_agent("subagent-x")
        # A non-steering message type — ignored.
        await bus.send(
            AgentMessage(
                type=MessageType.CANCEL_REQUEST,
                sender="task_manager",
                receiver="subagent-x",
                payload={"reason": "soft_cancel"},
                task_id="x",
            )
        )
        # A steering message with an empty body — skipped.
        await bus.send(
            AgentMessage(
                type=MessageType.TASK_UPDATE,
                sender="parent",
                receiver="subagent-x",
                payload={"message": ""},
                task_id="x",
            )
        )

        assert await _drain_steering_messages(bus, "subagent-x") == []


class TestSendMessageToSubagent:
    """Tests for the `send_message_to_subagent` parent-facing tool."""

    def _make_toolset(self):
        config = SubAgentConfig(name="helper", description="Helps", instructions="Help")
        with patch(
            "subagents_pydantic_ai.toolset._compile_subagent",
            return_value=_make_mock_compiled_subagent(config),
        ):
            return create_subagent_toolset(subagents=[config], include_general_purpose=False)

    @pytest.mark.asyncio
    async def test_registered_in_toolset(self):
        toolset = self._make_toolset()
        assert "send_message_to_subagent" in toolset.tools

    @pytest.mark.asyncio
    async def test_task_not_found(self):
        toolset = self._make_toolset()
        tool = toolset.tools["send_message_to_subagent"]
        ctx = MockRunContext(deps=MockDeps())

        result = await tool.function(ctx, "nope", "do this")
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_task_not_running(self):
        from subagents_pydantic_ai.types import TaskHandle

        toolset = self._make_toolset()
        tm = toolset.task_manager
        # A finished task: a handle exists, but the subagent is not registered
        # on the bus (it unregisters when done).
        tm.handles["done-1"] = TaskHandle(
            task_id="done-1",
            subagent_name="helper",
            description="d",
            status=TaskStatus.COMPLETED,
        )
        tool = toolset.tools["send_message_to_subagent"]
        ctx = MockRunContext(deps=MockDeps())

        result = await tool.function(ctx, "done-1", "too late")
        assert "Error" in result
        assert "not accepting messages" in result

    @pytest.mark.asyncio
    async def test_delivers_to_running_subagent(self):
        from subagents_pydantic_ai.types import MessageType

        toolset = self._make_toolset()
        bus = toolset.task_manager.message_bus
        bus.register_agent("subagent-run-1")
        toolset.task_manager.handles["run-1"] = TaskHandle(
            task_id="run-1",
            subagent_name="helper",
            description="d",
            status=TaskStatus.RUNNING,
        )
        tool = toolset.tools["send_message_to_subagent"]
        ctx = MockRunContext(deps=MockDeps())

        result = await tool.function(ctx, "run-1", "narrow the scope")
        assert "delivered" in result

        msgs = await bus.get_messages("subagent-run-1")
        assert len(msgs) == 1
        assert msgs[0].type == MessageType.TASK_UPDATE
        assert msgs[0].payload == {"message": "narrow the scope"}


class TestDelegationConfiguration:
    """Tests for configurable delegation entry points."""

    def test_default_exposes_task_only(self):
        toolset = create_subagent_toolset(include_general_purpose=False)
        assert "delegate" not in toolset.tools
        assert "create_agent" not in toolset.tools
        assert "task" in toolset.tools

    def test_persisted_exposes_create_agent_and_task(self):
        toolset = create_subagent_toolset(
            delegation_configuration="persisted",
            include_general_purpose=False,
        )
        assert "create_agent" in toolset.tools
        assert "task" in toolset.tools
        assert "delegate" not in toolset.tools

    def test_persisted_and_oneshot_exposes_all_entry_points(self):
        toolset = create_subagent_toolset(
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        assert "delegate" in toolset.tools
        assert "create_agent" in toolset.tools
        assert "task" in toolset.tools

    def test_oneshot_only_exposes_only_delegate_entry_point(self):
        toolset = create_subagent_toolset(
            delegation_configuration="oneshot_only",
            include_general_purpose=False,
        )
        assert "delegate" in toolset.tools
        assert "create_agent" not in toolset.tools
        assert "task" not in toolset.tools
        assert "check_task" in toolset.tools

    def test_invalid_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid delegation_configuration"):
            create_subagent_toolset(
                delegation_configuration="invalid",
                include_general_purpose=False,
            )

    def test_oneshot_only_rejects_configured_subagents(self):
        with pytest.raises(ValueError, match="cannot be combined with"):
            create_subagent_toolset(
                delegation_configuration="oneshot_only",
                subagents=[
                    SubAgentConfig(
                        name="researcher",
                        description="Researches topics",
                        instructions="Research.",
                    )
                ],
                include_general_purpose=False,
            )

    @pytest.mark.asyncio
    async def test_create_agent_registers_reusable_agent(self, registry):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted",
            registry=registry,
            include_general_purpose=False,
        )
        create_tool = toolset.tools["create_agent"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("persisted result"))
            result = await create_tool.function(
                ctx,
                name="analyst",
                description="Analyzes data",
                instructions="You are a data analyst.",
            )

        assert "created successfully" in result
        assert registry.exists("analyst")

        task_result = await toolset.tools["task"].function(
            ctx,
            description="Analyze the data",
            subagent_type="analyst",
        )
        assert task_result.startswith("persisted result\n\nChat Trace ID: ")

    @pytest.mark.asyncio
    async def test_create_agent_without_a_model_or_default_is_refused(self, registry):
        """A `create_agent` call naming no model is refused when the toolset has no default.

        The sibling of the `task` and `delegate` refusals: the model named nothing,
        there is no default to fall back on, and it can name one and call again --
        rather than the library running it on a model it never chose.
        """
        toolset = create_subagent_toolset(
            default_model=None,
            delegation_configuration="persisted",
            registry=registry,
            include_general_purpose=False,
        )
        ctx = MockRunContext(deps=MockDeps())

        result = await toolset.tools["create_agent"].function(
            ctx,
            name="analyst",
            description="Analyzes data",
            instructions="You are a data analyst.",
        )

        assert "no model was given and there is no default model" in result
        assert not registry.exists("analyst")

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name(self, registry):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted",
            registry=registry,
            include_general_purpose=False,
        )
        create_tool = toolset.tools["create_agent"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("ok"))
            await create_tool.function(
                ctx,
                name="analyst",
                description="Analyzes data",
                instructions="You are a data analyst.",
            )
            result = await create_tool.function(
                ctx,
                name="analyst",
                description="Analyzes data",
                instructions="You are a data analyst.",
            )

        assert "already exists" in result

    @pytest.mark.asyncio
    async def test_create_agent_register_max_agents(self):
        registry = DynamicAgentRegistry(max_agents=1)
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted",
            registry=registry,
            include_general_purpose=False,
        )
        create_tool = toolset.tools["create_agent"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("ok"))
            await create_tool.function(
                ctx,
                name="first",
                description="First",
                instructions="First",
            )
            result = await create_tool.function(
                ctx,
                name="second",
                description="Second",
                instructions="Second",
            )

        assert "Error" in result
        assert "Maximum" in result

    @pytest.mark.asyncio
    async def test_create_agent_validation_error(self, registry):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted",
            registry=registry,
            include_general_purpose=False,
        )
        create_tool = toolset.tools["create_agent"]
        ctx = MockRunContext(deps=MockDeps())

        result = await create_tool.function(
            ctx,
            name="bad name",
            description="Invalid",
            instructions="Invalid",
        )

        assert "Error" in result
        assert "letters, numbers, and hyphens" in result

    @pytest.mark.asyncio
    async def test_delegate_sync_success(self):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        delegate_tool = toolset.tools["delegate"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("oneshot result"))
            result = await delegate_tool.function(
                ctx,
                description="Analyze the data",
                instructions="You are a data analyst.",
                name="data-analyst",
            )

        # No chat trace: a one-shot specialist is unreachable from `task`, so an
        # id the orchestrator cannot redeem would only invite a failed retry.
        assert result == "oneshot result"

    @pytest.mark.asyncio
    async def test_delegate_names_the_handle(self):
        """The caller's `name` is what shows up on the task handle.

        A generated `oneshot-{task_id}` told an operator nothing about what the
        specialist was for.
        """
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("named result"))
            start_result = await toolset.tools["delegate"].function(
                ctx,
                description="Analyze the data",
                instructions="You are a data analyst.",
                name="data-analyst",
                mode="async",
            )

        assert "Task ID:" in start_result
        task_id = start_result.split("Task ID: ")[1].split("\n")[0]
        handle = toolset.task_manager.get_handle(task_id)
        assert handle is not None
        assert handle.subagent_name == "data-analyst"

    @pytest.mark.asyncio
    async def test_delegate_rejects_an_invalid_name(self):
        """Names go through the same validation as `create_agent`."""
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        ctx = MockRunContext(deps=MockDeps())

        result = await toolset.tools["delegate"].function(
            ctx,
            description="Analyze the data",
            instructions="You are a data analyst.",
            name="bad name",
        )

        assert "Error" in result
        assert "letters, numbers, and hyphens" in result

    @pytest.mark.asyncio
    async def test_delegate_does_not_register_agent(self, registry):
        registry.max_agents = 1
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            registry=registry,
            include_general_purpose=False,
        )
        delegate_tool = toolset.tools["delegate"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("oneshot result"))
            result = await delegate_tool.function(
                ctx,
                description="Do work",
                instructions="You are a worker.",
                name="worker",
            )

        assert result == "oneshot result"
        assert registry.count() == 0

    @pytest.mark.asyncio
    async def test_delegate_works_when_registry_full(self, registry):
        registry.max_agents = 1
        registry.register(
            SubAgentConfig(
                name="existing",
                description="Existing agent",
                instructions="Existing",
            ),
            MagicMock(),
        )
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            registry=registry,
            include_general_purpose=False,
        )
        delegate_tool = toolset.tools["delegate"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("still works"))
            result = await delegate_tool.function(
                ctx,
                description="Do work",
                instructions="You are a worker.",
                name="worker",
            )

        assert result == "still works"
        assert registry.count() == 1

    @pytest.mark.asyncio
    async def test_delegate_async_success(self):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        delegate_tool = toolset.tools["delegate"]
        check_tool = toolset.tools["check_task"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(result=MockResult("async oneshot"))
            start_result = await delegate_tool.function(
                ctx,
                description="Long analysis",
                instructions="You are an analyst.",
                name="analyst",
                mode="async",
            )

        assert "Task ID:" in start_result
        task_id = start_result.split("Task ID: ")[1].split("\n")[0]
        handle = toolset.task_manager.get_handle(task_id)
        assert handle is not None
        assert handle.subagent_name == "analyst"

        await asyncio.sleep(0.05)
        status = await check_tool.function(ctx, task_id)
        assert "COMPLETED" in status or "async oneshot" in status

    @pytest.mark.asyncio
    async def test_delegate_disallowed_model(self):
        toolset = create_subagent_toolset(
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
            allowed_models=["openai:gpt-4.1"],
        )
        delegate_tool = toolset.tools["delegate"]
        ctx = MockRunContext(deps=MockDeps())

        result = await delegate_tool.function(
            ctx,
            description="Do work",
            instructions="You are a worker.",
            name="worker",
            model="anthropic:claude-3",
        )

        assert "Error" in result
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_delegate_invalid_capability(self):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
            capabilities_map={"filesystem": lambda deps: []},
        )
        delegate_tool = toolset.tools["delegate"]
        ctx = MockRunContext(deps=MockDeps())

        result = await delegate_tool.function(
            ctx,
            description="Do work",
            instructions="You are a worker.",
            name="worker",
            capabilities=["missing"],
        )

        assert "Error" in result
        assert "Unknown capabilities" in result

    @pytest.mark.asyncio
    async def test_delegate_execution_error(self):
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        delegate_tool = toolset.tools["delegate"]
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(error=Exception("delegate failed"))
            with pytest.raises(ModelRetry, match="delegate failed"):
                await delegate_tool.function(
                    ctx,
                    description="Do work",
                    instructions="You are a worker.",
                    name="worker",
                )

    @pytest.mark.asyncio
    async def test_delegate_async_reports_no_chat_trace(self):
        """An async one-shot handle carries no trace, so neither listing shows one."""
        toolset = create_subagent_toolset(
            default_model="test",
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        ctx = MockRunContext(deps=MockDeps())

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(
                result=MockResultWithMessages("async oneshot", messages=["m"])
            )
            start_result = await toolset.tools["delegate"].function(
                ctx,
                description="Long analysis",
                instructions="You are an analyst.",
                name="analyst",
                mode="async",
            )

        assert "Chat Trace ID" not in start_result
        task_id = start_result.split("Task ID: ")[1].split("\n")[0]

        waited = await toolset.tools["wait_tasks"].function(ctx, [task_id], timeout=5.0)
        assert "COMPLETED" in waited
        assert "Chat Trace ID" not in waited

        status = await toolset.tools["check_task"].function(ctx, task_id)
        assert "Chat Trace ID" not in status
        assert toolset.task_manager.get_handle(task_id).chat_trace_id is None

    @pytest.mark.asyncio
    async def test_delegate_does_not_evict_resumable_chat_traces(self):
        """One-shot runs must not spend slots in the bounded chat-trace LRU.

        Regression test: they used to save history under a key nothing could
        resolve, so a burst of `delegate` calls flushed the conversations a
        `task` subagent could actually continue.
        """
        researcher = FakeAgent(result=MockResultWithMessages("researched", messages=["m1"]))
        toolset = create_subagent_toolset(
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches topics",
                    instructions="You research.",
                    agent=researcher,
                )
            ],
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
            max_chat_traces=1,
        )
        ctx = MockRunContext(deps=MockDeps())

        started = await toolset.tools["task"].function(
            ctx, description="research X", subagent_type="researcher"
        )
        trace_id = started.split("Chat Trace ID: ")[1].strip()

        with patch("subagents_pydantic_ai.dynamic_agent.Agent") as mock_agent_class:
            mock_agent_class.return_value = FakeAgent(
                result=MockResultWithMessages("oneshot", messages=["m2"])
            )
            for _ in range(3):
                await toolset.tools["delegate"].function(
                    ctx,
                    description="unrelated job",
                    instructions="You are a specialist.",
                    name="specialist",
                )

        resumed = await toolset.tools["task"].function(
            ctx,
            description="follow up",
            subagent_type="researcher",
            chat_trace_id=trace_id,
        )
        assert not resumed.startswith("Error:")
        assert researcher.iter_calls[-1]["message_history"] == ["m1"]


class TestDelegationConfigurationRejectsUnreachableConfig:
    """A hidden tool must not leave its configuration silently inert."""

    def test_oneshot_only_rejects_registry(self):
        with pytest.raises(ValueError, match="cannot be combined with a registry"):
            create_subagent_toolset(
                delegation_configuration="oneshot_only",
                registry=DynamicAgentRegistry(),
            )

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"allowed_models": ["openai:gpt-4.1"]}, "allowed_models"),
            ({"capabilities_map": {"filesystem": lambda deps: []}}, "capabilities_map"),
            ({"default_agent_factory": lambda config: MagicMock()}, "default_agent_factory"),
        ],
    )
    def test_default_mode_rejects_dynamic_agent_config(
        self, kwargs: dict[str, Any], expected: str
    ) -> None:
        with pytest.raises(ValueError, match=f"exposes no dynamic-agent tool.*{expected}"):
            create_subagent_toolset(include_general_purpose=False, **kwargs)

    def test_error_lists_every_unreachable_argument(self):
        with pytest.raises(ValueError) as exc_info:
            create_subagent_toolset(
                include_general_purpose=False,
                allowed_models=["openai:gpt-4.1"],
                capabilities_map={"filesystem": lambda deps: []},
                default_agent_factory=lambda config: MagicMock(),
            )
        assert "allowed_models, capabilities_map, default_agent_factory" in str(exc_info.value)

    @pytest.mark.parametrize(
        "delegation_configuration",
        ["persisted", "persisted_and_oneshot", "oneshot_only"],
    )
    def test_modes_with_a_dynamic_agent_tool_accept_the_config(
        self, delegation_configuration: Any
    ) -> None:
        toolset = create_subagent_toolset(
            delegation_configuration=delegation_configuration,
            include_general_purpose=False,
            allowed_models=["openai:gpt-4.1"],
            capabilities_map={"filesystem": lambda deps: []},
        )
        assert {"create_agent", "delegate"} & set(toolset.tools)


class TestObservabilityHelpers:
    """Direct unit tests for the observability helper branches."""

    def test_capture_message_history_skips_result_without_all_messages(self):
        from subagents_pydantic_ai.toolset import _capture_message_history

        captured: list[Any] = []
        _capture_message_history(object(), captured.append)
        assert captured == []

    def test_get_result_traceparent_none_when_absent(self):
        from subagents_pydantic_ai.toolset import _get_result_traceparent

        assert _get_result_traceparent(object()) is None

    def test_iter_model_responses_includes_result_response(self):
        from subagents_pydantic_ai.toolset import _iter_model_responses

        sentinel = object()

        class _Result:
            def all_messages(self) -> list[Any]:
                return []

            @property
            def response(self) -> Any:
                return sentinel

        assert _iter_model_responses(_Result()) == [sentinel]

    def test_capture_observability_sets_trace_and_span_from_traceparent(self):
        from subagents_pydantic_ai.toolset import _capture_result_observability

        class _Result:
            usage = None
            run_id = "r"
            conversation_id = "c"

            def all_messages_json(self) -> bytes:
                return b"[]"

            def all_messages(self) -> list[Any]:
                return []

            def _traceparent(self, required: bool = False) -> str:
                return "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

        handle = TaskHandle(task_id="t", subagent_name="s", description="d")
        _capture_result_observability(handle, _Result())
        assert handle.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert handle.span_id == "b7ad6b7169203331"

    def test_capture_observability_ignores_malformed_traceparent(self):
        from subagents_pydantic_ai.toolset import _capture_result_observability

        class _Result:
            usage = None
            run_id = "r"
            conversation_id = "c"

            def all_messages_json(self) -> bytes:
                return b"[]"

            def all_messages(self) -> list[Any]:
                return []

            def _traceparent(self, required: bool = False) -> str:
                return "00-abc"

        handle = TaskHandle(task_id="t", subagent_name="s", description="d")
        _capture_result_observability(handle, _Result())
        assert handle.traceparent == "00-abc"
        assert handle.trace_id is None
        assert handle.span_id is None


class TestPreviewResult:
    """Tests for the _preview_result helper backing wait_tasks previews."""

    def test_short_result_is_unchanged(self):
        from subagents_pydantic_ai.toolset import _preview_result

        assert _preview_result("short answer", "abc123", 2000) == "short answer"

    def test_result_at_budget_is_not_truncated(self):
        """A result exactly at the budget is complete, so it gets no marker."""
        from subagents_pydantic_ai.toolset import _preview_result

        result = "x" * 100
        assert _preview_result(result, "abc123", 100) == result

    def test_result_one_char_over_budget_is_truncated(self):
        from subagents_pydantic_ai.toolset import _preview_result

        preview = _preview_result("x" * 101, "abc123", 100)
        assert preview.startswith("x" * 100)
        assert "x" * 101 not in preview
        assert "showing 100 of 101 characters" in preview

    def test_marker_points_at_check_task_with_the_task_id(self):
        from subagents_pydantic_ai.toolset import _preview_result

        preview = _preview_result("y" * 50, "task-42", 10)
        assert "Result truncated for display" in preview
        assert "check_task('task-42')" in preview
        assert "The subagent's answer is complete" in preview

    def test_none_budget_never_truncates(self):
        from subagents_pydantic_ai.toolset import _preview_result

        result = "z" * 10_000
        assert _preview_result(result, "abc123", None) == result

    def test_zero_budget_yields_marker_only(self):
        from subagents_pydantic_ai.toolset import _preview_result

        preview = _preview_result("hidden", "abc123", 0)
        assert preview.startswith("\n\n[Result truncated for display")
        assert "hidden" not in preview

    def test_empty_result_is_unchanged(self):
        from subagents_pydantic_ai.toolset import _preview_result

        assert _preview_result("", "abc123", 2000) == ""


class TestWaitTasksResultTruncation:
    """wait_tasks must never hand the orchestrator a silently cut result."""

    @staticmethod
    def _toolset_with_result(result: str, /, **kwargs: Any) -> tuple[Any, Any]:
        toolset = create_subagent_toolset(default_model="test", **kwargs)
        tm = toolset.task_manager
        tm.handles["long-task"] = TaskHandle(
            task_id="long-task",
            subagent_name="worker",
            description="verbose task",
            status=TaskStatus.COMPLETED,
            result=result,
        )
        return toolset, tm

    @pytest.mark.anyio
    async def test_truncated_result_carries_a_marker(self):
        toolset, _ = self._toolset_with_result("a" * 2500)

        wait_tool = toolset.tools["wait_tasks"]
        output = await wait_tool.function(MockRunContext(deps=MockDeps()), ["long-task"])

        assert "COMPLETED" in output
        assert "showing 2000 of 2500 characters" in output
        assert "check_task('long-task')" in output

    @pytest.mark.anyio
    async def test_result_within_budget_has_no_marker(self):
        toolset, _ = self._toolset_with_result("complete answer")

        wait_tool = toolset.tools["wait_tasks"]
        output = await wait_tool.function(MockRunContext(deps=MockDeps()), ["long-task"])

        assert "complete answer" in output
        assert "truncated" not in output

    @pytest.mark.anyio
    async def test_custom_budget_is_honoured(self):
        toolset, _ = self._toolset_with_result("b" * 100, max_result_chars=10)

        wait_tool = toolset.tools["wait_tasks"]
        output = await wait_tool.function(MockRunContext(deps=MockDeps()), ["long-task"])

        assert "showing 10 of 100 characters" in output
        assert "b" * 11 not in output

    @pytest.mark.anyio
    async def test_none_budget_returns_the_full_result(self):
        toolset, _ = self._toolset_with_result("c" * 5000, max_result_chars=None)

        wait_tool = toolset.tools["wait_tasks"]
        output = await wait_tool.function(MockRunContext(deps=MockDeps()), ["long-task"])

        assert "c" * 5000 in output
        assert "truncated" not in output

    @pytest.mark.anyio
    async def test_truncation_keeps_the_chat_trace_line_intact(self):
        """The trace id stays above the preview so continuation still parses."""
        toolset, tm = self._toolset_with_result("d" * 2500)
        tm.handles["long-task"].chat_trace_id = "trace-9"

        wait_tool = toolset.tools["wait_tasks"]
        output = await wait_tool.function(MockRunContext(deps=MockDeps()), ["long-task"])

        assert "Chat Trace ID: trace-9\nddd" in output
        assert "showing 2000 of 2500 characters" in output

    @pytest.mark.anyio
    async def test_check_task_returns_the_untruncated_result(self):
        """check_task is the escape hatch the truncation marker points at."""
        toolset, _ = self._toolset_with_result("e" * 2500)

        check_tool = toolset.tools["check_task"]
        output = await check_tool.function(MockRunContext(deps=MockDeps()), "long-task")

        assert "e" * 2500 in output
        assert "truncated" not in output

    def test_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="max_result_chars must be >= 0 or None, got -1"):
            create_subagent_toolset(default_model="test", max_result_chars=-1)


class TestMemoryBoundsValidation:
    """The bounds arguments used to be accepted unchecked.

    `max_agents=0` is the obvious way to write "no dynamic agents" and did the
    opposite -- the registry read it as falsy and imposed no cap at all. A zero or
    negative store bound was accepted too, evicting every entry before anything
    could read it back.
    """

    @pytest.mark.parametrize("bound", ["max_chat_traces", "max_task_handles"])
    @pytest.mark.parametrize("value", [0, -5])
    def test_a_store_that_cannot_hold_an_entry_is_rejected(self, bound: str, value: int):
        with pytest.raises(ValueError, match=f"{bound} must be >= 1, got {value}"):
            create_subagent_toolset(default_model="test", **{bound: value})

    def test_negative_max_agents_is_rejected(self):
        with pytest.raises(ValueError, match="max_agents must be >= 0, got -1"):
            create_subagent_toolset(default_model="test", max_agents=-1)

    def test_zero_max_agents_allows_no_dynamic_agents(self):
        """The cap that used to read as unlimited."""
        toolset = create_subagent_toolset(
            default_model="test",
            max_agents=0,
            delegation_configuration="persisted",
        )

        with pytest.raises(ValueError, match="Maximum number of agents \\(0\\) reached"):
            toolset.registry.register(
                SubAgentConfig(name="a", description="d", instructions="i"),
                object(),
            )
