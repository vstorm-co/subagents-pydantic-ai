"""Regression tests for task lifecycle, run isolation, and model-facing output.

Each test here pins a defect that the previous implementation shipped while the
suite reported 100% branch coverage -- coverage proved the lines executed, not
that they produced the right answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

import pytest
from pydantic_ai import DeferredToolRequests, UsageLimits
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai import (
    AgentMessage,
    InMemoryMessageBus,
    MessageType,
    SubAgentCapability,
    SubAgentConfig,
    TaskHandle,
    TaskManager,
    TaskPriority,
    TaskStatus,
    create_subagent_toolset,
)
from subagents_pydantic_ai._chat_trace import ChatTraceStore
from subagents_pydantic_ai._execution import _question_budget, _run_async, _run_sync
from subagents_pydantic_ai._state import QuestionBudget, SubAgentState, bind_subagent_state
from subagents_pydantic_ai.types import utcnow

_UNSET = object()
"""Distinguishes "the agent was never run" from "it was handed no history"."""

# `asyncio_mode = "auto"` in pyproject.toml drives the async tests here.


@dataclass
class Deps:
    """Minimal deps satisfying `SubAgentDepsProtocol`."""

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> Deps:
        return Deps(subagents={} if max_depth <= 0 else dict(self.subagents))


@dataclass
class Ctx:
    """Stand-in for `RunContext`, carrying the fields the tools read."""

    deps: Deps = field(default_factory=Deps)
    run_id: str | None = None


class _Run:
    def __init__(self, output: Any) -> None:
        self.result = _Result(output)
        self.next_node: Any = object()

    async def next(self, node: Any) -> Any:
        from pydantic_graph import End

        return End(self.result)

    def all_messages(self) -> list[Any]:
        return []


class _Result:
    def __init__(self, output: Any) -> None:
        self.output = output


class _CM:
    def __init__(self, agent: StubAgent) -> None:
        self._agent = agent

    async def __aenter__(self) -> _Run:
        agent = self._agent
        if agent.block is not None:
            await agent.block.wait()
        if agent.error is not None:
            raise agent.error
        return _Run(agent.output)

    async def __aexit__(self, *exc: object) -> bool:
        return False


class StubAgent:
    """Agent stand-in driven through `.iter()`, like `Agent.run` drives one."""

    def __init__(
        self,
        output: Any = "done",
        error: BaseException | None = None,
        block: asyncio.Event | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.block = block

    def iter(self, prompt: Any = None, **kwargs: Any) -> _CM:
        return _CM(self)


def _config(name: str = "worker", **extra: Any) -> SubAgentConfig:
    config = SubAgentConfig(
        name=name,
        description=f"{name} description",
        instructions="Do the work.",
    )
    config.update(extra)  # type: ignore[typeddict-item]
    return config


def _toolset(**kwargs: Any) -> Any:
    return create_subagent_toolset(
        subagents=[_config()],
        include_general_purpose=False,
        default_model=TestModel(),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Model-facing status rendering
# --------------------------------------------------------------------------- #


class TestStatusRendering:
    """Statuses reach the model as their values, on every supported Python.

    Python 3.11 changed `Enum.__format__` for mixin enums, so `f"{status}"`
    produced `TaskStatus.WAITING_FOR_ANSWER` and the tool descriptions promised
    `waiting_for_answer`. No test asserted the rendered string, so the leak
    shipped.
    """

    async def test_check_task_renders_status_value(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.WAITING_FOR_ANSWER,
            pending_question="which repo?",
        )

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Status: waiting_for_answer" in result
        assert "TaskStatus." not in result

    async def test_check_task_renders_retrying_status(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.RETRYING,
            started_at=utcnow(),
        )

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Status: retrying" in result
        assert "TaskStatus." not in result

    async def test_check_task_reports_a_retry_instead_of_elapsed_time(self) -> None:
        """`RETRYING` used to fall through to the elapsed line, hiding the error."""
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.RETRYING,
            started_at=utcnow(),
            retry_count=2,
            error="Transient error (retry 2): 503",
        )

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Retry 2: Transient error (retry 2): 503" in result
        assert "Running for:" not in result

    async def test_check_task_reports_a_cancellation_instead_of_elapsed_time(self) -> None:
        """A cancelled task used to report a running-time that grew forever.

        The elapsed line read `utcnow() - started_at`, so a task cancelled hours
        ago told the orchestrating model it was still running, and never said why
        it stopped.
        """
        toolset = _toolset()
        handle = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.RUNNING,
            started_at=utcnow(),
        )
        handle.finish(TaskStatus.CANCELLED, error="Task was cancelled")
        toolset.task_manager.handles["t1"] = handle

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Status: cancelled" in result
        assert "Outcome: Task was cancelled" in result
        assert "Running for:" not in result

    async def test_list_active_tasks_renders_status_value(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="a long running job",
            deps=Deps(),
            task_id="t1",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
        )

        result = await toolset.tools["list_active_tasks"].function(Ctx())

        assert "(running)" in result
        assert "TaskStatus." not in result

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_wait_tasks_renders_pending_status_value(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.RUNNING,
        )

        result = await toolset.tools["wait_tasks"].function(Ctx(), ["t1"])

        assert "(worker): running" in result
        assert "TaskStatus." not in result

    async def test_wait_tasks_renders_terminal_status_value(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.CANCELLED,
            error="Task was cancelled",
        )

        result = await toolset.tools["wait_tasks"].function(Ctx(), ["t1"])

        assert "CANCELLED - Task was cancelled" in result
        assert "TaskStatus." not in result


# --------------------------------------------------------------------------- #
# What a status answer may say about a failure
# --------------------------------------------------------------------------- #


_KEYED_URL = "https://acme.example/v1/chat?api-key=sk-live-4f2b"


def _failed(status: TaskStatus = TaskStatus.FAILED, **extra: Any) -> TaskHandle:
    """A handle whose `error` embeds a model client's own message, as a real one does."""
    exception = RuntimeError(f"401 Unauthorized calling {_KEYED_URL}")
    return TaskHandle(
        task_id="t1",
        subagent_name="worker",
        description="dig",
        status=status,
        error=f"{type(exception).__name__}: {exception}",
        exception=exception,
        **extra,
    )


class TestAStatusAnswerNamesTheTypeNotTheMessage:
    """A task-status tool's answer is stored and shown, so it carries no vendor text.

    A host stores a `ToolReturnPart` whole -- it is the tool's own answer, not a
    retry prompt something can scrub -- and renders it to everyone who can read the
    run. A model client's message carries the failing request URL with the key still
    in its query string on a custom `base_url`, so handing `handle.error` to the
    model put that key on a transcript row. The class is named instead, and
    `handle.error` keeps the original for the host to log.
    """

    async def test_check_task_on_a_failed_task_names_the_class(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = _failed()

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Error: RuntimeError" in result
        assert _KEYED_URL not in result
        assert "401" not in result

    async def test_check_task_mid_retry_names_the_class_and_keeps_the_count(self) -> None:
        """This one leaks for a delegation that eventually succeeds.

        `finish` clears `error` on completion, but the model may have polled while
        the task was retrying, and by then the answer is already a transcript row.
        """
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = _failed(
            TaskStatus.RETRYING, started_at=utcnow(), retry_count=2
        )

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Retry 2: RuntimeError" in result
        assert _KEYED_URL not in result

    async def test_check_task_on_a_cancellation_that_carried_one_names_the_class(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = _failed(TaskStatus.CANCELLED)

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Outcome: RuntimeError" in result
        assert _KEYED_URL not in result

    async def test_wait_tasks_names_the_class(self) -> None:
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = _failed()

        result = await toolset.tools["wait_tasks"].function(Ctx(), ["t1"])

        assert "FAILED - RuntimeError" in result
        assert _KEYED_URL not in result

    async def test_a_budget_that_ran_out_still_says_so(self) -> None:
        """The marker is this library's own text, so replacing it would lose the
        one thing that tells a model to stop delegating rather than retry."""
        toolset = _toolset()
        exhausted = UsageLimitExceeded("The next request would exceed the limit")
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="dig",
            status=TaskStatus.FAILED,
            error=f"usage limit exceeded: {exhausted}",
            exception=exhausted,
        )

        result = await toolset.tools["check_task"].function(Ctx(), "t1")

        assert "Error: usage limit exceeded: UsageLimitExceeded" in result
        assert "The next request would exceed the limit" not in result

    async def test_the_handle_keeps_the_original_for_the_host_to_log(self) -> None:
        """Scrubbing the answer must not scrub the record. #699's contract is that
        `error` holds the whole thing and `exception` is how a host composes its own."""
        toolset = _toolset()
        handle = _failed()
        toolset.task_manager.handles["t1"] = handle

        await toolset.tools["check_task"].function(Ctx(), "t1")

        assert handle.error is not None
        assert _KEYED_URL in handle.error


# --------------------------------------------------------------------------- #
# Run isolation
# --------------------------------------------------------------------------- #


class TestRunIsolation:
    """One toolset instance serves every run of its agent, so tasks are scoped.

    Without scoping, a server sharing one agent across users let any run inspect,
    answer, steer, or cancel another run's task by id.
    """

    def _foreign_handle(self, toolset: Any, status: TaskStatus = TaskStatus.RUNNING) -> None:
        toolset.task_manager.handles["foreign"] = TaskHandle(
            task_id="foreign",
            subagent_name="worker",
            description="someone else's task",
            status=status,
            parent_run_id="run-a",
            pending_question="secret?",
        )

    async def test_check_task_hides_another_runs_task(self) -> None:
        toolset = _toolset()
        self._foreign_handle(toolset)

        result = await toolset.tools["check_task"].function(Ctx(run_id="run-b"), "foreign")

        assert result == "Error: Task 'foreign' not found"

    async def test_check_task_shows_own_task(self) -> None:
        toolset = _toolset()
        self._foreign_handle(toolset)

        result = await toolset.tools["check_task"].function(Ctx(run_id="run-a"), "foreign")

        assert "Subagent: worker" in result

    async def test_answer_subagent_refuses_another_runs_task(self) -> None:
        toolset = _toolset()
        self._foreign_handle(toolset, status=TaskStatus.WAITING_FOR_ANSWER)

        result = await toolset.tools["answer_subagent"].function(
            Ctx(run_id="run-b"), "foreign", "42"
        )

        assert result == "Error: Task 'foreign' not found"

    async def test_send_message_refuses_another_runs_task(self) -> None:
        toolset = _toolset()
        self._foreign_handle(toolset)
        toolset.task_manager.message_bus.register_agent("subagent-foreign")

        result = await toolset.tools["send_message_to_subagent"].function(
            Ctx(run_id="run-b"), "foreign", "stop"
        )

        assert result == "Error: Task 'foreign' not found"

    async def test_cancel_refuses_another_runs_task(self) -> None:
        toolset = _toolset()
        self._foreign_handle(toolset)

        soft = await toolset.tools["soft_cancel_task"].function(Ctx(run_id="run-b"), "foreign")
        hard = await toolset.tools["hard_cancel_task"].function(Ctx(run_id="run-b"), "foreign")

        assert soft == "Error: Task 'foreign' not found"
        assert hard == "Error: Task 'foreign' not found"

    async def test_cancel_refuses_another_runs_live_task(self) -> None:
        """The cancel tools once admitted a foreign task that was actually running.

        `_foreign_handle` registers no `asyncio.Task`, so the guard those tools used
        (`handle is None and task_id not in tasks`) short-circuited on the missing
        task and the test above passed without ever reaching the broken case. With a
        live task registered, the guard fell through and the cancel went ahead.
        """
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="theirs",
            deps=Deps(),
            task_id="victim",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )

        soft = await toolset.tools["soft_cancel_task"].function(Ctx(run_id="run-b"), "victim")
        hard = await toolset.tools["hard_cancel_task"].function(Ctx(run_id="run-b"), "victim")

        assert soft == "Error: Task 'victim' not found"
        assert hard == "Error: Task 'victim' not found"
        # The returned string alone passed against the broken code too: assert the
        # task actually survived.
        assert toolset.task_manager.handles["victim"].status == TaskStatus.RUNNING

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_cancel_allows_own_live_task(self) -> None:
        """Scoping a cancel must not stop the run that owns the task."""
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="mine",
            deps=Deps(),
            task_id="mine",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )

        result = await toolset.tools["hard_cancel_task"].function(Ctx(run_id="run-a"), "mine")

        assert result == "Task 'mine' has been cancelled"
        assert toolset.task_manager.handles["mine"].status == TaskStatus.CANCELLED

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_cancel_allows_an_unowned_task(self) -> None:
        """A task with no handle has no owner to compare against, so it stays cancellable."""
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="orphan",
            deps=Deps(),
            task_id="orphan",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )
        del toolset.task_manager.handles["orphan"]

        result = await toolset.tools["soft_cancel_task"].function(Ctx(run_id="run-b"), "orphan")

        assert result == "Cancellation requested for task 'orphan'"

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_list_active_tasks_omits_another_runs_task(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="theirs",
            deps=Deps(),
            task_id="t-a",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )

        result = await toolset.tools["list_active_tasks"].function(Ctx(run_id="run-b"))

        assert result == "No active background tasks."

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_wait_tasks_reports_another_runs_task_as_missing(self) -> None:
        toolset = _toolset()
        self._foreign_handle(toolset)

        result = await toolset.tools["wait_tasks"].function(Ctx(run_id="run-b"), ["foreign"])

        assert "- foreign: not found" in result


# --------------------------------------------------------------------------- #
# Terminal transitions
# --------------------------------------------------------------------------- #


class TestTerminalTransitions:
    """The first terminal transition wins."""

    def test_finish_is_idempotent(self) -> None:
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        assert handle.finish(TaskStatus.COMPLETED, result="the answer") is True
        completed_at = handle.completed_at

        assert handle.finish(TaskStatus.CANCELLED, error="too late") is False
        assert handle.status == TaskStatus.COMPLETED
        assert handle.result == "the answer"
        assert handle.error is None
        assert handle.completed_at == completed_at

    def test_finish_records_timezone_aware_timestamps(self) -> None:
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")
        handle.finish(TaskStatus.COMPLETED, result="x")

        assert handle.created_at.tzinfo is not None
        assert handle.completed_at is not None
        assert handle.completed_at.tzinfo == timezone.utc

    async def test_hard_cancel_cannot_overwrite_a_completed_task(self) -> None:
        """A task inside its `finally` is not `done()`, but its outcome is settled.

        The old guard was `if not task.done()`, which is still true while the
        task runs its cleanup, so a cancel arriving in that window replaced the
        real result with `CANCELLED`.
        """
        manager = TaskManager(message_bus=InMemoryMessageBus())
        in_cleanup = asyncio.Event()
        may_exit = asyncio.Event()
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        async def work() -> None:
            try:
                handle.finish(TaskStatus.COMPLETED, result="the answer")
            finally:
                in_cleanup.set()
                await may_exit.wait()

        task = manager.create_task("t1", work(), handle)
        await in_cleanup.wait()

        assert await manager.hard_cancel("t1") is True
        assert handle.status == TaskStatus.COMPLETED
        assert handle.result == "the answer"

        may_exit.set()
        await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Background tasks do not outlive their parent run
# --------------------------------------------------------------------------- #


class TestTaskCancellation:
    async def test_cancel_all_stops_live_tasks(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="never finishes on its own",
            deps=Deps(),
            task_id="t1",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )
        await asyncio.sleep(0)

        await toolset.cancel_run_tasks("run-a")

        assert toolset.task_manager.handles["t1"].status == TaskStatus.CANCELLED
        assert toolset.task_manager.list_active_tasks() == []

    async def test_cancelling_a_finished_task_reports_its_status(self) -> None:
        """ "Not found" for a task whose result is still readable is misleading."""
        toolset = _toolset()
        toolset.task_manager.handles["t1"] = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="d",
            status=TaskStatus.COMPLETED,
            result="the answer",
        )
        ctx = Ctx()

        soft = await toolset.tools["soft_cancel_task"].function(ctx, "t1")
        hard = await toolset.tools["hard_cancel_task"].function(ctx, "t1")

        for message in (soft, hard):
            assert "no longer running (status: completed)" in message
            assert "check_task('t1')" in message

    async def test_cancel_all_leaves_other_runs_alone(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        # `t-b` sits between two tasks of `run-a` so both the skip and the cancel
        # branch continue into another iteration.
        for task_id, run_id in (("t-a", "run-a"), ("t-b", "run-b"), ("t-c", "run-a")):
            await _run_async(
                agent=StubAgent(block=block),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id=task_id,
                task_manager=toolset.task_manager,
                message_bus=toolset.task_manager.message_bus,
                parent_run_id=run_id,
            )
        await asyncio.sleep(0)

        await toolset.cancel_run_tasks("run-a")

        assert toolset.task_manager.handles["t-a"].status == TaskStatus.CANCELLED
        assert toolset.task_manager.handles["t-c"].status == TaskStatus.CANCELLED
        assert toolset.task_manager.handles["t-b"].status == TaskStatus.RUNNING

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_cancel_all_without_run_id_stops_everything(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )
        await asyncio.sleep(0)

        await toolset.task_manager.cancel_all()

        assert toolset.task_manager.handles["t1"].status == TaskStatus.CANCELLED

    async def test_cancel_all_skips_finished_and_handleless_tasks(self) -> None:
        """`cancel_all` tolerates a registry that is not perfectly tidy.

        A task that already finished but whose entry has not been cleaned up must
        not be cancelled again, and a task tracked without a handle (created
        directly rather than through a delegation) must still be stopped.
        """
        manager = TaskManager(message_bus=InMemoryMessageBus())
        block = asyncio.Event()

        async def finished() -> None:
            return None

        async def forever() -> None:
            await block.wait()

        done_handle = TaskHandle(task_id="done", subagent_name="worker", description="d")
        manager.create_task("done", finished(), done_handle)
        await asyncio.sleep(0)
        assert manager.tasks["done"].done()
        done_handle.finish(TaskStatus.COMPLETED, result="kept")

        # A bare task with no handle, which `cancel_all` must still cancel.
        manager.tasks["bare"] = asyncio.create_task(forever())
        await asyncio.sleep(0)

        await manager.cancel_all()

        assert not manager.tasks["done"].cancelled()
        assert done_handle.status == TaskStatus.COMPLETED
        assert done_handle.result == "kept"
        assert manager.tasks["bare"].cancelled()

    async def test_cancel_all_stops_a_task_parked_in_ask_parent(self) -> None:
        """A task parked in `ask_parent` stops now, not after the ask timeout."""
        manager = TaskManager(message_bus=InMemoryMessageBus())
        handle = TaskHandle(
            task_id="t1",
            subagent_name="worker",
            description="d",
            parent_run_id="run-a",
        )
        answered = False

        async def work() -> None:
            nonlocal answered
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            manager.set_answer_future("t1", future)
            await asyncio.wait_for(future, timeout=300.0)
            answered = True

        task = manager.create_task("t1", work(), handle)
        await asyncio.sleep(0)

        await manager.cancel_all("run-a")

        assert task.done()
        assert answered is False
        assert handle.status == TaskStatus.CANCELLED

    async def test_soft_cancel_unblocks_a_task_parked_in_ask_parent(self) -> None:
        """Soft cancel resolves the pending question so the stop lands immediately.

        A task inside `ask_parent` sits in a tool call, not at a node boundary, so
        the cooperative cancel would otherwise only take effect after the ask
        timeout expired.
        """
        manager = TaskManager(message_bus=InMemoryMessageBus())
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")
        received: list[str] = []
        done = asyncio.Event()

        async def work() -> None:
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            manager.set_answer_future("t1", future)
            received.append(await future)
            done.set()

        task = manager.create_task("t1", work(), handle)
        await asyncio.sleep(0)

        assert await manager.soft_cancel("t1") is True
        await asyncio.wait_for(done.wait(), timeout=1.0)

        assert received == [
            "Your parent agent cancelled this task. Stop working and wrap up immediately."
        ]
        await task

    async def test_capability_cancels_its_run_tasks(self) -> None:
        """`SubAgentCapability.wrap_run` is the finalizer that guarantees this."""
        capability = SubAgentCapability(
            subagents=[_config()],
            include_general_purpose=False,
            default_model=TestModel(),
        )
        toolset = capability.get_toolset()
        assert toolset is not None
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="outlives the run unless cancelled",
            deps=Deps(),
            task_id="t1",
            task_manager=capability.task_manager,
            message_bus=capability.task_manager.message_bus,
            parent_run_id="run-a",
        )
        await asyncio.sleep(0)

        async def handler() -> Any:
            return "run result"

        result = await capability.wrap_run(Ctx(run_id="run-a"), handler=handler)

        assert result == "run result"
        assert capability.task_manager.handles["t1"].status == TaskStatus.CANCELLED

    async def test_capability_cancels_tasks_even_when_the_run_raises(self) -> None:
        capability = SubAgentCapability(
            subagents=[_config()],
            include_general_purpose=False,
            default_model=TestModel(),
        )
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            task_manager=capability.task_manager,
            message_bus=capability.task_manager.message_bus,
            parent_run_id="run-a",
        )
        await asyncio.sleep(0)

        async def handler() -> Any:
            raise RuntimeError("the run blew up")

        with pytest.raises(RuntimeError, match="the run blew up"):
            await capability.wrap_run(Ctx(run_id="run-a"), handler=handler)

        assert capability.task_manager.handles["t1"].status == TaskStatus.CANCELLED


# --------------------------------------------------------------------------- #
# Error contract
# --------------------------------------------------------------------------- #


class TestErrorContract:
    async def test_sync_propagates_usage_limit(self) -> None:
        """Containing an exhausted budget would let the parent keep delegating."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")
        exhausted = UsageLimitExceeded("out of tokens")

        with pytest.raises(UsageLimitExceeded):
            await _run_sync(
                agent=StubAgent(error=exhausted),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id="t1",
                handle=handle,
                usage_limits=UsageLimits(request_limit=1),
            )

        assert handle.status == TaskStatus.FAILED
        assert handle.error == "usage limit exceeded: out of tokens"
        assert handle.exception is exhausted

    async def test_async_records_usage_limit_under_the_same_marker(self) -> None:
        """A background delegation has no caller to raise into, so it records instead.

        The two modes used to write different strings for the same outcome, which
        made telemetry depend on knowing how a delegation had been dispatched.
        """
        toolset = _toolset()
        exhausted = UsageLimitExceeded("out of tokens")
        await _run_async(
            agent=StubAgent(error=exhausted),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
        )
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

        handle = toolset.task_manager.handles["t1"]
        assert handle.status == TaskStatus.FAILED
        assert handle.error == "usage limit exceeded: out of tokens"
        assert handle.exception is exhausted

    @pytest.mark.parametrize(
        "error",
        [ModelRetry("please rephrase"), UnexpectedModelBehavior("garbled tool call")],
    )
    async def test_sync_soft_failure_is_reported_as_failed_not_crashed(
        self, error: BaseException, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A soft child failure is expected, so it is not logged as a crash."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with caplog.at_level(logging.WARNING), pytest.raises(ModelRetry, match="failed"):
            await _run_sync(
                agent=StubAgent(error=error),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.FAILED
        assert "Contained crash" not in caplog.text

    async def test_sync_soft_failure_honours_on_failure(self) -> None:
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        result = await _run_sync(
            agent=StubAgent(error=ModelRetry("please rephrase")),
            config=_config(on_failure="Work with what you have."),
            description="work",
            deps=Deps(),
            task_id="t1",
            handle=handle,
        )

        assert result == "Work with what you have."
        assert handle.status == TaskStatus.FAILED

    async def test_config_can_override_contain_errors(self) -> None:
        """A subagent may opt out of containment while the toolset contains by default."""
        toolset = _toolset(contain_errors=True)
        toolset._compiled["worker"].agent = StubAgent(error=ValueError("bad argument"))
        toolset._compiled["worker"].config["contain_errors"] = False

        with pytest.raises(ValueError, match="bad argument"):
            await toolset.tools["task"].function(Ctx(), "work", "worker")

    async def test_a_failed_delegation_hands_the_host_the_exception_behind_the_text(self) -> None:
        """`TaskHandle.exception` carries the exception whose text `error` embeds.

        A provider error's message can carry the failing request URL with the
        key still in its query string. A host that must not surface foreign
        text reads the class from here, composes its own sentence and logs the
        original (agenticos#699).
        """
        boom = RuntimeError("401 from https://llm.example.com/v1?api_key=sk-secret")
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with pytest.raises(ModelRetry):
            await _run_sync(
                agent=StubAgent(error=boom),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.FAILED
        assert handle.exception is boom

    async def test_a_background_failure_records_the_exception_too(self) -> None:
        """Both dispatch modes fill `TaskHandle.exception`, like `error` itself."""
        boom = RuntimeError("401 from https://llm.example.com/v1?api_key=sk-secret")
        manager = TaskManager(message_bus=InMemoryMessageBus())

        await _run_async(
            agent=StubAgent(error=boom),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            task_manager=manager,
            message_bus=manager.message_bus,
        )
        await asyncio.gather(*manager.tasks.values(), return_exceptions=True)

        handle = manager.handles["t1"]
        assert handle.status == TaskStatus.FAILED
        assert handle.exception is boom

    async def test_sync_marks_handle_deferred_when_a_suspension_propagates(self) -> None:
        """A suspension is not a failure, and the handle has to say which it was.

        `FAILED` sent a caller looking for a defect when what the delegation
        needs is a person to decide.
        """
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with pytest.raises(CallDeferred):
            await _run_sync(
                agent=StubAgent(error=CallDeferred()),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.DEFERRED
        assert handle.error is not None
        assert handle.error.startswith("CallDeferred:")

    async def test_sync_marks_handle_failed_when_a_non_suspension_signal_propagates(self) -> None:
        """`UserError` and the `Skip*` signals still propagate as failures."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with pytest.raises(UserError):
            await _run_sync(
                agent=StubAgent(error=UserError("misconfigured")),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.FAILED
        assert handle.error == "propagated"

    @pytest.mark.parametrize("signal", [CallDeferred(), ApprovalRequired()])
    async def test_background_reports_human_in_the_loop_as_unsupported(
        self, signal: BaseException
    ) -> None:
        """A background task has no caller to hand deferred state back to."""
        manager = TaskManager(message_bus=InMemoryMessageBus())

        await _run_async(
            agent=StubAgent(error=signal),
            config=_config(),
            description="needs approval",
            deps=Deps(),
            task_id="t1",
            task_manager=manager,
            message_bus=manager.message_bus,
        )
        await asyncio.gather(*manager.tasks.values(), return_exceptions=True)

        handle = manager.handles["t1"]
        assert handle.status == TaskStatus.DEFERRED
        assert "cannot run in the background" in (handle.error or "")
        assert "mode='sync'" in (handle.error or "")


# --------------------------------------------------------------------------- #
# ask_parent timeout
# --------------------------------------------------------------------------- #


class TestAskTimeout:
    async def test_timeout_is_configurable(self) -> None:
        """The 300s wait used to be hardcoded, with no way to shorten it."""
        manager = TaskManager(message_bus=InMemoryMessageBus())
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")
        manager.handles["t1"] = handle

        from subagents_pydantic_ai._state import SubAgentState, bind_subagent_state
        from subagents_pydantic_ai.toolset import _create_ask_parent_toolset

        ask_parent = _create_ask_parent_toolset().tools["ask_parent"]
        state = SubAgentState(ask_timeout_seconds=0.01, task_manager=manager, task_id="t1")

        with bind_subagent_state(state):
            result = await ask_parent.function(Ctx(), "which repo?")

        assert result == "Error: Parent did not respond in time"
        assert handle.status == TaskStatus.RUNNING
        assert handle.pending_question is None

    def test_non_positive_timeout_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ask_timeout_seconds must be > 0"):
            _toolset(ask_timeout_seconds=0)


# --------------------------------------------------------------------------- #
# Message bus
# --------------------------------------------------------------------------- #


class TestMessageBusHandlers:
    async def test_failing_handler_is_logged_and_delivery_continues(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A raising observer used to be swallowed by a bare `except: pass`."""
        bus = InMemoryMessageBus()
        queue = bus.register_agent("worker")
        seen: list[AgentMessage] = []

        async def boom(message: AgentMessage) -> None:
            raise RuntimeError("handler exploded")

        async def record(message: AgentMessage) -> None:
            seen.append(message)

        bus.add_handler(boom)
        bus.add_handler(record)

        with caplog.at_level(logging.WARNING):
            await bus.send(
                AgentMessage(
                    type=MessageType.TASK_UPDATE,
                    sender="parent",
                    receiver="worker",
                    payload={"message": "hi"},
                    task_id="t1",
                )
            )

        assert queue.qsize() == 1
        assert len(seen) == 1
        assert "handler exploded" in caplog.text


# --------------------------------------------------------------------------- #
# Run isolation, part two: the stores the handle gate never covered
# --------------------------------------------------------------------------- #


class TestChatTraceIsolation:
    """A chat trace belongs to the run that started it.

    Handles were scoped by `parent_run_id`; traces were keyed by
    `(subagent_name, chat_trace_id)` with no owner, so one run could pass an id it
    had seen and have the subagent replay another run's whole conversation.
    """

    def _seeded(self) -> Any:
        toolset = _toolset()
        toolset._compiled["worker"].agent = _HistoryRecordingAgent()
        toolset._chat_traces.mark_active(("worker", "TRACE-A"), "run-a")
        toolset._chat_traces.save(("worker", "TRACE-A"), ["run-a's private conversation"])
        toolset._chat_traces.release(("worker", "TRACE-A"))
        return toolset

    async def _resume(self, toolset: Any, run_id: str | None) -> Any:
        return await toolset.tools["task"].function(
            Ctx(run_id=run_id),
            "continue",
            "worker",
            "sync",
            TaskPriority.NORMAL,
            None,
            False,
            False,
            "TRACE-A",
        )

    async def test_another_run_cannot_resume_a_trace(self) -> None:
        toolset = self._seeded()

        result = await self._resume(toolset, "run-b")

        assert "no saved conversation for chat_trace_id 'TRACE-A'" in result
        # The returned string alone is not enough: assert the history never
        # reached the subagent.
        assert toolset._compiled["worker"].agent.seen_history is _UNSET

    async def test_a_foreign_active_trace_still_reads_as_unknown(self) -> None:
        """The 'already has a running task' branch would confirm the id exists."""
        toolset = self._seeded()
        toolset._chat_traces.mark_active(("worker", "TRACE-A"), "run-a")

        result = await self._resume(toolset, "run-b")

        assert "no saved conversation" in result
        assert "already has a running task" not in result

    async def test_the_owning_run_still_resumes(self) -> None:
        toolset = self._seeded()

        result = await self._resume(toolset, "run-a")

        assert "no saved conversation" not in result
        assert toolset._compiled["worker"].agent.seen_history == ["run-a's private conversation"]

    async def test_an_unclaimed_trace_stays_open(self) -> None:
        """Matches `_handle_for`: state with no recorded owner is visible to all."""
        toolset = _toolset()
        toolset._compiled["worker"].agent = _HistoryRecordingAgent()
        toolset._chat_traces.save(("worker", "TRACE-A"), ["no owner"])

        result = await self._resume(toolset, "run-b")

        assert "no saved conversation" not in result
        assert toolset._compiled["worker"].agent.seen_history == ["no owner"]


class TestChatTraceOwnership:
    """Ownership bookkeeping in the store itself."""

    def test_the_first_claim_wins(self) -> None:
        store = ChatTraceStore(max_traces=10)

        store.mark_active(("w", "t"), "run-a")
        store.mark_active(("w", "t"), "run-b")

        assert store.owned_by(("w", "t"), "run-a") is True
        assert store.owned_by(("w", "t"), "run-b") is False

    def test_a_trace_claimed_without_a_run_id_stays_open(self) -> None:
        store = ChatTraceStore(max_traces=10)

        store.mark_active(("w", "t"), None)

        assert store.owned_by(("w", "t"), "anyone") is True

    def test_releasing_an_unsaved_trace_drops_its_claim(self) -> None:
        """A first run that failed saves nothing, so the id protects nothing."""
        store = ChatTraceStore(max_traces=10)
        store.mark_active(("w", "t"), "run-a")

        store.release(("w", "t"))

        assert store.owned_by(("w", "t"), "run-b") is True

    def test_releasing_a_saved_trace_keeps_its_claim(self) -> None:
        store = ChatTraceStore(max_traces=10)
        store.mark_active(("w", "t"), "run-a")
        store.save(("w", "t"), ["history"])

        store.release(("w", "t"))

        assert store.owned_by(("w", "t"), "run-b") is False

    def test_eviction_drops_the_claim(self) -> None:
        store = ChatTraceStore(max_traces=1)
        store.mark_active(("w", "old"), "run-a")
        store.save(("w", "old"), ["old"])
        store.release(("w", "old"))

        store.save(("w", "new"), ["new"])

        assert ("w", "old") not in store
        assert store.owned_by(("w", "old"), "run-b") is True

    def test_eviction_keeps_the_claim_of_a_running_trace(self) -> None:
        """A trace evicted mid-run is still live; `release` drops it afterwards."""
        store = ChatTraceStore(max_traces=1)
        store.mark_active(("w", "old"), "run-a")
        store.save(("w", "old"), ["old"])

        store.save(("w", "new"), ["new"])

        assert store.owned_by(("w", "old"), "run-b") is False


class TestWaitTasksIsolation:
    """`wait_tasks` reported another run's task as missing and awaited it anyway.

    The existing test used a handle with no `asyncio.Task` registered, which is the
    same fixture that hid the cancel-tool defect: with nothing in
    `task_manager.tasks` the await never happened, so the unscoped lookup that
    built the pending list was never exercised.
    """

    async def test_does_not_await_another_runs_live_task(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="theirs",
            deps=Deps(),
            task_id="victim",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )

        started = asyncio.get_running_loop().time()
        result = await toolset.tools["wait_tasks"].function(Ctx(run_id="run-b"), ["victim"], 5.0)
        elapsed = asyncio.get_running_loop().time() - started

        assert "- victim: not found" in result
        # Timing is the assertion that matters: the old code blocked for the full
        # timeout, which is also an existence oracle over foreign task ids.
        assert elapsed < 1.0

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)

    async def test_still_awaits_its_own_task(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="mine",
            deps=Deps(),
            task_id="mine",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )

        async def unblock() -> None:
            await asyncio.sleep(0.05)
            block.set()

        asyncio.ensure_future(unblock())
        result = await toolset.tools["wait_tasks"].function(Ctx(run_id="run-a"), ["mine"], 5.0)

        assert "1/1 finished" in result
        assert "COMPLETED" in result

    async def test_a_missing_task_is_not_counted_as_running(self) -> None:
        """The header said "1 still running" in the same breath as "not found"."""
        toolset = _toolset()

        result = await toolset.tools["wait_tasks"].function(Ctx(run_id="run-b"), ["nosuch"], 0.0)

        assert result.startswith("Task results (mode=all, 0/1 finished, 1 not found):")
        assert "still running" not in result

    async def test_running_and_missing_are_counted_separately(self) -> None:
        toolset = _toolset()
        block = asyncio.Event()
        await _run_async(
            agent=StubAgent(block=block),
            config=_config(),
            description="mine",
            deps=Deps(),
            task_id="mine",
            task_manager=toolset.task_manager,
            message_bus=toolset.task_manager.message_bus,
            parent_run_id="run-a",
        )

        result = await toolset.tools["wait_tasks"].function(
            Ctx(run_id="run-a"), ["mine", "nosuch"], 0.0
        )

        assert "0/2 finished" in result
        assert "1 still running" in result
        assert "1 not found" in result

        block.set()
        await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)


# --------------------------------------------------------------------------- #
# Question contracts
# --------------------------------------------------------------------------- #


class TestCanAskQuestions:
    """`can_ask_questions=False` has to remove the tool, not just discourage it.

    The flag was honoured for runtime-injected specialists and ignored when
    compiling a configured subagent, so a background task could park in
    WAITING_FOR_ANSWER for the full `ask_timeout_seconds` while the parent's own
    instructions said that subagent never asks.
    """

    def _tool_names(self, toolset: Any, name: str) -> list[str]:
        agent = toolset._compiled[name].agent
        return [
            tool_name
            for sub in agent.toolsets
            for tool_name in getattr(sub, "tools", {})
            if getattr(sub, "id", None) == "ask_parent"
        ]

    def test_a_quiet_subagent_has_no_ask_parent(self) -> None:
        toolset = create_subagent_toolset(
            subagents=[_config("quiet", can_ask_questions=False)],
            include_general_purpose=False,
            default_model=TestModel(),
        )

        assert self._tool_names(toolset, "quiet") == []

    def test_a_talkative_subagent_keeps_ask_parent(self) -> None:
        toolset = create_subagent_toolset(
            subagents=[_config("chatty", can_ask_questions=True)],
            include_general_purpose=False,
            default_model=TestModel(),
        )

        assert self._tool_names(toolset, "chatty") == ["ask_parent"]

    def test_the_default_keeps_ask_parent(self) -> None:
        toolset = _toolset()

        assert self._tool_names(toolset, "worker") == ["ask_parent"]

    def test_a_quiet_subagent_keeps_its_own_toolsets(self) -> None:
        """Dropping `ask_parent` must not drop the configured toolsets with it."""
        extra: FunctionToolset[Any] = FunctionToolset(id="extra")
        toolset = create_subagent_toolset(
            subagents=[_config("quiet", can_ask_questions=False, toolsets=[extra])],
            include_general_purpose=False,
            default_model=TestModel(),
        )

        assert extra in toolset._compiled["quiet"].agent.toolsets


class TestQuestionBudget:
    """`max_questions` reached the subagent only as a sentence in its prompt.

    A model that ignores it asks forever, and every unanswered question costs the
    task up to `ask_timeout_seconds`.
    """

    def _ask_parent(self, toolset: Any, name: str = "asker") -> Any:
        agent = toolset._compiled[name].agent
        ask = next(t for t in agent.toolsets if getattr(t, "id", None) == "ask_parent")
        return ask.tools["ask_parent"].function

    def _toolset_with_budget(self, limit: int | None) -> Any:
        extra: dict[str, Any] = {"max_questions": limit} if limit is not None else {}
        return create_subagent_toolset(
            subagents=[_config("asker", **extra)],
            include_general_purpose=False,
            default_model=TestModel(),
            ask_user=self._answerer,
        )

    @staticmethod
    async def _answerer(question: str) -> str:
        return f"answer to {question}"

    async def test_questions_past_the_limit_are_refused(self) -> None:
        toolset = self._toolset_with_budget(2)
        ask_parent = self._ask_parent(toolset)
        state = SubAgentState(
            ask_timeout_seconds=1.0,
            ask_callback=self._answerer,
            questions=QuestionBudget(limit=2),
        )

        with bind_subagent_state(state):
            first = await ask_parent(Ctx(), "one?")
            second = await ask_parent(Ctx(), "two?")
            third = await ask_parent(Ctx(), "three?")

        assert first == "answer to one?"
        assert second == "answer to two?"
        assert third == "Error: question limit reached (2 for this task). " + (
            "Finish with the information you already have."
        )

    async def test_no_limit_means_unlimited(self) -> None:
        toolset = self._toolset_with_budget(None)
        ask_parent = self._ask_parent(toolset)
        state = SubAgentState(ask_timeout_seconds=1.0, ask_callback=self._answerer)

        with bind_subagent_state(state):
            answers = [await ask_parent(Ctx(), f"q{i}?") for i in range(5)]

        assert answers == [f"answer to q{i}?" for i in range(5)]

    def test_the_budget_is_built_from_the_config(self) -> None:
        assert _question_budget(_config("a", max_questions=3)) == QuestionBudget(limit=3)
        assert _question_budget(_config("a")) is None

    def test_consume_counts_and_stops(self) -> None:
        budget = QuestionBudget(limit=1)

        assert budget.consume() is True
        assert budget.asked == 1
        assert budget.consume() is False
        assert budget.asked == 1

    def test_an_unlimited_budget_never_refuses(self) -> None:
        budget = QuestionBudget()

        assert all(budget.consume() for _ in range(10))


class _HistoryRecordingAgent:
    """Records the `message_history` it was handed, so a leak is assertable."""

    def __init__(self) -> None:
        self.seen_history: Any = _UNSET

    def iter(self, prompt: Any = None, **kwargs: Any) -> Any:
        self.seen_history = kwargs.get("message_history")
        return _CM(StubAgent())


# --------------------------------------------------------------------------- #
# Deferred output
# --------------------------------------------------------------------------- #


def _deferred(*, approvals: bool = False, calls: bool = False) -> DeferredToolRequests:
    """A `DeferredToolRequests` holding whichever kind of parked call is asked for."""
    return DeferredToolRequests(
        calls=[ToolCallPart(tool_name="fetch", args={}, tool_call_id="c1")] if calls else [],
        approvals=(
            [ToolCallPart(tool_name="send_email", args={}, tool_call_id="a1")] if approvals else []
        ),
    )


class TestDeferredOutput:
    """A suspended subagent must not be reported to the parent as a finished one.

    `_ALWAYS_PROPAGATE` guards the exception route, but an agent whose
    `output_type` includes `DeferredToolRequests` never raises: the run ends
    normally with the parked calls as its output. That reached `serialize_output`,
    so the parent read `{"calls": [], "approvals": [...]}` as the subagent's
    answer and the handle said `completed` -- the exact outcome the module
    docstring says the design prevents, arriving by the other route.
    """

    async def test_sync_raises_approval_required_instead_of_answering(self) -> None:
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with pytest.raises(ApprovalRequired):
            await _run_sync(
                agent=StubAgent(output=_deferred(approvals=True)),
                config=_config(),
                description="send the email",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.DEFERRED
        assert handle.result is None

    async def test_sync_raises_call_deferred_when_nothing_needs_approval(self) -> None:
        """Only deferred calls means nobody has to be asked, so `CallDeferred` is the signal."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with pytest.raises(CallDeferred):
            await _run_sync(
                agent=StubAgent(output=_deferred(calls=True)),
                config=_config(),
                description="fetch it",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.DEFERRED

    async def test_an_approval_outranks_a_deferred_call(self) -> None:
        """A parent told only that a call was deferred would never ask anyone."""
        with pytest.raises(ApprovalRequired):
            await _run_sync(
                agent=StubAgent(output=_deferred(approvals=True, calls=True)),
                config=_config(),
                description="both",
                deps=Deps(),
                task_id="t1",
            )

    async def test_the_parked_calls_survive_on_the_handle(self) -> None:
        """The raise carries no result, so the handle is the only place they can be read."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        with pytest.raises(ApprovalRequired):
            await _run_sync(
                agent=StubAgent(output=_deferred(approvals=True)),
                config=_config(),
                description="send the email",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.deferred_requests is not None
        assert [call.tool_name for call in handle.deferred_requests.approvals] == ["send_email"]

    async def test_a_suspended_run_does_not_save_its_chat_trace(self) -> None:
        """Resuming a trace saved mid-suspension would replay a point that never resolved."""
        saved: list[list[Any]] = []

        with pytest.raises(ApprovalRequired):
            await _run_sync(
                agent=StubAgent(output=_deferred(approvals=True)),
                config=_config(),
                description="send the email",
                deps=Deps(),
                task_id="t1",
                on_message_history=saved.append,
            )

        assert saved == []

    async def test_an_empty_deferred_output_is_an_ordinary_result(self) -> None:
        """Nothing is parked, so there is nothing to strand the delegation on."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")

        result = await _run_sync(
            agent=StubAgent(output=DeferredToolRequests()),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            handle=handle,
        )

        assert handle.status == TaskStatus.COMPLETED
        assert "calls" in result

    async def test_background_records_deferred_rather_than_completed(self) -> None:
        """No caller is left to hand the parked calls back to, so this is as far as it goes."""
        manager = TaskManager(message_bus=InMemoryMessageBus())

        await _run_async(
            agent=StubAgent(output=_deferred(approvals=True)),
            config=_config(),
            description="send the email",
            deps=Deps(),
            task_id="t1",
            task_manager=manager,
            message_bus=manager.message_bus,
        )
        await asyncio.gather(*manager.tasks.values(), return_exceptions=True)

        handle = manager.handles["t1"]
        assert handle.status == TaskStatus.DEFERRED
        assert handle.result is None
        assert "mode='sync'" in (handle.error or "")
        assert handle.deferred_requests is not None

    async def test_a_cancel_that_already_landed_outranks_the_suspension(self) -> None:
        """First terminal transition wins: a cancelled task does not become deferred."""
        handle = TaskHandle(task_id="t1", subagent_name="worker", description="d")
        handle.finish(TaskStatus.CANCELLED, error="Task was cancelled")

        with pytest.raises(ApprovalRequired):
            await _run_sync(
                agent=StubAgent(output=_deferred(approvals=True)),
                config=_config(),
                description="send the email",
                deps=Deps(),
                task_id="t1",
                handle=handle,
            )

        assert handle.status == TaskStatus.CANCELLED
        assert handle.deferred_requests is None

    async def test_a_background_cancel_that_already_landed_outranks_the_suspension(self) -> None:
        """The same invariant on the background path, where the cancel arrives mid-run."""
        manager = TaskManager(message_bus=InMemoryMessageBus())
        block = asyncio.Event()

        await _run_async(
            agent=StubAgent(output=_deferred(approvals=True), block=block),
            config=_config(),
            description="send the email",
            deps=Deps(),
            task_id="t1",
            task_manager=manager,
            message_bus=manager.message_bus,
        )
        await asyncio.sleep(0)
        manager.handles["t1"].finish(TaskStatus.CANCELLED, error="Task was cancelled")
        block.set()
        await asyncio.gather(*manager.tasks.values(), return_exceptions=True)

        handle = manager.handles["t1"]
        assert handle.status == TaskStatus.CANCELLED
        assert handle.deferred_requests is None

    async def test_check_task_tells_the_model_a_delegation_is_waiting_on_a_person(self) -> None:
        """The model reads the outcome, not the status name, so the guidance has to be in it."""
        toolset = _toolset()
        toolset._compiled["worker"].agent = StubAgent(output=_deferred(approvals=True))

        with pytest.raises(ApprovalRequired):
            await toolset.tools["task"].function(Ctx(), "send the email", "worker")

        task_id = next(iter(toolset.task_manager.handles))
        rendered = await toolset.tools["check_task"].function(Ctx(), task_id)

        assert "Status: deferred" in rendered
        assert "human decision" in rendered


# --------------------------------------------------------------------------- #
# Event streaming
# --------------------------------------------------------------------------- #


class _StreamingAgent:
    """Agent stand-in that reports whichever handler `run_with_retry` resolved."""

    def __init__(self, own_handler: Any = None) -> None:
        if own_handler is not None:
            self.event_stream_handler = own_handler
        self.seen_handler: Any = _UNSET

    def iter(self, prompt: Any = None, **kwargs: Any) -> _CM:
        return _CM(StubAgent())


class TestEventStreamHandler:
    """Streaming a delegation must not require reaching into each agent instance.

    A handler could only be set on the agent object, which a dynamically created
    specialist does not have -- the library builds it -- so "stream subagents"
    and "let the model create specialists" were mutually exclusive, undocumented.
    """

    @staticmethod
    def _record(calls: list[tuple[str, str]], label: str) -> Any:
        async def handler(ctx: Any, events: Any) -> None:  # pragma: no cover - never awaited
            calls.append((label, "streamed"))

        return handler

    async def test_the_toolset_handler_reaches_the_run(self, monkeypatch: Any) -> None:
        seen: dict[str, Any] = {}

        async def fake_run_with_retry(*args: Any, **kwargs: Any) -> Any:
            seen["handler"] = kwargs.get("event_stream_handler")
            return _Run("done").result

        monkeypatch.setattr("subagents_pydantic_ai._execution.run_with_retry", fake_run_with_retry)
        handler = self._record([], "toolset")

        await _run_sync(
            agent=StubAgent(),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            event_stream_handler=handler,
        )

        assert seen["handler"] is handler

    async def test_an_agents_own_handler_wins_over_the_toolsets(self, monkeypatch: Any) -> None:
        """A handler set on one specialist is a deliberate choice about that specialist."""
        seen: dict[str, Any] = {}

        async def fake_run_with_retry(*args: Any, **kwargs: Any) -> Any:
            seen["handler"] = kwargs.get("event_stream_handler")
            return _Run("done").result

        monkeypatch.setattr("subagents_pydantic_ai._execution.run_with_retry", fake_run_with_retry)
        own = self._record([], "own")
        toolset_handler = self._record([], "toolset")

        await _run_sync(
            agent=_StreamingAgent(own_handler=own),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            event_stream_handler=toolset_handler,
        )

        assert seen["handler"] is own

    async def test_a_background_delegation_streams_too(self, monkeypatch: Any) -> None:
        seen: dict[str, Any] = {}

        async def fake_run_with_retry(*args: Any, **kwargs: Any) -> Any:
            seen["handler"] = kwargs.get("event_stream_handler")
            return _Run("done").result

        monkeypatch.setattr("subagents_pydantic_ai._execution.run_with_retry", fake_run_with_retry)
        manager = TaskManager(message_bus=InMemoryMessageBus())
        handler = self._record([], "toolset")

        await _run_async(
            agent=StubAgent(),
            config=_config(),
            description="work",
            deps=Deps(),
            task_id="t1",
            task_manager=manager,
            message_bus=manager.message_bus,
            event_stream_handler=handler,
        )
        await asyncio.gather(*manager.tasks.values(), return_exceptions=True)

        assert seen["handler"] is handler

    async def test_the_factory_receives_the_task_id_that_labels_the_fan_out(self) -> None:
        """Events carry nothing to tell concurrent specialists apart; the task id does."""
        seen: list[tuple[str, str]] = []

        def factory(ctx: Any, config: SubAgentConfig, task_id: str) -> Any:
            seen.append((config["name"], task_id))
            return None

        toolset = _toolset(event_stream_handler_factory=factory)
        toolset._compiled["worker"].agent = StubAgent()

        await toolset.tools["task"].function(Ctx(), "work", "worker")

        assert len(seen) == 1
        name, task_id = seen[0]
        assert name == "worker"
        assert task_id in toolset.task_manager.handles

    def test_a_handler_and_a_factory_together_are_refused(self) -> None:
        """Both are callables, so nothing downstream could tell them apart."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            _toolset(
                event_stream_handler=self._record([], "h"),
                event_stream_handler_factory=lambda ctx, config, task_id: None,
            )


# --------------------------------------------------------------------------- #
# Cancellation is bounded
# --------------------------------------------------------------------------- #


class TestCancelGrace:
    """`cancel_all` runs in the finalizer of a parent run, so its wait must end.

    `CancelledError` can be caught, and a subagent's toolset is arbitrary consumer
    code. Awaiting each cancelled task with no bound meant one subagent that
    swallowed the cancel held the parent run's teardown open forever, with
    nothing logged.
    """

    @staticmethod
    async def _uncancellable(started: asyncio.Event, release: asyncio.Event) -> None:
        started.set()
        while True:
            try:
                await release.wait()
                return
            except asyncio.CancelledError:
                # Exactly the shape a too-wide `suppress` or a shielded client has.
                continue

    async def test_a_task_that_ignores_cancellation_does_not_hang_the_caller(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = TaskManager(message_bus=InMemoryMessageBus(), cancel_grace_seconds=0.01)
        started, release = asyncio.Event(), asyncio.Event()
        handle = TaskHandle(task_id="stuck", subagent_name="worker", description="d")
        manager.create_task("stuck", self._uncancellable(started, release), handle)
        await started.wait()

        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(manager.cancel_all(), timeout=2)

        assert "did not unwind" in caplog.text
        assert "stuck" in caplog.text
        assert handle.status == TaskStatus.CANCELLED

        release.set()
        await asyncio.sleep(0)

    async def test_a_well_behaved_task_is_still_awaited(self) -> None:
        """The bound must not turn a clean unwind into a leak report."""
        manager = TaskManager(message_bus=InMemoryMessageBus(), cancel_grace_seconds=5.0)
        cleaned = asyncio.Event()

        async def polite() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaned.set()
                raise

        handle = TaskHandle(task_id="polite", subagent_name="worker", description="d")
        manager.create_task("polite", polite(), handle)
        await asyncio.sleep(0)

        await manager.cancel_all()

        assert cleaned.is_set()
        assert manager.tasks["polite"].cancelled()

    def test_a_non_positive_grace_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cancel_grace_seconds must be > 0"):
            _toolset(cancel_grace_seconds=0)
