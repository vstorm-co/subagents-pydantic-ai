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
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.models.test import TestModel

from subagents_pydantic_ai import (
    AgentMessage,
    InMemoryMessageBus,
    MessageType,
    SubAgentCapability,
    SubAgentConfig,
    TaskHandle,
    TaskManager,
    TaskStatus,
    create_subagent_toolset,
)
from subagents_pydantic_ai._execution import _run_async, _run_sync
from subagents_pydantic_ai.types import utcnow

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
    def __init__(self, output: str) -> None:
        self.result = _Result(output)
        self.next_node: Any = object()

    async def next(self, node: Any) -> Any:
        from pydantic_graph import End

        return End(self.result)

    def all_messages(self) -> list[Any]:
        return []


class _Result:
    def __init__(self, output: str) -> None:
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
        output: str = "done",
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

        with pytest.raises(UsageLimitExceeded):
            await _run_sync(
                agent=StubAgent(error=UsageLimitExceeded("out of tokens")),
                config=_config(),
                description="work",
                deps=Deps(),
                task_id="t1",
                handle=handle,
                usage_limits=UsageLimits(request_limit=1),
            )

        assert handle.status == TaskStatus.FAILED
        assert handle.error == "usage limit exceeded: out of tokens"

    async def test_async_records_usage_limit_under_the_same_marker(self) -> None:
        """A background delegation has no caller to raise into, so it records instead.

        The two modes used to write different strings for the same outcome, which
        made telemetry depend on knowing how a delegation had been dispatched.
        """
        toolset = _toolset()
        await _run_async(
            agent=StubAgent(error=UsageLimitExceeded("out of tokens")),
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

    async def test_sync_marks_handle_failed_when_signal_propagates(self) -> None:
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
        assert handle.status == TaskStatus.FAILED
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
