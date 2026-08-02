"""Running one delegated task, synchronously or in the background.

## Error contract

A delegation can fail in five distinct ways, and collapsing them into one string
result -- which is what this module used to do -- loses information the parent run
needs:

- **Control-flow signals** (`CallDeferred`, `ApprovalRequired`, `Skip*`) are how
  pydantic-ai suspends a run for deferred tools or human approval. Catching them
  turns a suspended run into a tool result the parent reads as a finished task, so
  they always propagate.
- **A deferred *output*** is the same suspension arriving by the other route. An
  agent whose `output_type` includes `DeferredToolRequests` does not raise: the
  run ends normally with the parked calls as its output, exactly as a top-level
  run does for a caller that is expected to resume it. Nothing about that is a
  failure, so the guard above never sees it -- and without the check in
  `_deferred_requests` the parent is handed a serialized dataclass as the
  subagent's answer and the handle says `completed`. It is treated as the
  signal it is: `DEFERRED` on the handle, the parked calls kept on
  `TaskHandle.deferred_requests`, and the matching exception raised so the
  parent run suspends too.
- **`UserError`** is a setup mistake no retry can fix. Reporting it to the model as
  a task failure hides it, so it always propagates.
- **`UsageLimitExceeded`** means a delegation ran out of budget. Every subagent
  budget is its own -- a child run never gets the parent's usage tally -- but
  containing it would let the parent keep fanning out into an empty wallet, so it
  propagates too.
- **Everything else** is a failure the parent can react to. It reaches the parent as
  `ModelRetry`, which engages pydantic-ai's retry budget, rather than as a normal
  tool result whose text happens to start with `Error`.

Background delegations are always soft: the parent already received its tool result
(the task id), so an outcome can only be delivered as a status transition.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from pydantic_ai import DeferredToolRequests, UsageLimits
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)

from subagents_pydantic_ai._observability import (
    capture_message_history,
    capture_observability,
    serialize_output,
)
from subagents_pydantic_ai._state import QuestionBudget, SubAgentState, bind_subagent_state
from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
from subagents_pydantic_ai.prompts import get_task_instructions_prompt
from subagents_pydantic_ai.retry import OnRetryCallback, RetryConfig, run_with_retry
from subagents_pydantic_ai.types import (
    AskUserCallback,
    MessageType,
    SubAgentConfig,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    utcnow,
)

logger = logging.getLogger(__name__)

DEFAULT_ASK_TIMEOUT_SECONDS = 300.0
"""How long `ask_parent` waits for the parent before proceeding on its own."""

_ALWAYS_PROPAGATE: tuple[type[Exception], ...] = (
    CallDeferred,
    ApprovalRequired,
    SkipModelRequest,
    SkipToolValidation,
    SkipToolExecution,
    UserError,
)
"""Signals that must reach the parent run even when errors are contained.

Containing the first five breaks the agent graph -- they are deferred/approval
control flow, not failures. `UserError` is a configuration bug that no retry fixes.
"""

_USAGE_LIMIT_MARKER = "usage limit exceeded"
"""Prefix both modes record on `handle.error` when a budget runs out.

Sync propagates the exception and background mode contains it, but telemetry
should not have to know which mode a delegation ran in to spot the same outcome.
"""

_HUMAN_IN_THE_LOOP: tuple[type[Exception], ...] = (CallDeferred, ApprovalRequired)
"""Signals a background delegation cannot deliver: they need a caller to hand the
deferred state back to, and the delegating tool returned long ago."""

_SUSPENSION_ERROR = "the subagent stopped for a human decision and produced no answer"
"""What a `DEFERRED` handle says happened, whichever route the suspension arrived by."""

_BACKGROUND_SUSPENSION_ERROR = (
    "a subagent that needs approval or defers a tool call cannot run in the "
    "background; delegate it with mode='sync'."
)
"""Why a background delegation cannot suspend, whichever route it arrived by.

Shared so the exception and the deferred-output branch cannot drift into two
explanations of one rule.
"""


def _deferred_requests(output: object) -> DeferredToolRequests | None:
    """The parked tool calls in a run's output, when it suspended rather than answered.

    An empty `DeferredToolRequests` -- no calls and no approvals -- is not a
    suspension. Core does not produce one, but treating it as deferred would
    strand a delegation on nothing to decide, so it is read as an ordinary
    output and serialized like any other.
    """
    if not isinstance(output, DeferredToolRequests):
        return None
    if not output.calls and not output.approvals:
        return None
    return output


def _resolve_event_stream_handler(
    agent: Any, configured: EventStreamHandler[Any] | None
) -> EventStreamHandler[Any] | None:
    """The handler for this delegation: the agent's own first, the toolset's as a default.

    An agent handed over as `SubAgentConfig["agent"]` with a handler already on it
    was configured for that one specialist deliberately, so it wins. A
    toolset-level handler is the application saying "stream everything I did not
    configure individually" -- it fills the gap rather than overriding the
    specific choice, which is also the only way an agent the library builds
    itself (a dynamic specialist) can stream at all.
    """
    own: EventStreamHandler[Any] | None = getattr(agent, "event_stream_handler", None)
    return own or configured


def _suspension_signal(deferred: DeferredToolRequests) -> Exception:
    """The exception that reproduces a child's suspension in its parent run.

    Approvals outrank deferred calls: a parent told only that a call was
    deferred would resume it with a tool result, and nobody would ever be asked
    the question the child stopped for.
    """
    if deferred.approvals:
        return ApprovalRequired()
    return CallDeferred()


def _build_run_kwargs(
    deps: Any,
    *,
    extra_toolsets: list[Any] | None,
    usage_limits: UsageLimits | None,
    message_history: list[Any] | None,
    conversation_id: str | None,
) -> dict[str, Any]:
    run_kwargs: dict[str, Any] = {"deps": deps}
    if extra_toolsets:
        run_kwargs["toolsets"] = extra_toolsets
    if usage_limits is not None:
        run_kwargs["usage_limits"] = usage_limits
    if message_history is not None:
        run_kwargs["message_history"] = message_history
    if conversation_id is not None:
        run_kwargs["conversation_id"] = conversation_id
    return run_kwargs


def _task_prompt(config: SubAgentConfig, description: str) -> str:
    return get_task_instructions_prompt(
        description,
        can_ask_questions=config.get("can_ask_questions", True),
        max_questions=config.get("max_questions"),
    )


def _question_budget(config: SubAgentConfig) -> QuestionBudget | None:
    """The delegation's `max_questions` budget, or `None` when it is unlimited.

    One budget per delegation, so a subagent continuing a chat trace gets a fresh
    allowance rather than inheriting a spent one -- `max_questions` is documented
    per task, not per conversation.
    """
    limit = config.get("max_questions")
    return QuestionBudget(limit=limit) if limit is not None else None


async def _run_sync(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    extra_toolsets: list[Any] | None = None,
    ask_user: AskUserCallback | None = None,
    usage_limits: UsageLimits | None = None,
    handle: TaskHandle | None = None,
    message_history: list[Any] | None = None,
    on_message_history: Callable[[list[Any]], None] | None = None,
    ask_timeout_seconds: float = DEFAULT_ASK_TIMEOUT_SECONDS,
    contain_errors: bool = True,
    event_stream_handler: EventStreamHandler[Any] | None = None,
) -> str:
    """Run a subagent task synchronously, blocking until it finishes.

    Args:
        agent: The agent to run.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        extra_toolsets: Additional toolsets for this run.
        ask_user: Callback backing `ask_parent`. Required for questions in sync
            mode: the parent's run loop is blocked here, so it cannot answer
            through `answer_subagent`.
        usage_limits: Usage limits forwarded to the run, honoured on every retry.
        handle: Task handle populated for observability.
        message_history: Prior history when continuing a chat trace.
        on_message_history: Receives the successful run's full message history.
        ask_timeout_seconds: How long `ask_parent` waits for `ask_user`.
        contain_errors: Whether a subagent crash is converted into a `ModelRetry`
            for the parent instead of propagating and aborting the parent run.
            Signals in `_ALWAYS_PROPAGATE` ignore this.
        event_stream_handler: Streams this delegation's events. Used only when
            the agent carries no handler of its own.

    Returns:
        The subagent's output, or the configured `on_failure` message.

    Raises:
        ModelRetry: When the subagent failed and no `on_failure` message is set.
        ApprovalRequired: When the subagent suspended on a tool needing approval.
        CallDeferred: When the subagent suspended on a deferred tool call.
        Exception: Control-flow signals, `UserError`, `UsageLimitExceeded`,
            and -- when `contain_errors` is false -- any other subagent exception.
    """
    prompt = _task_prompt(config, description)
    run_kwargs = _build_run_kwargs(
        deps,
        extra_toolsets=extra_toolsets,
        usage_limits=usage_limits,
        message_history=message_history,
        conversation_id=handle.chat_trace_id if handle is not None else None,
    )
    state = SubAgentState(
        ask_timeout_seconds=ask_timeout_seconds,
        ask_callback=ask_user,
        questions=_question_budget(config),
    )

    try:
        with bind_subagent_state(state):
            result = await run_with_retry(
                agent,
                prompt,
                run_kwargs=run_kwargs,
                retry=RetryConfig.from_config(config),
                on_retry=_retry_recorder(handle),
                event_stream_handler=_resolve_event_stream_handler(agent, event_stream_handler),
            )
    except _HUMAN_IN_THE_LOOP as exc:
        # Before `_ALWAYS_PROPAGATE`, which also matches these: a suspension is
        # not a failure, and reporting one as `FAILED` sends a caller looking for
        # a defect instead of for the person who has to decide.
        _suspend(handle, exc)
        raise
    except _ALWAYS_PROPAGATE:
        _fail(handle, "propagated")
        raise
    except UsageLimitExceeded as exc:
        _fail(handle, f"{_USAGE_LIMIT_MARKER}: {exc}")
        raise
    except (ModelRetry, UnexpectedModelBehavior) as exc:
        return _degrade(handle, config, task_id, exc, crashed=False)
    except Exception as exc:
        if not contain_errors:
            _fail(handle, str(exc))
            raise
        return _degrade(handle, config, task_id, exc, crashed=True)

    deferred = _deferred_requests(result.output)
    if deferred is not None:
        signal = _suspension_signal(deferred)
        # Deliberately not `capture_message_history`: a suspended run has not
        # finished, and saving it would let a later `chat_trace_id` resume from a
        # point whose deferred results were never supplied.
        _suspend(handle, signal, deferred=deferred, result=result)
        raise signal

    capture_message_history(result, on_message_history)
    output = serialize_output(result.output)
    if handle is not None:
        handle.finish(TaskStatus.COMPLETED, result=output)
        # Telemetry runs after the run is marked complete, so a capture failure
        # can never flip a successful run to FAILED.
        capture_observability(handle, result)
    return output


def _retry_recorder(handle: TaskHandle | None) -> OnRetryCallback | None:
    """Record each transient retry on the handle, or `None` when there is no handle.

    Both execution modes need this: `retry_count` is documented without a mode
    qualifier, and `sync` is the default, so leaving it out there reported zero
    retries for the path most delegations take.
    """
    if handle is None:
        return None

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        handle.status = TaskStatus.RETRYING
        handle.retry_count = attempt
        handle.error = f"Transient error (retry {attempt}): {exc}"

    return on_retry


def _fail(handle: TaskHandle | None, error: str) -> None:
    if handle is not None:
        handle.finish(TaskStatus.FAILED, error=error)


def _suspend(
    handle: TaskHandle | None,
    signal: Exception,
    *,
    deferred: DeferredToolRequests | None = None,
    result: Any = None,
) -> None:
    """Record a delegation that stopped for a human instead of answering.

    `deferred` is set only on the output route; the exception route carries no
    result to read the parked calls off, which is why the two are distinguishable
    on the handle but not in its status.

    Telemetry is captured where there is a result to capture it from: the
    suspended run still spent tokens, and a cost that only lands for delegations
    that happened to finish is a cost report that under-reports exactly the runs
    a human is about to make more expensive.
    """
    if handle is None:
        return
    if handle.finish(TaskStatus.DEFERRED, error=f"{type(signal).__name__}: {_SUSPENSION_ERROR}"):
        handle.deferred_requests = deferred
        if result is not None:
            capture_observability(handle, result)


def _degrade(
    handle: TaskHandle | None,
    config: SubAgentConfig,
    task_id: str,
    exc: BaseException,
    *,
    crashed: bool,
) -> str:
    """Record a failed delegation and tell the parent about it.

    Raises `ModelRetry` unless the subagent configured an `on_failure` message, in
    which case the parent receives that as an ordinary tool result.
    """
    name = config["name"]
    _fail(handle, f"{type(exc).__name__}: {exc}")
    if crashed:
        # Contained, but loud: the exception rides the retry message and is logged,
        # and the tool's retry budget still turns repeated crashes into an abort.
        logger.warning("Contained crash from subagent %r (task %s)", name, task_id, exc_info=exc)
    on_failure = config.get("on_failure")
    if on_failure is not None:
        return on_failure
    verb = "crashed" if crashed else "failed"
    raise ModelRetry(
        f"Subagent {name!r} {verb}: {type(exc).__name__}: {exc}. "
        f"Treat this as a recoverable failure and decide from the evidence you have."
    ) from exc


async def _run_async(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    task_manager: TaskManager,
    message_bus: InMemoryMessageBus,
    priority: TaskPriority = TaskPriority.NORMAL,
    extra_toolsets: list[Any] | None = None,
    usage_limits: UsageLimits | None = None,
    chat_trace_id: str | None = None,
    message_history: list[Any] | None = None,
    on_message_history: Callable[[list[Any]], None] | None = None,
    on_run_finished: Callable[[], None] | None = None,
    ask_timeout_seconds: float = DEFAULT_ASK_TIMEOUT_SECONDS,
    parent_run_id: str | None = None,
    event_stream_handler: EventStreamHandler[Any] | None = None,
) -> str:
    """Start a subagent task in the background and return its handle text.

    Args:
        agent: The agent to run.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        task_manager: Owns the task, its handle, and its cancellation state.
        message_bus: Delivers steering messages to the running subagent.
        priority: Task priority recorded on the handle.
        extra_toolsets: Additional toolsets for this run.
        usage_limits: Usage limits forwarded to the run, honoured on every retry.
        chat_trace_id: Chat trace this conversation belongs to, when continuable.
        message_history: Prior history when continuing a chat trace.
        on_message_history: Receives the successful run's full message history.
        on_run_finished: Invoked once when the run finishes, however it finishes.
        ask_timeout_seconds: How long `ask_parent` waits for the parent's answer.
        parent_run_id: `run_id` of the parent run, so the task can be scoped to it.
        event_stream_handler: Streams this delegation's events. Used only when
            the agent carries no handler of its own.

    Returns:
        Text telling the parent the task id and how to check on it.
    """
    handle = TaskHandle(
        task_id=task_id,
        subagent_name=config["name"],
        description=description,
        status=TaskStatus.PENDING,
        priority=priority,
        chat_trace_id=chat_trace_id,
        parent_run_id=parent_run_id,
    )

    agent_id = f"subagent-{task_id}"
    try:
        message_bus.register_agent(agent_id)
    except ValueError:
        pass  # Already registered

    prompt = _task_prompt(config, description)
    run_kwargs = _build_run_kwargs(
        deps,
        extra_toolsets=extra_toolsets,
        usage_limits=usage_limits,
        message_history=message_history,
        conversation_id=chat_trace_id,
    )
    state = SubAgentState(
        ask_timeout_seconds=ask_timeout_seconds,
        task_manager=task_manager,
        task_id=task_id,
        questions=_question_budget(config),
    )

    async def run_task() -> None:
        def cancel_requested() -> bool:
            # Cooperative cancellation: `soft_cancel` sets this event and the run
            # loop polls it between graph nodes, so the subagent stops at a clean
            # boundary with its partial progress intact.
            cancel_event = task_manager.get_cancel_event(task_id)
            return cancel_event is not None and cancel_event.is_set()

        async def pending_steering() -> list[str]:
            return await drain_steering_messages(message_bus, agent_id)

        try:
            with bind_subagent_state(state):
                result = await run_with_retry(
                    agent,
                    prompt,
                    run_kwargs=run_kwargs,
                    retry=RetryConfig.from_config(config),
                    on_retry=_retry_recorder(handle),
                    cancel_check=cancel_requested,
                    inject_messages=pending_steering,
                    event_stream_handler=_resolve_event_stream_handler(agent, event_stream_handler),
                )
            deferred = _deferred_requests(result.output)
            if deferred is not None:
                # No caller left to hand the parked calls back to, so this is as
                # far as the delegation goes -- but the calls are kept on the
                # handle, which is the only way an application can see what the
                # subagent stopped on and re-run it where a human can answer.
                if handle.finish(
                    TaskStatus.DEFERRED,
                    error=(
                        f"{type(_suspension_signal(deferred)).__name__}: "
                        f"{_BACKGROUND_SUSPENSION_ERROR}"
                    ),
                ):
                    handle.deferred_requests = deferred
                    capture_observability(handle, result)
                return
            capture_message_history(result, on_message_history)
            if handle.finish(TaskStatus.COMPLETED, result=serialize_output(result.output)):
                capture_observability(handle, result)
        except asyncio.CancelledError:
            handle.finish(TaskStatus.CANCELLED, error="Task was cancelled")
            raise
        except UsageLimitExceeded as exc:
            handle.finish(TaskStatus.FAILED, error=f"{_USAGE_LIMIT_MARKER}: {exc}")
        except _HUMAN_IN_THE_LOOP as exc:
            handle.finish(
                TaskStatus.DEFERRED,
                error=f"{type(exc).__name__}: {_BACKGROUND_SUSPENSION_ERROR}",
            )
        except Exception as exc:
            logger.warning(
                "Background subagent %r failed (task %s)", config["name"], task_id, exc_info=exc
            )
            handle.finish(TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        finally:
            if handle.completed_at is None:  # pragma: no cover - every path finishes above
                handle.completed_at = utcnow()
            message_bus.unregister_agent(agent_id)
            task_manager.clear_answer_future(task_id)
            task_manager.cleanup_task(task_id)
            if on_run_finished is not None:
                on_run_finished()

    task_manager.create_task(task_id, run_task(), handle)

    response = f"Task started in background.\nTask ID: {task_id}\nSubagent: {config['name']}\n"
    if chat_trace_id is not None:
        response += f"Chat Trace ID: {chat_trace_id}\n"
    response += f"Use check_task('{task_id}') to check status."
    return response


async def drain_steering_messages(message_bus: InMemoryMessageBus, agent_id: str) -> list[str]:
    """Take the parent-to-child steering messages queued for a running subagent.

    Only `TASK_UPDATE` carries steering. Other message types on the queue (an
    unused `CANCEL_REQUEST` -- soft cancel runs off the cancel event, not the bus)
    and empty payloads are ignored.

    Args:
        message_bus: The bus the running subagent is registered on.
        agent_id: The subagent's bus id (`subagent-{task_id}`).

    Returns:
        Steering instructions in delivery order, possibly empty.
    """
    pending = await message_bus.get_messages(agent_id, timeout=0)
    steering: list[str] = []
    for msg in pending:
        if msg.type != MessageType.TASK_UPDATE:
            continue
        payload = msg.payload
        text = payload.get("message") if isinstance(payload, dict) else payload
        if text:
            steering.append(str(text))
    return steering
