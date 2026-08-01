"""Auto-retry for subagent runs over flaky networking.

Wraps a subagent's execution so transient failures (gateway 5xx, rate
limits, connection drops from proxies such as LiteLLM) are retried with
exponential backoff. Each retry resumes with the full accumulated message
history, so partial progress (model turns, tool calls) is not lost. All
attempts share one `RunUsage`, so a configured `usage_limits` caps the
whole delegation rather than being granted again per attempt.

The retry path deliberately uses :meth:`Agent.iter` rather than
`capture_run_messages()` to recover the failed run's messages, because
nested `capture_run_messages` contexts do not work
(https://github.com/pydantic/pydantic-ai/issues/1568) and subagents
always run nested inside a parent agent's run. `Agent.iter` exposes the
accumulated history directly, sidestepping that limitation entirely.

Driving the run is more than `async for _ in run`: a bare loop uses
`AgentRun.__anext__`, which skips capability node hooks and never streams
the model response, so the configured `event_stream_handler` (and any
`wrap_run_event_stream` capability, e.g. a UI event stream) never fires —
tool-call and reasoning events would silently be lost. :func:`run_with_retry`
therefore drives the run exactly the way :meth:`Agent.run` does, streaming
each model-request/tool node so the handler receives every event.

When retrying is disabled (`max_retries == 0`) the legacy `agent.run()`
call path is used unchanged (it already honours the handler), so behaviour
only differs when retrying is opted in.

## Coupling to pydantic-ai internals

:func:`_drive_run` mirrors the driving loop inside :meth:`Agent.run`, which
means it uses private names (`run._advance_graph`, `run._run_node_with_hooks`,
`_agent_graph.build_run_context`, `run.ctx.deps.root_capability`). There is no
public way to drive a run with a custom step while keeping event streaming, so
the copy is deliberate — but it can drift. `tests/test_retry.py` asserts the
observable contract the copy has to preserve (node hooks fire, the handler
receives every event, the wrapped stream is fully drained), and
`pyproject.toml` pins a `pydantic-ai-slim` lower bound for the names above.
Removing the copy needs a public primitive in core, not a workaround here.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import _agent_graph
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.usage import RunUsage
from pydantic_graph import End

from subagents_pydantic_ai.types import SubAgentConfig

# HTTP status codes worth retrying: request timeout, rate limit, and 5xx
# server/gateway errors (502/503/504 are typical of an overloaded LiteLLM
# proxy). 409 Conflict and 425 Too Early are deliberately absent — they signal
# a request the server rejected on its merits, and replaying it unchanged is
# not expected to succeed.
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})

RetryPredicate = Callable[[BaseException], bool]
"""Predicate deciding whether an exception is a transient, retryable error."""

OnRetryCallback = Callable[[int, BaseException, float], "Awaitable[None] | None"]
"""Callback invoked before each retry sleep with `(attempt, exc, delay)`."""


def is_transient_error(exc: BaseException) -> bool:
    """Return `True` if *exc* looks like a transient networking failure.

    Treated as transient (worth retrying):

    - `ModelHTTPError` with a 408/429/5xx status code — gateway hiccups,
      rate limits or upstream overload, typical with proxies such as
      LiteLLM.
    - `ModelAPIError` that is *not* an HTTP error — connection resets,
      read timeouts and other transport-level problems surfaced by the
      model client.

    Everything else (auth/4xx, `UnexpectedModelBehavior`,
    `UsageLimitExceeded`, `UserError`, validation errors, task
    cancellation, ...) is treated as non-transient and is not retried.
    """
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _TRANSIENT_STATUS_CODES
    # A bare ModelAPIError (no HTTP status) is a transport/connection
    # error from the model client — safe to retry.
    return isinstance(exc, ModelAPIError)


@dataclass(frozen=True)
class RetryConfig:
    """Resolved retry policy for a subagent run.

    Attributes:
        max_retries: Number of *additional* attempts after the first
            failure. Defaults to `3` so subagents are resilient to
            flaky model gateways/networks out of the box. Set `0` to
            disable retrying entirely (the legacy `agent.run()`
            opt-out path).
        initial_delay: Seconds to wait before the first retry.
        max_delay: Upper bound for the backoff delay, in seconds.
        backoff_multiplier: The delay is multiplied by this each attempt.
        jitter: When `True`, the delay is randomised in
            `[0, computed_delay]` (full jitter) to avoid a thundering
            herd across many concurrent subagents.
        retry_on: Predicate deciding whether an exception is transient.
            `None` uses :func:`is_transient_error`.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on: RetryPredicate | None = None

    @classmethod
    def from_config(cls, config: SubAgentConfig) -> RetryConfig:
        """Build a :class:`RetryConfig` from a :class:`SubAgentConfig`.

        Missing keys fall back to the dataclass defaults, so a config
        without any `retry_*` keys yields the default policy (3
        retries with exponential backoff).
        """
        return cls(
            max_retries=config.get("max_retries", 3),
            initial_delay=config.get("retry_initial_delay", 1.0),
            max_delay=config.get("retry_max_delay", 30.0),
            backoff_multiplier=config.get("retry_backoff_multiplier", 2.0),
            jitter=config.get("retry_jitter", True),
            retry_on=config.get("retry_on"),
        )

    def should_retry(self, exc: BaseException) -> bool:
        """Return whether *exc* is retryable under this policy."""
        predicate = self.retry_on or is_transient_error
        return predicate(exc)


def compute_backoff_delay(
    attempt: int,
    cfg: RetryConfig,
    rng: Callable[[float, float], float] = random.uniform,
) -> float:
    """Compute the delay (seconds) before retry *attempt* (1-based).

    Exponential backoff (`initial_delay * multiplier ** (attempt - 1)`)
    capped at `cfg.max_delay`. With `cfg.jitter` the result is
    randomised in `[0, delay]` (full jitter). `rng` is injectable for
    deterministic tests.
    """
    base = cfg.initial_delay * (cfg.backoff_multiplier ** (attempt - 1))
    delay = min(base, cfg.max_delay)
    if cfg.jitter:
        delay = rng(0.0, delay)
    return delay


def _wants_event_stream(run: Any) -> bool:
    """Return `True` when a capability needs the run streamed.

    Mirrors the `agent_run.ctx.deps.root_capability.has_wrap_run_event_stream`
    check in :meth:`Agent.run`: a capability such as a UI event stream overrides
    `wrap_run_event_stream` and must see events even without an explicit
    `event_stream_handler`. Defensive against fakes and older pydantic-ai
    builds that lack the capability surface (treated as "no streaming").
    """
    try:
        return bool(run.ctx.deps.root_capability.has_wrap_run_event_stream)
    except AttributeError:
        return False


async def _drive_run(
    agent: Any,
    run: Any,
    event_stream_handler: Any,
    cancel_check: Callable[[], bool] | None = None,
    inject_messages: Callable[[], Awaitable[list[str]]] | None = None,
) -> None:
    """Advance *run* to completion, firing the same hooks as :meth:`Agent.run`.

    A bare `async for _ in run` uses `AgentRun.__anext__`, which skips the
    node hooks (`before_node_run` / `after_node_run` / `wrap_node_run` /
    `on_node_run_error`). :meth:`Agent.run` always drives via
    `AgentRun.next` (or the streaming step) so those hooks fire. We mirror it
    exactly: when a streaming consumer is present we replicate the streaming
    step (forwarding events through `wrap_run_event_stream`); otherwise we
    advance via `run.next(node)` — same node hooks, no streaming overhead.

    `cancel_check` enables cooperative (soft) cancellation: it is polled
    between graph nodes and, when it returns `True`, the loop stops by
    raising :class:`asyncio.CancelledError`. The run is left at a clean node
    boundary, so partial progress (completed model turns / tool calls) is
    preserved on the accumulated history. Because `CancelledError` is a
    `BaseException` it is not caught by the retry loop and propagates to the
    caller's cancellation handling unchanged.

    `inject_messages` enables unprompted parent -> child steering: it is
    awaited between nodes and its messages are handed to `AgentRun.enqueue`,
    pydantic-ai's own primitive for injecting content into a live run. Core
    delivers them before the next model request, so the subagent sees them as
    extra user instructions while keeping all partial progress, and the parts
    can never be spliced into a tool-call/tool-return pair. Splicing
    `UserPromptPart`s into `node.request.parts` by hand did the same job before
    `enqueue` existed and is no longer needed.
    """

    _stream_step: Any = None
    if event_stream_handler is not None or _wants_event_stream(run):

        async def _stream_and_advance(node: Any) -> Any:
            if agent.is_model_request_node(node) or agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    run_ctx = _agent_graph.build_run_context(run.ctx)
                    wrapped = run.ctx.deps.root_capability.wrap_run_event_stream(
                        run_ctx, stream=stream
                    )
                    if event_stream_handler is not None:
                        await event_stream_handler(run_ctx, wrapped)
                    # Drain whatever the handler left unconsumed so the node can
                    # finish through any stream wrappers. Matches `Agent.run`,
                    # which drains unconditionally; draining only in the
                    # no-handler case left wrappers unfinished for a handler that
                    # stops reading early.
                    async for _ in wrapped:
                        pass
            return await run._advance_graph(node)

        _stream_step = _stream_and_advance

    node = run.next_node
    while not isinstance(node, End):
        # Cooperative cancellation: stop at a clean node boundary when a soft
        # cancel has been requested, before advancing into the next node.
        if cancel_check is not None and cancel_check():
            raise asyncio.CancelledError("Task soft-cancelled")
        # A capability's wrap_run short-circuit can publish the result early.
        if run.result is not None:
            break
        if inject_messages is not None:
            steering = await inject_messages()
            if steering:
                run.enqueue(*steering, priority="asap")
        if _stream_step is not None:
            node = await run._run_node_with_hooks(node, _stream_step)
        else:
            node = await run.next(node)


async def run_with_retry(
    agent: Any,
    user_prompt: str | None,
    *,
    run_kwargs: dict[str, Any],
    retry: RetryConfig,
    on_retry: OnRetryCallback | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    event_stream_handler: Any | None = None,
    cancel_check: Callable[[], bool] | None = None,
    inject_messages: Callable[[], Awaitable[list[str]]] | None = None,
) -> Any:
    """Run *agent* with auto-retry on transient errors.

    When `retry.max_retries <= 0` this is exactly `agent.run(...)` —
    the legacy path, unchanged. Otherwise the agent is driven via
    `agent.iter()` so that, on a transient failure, the accumulated
    message history from the failed attempt is replayed via
    `message_history` on the next attempt and the subagent resumes
    instead of restarting from scratch.

    Args:
        agent: The pydantic-ai `Agent` to run.
        user_prompt: Initial prompt. After a retry that captured history
            it is set to `None` because the prompt is already replayed
            inside `message_history`.
        run_kwargs: Extra kwargs forwarded to `agent.run`/`agent.iter`
            (`deps`, `toolsets`, ...). A caller-supplied
            `message_history` is honoured as the starting history, and a
            caller-supplied `usage` as the tally every attempt adds to.
        retry: The resolved retry policy.
        on_retry: Optional callback invoked before each retry sleep with
            `(attempt, exc, delay)`. May be sync or async.
        sleep: Async sleep function, injectable for tests.
        event_stream_handler: Optional override for the agent's configured
            `event_stream_handler`. When `None` the agent's own handler
            (`agent.event_stream_handler`) is used, so streaming to a
            platform (e.g. tool-call/reasoning events to Kafka) keeps working
            across retries — matching `agent.run()` semantics.
        cancel_check: Optional callable polled between graph nodes for
            cooperative (soft) cancellation. When it returns `True` the run
            stops at the next node boundary by raising
            `asyncio.CancelledError`. Only honoured on the retry-driven path
            (`max_retries > 0`); the legacy `agent.run()` fast path
            (`max_retries <= 0`) does not expose node boundaries, so soft
            cancel is best-effort there.
        inject_messages: Optional async callable awaited before each model
            request; its returned strings are appended to that request as
            user instructions (unprompted parent -> child steering). Like
            `cancel_check`, only honoured on the retry-driven path
            (`max_retries > 0`); the legacy `agent.run()` fast path does not
            expose node boundaries, so steering messages stay queued there.

    Returns:
        The `AgentRunResult` of the first successful attempt.

    Raises:
        Exception: The last exception when retries are exhausted or the error
            is not transient. `asyncio.CancelledError` is a `BaseException`
            and is never caught here, so cooperative/hard task cancellation
            propagates unchanged.
    """
    # An explicit handler overrides the agent's; otherwise inherit the agent's
    # own, exactly as agent.run() does (event_stream_handler or self.…).
    handler = event_stream_handler or getattr(agent, "event_stream_handler", None)

    if retry.max_retries <= 0:
        # Fast path: agent.run() already drives streaming and honours the
        # agent's handler. Only forward an explicit override.
        if event_stream_handler is not None:
            run_kwargs = {**run_kwargs, "event_stream_handler": event_stream_handler}
        return await agent.run(user_prompt, **run_kwargs)

    message_history = run_kwargs.pop("message_history", None)
    # One tally across every attempt. `Agent.iter` builds a fresh `RunUsage` when
    # `usage` is `None`, so without this each attempt starts counting from zero
    # while the replayed history genuinely re-spends the tokens -- and a
    # `usage_limits` ceiling would be granted again per attempt, multiplying the
    # caller's budget by `max_retries + 1`.
    caller_usage: RunUsage | None = run_kwargs.pop("usage", None)
    run_usage = caller_usage if caller_usage is not None else RunUsage()
    prompt = user_prompt
    attempt = 0
    while True:
        run = None
        try:
            async with agent.iter(
                prompt, message_history=message_history, usage=run_usage, **run_kwargs
            ) as run:
                await _drive_run(agent, run, handler, cancel_check, inject_messages)
            return run.result
        except Exception as exc:
            if attempt >= retry.max_retries or not retry.should_retry(exc):
                raise
            attempt += 1
            # Resume from wherever the failed attempt got to. `run` is
            # None only if `agent.iter()` failed before yielding.
            if run is not None:
                accumulated = run.all_messages()
                if accumulated:
                    message_history = accumulated
                    prompt = None
            delay = compute_backoff_delay(attempt, retry)
            if on_retry is not None:
                maybe_coro = on_retry(attempt, exc, delay)
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            await sleep(delay)
