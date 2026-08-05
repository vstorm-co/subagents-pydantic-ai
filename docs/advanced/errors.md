# Failure handling

A delegation can fail in several ways, and they are not interchangeable. A crashed
subagent is a setback the parent can work around; a suspended run waiting for human
approval is not a failure at all. This page covers how each one reaches the parent.

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        SubAgentCapability(
            default_model="openai:gpt-4.1",
            subagents=[
                SubAgentConfig(
                    name="scraper",
                    description="Fetches pages",
                    instructions="You fetch and summarise web pages.",
                    on_failure="The scraper is unavailable. Answer from what you have.",
                ),
            ],
        )
    ],
)
```

## The five outcomes

| What happened | How it reaches the parent |
|---|---|
| Control-flow signal (`CallDeferred`, `ApprovalRequired`, `Skip*`) | Propagates unchanged |
| A `DeferredToolRequests` **output** | The matching signal is raised, so it propagates too |
| `UserError` | Propagates unchanged |
| `UsageLimitExceeded` | Propagates unchanged from a sync delegation |
| Anything else | `ModelRetry`, or the subagent's `on_failure` message |

### Control-flow signals propagate

`CallDeferred` and `ApprovalRequired` are how pydantic-ai suspends a run so a human
can approve a tool call or a deferred tool can be resolved later. They are not
errors. Catching them would turn a suspended run into a tool result the parent
reads as a finished task, which breaks approval and deferred-tool flows outright,
so they always reach the parent run.

`UserError` propagates for a different reason: it means the setup is wrong, and no
retry fixes a misconfiguration. Reporting it to the model as a task failure hides a
bug you need to see.

### A deferred output is the same suspension, arriving differently

A subagent whose `output_type` includes `DeferredToolRequests` does not raise. Its
run ends normally, with the parked calls as its output -- exactly as a top-level run
does for a caller that is expected to resume it. Nothing about that is an
exception, so the guard above never sees it.

It is treated as the suspension it is. The handle is marked `deferred`, the parked
calls are kept on `TaskHandle.deferred_requests`, and the matching signal is raised
so the parent run suspends too: `ApprovalRequired` when anything needs approving,
`CallDeferred` otherwise. A parent told only that a call was deferred would resume
it with a tool result, and nobody would ever be asked the question the child
stopped for.

```python
handle = toolset.task_manager.get_handle(task_id)
if handle is not None and handle.status is TaskStatus.DEFERRED:
    assert handle.deferred_requests is not None
    for call in handle.deferred_requests.approvals:
        print("waiting on", call.tool_name)
```

The suspended run's chat trace is deliberately **not** saved: continuing it later
would resume from a point whose deferred results were never supplied.

!!! warning "Background mode cannot suspend"
    A background delegation returned its tool result (the task id) long before the
    suspension arrives, so there is no caller left to hand the deferred state back
    to. The task is marked `deferred` with an error saying to delegate it with
    `mode="sync"` instead, and the parked calls are still kept on the handle.
    Approval and deferred tools need synchronous delegation.

### Usage limits stop the parent run

A `UsageLimitExceeded` from a sync delegation propagates. Every subagent budget is
its own -- the library never hands a child run the parent's usage tally -- but a
subagent that ran out of budget is still the parent's problem: reporting it as one
delegation's bad luck lets the parent keep fanning out into an empty wallet. See
[Usage limits](usage-limits.md).

A background delegation contains it, like every other failure in background mode:
the tool result (the task id) went back to the parent long ago, so the only outcome
left to deliver is a status transition. Both modes record the same
`usage limit exceeded: ...` marker on `handle.error`, so telemetry does not have to
know which mode a delegation ran in.

### Everything else becomes a retry

A subagent that crashed, timed out, or returned something unusable reaches the
parent as `ModelRetry`:

```text
Subagent 'scraper' crashed: ConnectionError: Connection reset by peer.
Treat this as a recoverable failure and decide from the evidence you have.
```

`ModelRetry` is what engages pydantic-ai's retry budget: the parent model sees the
failure, can react to it, and repeated crashes eventually abort the tool call
rather than looping forever. A plain string result whose text happens to start with
`Error` would look to the model like a *successful* delegation, which is how a
failure gets quietly folded into a final answer.

## Containment

By default a crash cannot abort the parent run -- it is converted into that
`ModelRetry`, logged with its traceback at `WARNING`. Turn that off when you want a
subagent crash to surface as an exception in your application:

```python
from subagents_pydantic_ai import create_subagent_toolset

toolset = create_subagent_toolset(default_model="openai:gpt-4.1", contain_errors=False)
```

Per-subagent overrides win over the toolset default:

```python
from subagents_pydantic_ai import SubAgentConfig

SubAgentConfig(
    name="critical-writer",
    description="Writes the final report",
    instructions="You write the final report.",
    contain_errors=False,  # a failure here should stop the run
)
```

Containment never applies to the signals in the table above.

## Steering instead of retrying

`on_failure` replaces the `ModelRetry` with an ordinary tool result, which tells
the parent what to do next instead of inviting it to try the same delegation again:

```python
SubAgentConfig(
    name="scraper",
    description="Fetches pages",
    instructions="You fetch and summarise web pages.",
    on_failure=(
        "The scraper is unavailable. Summarise from the sources you already have "
        "and note the gap; do not delegate to the scraper again."
    ),
)
```

Use it when the failure is expected and re-delegating is pointless -- a flaky
external service, a paywalled source, a subagent whose dependency is down. The
handle still records `failed` with the real exception, so your telemetry sees the
truth even though the model was steered around it.

## Background failures are always soft

The parent already received the task id, so a background outcome can only be
delivered as a status transition. The task reaches `failed`, `handle.error` carries
the exception, and the failure is logged. `check_task` reports it:

```text
Task: a1b2c3d4
Subagent: scraper
Status: failed
Description: Fetch the pricing page
Error: ConnectionError: Connection reset by peer
```

Cancellation is separate: a cancelled task reaches `cancelled`, and
`asyncio.CancelledError` is re-raised so the task really stops. See
[Cancellation](cancellation.md).

## Transient failures never get this far

A flaky gateway, a rate limit, or a dropped connection is retried with exponential
backoff before it is treated as a failure, resuming from the accumulated message
history rather than restarting. Only an exhausted retry budget produces the
outcomes on this page. See [Retries](retries.md).

## Next steps

- [Retries](retries.md) -- transient failures and backoff
- [Usage limits](usage-limits.md) -- per-task budgets
- [Cancellation](cancellation.md) -- stopping a task on purpose
