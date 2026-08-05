# Observability

Every delegation records what it cost, which model ran it, which tools it called,
and where its span sits in your trace. That data lands on a `TaskHandle` and stays
queryable after the task finishes.

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

capability = SubAgentCapability(
    default_model="openai:gpt-4.1",
    subagents=[
        SubAgentConfig(
            name="researcher",
            description="Researches topics",
            instructions="You research topics thoroughly.",
        ),
    ],
)
agent = Agent("openai:gpt-4.1", capabilities=[capability])

result = await agent.run("Research Python async patterns")

for handle in capability.task_manager.list_handles():
    print(handle.subagent_name, handle.status, handle.cost, handle.tool_call_counts)
```

## What a handle carries

A `TaskHandle` is created for every delegation, sync or background, and populated
when the run finishes.

| Field | What it tells you |
|---|---|
| `task_id` | Identifier the model uses with `check_task`, `wait_tasks`, and the cancel tools |
| `subagent_name` | Which specialist ran |
| `status` | `pending`, `running`, `waiting_for_answer`, `retrying`, `completed`, `failed`, `cancelled` |
| `result` / `error` | The output, or why there isn't one |
| `created_at` / `started_at` / `completed_at` | Timezone-aware UTC timestamps |
| `usage` | pydantic-ai's `RunUsage` for the subagent run |
| `cost` | Total model cost as a `Decimal`, from genai-prices |
| `tool_call_counts` | `{tool_name: calls}`, summed across every model response |
| `model_name`, `provider_name`, `provider_url`, `provider_response_id`, `provider_details`, `finish_reason` | Metadata from the response that produced the output |
| `traceparent`, `trace_id`, `span_id` | W3C trace identifiers, when instrumentation is on |
| `conversation_id`, `run_id` | pydantic-ai's own identifiers for the subagent run |
| `message_history` | The full run serialised as JSON |
| `retry_count` | Transient failures retried before the run succeeded |
| `chat_trace_id` | The conversation this run belongs to, if it can be continued |
| `parent_run_id` | The parent run that started the task |

!!! note "Telemetry never fails a task"
    Collection runs *after* the run is marked complete and is wrapped
    best-effort, matching pydantic-ai's own instrumentation. A subagent that
    produced an answer is never reported as `failed` because a cost lookup or a
    serialisation raised -- the failure is logged at `WARNING` instead.

## Cost and tokens across a fan-out

`get_total_usage()` aggregates every task the toolset has run, including tasks
whose handles were already evicted by `max_task_handles`:

```python
from subagents_pydantic_ai import create_subagent_toolset

toolset = create_subagent_toolset(default_model="openai:gpt-4.1", max_task_handles=500)

totals = toolset.get_total_usage()
print(totals["input_tokens"], totals["output_tokens"], totals["requests"])
```

Cost is per-handle rather than aggregated, because a fan-out across several models
rarely wants one number:

```python
from decimal import Decimal

spend = sum(
    (h.cost for h in toolset.task_manager.list_handles() if h.cost is not None),
    Decimal("0"),
)
```

`cost` is `None` when genai-prices has no entry for the model and provider pair.
That is a missing price, not a free run, so treat `None` as unknown rather than
zero.

!!! tip "Aggregates versus the final response"
    `cost` and `tool_call_counts` are summed over every model response in the run.
    `model_name` and the `provider_*` fields come from the *last* response -- the
    one that produced the returned output. In a run that switched models, earlier
    responses' model metadata is not on the handle, though their cost is.

## Correlating with Logfire

When the parent agent is instrumented, each subagent run gets its own span and the
handle carries the traceparent, so you can jump from a task to its trace:

```python
import logfire
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability

logfire.configure()
logfire.instrument_pydantic_ai()

capability = SubAgentCapability(default_model="openai:gpt-4.1")
agent = Agent("openai:gpt-4.1", capabilities=[capability])

await agent.run("Delegate something")

for handle in capability.task_manager.list_handles():
    if handle.trace_id is not None:
        print(f"{handle.subagent_name}: trace {handle.trace_id} span {handle.span_id}")
```

`traceparent` is the raw W3C header (`version-trace_id-span_id-flags`);
`trace_id` and `span_id` are parsed out of it for convenience. All three are
`None` when no instrumentation is active.

## Watching background tasks from outside the agent

`task_manager.handles` is a live mapping, so a UI or a monitoring loop can poll it
while the agent is still running:

```python
import asyncio

from subagents_pydantic_ai import TaskStatus


async def report(task_manager, poll_seconds: float = 1.0) -> None:
    """Print each background task as it reaches a terminal state."""
    reported: set[str] = set()
    while True:
        for task_id, handle in list(task_manager.handles.items()):
            if handle.is_finished and task_id not in reported:
                reported.add(task_id)
                print(f"{task_id} {handle.status}: {handle.result or handle.error}")
        await asyncio.sleep(poll_seconds)
```

!!! warning "Handles are evicted"
    Finished handles past `max_task_handles` (default 500) are dropped, oldest
    first, so a long-lived session does not grow without bound. Their token usage
    is folded into `get_total_usage()` before they go, but their `result` and
    `cost` are lost. Raise the limit, or persist what you need as tasks finish.

## Bounding what stays in memory

Two limits keep a long-running orchestrator flat:

```python
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    max_task_handles=500,  # finished handles retained for status queries
    max_chat_traces=100,  # conversations that a chat_trace_id can still resume
)
```

Both evict least-recently-used entries. Continuing an evicted chat trace returns
an error telling the model to start a new conversation rather than silently
starting one.

## Next steps

- [Chat traces](../advanced/chat-traces.md) -- continue a subagent's conversation
- [Retries](../advanced/retries.md) -- what `retry_count` is counting
- [Types](types.md) -- the full `TaskHandle` reference
