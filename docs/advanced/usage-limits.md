# Usage limits

A fan-out of subagents can burn a lot of tokens before the parent notices. Usage
limits put a ceiling on every delegated run, either the same ceiling everywhere
or one computed per task.

```python
from pydantic_ai import Agent, UsageLimits
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        SubAgentCapability(
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches topics",
                    instructions="You research topics thoroughly.",
                ),
            ],
            usage_limits=UsageLimits(request_limit=10, output_tokens_limit=20_000),
        )
    ],
)
```

Every delegation now runs under those limits, retries included -- a retried attempt
does not get a fresh allowance.

## Per-task limits

Pass a factory instead of an instance to decide per delegation. It receives the
parent's run context and the selected subagent's config:

```python
from pydantic_ai import RunContext, UsageLimits
from subagents_pydantic_ai import SubAgentConfig, create_subagent_toolset


def limits_for(ctx: RunContext[object], config: SubAgentConfig) -> UsageLimits | None:
    if config["name"] == "summariser":
        return UsageLimits(request_limit=3)
    if config.get("typical_complexity") == "complex":
        return UsageLimits(request_limit=30, output_tokens_limit=100_000)
    return UsageLimits(request_limit=10)


toolset = create_subagent_toolset(usage_limits=limits_for)
```

Returning `None` runs that task without explicit limits, which is how you exempt a
specific subagent from a global ceiling.

The factory is called once per delegation, before the subagent starts, so it can
read anything on the parent context -- the tenant on `ctx.deps`, how much budget the
run has already spent, the time of day:

```python
def limits_for(ctx: RunContext[MyDeps], config: SubAgentConfig) -> UsageLimits | None:
    remaining = ctx.deps.budget.remaining_tokens()
    if remaining <= 0:
        return UsageLimits(request_limit=1)
    return UsageLimits(output_tokens_limit=min(remaining, 50_000))
```

## What happens when a limit is hit

`UsageLimitExceeded` from a sync delegation propagates to the parent run rather
than being contained. Quietly reporting an exhausted budget as one subagent's bad
luck would let the parent keep delegating into an empty wallet. A background
delegation has no caller left to raise into, so it records the failure on the
handle instead -- see [Failure handling](errors.md).

Either way the task handle records `failed`, so your telemetry sees which
delegation hit the ceiling:

```python
for handle in toolset.task_manager.list_handles():
    if handle.error is not None and handle.error.startswith("usage limit exceeded"):
        print(f"{handle.subagent_name} ran out of budget")
```

See [Failure handling](errors.md) for how this differs from a crash.

## Limits versus accounting

A limit caps a run. It does not tell you what the run cost -- for that, read
`handle.cost` and `handle.usage`, or aggregate with `get_total_usage()`:

```python
totals = toolset.get_total_usage()
print(totals["input_tokens"], totals["output_tokens"], totals["requests"])
```

See [Observability](../concepts/observability.md).

!!! tip "Pair limits with `max_result_chars`"
    A token limit stops a subagent from *generating* too much. It does nothing about
    a fan-out of ten verbose subagents flooding the orchestrator's context when
    `wait_tasks` reports them. `max_result_chars` (default 2000) caps each result in
    that listing and points the model at `check_task` for the full text.

## Next steps

- [Failure handling](errors.md) -- what propagates and what is contained
- [Observability](../concepts/observability.md) -- cost and token accounting
- [Execution modes](execution-modes.md) -- limits apply to both modes
