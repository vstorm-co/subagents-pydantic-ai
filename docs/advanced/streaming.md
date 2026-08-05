# Streaming subagent output

A delegation can take a while. Without streaming, everything a specialist does
between "task started" and its final answer is invisible -- the user watches a
spinner, and an application that wants to show the work has nothing to show.

Pass an `event_stream_handler` and every delegation's events -- model text,
thinking, tool calls and their results -- arrive as they happen.

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig


async def on_events(ctx: RunContext[Any], events: Any) -> None:
    async for event in events:
        print(event)


agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        SubAgentCapability(
            default_model="openai:gpt-4.1",
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches topics",
                    instructions="You research topics thoroughly.",
                ),
            ],
            event_stream_handler=on_events,
        )
    ],
)
```

The handler is pydantic-ai's own
[`EventStreamHandler`](https://ai.pydantic.dev/agents/#streaming-all-events) --
the same callable an `Agent` takes -- so anything already written for a top-level
run works unchanged for a subagent.

## Labelling a fan-out

Three specialists streaming into one callback are indistinguishable: the events
themselves carry nothing that says which delegation produced them. Pass a
factory instead, and it is handed the task id.

```python
from subagents_pydantic_ai import SubAgentConfig, create_subagent_toolset


def handler_for(ctx: RunContext[Any], config: SubAgentConfig, task_id: str) -> Any:
    async def on_events(run_ctx: RunContext[Any], events: Any) -> None:
        async for event in events:
            await socket.send_json(
                {"task_id": task_id, "subagent": config["name"], "event": repr(event)}
            )

    return on_events


toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    event_stream_handler_factory=handler_for,
)
```

The factory runs once per delegation, before the subagent starts, so it can read
anything on the parent context as well. Return `None` to run that one without
streaming.

`event_stream_handler` and `event_stream_handler_factory` are mutually exclusive:
both are callables, so nothing downstream could tell them apart and one would
silently win.

## Which handler wins

| The agent for this delegation | Toolset handler | What streams |
|---|---|---|
| Has its own `event_stream_handler` | set | The agent's own |
| Has its own `event_stream_handler` | unset | The agent's own |
| Has none | set | The toolset's |
| Has none | unset | Nothing |

An agent passed in as `SubAgentConfig["agent"]` with a handler already on it was
configured for that one specialist deliberately, so it wins. The toolset's
handler is the application saying "stream everything I did not configure
individually".

## Dynamically created specialists

An agent the model creates through `create_agent` or `delegate` is built by this
library, so there is no instance for an application to attach a handler to. That
is the reason the handler is resolved per delegation rather than baked onto an
agent at construction: a dynamic specialist streams on exactly the same terms as
a configured one, with no extra wiring.

```python
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    delegation_configuration="persisted_and_oneshot",
    allowed_models=["openai:gpt-4o-mini"],
    event_stream_handler_factory=handler_for,
)
```

## Background delegations

Streaming works in both execution modes. A background delegation streams while it
runs, which means its events keep arriving after the delegating tool has already
returned its task id -- and, if the parent finishes first, after the parent's own
answer. An application that tears its display down when the parent run ends will
drop the last thing a specialist said.

## Retries

A retried attempt streams too. The handler is resolved once and honoured on every
attempt, so a transient failure mid-delegation does not silently stop the stream.
See [Retries](retries.md).

## Next steps

- [Execution modes](execution-modes.md) -- sync, background, and `auto`
- [Observability](../concepts/observability.md) -- what a finished delegation cost
- [Cancellation](cancellation.md) -- stopping a task on purpose
