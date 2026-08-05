# Chat traces

By default every delegation starts a subagent from nothing. A chat trace lets the
next delegation continue where the last one stopped, so a follow-up does not have
to restate the context.

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
                    name="analyst",
                    description="Analyses datasets",
                    instructions="You analyse datasets and explain your reasoning.",
                ),
            ],
        )
    ],
)

result = await agent.run(
    "Have the analyst look at Q3 revenue, then ask the same analyst "
    "to compare it with Q2 without re-reading Q3."
)
```

## What the model sees

Every result from a continuable delegation ends with the trace id:

```text
Revenue grew 12% quarter on quarter, driven by ...

Chat Trace ID: 4f2c8a1e9b7d4c5eab3f
```

Passing that id back continues the same conversation:

```text
task(description="Now compare with Q2", subagent_type="analyst",
     chat_trace_id="4f2c8a1e9b7d4c5eab3f")
```

Omitting it starts a fresh one. The tool description tells the model to continue
*intentionally* -- a follow-up that genuinely builds on the previous answer, not
every call to the same subagent.

## The rules

A trace belongs to one subagent and one task at a time.

- **Same subagent only.** A trace records one agent's messages; replaying them
  into a different agent produces a conversation that agent never had. Continuing
  with the wrong subagent returns an error.
- **One task at a time.** A trace with a running task cannot be continued. Two
  concurrent tasks would both save their history at the end, and the slower save
  would silently discard the faster one's work. The error tells the model to wait
  with `check_task` or `wait_tasks`.
- **Only successful runs are stored.** A failed first run saves nothing, so its id
  would resume an empty conversation. Neither the result text nor `check_task`
  advertises a trace in that case.
- **One run only.** A trace belongs to the run that started it, the same way a task
  handle does. One toolset instance serves every run of its agent, and a trace
  holds a whole subagent conversation, so a trace from another run reads exactly
  like an unknown one — including while its task is still running, so the error
  cannot be used to probe which ids exist. A trace created without a `run_id`
  stays continuable by anyone.

## Where history lives

Histories are kept in memory, bounded by an LRU:

```python
from subagents_pydantic_ai import create_subagent_toolset

toolset = create_subagent_toolset(default_model="openai:gpt-4.1", max_chat_traces=100)
```

Reading a trace refreshes its recency, so a conversation the orchestrator keeps
continuing is not evicted underneath it. Past the limit the least recently used
trace is dropped, and continuing an evicted one returns an error telling the model
to start a new conversation rather than silently starting one.

The store is readable from Python, keyed by `(subagent_name, chat_trace_id)`:

```python
for (subagent, trace_id), messages in toolset.message_history_store.items():
    print(subagent, trace_id, len(messages))
```

!!! warning "In-process only"
    Traces live in the toolset instance. They do not survive a restart and are not
    shared between workers. For a conversation that must outlive the process, keep
    pydantic-ai's own `message_history` in your own store and pass it when you
    construct the subagent.

## One-shot specialists have no trace

`delegate` builds an ephemeral specialist that is never registered, so `task`
could not resolve it later. Handing out a trace id for it would promise a
continuation that cannot happen, so one-shot runs report no trace and store no
history -- which also stops them from consuming LRU slots that a continuable
conversation would use.

See [Dynamic agents](dynamic-agents.md) for the difference between `create_agent`
and `delegate`.

## Background tasks

An async delegation reports its trace when the task starts, and `check_task` and
`wait_tasks` repeat it only once the task has **completed** -- a failed or still
running task has not saved this run's history yet, so advertising the id earlier
would invite the model to resume nothing.

```text
Task started in background.
Task ID: a1b2c3d4
Subagent: analyst
Chat Trace ID: 4f2c8a1e9b7d4c5eab3f
Use check_task('a1b2c3d4') to check status.
```

## Next steps

- [Execution modes](execution-modes.md) -- sync, background, and auto
- [Observability](../concepts/observability.md) -- `chat_trace_id` on the handle
- [Dynamic agents](dynamic-agents.md) -- reusable versus one-shot specialists
