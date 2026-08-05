# Toolset

The `create_subagent_toolset()` function creates a toolset that adds delegation capabilities to your Pydantic AI agent.

## Creating a Toolset

```python
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics",
        instructions="You are a research assistant.",
    ),
]

toolset = create_subagent_toolset(default_model="openai:gpt-4.1", subagents=subagents)
```

## Factory Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subagents` | `list[SubAgentConfig]` | `[]` | List of subagent configurations |
| `default_model` | `str \| Model \| None` | `None` | Model a subagent falls back on when it names none. No implicit default: when unset, a subagent or dynamic call naming no model is refused rather than run on a library-chosen one |
| `toolsets_factory` | `Callable` | `None` | Factory to create toolsets for subagents |
| `include_general_purpose` | `bool` | `True` | Include the general-purpose subagent. Needs a `default_model` or `default_agent_factory` to build it from; construction raises without one |
| `max_nesting_depth` | `int` | `0` | Maximum subagent nesting depth |
| `registry` | `DynamicAgentRegistry \| None` | `None` | Registry for dynamically created agents |
| `descriptions` | `dict[str, str] \| None` | `None` | Override default tool descriptions by tool name |
| `delegation_configuration` | `DelegationConfiguration` | `"default"` | Controls whether task-only, persisted, one-shot, or combined entry points are exposed |
| `allowed_models` | `list[str] \| None` | `None` | Model allow-list for dynamic specialists |
| `capabilities_map` | `dict[str, Callable] \| None` | `None` | Capability factories for dynamic specialists |
| `default_agent_factory` | `Callable \| None` | `None` | Custom agent factory for dynamic specialists |
| `max_agents` | `int` | `10` | Maximum persistent agents in an internally created registry |
| `max_chat_traces` | `int` | `100` | Max subagent conversations kept for `chat_trace_id` continuation (LRU-evicted) |
| `max_task_handles` | `int` | `500` | Max finished task handles retained for status/observability (oldest evicted; usage totals preserved) |
| `max_result_chars` | `int \| None` | `2000` | Character budget for a result shown by `wait_tasks`; longer results are cut with an explicit marker. `None` disables truncation |

## Adding to an Agent

```python
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate tasks to specialized subagents.",
)
```

## Custom Agent Resolution

When the toolset compiles a `SubAgentConfig`, the internal `_compile_subagent` function
decides which agent instance to use. The resolution priority is:

1. **`config["agent"]`** -- a pre-built agent instance, used as-is.
2. **`config["agent_factory"]`** -- a `(config: SubAgentConfig) -> Agent` callable, invoked at compile time.
3. **Default** -- a new `pydantic_ai.Agent` is created from the config's `model`, `instructions`, `toolsets`, and `agent_kwargs` fields.

This means you can mix pre-built agents, factory-built agents, and default agents in the
same subagent list:

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=[
        SubAgentConfig(
            name="custom",
            description="Uses a pre-built agent",
            instructions="...",
            agent=my_prebuilt_agent,
        ),
        SubAgentConfig(
            name="factory-built",
            description="Built via factory",
            instructions="...",
            agent_factory=lambda cfg: build_agent(cfg),
        ),
        SubAgentConfig(
            name="default",
            description="Default Agent created automatically",
            instructions="...",
        ),
    ],
)
```

See [SubAgentConfig](types.md#subagentconfig) for full documentation of the `agent` and
`agent_factory` fields.

A caller-supplied agent (`agent` or `agent_factory`) is used exactly as given, so the
`ask_parent` tool the default branch would attach is not compiled into it. `task`
injects that tool at run time instead when the subagent's `can_ask_questions` allows it,
so a pre-built or factory-built subagent asks questions on the same terms as a
default-built one. If your factory attaches its own `ask_parent`, leave
`can_ask_questions` off to avoid a duplicate.

## Available Tools

The toolset provides these tools to your agent:

### Delegation modes

| Mode | Entry-point tools |
|------|-------------------|
| `"default"` | `task` |
| `"persisted"` | `create_agent`, `task` |
| `"persisted_and_oneshot"` | `create_agent`, `task`, `delegate` |
| `"oneshot_only"` | `delegate` |

Task lifecycle tools remain available in every mode so async work can be checked,
steered, awaited, answered, or cancelled.

### create_agent

Create and register a reusable specialist. The resulting name can be passed to
`task` multiple times.

### task

Delegate a task to a subagent.

```python
# The agent calls:
task(
    description="Research Python async patterns",
    subagent_type="researcher",
    mode="sync",  # or "async" or "auto"
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | `str` | What the subagent should do |
| `subagent_type` | `str` | Name of the subagent to use |
| `mode` | `str` | `"sync"`, `"async"`, or `"auto"` |
| `chat_trace_id` | `str \| None` | Continue a previous subagent conversation (see below) |

#### Stateful conversations (`chat_trace_id`)

Every successful `task()` result ends with a `Chat Trace ID: <id>` line. Passing
that ID back to `task()` resumes the same subagent conversation: the subagent
sees the full message history of its previous run and continues from there.
Omit `chat_trace_id` to start a fresh conversation.

`delegate()` results carry no trace ID, because a one-shot specialist is never
registered and so cannot be named in a follow-up `task()` call.

```python
# First task — a new conversation is created automatically:
task(description="Read the auth module and summarize it", subagent_type="researcher")
# -> "...summary...\n\nChat Trace ID: 3f2a..."

# Follow-up in the same conversation — the subagent remembers the module:
task(
    description="Now list the security issues you noticed",
    subagent_type="researcher",
    chat_trace_id="3f2a...",
)
```

Rules and limits:

- A trace belongs to one subagent: the history is stored per
  `(subagent_name, chat_trace_id)`.
- A trace can only be continued after its current task finishes. Continuing a
  trace that still has a running task returns an error — wait via
  `check_task`/`wait_tasks` first.
- Passing an unknown `chat_trace_id` (typo, evicted trace, or a trace whose
  first run failed) returns an error instead of silently starting over.
- Only the `max_chat_traces` most recently used conversations are kept
  (default 100, configurable on `create_subagent_toolset`); older ones are
  evicted to bound memory in long-lived sessions. One-shot `delegate` runs are
  not stored, so they never evict a conversation you can still continue.
- History grows with every continuation — the full prior conversation is
  replayed on each resumed run, so long traces cost more tokens per call.

### delegate

Create an ephemeral specialist and delegate a task in one call. Available in
`"persisted_and_oneshot"` and `"oneshot_only"` modes.

```python
# The agent calls:
delegate(
    description="Analyze this dataset and summarize key trends",
    instructions="You are a data analyst. Return concise findings.",
    name="data-analyst",
    mode="sync",
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | `str` | Task prompt for the specialist |
| `instructions` | `str` | Specialist system prompt |
| `name` | `str` | Label for logs and async task handles (letters, numbers, hyphens) |
| `model` | `str \| None` | Optional model override |
| `capabilities` | `list[str] \| None` | Optional capability names |
| `can_ask_questions` | `bool` | Whether the specialist can ask the parent |
| `mode` | `str` | `"sync"`, `"async"`, or `"auto"` |

The specialist is not registered and cannot be reused via `task`, even though it
has a `name`.

### check_task

Check the status of a background task.

```python
# The agent calls:
check_task(task_id="abc123")
```

**Returns:** Status, result (if complete), or pending question (if waiting).

### answer_subagent

Answer a question from a blocked subagent.

```python
# The agent calls:
answer_subagent(task_id="abc123", answer="Use PostgreSQL for this project")
```

### send_message_to_subagent

Steer a running **async** subagent mid-flight, without cancelling it. Use this
when you learn something new while a long task is in progress and want to
redirect or narrow it — the subagent keeps all its partial progress.

```python
# The agent calls:
send_message_to_subagent(
    task_id="abc123",
    message="narrow the search to packages/sparta/ — it isn't in core/",
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | Task ID of the running async subagent |
| `message` | `str` | Steering instruction to deliver |

The message is folded into the subagent's **next model request** as an extra
user instruction, so it adapts on its next step. This is unprompted parent ->
child steering, distinct from `answer_subagent` (which only replies to a
question the subagent already asked via `ask_parent`). It applies to async
tasks that are still running; messages to a finished or unknown task return an
error.

!!! note
    Steering is delivered at model-request boundaries on the retry-driven run
    path, which is the default (`max_retries > 0`). If you explicitly set
    `max_retries=0` on a subagent, it runs via the legacy single-shot path and
    steering messages stay queued instead of being applied.

### list_active_tasks

List all running background tasks.

```python
# The agent calls:
list_active_tasks()
```

**Returns:** List of task IDs, subagent names, and statuses.

### wait_tasks

Wait for one or more background tasks to finish.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_ids` | `list[str]` | — | Task IDs to wait on |
| `timeout` | `float` | `300.0` | Max seconds to wait |
| `mode` | `"all" \| "any"` | `"all"` | When to return |

A task is considered "finished" when it is completed, failed, or cancelled.

#### `mode="all"` (default)

Block until every task in `task_ids` is finished, or the timeout fires.
Use when you need every result together before the next step (e.g. a
final synthesis across all subagents).

```python
# Wait for both research subagents before writing the report
result = wait_tasks(task_ids=["abc123", "def456"], mode="all")
```

#### `mode="any"` — reactive orchestration

Return as soon as ONE task finishes. Use when the subagents are
independent and you can act on each result as it arrives — this avoids
stalling on the slowest task.

```python
# Dispatch 4 independent research tasks in parallel
ids = [task(...) for _ in range(4)]

remaining = list(ids)
while remaining:
    # Return as soon as one finishes — don't wait on the slowest
    result = wait_tasks(task_ids=remaining, mode="any")
    # ... react to the finished task (synthesize, dispatch follow-up, etc.)
    remaining = [tid for tid in remaining if check_task(tid).is_running]
```

The output includes a header like
`Task results (mode=any, 1/4 finished, 3 still running):` so the
orchestrator can see which tasks are still in flight and decide whether
to keep waiting or do other work first.

#### Long results are truncated, and say so

A fan-out of verbose subagents can flood the orchestrator's context, so
`wait_tasks` shows at most `max_result_chars` (default 2000) of each result.
A cut result always ends with an explicit marker:

```text
...the last of the 2000 shown characters

[Result truncated for display: showing 2000 of 5231 characters. The subagent's
answer is complete and stored in full. Call check_task('abc123') to read all of it.]
```

The marker matters as much as the limit. Without it an orchestrator reads a
result that stops mid-sentence, concludes the subagent failed to finish, and
delegates the same work again — a wasted round-trip caused entirely by the
display limit. `check_task` always returns the untruncated result, and passing
`max_result_chars=None` to `create_subagent_toolset` turns truncation off.

### soft_cancel_task

Request cooperative cancellation.

```python
# The agent calls:
soft_cancel_task(task_id="abc123")
```

The subagent will receive a cancellation request and should stop gracefully.

### hard_cancel_task

Immediately cancel a task.

```python
# The agent calls:
hard_cancel_task(task_id="abc123")
```

Forces immediate termination.

## Toolsets Factory

Provide tools to your subagents:

```python
from pydantic_ai_backends import create_console_toolset

def my_toolsets_factory(deps):
    """Create toolsets for each subagent."""
    return [
        create_console_toolset(),  # File operations
    ]

toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    toolsets_factory=my_toolsets_factory,
)
```

The factory is called for each subagent with cloned dependencies.

## Nesting Depth

Control how deep subagents can nest:

```python
# Allow subagents to have their own subagents (2 levels deep)
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    max_nesting_depth=2,
)

# No nesting - subagents can't delegate
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    max_nesting_depth=0,
)
```

## General-purpose subagent

A `general-purpose` subagent is added by default, so the model has somewhere to
send work that matches none of your specialists. It is built from `default_model`
(or from a `default_agent_factory`), so one of those must be set while it is on --
construction raises otherwise. Turn it off when you want the model restricted to
the subagents you defined:

```python
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    include_general_purpose=False,
)
```

To replace it rather than remove it, disable the built-in one and define your own
with whatever instructions you want:

```python
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=[
        *subagents,
        SubAgentConfig(
            name="general",
            description="Handles miscellaneous tasks",
            instructions="You are a general-purpose assistant.",
        ),
    ],
    include_general_purpose=False,
)
```

## Custom Tool Descriptions

Override the default tool descriptions to better guide LLM behavior. This is useful when you want descriptions that are more specific to your use case:

```python
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    descriptions={
        "task": "Assign a task to a specialized subagent",
        "check_task": "Check the status of a delegated task",
        "list_active_tasks": "Show all currently running background tasks",
    },
)
```

Only the tool names you include in the dictionary are overridden; the rest keep their built-in defaults. Available tool names:

| Tool Name | Description |
|-----------|-------------|
| `task` | Delegate a task to a subagent |
| `check_task` | Check status of a background task |
| `answer_subagent` | Answer a question from a blocked subagent |
| `list_active_tasks` | List all running background tasks |
| `wait_tasks` | Wait for background tasks to complete |
| `soft_cancel_task` | Request cooperative cancellation |
| `hard_cancel_task` | Immediately cancel a task |

## System Prompt

Add context about available subagents to your agent's system prompt:

```python
from subagents_pydantic_ai import SubAgentConfig, get_subagent_system_prompt

configs = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics and gathers information",
        instructions="You are a research assistant.",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes content based on research",
        instructions="You are a writer.",
    ),
]

# Generate prompt listing available subagents
prompt = get_subagent_system_prompt(configs)
```

The [`get_subagent_system_prompt`][subagents_pydantic_ai.prompts.get_subagent_system_prompt]
function takes a list of [`SubAgentConfig`][subagents_pydantic_ai.types.SubAgentConfig]
dicts (not deps) and an optional `include_dual_mode` flag. It generates text like:

```
## Available Subagents

Use the `task` tool to delegate work to these subagents:

- **researcher**: Researches topics and gathers information
- **writer**: Writes content based on research
```

Subagents configured with `can_ask_questions=False` are annotated with
*(cannot ask clarifying questions)*.

## Next Steps

- [Types](types.md) - Data structures and enums
- [Execution Modes](../advanced/execution-modes.md) - Sync vs async
- [Examples](../examples/index.md) - Working examples
