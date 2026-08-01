# Should we ask pydantic-ai for a public streaming step on `AgentRun`?

Status: analysis, for a decision
Verified against: `pydantic-ai` 2.18.0 / `pydantic-ai-slim` 2.0.0, Python 3.14

`retry.py` drives a subagent run using four private names from pydantic-ai. This
is the case for and against asking core for a public alternative. Short answer:
**the coupling is real and unavoidable today, but it is not urgent** — we can live
with it, and one small upstream addition would delete it.

There is also a smaller finding that stands on its own: the docstring's stated
reason for the design is **obsolete**, and should be corrected whatever we decide
about upstream (see §5).

## 1. What we actually use

`_drive_run` in `src/subagents_pydantic_ai/retry.py`:

| Private name | Used for |
|---|---|
| `run._run_node_with_hooks(node, step)` | Advance one node with a custom step, keeping capability hooks |
| `run._advance_graph(node)` | Advance without the default (non-streaming) step |
| `_agent_graph.build_run_context(run.ctx)` | Build the `RunContext` the handler receives |
| `run.ctx.deps.root_capability.wrap_run_event_stream(...)` | Route events through capabilities |

It is a hand copy of the driving loop inside `Agent.run`.

## 2. Why we drive the run ourselves at all

Three shipped features need node-level control:

1. **Retry that resumes.** On a transient failure we replay the accumulated
   history instead of restarting, so a subagent that already made five tool calls
   does not pay for them twice.
2. **Cooperative cancellation.** `soft_cancel_task` sets an event that the loop
   polls *between nodes*, so the subagent stops at a clean boundary with partial
   progress intact, rather than being killed mid-tool-call.
3. **Steering.** `send_message_to_subagent` delivers through `AgentRun.enqueue`,
   which needs a live `AgentRun` handle.

Only `agent.iter()` gives that handle. `agent.run()` does not expose one.

## 3. The gap, precisely

`agent.iter()` cannot stream events.

```python
>>> import inspect
>>> from pydantic_ai import Agent
>>> "event_stream_handler" in inspect.signature(Agent.iter).parameters
False
```

`AgentRun`'s public surface has no streaming affordance either — `next(node)` is
the only drive method, and it uses the default non-streaming step:

```python
>>> from pydantic_ai.run import AgentRun
>>> [n for n in dir(AgentRun) if not n.startswith("_")]
['all_messages', 'all_messages_json', 'conversation_id', 'ctx', 'enqueue',
 'metadata', 'new_messages', 'new_messages_json', 'next', 'next_node',
 'pending_messages', 'result', 'run_id', 'usage']
```

Meanwhile `Agent.run` *does* stream, by building the step itself out of the four
private names above. So the capability exists in core; it is simply not reachable
from `iter()`.

**There is no public combination of node control and event streaming.** That is
the whole issue.

## 4. What each alternative costs

| Option | Streaming | Soft cancel | Steering | Private API |
|---|---|---|---|---|
| **A. Status quo** — copy the loop | yes | yes | yes | 4 names |
| **B. Public `run.next()` only** | **no** | yes | yes | none |
| **C. `agent.run()` + `capture_run_messages`** | yes | **no** | **no** | none |

Option B is what `pydantic-ai-harness` does, and it is explicit that
`event_stream_handler` is "synchronous delegations only". For us the loss is
larger: retrying is **on by default** (`max_retries=3`), so choosing B means every
subagent silently stops emitting tool-call and reasoning events to any UI stream
unless the app opts out of retries. That is a bad trade for a UI like
deepresearch's.

Option C is newly viable (see §5) but gives up two shipped features.

## 5. The obsolete justification (fix this regardless)

`retry.py`'s module docstring says:

> The retry path deliberately uses `Agent.iter` rather than
> `capture_run_messages()` to recover the failed run's messages, because nested
> `capture_run_messages` contexts do not work
> (<https://github.com/pydantic/pydantic-ai/issues/1568>) and subagents always run
> nested inside a parent agent's run.

**That is no longer true.** A nested capture returns the child's own partial
history in 2.18:

```python
@parent.tool_plain
async def delegate() -> str:
    with capture_run_messages() as msgs:
        try:
            await child.run("child task")   # fails on its second model call
        except ModelHTTPError:
            pass
    return str([type(m).__name__ for m in msgs])

# nested capture kinds: ['ModelRequest', 'ModelResponse', 'ModelRequest']
# nested capture user prompts: ['child task']   <- the child's own history
```

So message recovery is *not* a reason to use `iter` any more. The real reasons are
cooperative cancellation and steering (§2). The docstring should say that, because
as written it points a future maintainer at a closed issue and hides the actual
constraint.

## 6. The upstream ask, if we make it

Narrow and additive, either form:

```python
# Option 1 — the smaller change: let iter() take the handler run() already accepts.
async with agent.iter(prompt, event_stream_handler=handler) as run:
    node = run.next_node
    while not isinstance(node, End):
        node = await run.next(node)      # streams, hooks fire

# Option 2 — expose the step, so callers can compose their own.
async with agent.iter(prompt) as run:
    node = await run.next(node, step=run.streaming_step(handler))
```

Option 1 is what we want: it makes `iter()` and `run()` consistent about a
parameter `run()` already has, and it deletes ~40 lines of copied internals here.
Anyone else driving `iter()` inside an app with a UI stream has the same problem —
this is not a niche need.

## 7. Recommendation

**File it, but treat it as non-blocking.** The case is concrete, the fix is small,
and the benefit is not only ours. Nothing is broken today: the copy works, a drift
test pins the observable contract (hooks fire, the handler receives every event,
the wrapped stream is drained), and `pydantic-ai-slim` has a lower bound.

The risk we are carrying is maintenance, not correctness — and it has already
materialised once. The copy had drifted from core: `Agent.run` drains the wrapped
stream unconditionally after the handler returns, while our copy drained only when
there was no handler, leaving stream wrappers unfinished for a handler that stopped
reading early. That was fixed in 0.2.12, but it is the exact failure mode this
coupling produces, and nothing warns us the next time core changes the loop.

If we do not file it, the honest fallback is to accept option B — drop streaming on
the retry path — rather than keep a copy nobody is watching. I would not recommend
that while retries are the default.

## 8. What is already in place either way

- The coupling is documented in `retry.py`'s module docstring and in
  `agent_docs/core-boundary.md`, which names it as the standing exception to the
  "propose the core change" rule.
- `tests/test_retry.py` asserts the observable contract the copy must preserve.
- §5's docstring correction is a separate, small fix worth doing now.
