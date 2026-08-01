# Delegation contracts

The invariants a change to delegation must preserve. Each one exists because
breaking it produced a real defect.

## What the model sees is part of the API

Tool return strings are the interface to the orchestrating model, and it acts on
them. `TaskStatus` is a `str`-mixin enum whose `__format__` changed in Python 3.11,
which turned `Status: waiting_for_answer` into `Status: TaskStatus.WAITING_FOR_ANSWER`
on 3.11+ with no test noticing.

- Assert the exact rendered string, not that a branch ran.
- Render enums as `.value`, or through `_ValueStrEnum`, never bare in an f-string.
- A truncated result carries an explicit marker saying the cut is ours and the
  stored answer is complete. A silent cut reads as a subagent that stopped
  mid-sentence, and the model "recovers" by re-delegating the same work.

## Signals are not failures

`CallDeferred`, `ApprovalRequired`, and the `Skip*` exceptions are how core suspends
a run for approval or a deferred tool. `UserError` is a setup bug. A shared
`UsageLimitExceeded` means the whole tree is out of budget. All of these propagate;
`_ALWAYS_PROPAGATE` in `_execution.py` is the list, and `contain_errors` does not
apply to it.

Everything else reaches the parent as `ModelRetry`, so pydantic-ai's retry budget
engages. Never return a failure as a plain string result: to the model that is a
successful tool call whose text happens to start with `Error`.

A background delegation cannot suspend -- its tool result was returned long ago --
so a human-in-the-loop signal there becomes a `failed` status telling the model to
delegate synchronously.

## A task never outlives its parent run

Background delegations are `asyncio.Task`s nobody awaits. `SubAgentCapability`
cancels its run's tasks in a `wrap_run` finalizer. If you add another entry point
that spawns tasks, it needs the same guarantee, or an orphan keeps running against
torn-down deps.

## Tasks are scoped to the run that started them

One toolset instance is typically built once per agent and shared by every run it
serves. Handles record `parent_run_id`, and `_handle_for` refuses ids from another
run. Any new tool that takes a `task_id` must go through it; a direct
`task_manager.get_handle` lookup reintroduces cross-run access.

Scoping the *output* is not enough. `wait_tasks` filtered its listing correctly and
built the set it awaited straight off `task_manager.tasks`, so a foreign id still
blocked the caller for the full timeout -- which is also an existence oracle,
because a genuinely unknown id returns instantly. Anything that waits on, cancels,
or otherwise touches a task must be scoped, not only what gets rendered.

`TaskManager.handles` stays globally readable on purpose -- external monitoring
depends on it.

## Chat traces are scoped the same way

`ChatTraceStore` records the run that first claimed a trace (`mark_active`), and
`owned_by` gates the lookup in `_execute`. A trace holds a whole subagent
conversation, so an unscoped `chat_trace_id` was a cross-run read, not just a
cross-run reference. The ownership check runs *before* the "already has a running
task" branch, so that branch cannot confirm a foreign id exists. A trace claimed
without a `run_id` stays open, matching `_handle_for`.

The registry behind `create_agent` is deliberately **not** scoped -- a persistent
agent is meant to outlive one run. The unknown-subagent error still must not
enumerate it: those names are model-authored and describe the work.

## Terminal transitions are idempotent

`TaskHandle.finish()` lets the first terminal transition win. A task inside its
`finally` is not `asyncio.Task.done()`, so a cancel arriving in that window used to
overwrite a real result. Do not assign `status` and `completed_at` directly on a
finishing path.

## The caller's deps are read-only

`SubAgentDepsProtocol` permits a `frozen=True` or `slots=True` deps class. The
library never writes attributes onto deps; per-delegation state is a typed
`SubAgentState` in a `ContextVar` (`_state.py`). `clone_for_subagent` must return a
new instance -- returning `self` shares state between concurrent subagents.

## Timestamps are timezone-aware UTC

`utcnow()` in `types.py`. Handles are compared and subtracted for elapsed time and
eviction order, and naive local timestamps make that arithmetic wrong across a DST
transition.

## Memory is bounded

`max_task_handles` and `max_chat_traces` are LRU limits. Eviction folds usage into
`get_total_usage()` first, so aggregates survive it. Anything new that accumulates
per delegation needs a bound and a documented eviction rule.

## A chat trace has one owner and one writer

A trace belongs to one subagent, and only one task may run on it at a time --
concurrent tasks would both save at the end and the slower save would discard the
faster one's history. A one-shot specialist gets no trace: `task` cannot resolve an
unregistered agent, so the id would be unredeemable.
