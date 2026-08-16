# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.19] - 2026-08-16

### Added

- **`TaskHandle.exception` carries the exception behind `error`.** It is set
  wherever `error` embeds an exception's own text -- a contained crash, an
  exhausted retry budget, a usage limit, a propagated crash with
  `contain_errors=False`, and each transient retry -- and cleared by the
  winning terminal transition, so a completion leaves it `None`. A host that
  must not surface a foreign message (a provider error can carry the failing
  request URL, key included, in its text) reads the class from here, composes
  its own sentence, and logs the original instead of parsing `error`.
  `finish()` takes the exception as a keyword for the same reason.

## [0.2.18] - 2026-08-05

`default_model` no longer defaults to a model of the library's choosing. There is
now **no implicit default**: it defaults to `None`, and anything that would have
relied on the old fallback is refused instead of run.

The old default was `"openai:gpt-4.1"`. A consumer that never set `default_model`
got a delegate compiled against that string, which resolves whatever provider
credential the process environment happens to hold. On a deployment holding no
such key the build raised; on one that had `OPENAI_API_KEY` set it ran silently --
in a multi-tenant host, a caller's work on a credential that was not theirs, off
any per-caller model, budget or vault the host meant to enforce. The general-purpose
delegate was the sharpest edge: it was always compiled from `default_model` at
construction, so a host resolving a model per tenant had no single value to give it.

### Changed

- **`default_model` defaults to `None`** on `create_subagent_toolset`,
  `SubAgentToolset`, `SubAgentCapability` and `create_agent_factory_toolset`. Pass
  it to name the model subagents fall back on, or leave it unset to require every
  subagent and every dynamic call to name one.
- **The general-purpose delegate is now built the way every other subagent is** --
  through `_compile_subagent`, from `default_model` or from `default_agent_factory`
  when one is given. A host that resolves a model, a credential and a budget per
  caller passes a `default_agent_factory` and gets a delegate it can account for,
  rather than one the library compiled behind its back.

### Fixed

- **`include_general_purpose=True` with neither `default_model` nor
  `default_agent_factory` now raises at construction**, naming the three ways out,
  instead of failing later inside the model call (or not failing, on the wrong
  credential).
- **A `create_agent`, `delegate` or `task` call that names no model is refused with
  a tool result** when there is no default -- the model can name one and call again
  -- instead of the library choosing one for it.
- **A configured subagent that names no model, and supplies no `agent` or
  `agent_factory`, is refused when `default_model` is unset**, with a message
  saying what it needs.

Migration: a consumer relying on the old implicit `"openai:gpt-4.1"` should pass
`default_model="openai:gpt-4.1"` explicitly. One that already names a model on
every subagent and every call, or builds its own agents through a factory, needs
no change.

## [0.2.16] - 2026-08-03

Security housekeeping. **Nothing in the published package changes**: all five
advisories are against development and documentation dependencies -- `pytest`,
`requests`, `urllib3`, `pygments`, `pymdown-extensions` -- none of which this
library depends on at run time. Its runtime dependencies are still `pydantic`,
`pydantic-ai-slim` and `typing-extensions`. The exposure was this repository's CI
and a maintainer's docs build, not anybody's install.

### Security

- **`urllib3` 2.6.3 -> 2.7.0**, closing two high-severity advisories:
  decompression-bomb safeguards bypassed in parts of the streaming API, and
  sensitive headers forwarded across origins in proxied low-level redirects.
- **`pymdown-extensions` 10.20 -> 11.0.1**, closing two: path traversal in the
  `b64` extension letting `<img src>` read files outside `base_path`, and a
  regression reintroducing the sibling-prefix traversal bypass in
  `pymdownx.snippets` despite `restrict_base_path`. A major bump, so the docs
  build was checked for rendered output rather than just an exit code -- the
  changelog page is a `pymdownx.snippets` include of `CHANGELOG.md` from outside
  `docs/`, which is exactly what the second advisory tightened.
- **`requests` 2.32.5 -> 2.34.2**, closing insecure temporary-file reuse in
  `extract_zipped_paths()`.
- **`pytest` 9.0.2 -> 9.1.1**, closing vulnerable `tmpdir` handling.
- **`pygments` 2.19.2 -> 2.20.0**, closing a ReDoS in the GUID-matching regex.
- **`.github/workflows/ci.yml` declares `permissions: contents: read`.** It had no
  permissions block at all, so all four of its jobs inherited the repository
  default -- which can be write -- and an action compromised anywhere in the test
  matrix would have inherited it too. Four CodeQL `actions/missing-workflow-permissions`
  alerts, closed by one top-level block. The other three workflows already
  declared theirs.

Version constraints in `pyproject.toml` are deliberately unchanged: CI installs
from the lockfile, which is what both Dependabot and Renovate patch, and raising
a floor for a transitive dependency states a fact about a resolution rather than
about this package.

## [0.2.15] - 2026-08-02

Three defects found integrating the library into a platform where every agent is
built with `output_type=[str, DeferredToolRequests]`, which made the first one the
default path rather than an edge case.

One entry changes observable behaviour: a suspended delegation now reports
`deferred` where it used to report `completed` (a real defect) or `failed` (an
honest but wrong label). Code branching on `TaskStatus.FAILED` to detect a
human-in-the-loop delegation needs updating.

### Fixed

- **A suspended subagent was reported to the parent as a completed task.**
  `_ALWAYS_PROPAGATE` guards the exception route, but an agent whose `output_type`
  includes `DeferredToolRequests` never raises: pydantic-ai ends its run normally
  with the parked calls as the output, exactly as it does for a top-level run a
  caller is expected to resume. That reached `serialize_output`, so the parent
  agent received `{"calls": [], "approvals": [...]}` as the specialist's answer,
  summarised it, and carried on -- with `handle.status` saying `completed` and the
  approval surfaced nowhere. It is the outcome the module docstring says the
  design prevents, arriving by the output instead of the exception. A deferred
  output is now the fifth entry in the error contract: the handle is marked
  `DEFERRED`, the parked calls are kept on `TaskHandle.deferred_requests`, and the
  matching signal is raised (`ApprovalRequired` when anything needs approving,
  `CallDeferred` otherwise) so the parent run suspends too. The suspended run's
  chat trace is deliberately not saved -- continuing it would resume from a point
  whose deferred results were never supplied.
- **`cancel_all` waited on cancelled tasks without a bound, inside a `finally`.**
  `SubAgentCapability.wrap_run` calls it when a run ends, and it awaited each
  cancelled task indefinitely. A `CancelledError` can be caught, and a subagent's
  toolset is arbitrary consumer code, so one task that swallowed the cancel held
  the parent run's teardown open forever with nothing logged. The wait is now
  bounded by `cancel_grace_seconds` (5 s by default, configurable on
  `TaskManager`, `create_subagent_toolset` and `SubAgentCapability`); a task still
  alive after it is logged with its task id and left to the event loop. The
  suppression around the wait is also no longer able to hide an *outer*
  cancellation for longer than the grace period, which is the case that matters:
  the parent is usually being cancelled from outside when this runs.

### Added

- **`event_stream_handler` on `create_subagent_toolset` and `SubAgentCapability`.**
  Streaming a delegation was possible but only by setting the handler on each
  agent instance -- which a dynamically created specialist does not have, since the
  library builds it, making "stream subagents" and "let the model create
  specialists" quietly incompatible. The handler is now resolved per delegation,
  so both work together. An agent supplied as `SubAgentConfig["agent"]` keeps its
  own handler: the specific choice wins, and the toolset's is the default for
  everything else.
- **`event_stream_handler_factory`**, resolved per delegation from the parent run
  context, the subagent config and the task id. The task id is the argument that
  makes a fan-out readable -- three specialists streaming into one callback are
  otherwise indistinguishable. Mutually exclusive with `event_stream_handler`,
  which is refused at construction: both are callables, so nothing downstream
  could tell them apart.
- **`TaskStatus.DEFERRED`** and **`TaskHandle.deferred_requests`**, and
  `DEFERRED` joins `TERMINAL_STATUSES`.
- **`docs/advanced/streaming.md`**, which is also the first documentation that
  streaming a subagent is possible at all.

### Changed

- **A human-in-the-loop delegation reports `DEFERRED`, not `FAILED`.** Both the
  sync route (where the signal propagates) and the background route (where it
  cannot be delivered) previously recorded `FAILED`, sending a caller looking for
  a defect when what the delegation needs is a person to decide. The background
  message is unchanged and still tells the model to delegate with `mode="sync"`.
  `UserError` and the `Skip*` signals still record `FAILED`.

## [0.2.14] - 2026-08-01

A re-audit of 0.2.13, verifying each of that release's fixes and then sweeping the
areas the audit behind it had listed as uncovered. The fixes in 0.2.13 were all
correct; they had been applied to the sites the report named rather than to the
class of defect, and the class was still open elsewhere.

Three entries change observable behaviour, in each case to match what the docs
already promised: `can_ask_questions=False` now removes `ask_parent`,
`max_questions` is enforced, and a `chat_trace_id` from another run is refused.
Code relying on the old behaviour was relying on a defect, but it is worth knowing
about before upgrading.

### Fixed

- **`can_ask_questions=False` did not disable `ask_parent`.** The flag was honoured
  when `_execute` injects the tool for a runtime specialist and ignored when
  `_compile_subagent` builds a configured subagent, so the tool was attached
  regardless. In background mode that is a stall, not a cosmetic issue: the
  subagent parks the task in `WAITING_FOR_ANSWER` for the full
  `ask_timeout_seconds` (300 s by default) while the parent's own instructions
  describe that subagent as one that *cannot ask clarifying questions*, so nothing
  ever calls `answer_subagent`. The tool is now attached only when the flag allows
  it.
- **Any run could resume any other run's chat trace.** Task handles record
  `parent_run_id` and every tool that takes a `task_id` is scoped, but traces were
  keyed by `(subagent_name, chat_trace_id)` with no owner at all — so a run passing
  an id it had seen got the other run's whole conversation replayed into its
  subagent's `message_history`. `ChatTraceStore` now records the claiming run, and a
  foreign trace reads exactly like an unknown one. The check runs before the
  "already has a running task" branch, which would otherwise confirm the id exists.
  A trace claimed without a `run_id` stays open, matching `_handle_for`.
- **`wait_tasks` awaited another run's live task.** It filtered its listing through
  `_handle_for` and built the set it awaited from `task_manager.tasks` directly, so
  a foreign id blocked the caller for the whole `timeout` and then reported "not
  found". The wall-clock difference against a genuinely unknown id (which returns
  instantly) was an existence oracle over foreign task ids. The isolation test
  missed it for the same reason the 0.2.13 cancel defect shipped: its fixture
  registers a handle with no `asyncio.Task`, so the await never happened.
- **The unknown-subagent error enumerated every run's dynamic agents.** One registry
  is shared by every run of an agent, and `create_agent` names are model-authored
  and describe the work, so the list told one tenant what the others were doing. The
  error names the configured subagents and says dynamic agents exist without naming
  them. The registry itself stays shared — a persistent agent is meant to outlive
  one run.
- **`wait_tasks` counted a missing task as still running.** `still running` was
  `total - finished`, so the same message said `not found` and `1 still running`
  about one id and the orchestrator kept polling something that will never resolve.
  Missing ids are now counted and reported separately.
- **`max_questions` was prompt text, documented as a limit.** It reached the
  subagent only as a sentence in its task prompt, which a model is free to ignore —
  and each ignored question costs up to `ask_timeout_seconds`. It is now a counter
  on the delegation: past the limit, `ask_parent` returns immediately without
  waiting for the parent. The budget is per delegation, so a task continuing a chat
  trace gets a full allowance.
- **`check_task`'s tool description named four of seven statuses.** 0.2.13 added the
  `cancelled` and `retrying` renderings without updating the description that
  enumerated the statuses a model should expect. It now describes what each terminal
  state returns instead of listing values that drift.
- **`typing-extensions` was imported but not declared.** `types.py` imports it at
  module scope for `NotRequired`/`TypedDict`; it was only present transitively via
  `pydantic`. Now a direct dependency.

## [0.2.13] - 2026-08-01

Findings from a full-repo audit. Every defect here sat in a place the tooling
could not see: a disagreement between two call sites that each read fine alone, or
documentation describing behaviour the code did not implement.

### Fixed

- **Any run could cancel another run's live background task.** `soft_cancel_task`
  and `hard_cancel_task` guarded with `handle is None and task_id not in tasks` —
  and a foreign task that is actually running satisfies exactly that, so the guard
  fell through and the cancel went ahead. `check_task` hid the same task correctly,
  so one tenant on a shared agent could kill another tenant's work with an id it
  read out of tool output. A handle scoped to another run now reads exactly like a
  missing one. The existing isolation test passed against the defect because its
  fixture registered no `asyncio.Task`; the new test starts a real background
  delegation and asserts the task survives.
- **`check_task` told the model a cancelled task was still running.** `CANCELLED`
  and `RETRYING` fell through to an elapsed-time line computed from `started_at`,
  so a task cancelled two hours ago reported `Running for: 7200.0s` and never said
  why it stopped. Both statuses now report their outcome.
- **A retried attempt got a fresh usage allowance.** `run_with_retry` forwarded
  `usage_limits` into every attempt but no `usage`, so `Agent.iter` built a new
  `RunUsage` each time while the replayed history genuinely re-spent the tokens —
  with the default `max_retries=3` the real ceiling was 4× what the caller
  configured. Attempts now share one tally, which is what
  `docs/advanced/usage-limits.md` already promised.
- **`retry_count` was always 0 for sync delegations.** `_run_sync` passed no
  `on_retry`, and `sync` is the default mode for both `task` and `delegate`, so the
  handle field documented for spotting a flaky gateway reported nothing from the
  path most delegations take.
- **`SubAgentCapability` could not reach `ask_user`.** It forwarded 15 of
  `create_subagent_toolset`'s parameters and silently dropped `ask_user`,
  `max_chat_traces`, and `max_task_handles`. `ask_user` is the only channel a
  sync-mode subagent has for `ask_parent`, so the advertised question feature was
  off by construction on the primary entry point — and the error text pointed at a
  remedy the capability did not accept. All three are now fields, and a parity test
  fails if the toolset grows an argument the capability cannot reach.
- **`max_agents=0` meant unlimited.** `DynamicAgentRegistry.register` tested
  `if self.max_agents and ...`, reading `0` as falsy. It now tests `is not None`, so
  `0` rejects every registration and `None` stays unlimited. `max_agents`,
  `max_chat_traces`, and `max_task_handles` are validated at construction:
  `max_agents >= 0`, and the two stores `>= 1` since a store that cannot hold one
  entry evicts everything before it can be read back.
- **`validate_agent_name` enforced a wider rule than it stated.** `str.isalnum` is
  Unicode-aware, so Cyrillic and fullwidth-digit names passed an allow-list the
  model was told read "letters, numbers, and hyphens", while `café` was rejected
  only over a combining accent. It is an explicit ASCII match now.
- **Sync and async recorded different errors for the same exhausted budget.**
  `handle.error` was `usage limit exceeded` in sync mode and
  `UsageLimitExceeded: ...` in background mode, so telemetry had to know how a
  delegation had been dispatched. Both modes now write `usage limit exceeded: ...`.

### Changed

- `docs/advanced/errors.md` no longer claims a per-task usage budget is a soft
  outcome, or distinguishes a "shared" `UsageLimitExceeded`: the library never
  hands a child run the parent's tally, so every subagent budget is its own. It now
  states what actually differs — sync propagates, background contains.

## [0.2.12] - 2026-08-01

Correctness, typing, and documentation pass over the whole library. Every public
entry point keeps its name and signature, so no import or call site changes.

### Fixed

- **Task statuses leaked their enum member name to the model.** `TaskStatus` is a
  `str`-mixin `Enum`, and Python 3.11 changed `Enum.__format__` for mixin enums, so
  `check_task` reported `Status: TaskStatus.WAITING_FOR_ANSWER` on 3.11+ while the
  tool descriptions and docs promised `waiting_for_answer`. `list_active_tasks` and
  `wait_tasks` had the same leak. Statuses, priorities, and message types now render
  as their values on every supported Python.
- **Deferred tools and human approval could not work inside a subagent.**
  `except Exception` around the run swallowed pydantic-ai's control-flow signals
  (`CallDeferred`, `ApprovalRequired`, `SkipModelRequest`, `SkipToolValidation`,
  `SkipToolExecution`) and turned them into a string result the parent read as a
  finished task. Those signals, `UserError`, and a shared `UsageLimitExceeded` now
  propagate. A background delegation cannot suspend at all, so it reports a `failed`
  status explaining to delegate with `mode="sync"` instead.
- **A failed delegation looked like a successful one.** A crash returned
  `"Error executing task: ..."` as a normal tool result, so pydantic-ai's retry
  budget never engaged and the failure could be folded into a final answer. Failures
  now reach the parent as `ModelRetry`. See `contain_errors` and `on_failure` under
  *Added*.
- **Background tasks outlived their parent run.** Nothing cancelled the
  `asyncio.Task` behind a background delegation when the run ended, so it kept
  executing against torn-down deps, and one blocked in `ask_parent` waited out the
  full timeout. `SubAgentCapability` now cancels its run's tasks in a `wrap_run`
  finalizer, and `TaskManager.cancel_all()` exposes the same thing to the toolset API.
- **Tasks were not isolated per run.** One toolset instance is typically built per
  agent and shared by every run it serves, so any run could inspect, answer, steer,
  or cancel another run's task by id. Handles record `parent_run_id` and the tools
  refuse ids belonging to another run.
- **`hard_cancel` could overwrite a completed task's result.** The guard was
  `if not task.done()`, which is still true while the task runs its `finally`, so a
  cancel arriving in that window replaced the real result with `cancelled`.
  `TaskHandle.finish()` makes the first terminal transition win.
- **The library wrote a private attribute onto the caller's deps object.**
  `deps._subagent_state = {...}` raised `AttributeError` for a deps class declared
  `frozen=True` or `slots=True`, both of which `SubAgentDepsProtocol` allows. The
  state is a typed `SubAgentState` carried in a `ContextVar` instead. Reading a
  caller-injected `deps._subagent_state` still works.
- **Timestamps were naive local time.** `TaskHandle.created_at`, `started_at`, and
  `completed_at` are timezone-aware UTC, so elapsed time and eviction order stay
  correct across a DST transition.
- **`get_subagent_system_prompt(include_dual_mode=...)` was accepted and ignored.**
  It now appends `DUAL_MODE_SYSTEM_PROMPT` when asked. The default changed to
  `False`, so output is unchanged for callers that never passed it.
- **Steering could be spliced into the wrong place.** Parent-to-child steering
  appended `UserPromptPart`s directly into a graph node's request. It now goes
  through pydantic-ai's `AgentRun.enqueue`, so core places the parts and they can
  never land between a tool call and its return.
- **The retry driver had drifted from the loop it mirrors.** `Agent.run` drains the
  wrapped event stream after the handler returns, unconditionally; the copy drained
  only when there was no handler, leaving stream wrappers unfinished for a handler
  that stopped reading early.
- **409 Conflict and 425 Too Early were retried as transient.** Both signal a request
  the server rejected on its merits, so replaying it unchanged is not expected to
  help.
- **A message-bus handler that raised was silently swallowed** by a bare
  `except Exception: pass`. Failures are logged and delivery continues.
- **`asyncio.get_event_loop()` inside a coroutine** (deprecated since 3.10) is now
  `get_running_loop()`.
- **Cancelling a finished task reported "not found"**, inviting the model to conclude
  the work was lost. It now reports the task's status and points at `check_task`.
- **`make typecheck-mypy` was broken.** The `module = "tests.*"` override never
  matched, because `tests/` had no `__init__.py` and mypy named the modules
  `test_toolset` rather than `tests.test_toolset`; the target reported 583 errors the
  config intended to relax. It is green and runs in CI.

### Added

- **`contain_errors`** on `create_subagent_toolset`, `SubAgentCapability`, and
  `SubAgentConfig`. Defaults to `True`: an unexpected subagent crash becomes a
  `ModelRetry` for the parent, logged with its traceback, so one failed delegation
  cannot abort the run. Set `False` to let crashes propagate.
- **`on_failure`** on `SubAgentConfig`. Returns a steering message to the parent as
  an ordinary tool result instead of raising `ModelRetry`, for a failure where
  re-delegating is pointless.
- **`ask_timeout_seconds`** on `create_subagent_toolset` and `SubAgentCapability`,
  replacing a hardcoded 300-second wait in `ask_parent`.
- **`SubAgentToolset` is a real class.** It was an alias for
  `create_subagent_toolset`, whose result had `task_manager`,
  `message_history_store`, and `get_total_usage` attached afterwards behind three
  `type: ignore` comments. It is now a `FunctionToolset` subclass with those as typed
  members, and `create_subagent_toolset()` returns an instance —
  `SubAgentToolset(subagents=[...])`, `toolset.task_manager`, and
  `isinstance(t, FunctionToolset)` all keep working.
- **`TaskHandle.finish()` and `TaskHandle.is_finished`** for idempotent terminal
  transitions, plus `TERMINAL_STATUSES` and `utcnow` as exports.
- **`TaskManager.cancel_all()`** and **`TaskManager.resolve_answer()`**.
- **`SubAgentToolset.answer_task()` and `SubAgentToolset.steer_task()`** — the
  Python halves of the `answer_subagent` and `send_message_to_subagent` tools, for
  an application that drives delegation itself instead of letting a model call the
  tools. Both return a `bool` rather than raising. The tools now delegate to them,
  so there is one implementation.
- **`SubAgentSpec` covers the whole serialisable config.** It mirrored 11 keys, so a
  YAML-defined subagent could not set `max_retries`, any `retry_*` option,
  `agent_kwargs`, `on_failure`, or `contain_errors` — the loader silently ignored
  what it had no field for. It also validates now: `max_retries` cannot be negative,
  `retry_backoff_multiplier` cannot shrink the delay, and `retry_max_delay` cannot
  sit below `retry_initial_delay` (which pinned every retry to the cap instead of
  backing off). `tests/test_spec.py` fails if the config gains a serialisable key
  the spec cannot carry.

### Removed

- **`SubAgentDepsProtocol.subagents`.** The library never read it, so every
  application carried a `dict` for nothing. Dropping a requirement only widens what
  satisfies the protocol, so a deps class that still declares the field is
  unaffected.

### Changed (breaking)

- **`delegate` requires a `name`.** A one-shot specialist was labelled
  `oneshot-{task_id}`, which told an operator reading logs or a `TaskHandle`
  nothing about what the specialist was for. The caller now supplies the label
  (letters, numbers, hyphens, validated the same way as `create_agent`), and it
  becomes `TaskHandle.subagent_name`. Naming a one-shot still does not register it:
  it does not count toward `max_agents`, cannot be reached via `task`, and reports
  no chat trace. Any code or prompt that calls `delegate` must pass `name`.

### Changed

- **`SubAgentConfig` enforces its required keys.** `name`, `description`, and
  `instructions` were documented as required but optional to the type checker, and
  the library indexed them directly, so a config missing one raised `KeyError` mid
  delegation. Call sites are unchanged; both type checkers now catch it.
- **Typing bar raised to match `pydantic-ai-harness`.** pyright runs in **strict**
  mode, `src/` has no `type: ignore`, mypy strict covers tests as well as source, and
  ruff's complexity ceiling dropped from 30 to 15 with no per-function `noqa`.
  `TaskHandle.usage` is `RunUsage | None`, `finish_reason` is `FinishReason | None`,
  and `TaskManager.handles` is `dict[str, TaskHandle]`.
- **`ToolsetFactory` returns a `Sequence`**, so a factory annotated
  `list[FunctionToolset[MyDeps]]` satisfies it — `list` is invariant.
- **`toolset.py` split into focused modules** (`_execution`, `_observability`,
  `_chat_trace`, `_state`), with the historical names still importable from
  `subagents_pydantic_ai.toolset`.
- **Documentation.** New pages for [observability](https://vstorm-co.github.io/subagents-pydantic-ai/concepts/observability/),
  [steering](https://vstorm-co.github.io/subagents-pydantic-ai/advanced/steering/), [chat traces](https://vstorm-co.github.io/subagents-pydantic-ai/advanced/chat-traces/),
  [failure handling](https://vstorm-co.github.io/subagents-pydantic-ai/advanced/errors/), and
  [usage limits](https://vstorm-co.github.io/subagents-pydantic-ai/advanced/usage-limits/); API reference pages for the
  registry, message bus, retry, spec, and dynamic-agent helpers; a changelog page.
  Corrected the stale tool and feature tables on the index, the
  `general_purpose_config` parameter that never existed, and the nesting guide's
  claim that `max_nesting_depth` enforces a limit — it does not, the gate is what
  `toolsets_factory` hands the child. Snippets in `docs/` and `README.md` are now
  checked for syntax and API drift by `tests/test_docs.py`.

## [0.2.11] - 2026-07-31

### Added

- **Configurable delegation modes, including one-shot `delegate`** ([#50](https://github.com/vstorm-co/subagents-pydantic-ai/pull/50)). `delegation_configuration` on `create_subagent_toolset` and `SubAgentCapability` picks which delegation entry points the orchestrator sees, so an application exposes only the delegation behaviour it needs instead of every creation and execution option at once:
  - `"default"`: `task` only
  - `"persisted"`: `create_agent` + `task`
  - `"persisted_and_oneshot"`: `create_agent` + `task` + `delegate`
  - `"oneshot_only"`: `delegate` only

  `"default"` is the existing tool surface, so upgrading adds nothing to a deployed orchestrator. Async lifecycle tools (`check_task`, `wait_tasks`, `answer_subagent`, `send_message_to_subagent`, cancellation) remain available in every mode.
- **One-shot delegation via `delegate`.** Builds an ephemeral specialist from `instructions` and runs its task in a single call, for ad-hoc work that does not deserve a named, reusable agent. A one-shot never enters the registry, never counts against `max_agents`, and cannot collide with a persisted agent's name. It also reports no `Chat Trace ID` and stores no history: `task` can never resolve an unregistered specialist, so the id would be unredeemable, and keeping its history would let a one-shot fan-out evict genuinely continuable conversations from the `max_chat_traces` LRU.
- **Dynamic specialists configurable from the subagent toolset.** `create_subagent_toolset` and `SubAgentCapability` now take `allowed_models`, `capabilities_map`, `default_agent_factory`, and `max_agents`, previously reachable only through `create_agent_factory_toolset`. The shared validation and construction path lives in the new `dynamic_agent.py`, so both entry points enforce identical rules. Note that `"persisted"` and `"persisted_and_oneshot"` cannot be combined with `create_agent_factory_toolset` on one agent — both define a `create_agent` tool and pydantic-ai rejects duplicate tool names across toolsets. Pair `"default"` with the factory toolset over a shared registry when the parent also needs `list_agents` / `remove_agent`; see the dynamic-agents guide.
- **Configuration a mode cannot reach is rejected at construction.** Hiding a tool also hides everything only that tool reads, so `create_subagent_toolset` and `SubAgentCapability` raise `ValueError` rather than silently dropping arguments that could never take effect: `"oneshot_only"` rejects `subagents` and `registry` (both reachable only through `task`), and `"default"` rejects `allowed_models`, `capabilities_map`, and `default_agent_factory` (read only by `create_agent` and `delegate`).

### Fixed

- **`can_ask_questions` was a no-op for dynamically created agents.** Agents built at runtime never received the `ask_parent` toolset that statically compiled subagents get, so a dynamic agent configured to ask its parent had no tool with which to do it. The toolset now injects `ask_parent` when executing a registry-backed or one-shot agent. A custom `default_agent_factory` must not attach its own `ask_parent`, or the tool name is duplicated at run time.

### Changed

- **`registry` and `default_agent_factory` are precisely typed.** `registry` is now `DynamicAgentRegistry | None` (was `Any`), since the toolset calls `get_compiled`, `list_agents`, `exists`, and `register` on it directly, and `default_agent_factory` uses the new public `AgentFactory` alias (was `Any`) across `create_subagent_toolset`, `create_agent_factory_toolset`, and `SubAgentCapability`. Runtime behaviour is unchanged, but a downstream type-checker may now flag a duck-typed registry object.

## [0.2.10] - 2026-07-24

### Fixed

- **`wait_tasks` truncated results silently, making orchestrators re-delegate finished work** ([#55](https://github.com/vstorm-co/subagents-pydantic-ai/issues/55)). A completed task's result was hard-sliced to 2000 characters with no ellipsis, length, or marker of any kind. The orchestrator saw a well-formed answer that stopped mid-sentence, concluded the subagent had been cut off, and dispatched a *new* task asking it to finish — burning a full extra round-trip on work that was already complete and stored intact. Truncated results now end with an explicit marker stating that the cut is a display limit, that the stored answer is complete, and which `check_task(...)` call returns the full text; `check_task` and the `wait_tasks` tool description say the same, so the orchestrator knows the rule before it ever meets a cut result.

### Added

- **`max_result_chars` on `create_subagent_toolset` and `SubAgentCapability`** (default `2000`, matching the previous hard-coded limit). Sets the per-result character budget in the `wait_tasks` listing, so a fan-out of verbose subagents can't flood the orchestrator's context. Pass `None` to never truncate; a negative value raises `ValueError`.

## [0.2.9] - 2026-07-19

### Added

- **Stateful subagent conversations via `chat_trace_id`** ([#44](https://github.com/vstorm-co/subagents-pydantic-ai/pull/44)). Every successful `task()` result now ends with a `Chat Trace ID: <id>` line; passing that ID back to `task()` resumes the same subagent conversation with its full message history (stored per `(subagent_name, chat_trace_id)`). Guard rails: a trace can only be continued once its current task has finished (continuing a busy trace returns an error instead of racing and losing one branch of history), continuing an unknown/evicted trace returns an error instead of silently starting a fresh conversation, and a failed first run does not advertise a trace ID. The store is LRU-bounded by the new `max_chat_traces` parameter on `create_subagent_toolset` (default 100) so long-lived sessions don't grow memory without bound.
- **Rich per-task observability on `TaskHandle`** ([#44](https://github.com/vstorm-co/subagents-pydantic-ai/pull/44)). Both sync and async runs now populate the handle with `usage` (including provider detail counters), `message_history` (JSON), `run_id`, `conversation_id`, `traceparent`/`trace_id`/`span_id`, final-response `model_name`/`provider_name`/`provider_url`/`provider_response_id`/`provider_details`/`finish_reason`, summed `cost` (via genai-prices), and `tool_call_counts`. Capture is best-effort by design: the run is marked `COMPLETED` before telemetry is collected, and any capture failure (including message-history capture) logs a warning instead of flipping a successful run to `FAILED`. Sync tasks now register handles too, so `get_total_usage()` finally includes sync runs. Retained finished handles are bounded by the new `max_task_handles` parameter (default 500); evicted handles fold their token usage into `get_total_usage()` totals so aggregates stay correct.

### Changed

- **`check_task()` and `wait_tasks()` no longer embed usage details in tool-return text.** Observability data lives on the `TaskHandle` (inspect `toolset.task_manager`) instead of being fed back into the parent's context. `check_task` shows the `Chat Trace ID` only for completed tasks, matching `wait_tasks`, so continuation is never advertised for a run whose history was not saved.

### Documentation

- **Brand refresh** ([#48](https://github.com/vstorm-co/subagents-pydantic-ai/pull/48)): new social preview card, Pydantic favicon/logo in mkdocs, unified README header with the "Part of Pydantic Deep Agents" callout and Vstorm OSS ecosystem section.
- New "Stateful conversations (`chat_trace_id`)" section in `docs/concepts/toolset.md`, plus the `max_chat_traces` / `max_task_handles` factory parameters.

## [0.2.8] - 2026-06-26

### Fixed

- **pydantic-ai 2.0 compatibility: `'RunUsage' object is not callable`.** The post-run observability step captured usage via `result.usage()`, but pydantic-ai 2.0 turned `AgentRunResult.usage` from a method into a property returning `RunUsage`. Calling it raised inside the `try` that marks a task complete, flipping an otherwise-successful subagent run to `FAILED` — so every delegated task errored. Usage is now read as the `result.usage` property.

### Changed

- **Require `pydantic-ai-slim>=2.0`** (was `>=1.74.0`): the package now targets the 2.0 API (`result.usage` property; context-less tools registered via `tool_plain`, which 2.0 made a hard requirement rather than a deprecation).

## [0.2.7] - 2026-06-04

### Added

- **Unprompted parent -> child steering via `send_message_to_subagent`** ([#28](https://github.com/vstorm-co/subagents-pydantic-ai/issues/28)). A parent agent can now steer a running **async** subagent mid-flight without cancelling it — e.g. "narrow the search to `packages/sparta/`, it isn't in `core/`" — so the subagent adapts on its next step while keeping all partial progress, instead of the lossy cancel-and-respawn pattern. The new `send_message_to_subagent(task_id, message)` tool enqueues a `TASK_UPDATE` on the message bus for `subagent-{task_id}`; the run loop drains it at the next model-request boundary (`_drive_run` gained an `inject_messages` hook alongside the existing `cancel_check`) and folds each message into that request as an extra `UserPromptPart`. Injecting only at model-request boundaries guarantees a steering part is never spliced into a tool-call/tool-return pair. This is distinct from `answer_subagent`, which only replies to a question the subagent already asked via `ask_parent`. Sending to a finished or unknown task returns a clear error. Honoured on the retry-driven run path (`max_retries > 0`, the default); the legacy `agent.run()` fast path (`max_retries == 0`) does not expose node boundaries, so steering messages stay queued there.

## [0.2.6] - 2026-06-01

### Changed

- **Docstring and import hygiene (internal; no behavior change).** Converted reStructuredText-style double-backtick inline code in docstrings and comments to single-backtick Markdown (185 occurrences), so it renders correctly under the mkdocstrings Markdown handler. Hoisted 27 function-local imports to module top where safe; intentionally-lazy, conditional, optional-dependency (`try`/`except ImportError`), and circular-import-avoidance imports were left in place.

### Fixed

- **Retry path skipped node lifecycle hooks without a streaming consumer** (`retry.py`). The no-streaming branch of `_drive_run` advanced the run with a bare `async for _ in run` (`AgentRun.__anext__`), which fires none of the node hooks (`before_node_run` / `after_node_run` / `wrap_node_run` / `on_node_run_error`). With the default `max_retries=3`, any capability that recovers from a node error via `on_node_run_error` was therefore bypassed. The no-streaming branch now drives via `run.next(node)` exactly like `Agent.run` (hooks fire; no streaming overhead).
- **`soft_cancel` sent the cancel request to an unregistered receiver** (`message_bus.py`). It addressed `handle.subagent_name` (e.g. `"researcher"`), but the running subagent registers on the bus as `subagent-{task_id}`, so the send raised a swallowed `KeyError` and the cooperative-cancel message never arrived. Now sends to `subagent-{task_id}`.
- **`create_agent` tool had no description** (`factory.py`). The function body opened with an f-string instead of a string literal, so `__doc__` was `None` and the computed allowed-models / capabilities text was evaluated and discarded on every call. The model-facing description (with models / capabilities / default model interpolated) is now supplied via the `@toolset.tool(description=...)` decorator, and the function carries a normal docstring.
- **Soft cancellation was non-functional - the cancel event was never consumed** (`retry.py`, `toolset.py`). `soft_cancel` set a per-task `asyncio.Event` and sent a `CANCEL_REQUEST`, but nothing in the subagent run path ever checked it, so `soft_cancel` reported success while the task kept running. `run_with_retry`/`_drive_run` now accept an optional `cancel_check` callable that is polled between graph nodes; `run_task` wires it to the task's cancel event so a soft-cancelled subagent stops cooperatively at the next node boundary (raising `asyncio.CancelledError`, which surfaces as `TaskStatus.CANCELLED`). Honoured on the retry-driven path (`max_retries > 0`); the legacy `agent.run()` fast path (`max_retries == 0`) does not expose node boundaries, so soft cancel is best-effort there.
- **`create_agent` silently dropped capabilities with a custom `default_agent_factory`** (`factory.py`). When a custom `default_agent_factory` was configured the factory was called with only `config`, so any requested capabilities/toolsets were discarded even though the success message still reported them as enabled. `create_agent` now returns an error when capabilities are requested alongside a custom factory, since the factory owns the whole agent build and cannot receive injected toolsets.
- **`create_task` assigned a raw string status instead of the enum** (`message_bus.py`). It set `handle.status = "running"` rather than `TaskStatus.RUNNING`; equal via the str-Enum but inconsistent with the rest of the code and breaking any `isinstance(status, TaskStatus)` check. Now assigns `TaskStatus.RUNNING`.
- **`hard_cancel` clobbered the outcome of an already-finished task** (`message_bus.py`). It unconditionally set `handle.status = "cancelled"` and `completed_at` even when the task had already completed or failed, overwriting the real outcome and racing with `run_task`'s teardown. The handle update is now guarded under `not task.done()`, so a finished task keeps its `COMPLETED`/`FAILED` status and `completed_at`.

### Documentation

- **Documentation accuracy pass and new pages.** Fixed the wrong `get_subagent_system_prompt` signature and sample output, corrected the dynamic-agents factory defaults (`default_model="openai:gpt-4.1"`) and its missing options, and corrected the `can_ask_questions`/`max_questions` defaults in the config reference. Added a new **Retries** guide and API entries for `SubAgentSpec`, `UsageLimitsFactory`, `AskUserCallback`, the prompt/description constants, and the retry helpers (`RetryConfig`, `run_with_retry`, `is_transient_error`, `compute_backoff_delay`); completed the `create_subagent_toolset` tool list (added `wait_tasks`). Documented the `SubAgentCapability` `ask_user` limitation, usage limits, and the `max_nesting_depth`/`clone_for_subagent` deps contract, and clarified the `Agent.from_file` / `SubAgentSpec` YAML-loading paths. `mkdocs build --strict` passes with zero warnings.

## [0.2.5] - 2026-05-24

### Infrastructure

Pure CI / dependency-bot housekeeping — no source-code changes, no behaviour change since 0.2.4. Consolidates the two open Renovate auto-PRs plus the preemptive `setup-uv` / `setup-python` major bumps (same set Renovate has been gradually surfacing across the sibling repos) into a single release so downstream consumers see one bump instead of four.

- **CI: bump `actions/checkout` to `v6`** across `ci.yml` (×3), `docs.yml`, `publish.yml` ([#34](https://github.com/vstorm-co/subagents-pydantic-ai/pull/34), Renovate auto-PR — folded in here).
- **CI: bump `docs.yml` Python to `3.14`** ([#33](https://github.com/vstorm-co/subagents-pydantic-ai/pull/33), Renovate auto-PR — folded in here).
- **CI: bump `astral-sh/setup-uv` to `v8.1.0`** across `ci.yml` (×3) and `publish.yml`. Pinned to the specific patch because `astral-sh/setup-uv` does not maintain a rolling `v8` tag (only `v8.0.0` / `v8.1.0`; `v7` and earlier do have rolling majors).
- **CI: bump `actions/setup-python` to `v6`** in `docs.yml` — `v6` has a rolling tag so plain `@v6` is used.

The `ci.yml` test matrix is unchanged.

## [0.2.4] - 2026-05-24

### Added

- **`wait_tasks(mode="any")` for reactive orchestration** ([#29](https://github.com/vstorm-co/subagents-pydantic-ai/issues/29), [#30](https://github.com/vstorm-co/subagents-pydantic-ai/pull/30) by [@Gby56](https://github.com/Gby56)) — new `mode: Literal["all", "any"] = "all"` parameter on `wait_tasks`. `mode="any"` returns as soon as the first task reaches a terminal state (completed/failed/cancelled), so an orchestrator can act on the first finisher instead of stalling on the slowest. Default `mode="all"` is backward-compatible. Output now includes a header (`Task results (mode=any, X/Y finished, Z still running):`) and explicitly labels `CANCELLED` tasks.

### Fixed

- **`wait_tasks` no longer cascades cancellation to its workers.** Previously the default (`mode="all"`) path used `asyncio.wait_for(asyncio.gather(...))`, both of which propagate cancellation to their constituent tasks. When pydantic-ai's `_call_tools` sibling-cancel hit the `wait_tasks` tool call (e.g. another tool raised during a parallel turn), or any outer cancel reached the orchestrator, the cascade silently killed every in-flight subagent — they surfaced as `TaskStatus.CANCELLED` with an empty `error` string even though the parent never requested it. Both modes now use `asyncio.wait(..., return_when=...)`, which does not cancel its awaitees on timeout or caller cancellation. Workers keep owning their own lifecycle. Diagnosed by [@Gby56](https://github.com/Gby56).

## [0.2.3] - 2026-05-17

### Added

- **Auto-retry for transient subagent failures** — subagents are resilient to flaky model gateways/proxies (e.g. a LiteLLM gateway returning 502/503/429 or dropping connections) **by default**. New `subagents_pydantic_ai.retry` module:
  - `is_transient_error(exc)` — classifies retryable failures: `ModelHTTPError` with a 408/409/425/429/5xx status, and non-HTTP `ModelAPIError` (transport/connection errors). Auth/4xx, `UnexpectedModelBehavior`, `UsageLimitExceeded`, validation errors and task cancellation are **not** retried.
  - `RetryConfig` (frozen dataclass) + `RetryConfig.from_config()` — exponential backoff with configurable initial/max delay, multiplier, full jitter, and an optional custom `retry_on` predicate. Defaults to **3 retries** (`max_retries=3`).
  - `compute_backoff_delay()` — pure backoff helper with an injectable RNG.
  - `run_with_retry()` — drives the subagent and, on a transient failure, **replays the accumulated `message_history` so the subagent resumes instead of restarting from scratch**. Uses `Agent.iter()` rather than `capture_run_messages()` to recover the failed run's messages, sidestepping [pydantic/pydantic-ai#1568](https://github.com/pydantic/pydantic-ai/issues/1568) (nested `capture_run_messages` contexts do not work, and subagents always run nested inside the parent agent's run).
  - Exported: `RetryConfig`, `run_with_retry`, `is_transient_error`, `compute_backoff_delay`.
- **Retry configuration on `SubAgentConfig`** — `max_retries` (default `3`), `retry_initial_delay`, `retry_max_delay`, `retry_backoff_multiplier`, `retry_jitter`, `retry_on`. Set `max_retries=0` to disable retrying (the legacy `agent.run()` opt-out path). Consumers like pydantic-deep get this for free through the re-exported `SubAgentConfig` with no code change.
- **`TaskStatus.RETRYING`** and **`TaskHandle.retry_count`** — async-mode tasks surface in-progress retries via `check_task`; the transient error message is cleared from the handle once a retry eventually succeeds.
- **Usage-limits forwarding for delegated subagents** ([#25](https://github.com/vstorm-co/subagents-pydantic-ai/pull/25)) — `usage_limits` on `create_subagent_toolset()` and `SubAgentCapability`, accepting a static `pydantic_ai.UsageLimits` or a per-task `UsageLimitsFactory` `(RunContext, SubAgentConfig) -> UsageLimits | None` resolved once per delegated task. Limits are forwarded to sync and async runs and are **honoured on every retry attempt**. New public `UsageLimitsFactory` type alias (exported).

### Changed

- **`_run_sync` / `_run_async` now execute the subagent through `run_with_retry`.** With retries enabled (the default, `max_retries=3`) execution is driven via `Agent.iter()` from the first attempt, so a transient failure resumes with the full accumulated message history. Only genuinely transient errors are retried; non-transient errors fail immediately exactly as before. With `max_retries=0` it is **exactly the legacy `agent.run()` path** (opt-out, no behaviour change). `asyncio.CancelledError` is never caught by the retry loop, so soft/hard task cancellation is unaffected.

## [0.2.2] - 2026-04-20

### Added

- **`ask_user` parameter on `create_subagent_toolset`** — `Callable[[str], Awaitable[str]]` invoked when a subagent calls `ask_parent` in sync mode. The callback is attached to the cloned subagent deps via `_subagent_state["ask_callback"]`, so `ask_parent` resolves through the same path as async mode. Required for sync-mode subagents with `can_ask_questions=True`. Exported as `AskUserCallback`.

### Fixed

- **`ask_parent` no longer silently fails in sync mode** — previously, when a subagent with `can_ask_questions=True` ran in sync mode without an `ask_user` method on deps, `ask_parent` returned `"Error: Cannot ask parent - no communication channel configured"` — which the subagent LLM tended to launder into an invented answer. The error message now points to the fix and a first-class `ask_user` hook exists. ([#23](https://github.com/vstorm-co/subagents-pydantic-ai/issues/23))

### Changed

- **Docs: corrected sync-mode question semantics** — `docs/advanced/questions.md` previously claimed the parent could respond via `answer_subagent` in sync mode. That is architecturally impossible because the parent's run loop is blocked inside the subagent's `task` call. The docs now describe the `ask_user` callback flow.

## [0.2.1] - 2026-03-31

### Changed

- Bump minimum `pydantic-ai-slim` to `>=1.74.0` for compatibility with async `get_instructions` on toolsets

## [0.2.0] - 2026-03-30

### Added

- **Custom agent support** via `agent` and `agent_factory` fields on `SubAgentConfig`:
  ```python
  SubAgentConfig(
      name="researcher",
      description="Deep research agent",
      instructions="...",
      agent=my_prebuilt_agent,  # pre-built agent, used as-is
  )
  # OR
  SubAgentConfig(
      name="researcher",
      description="Deep research agent",
      instructions="...",
      agent_factory=lambda cfg: create_deep_agent(  # factory creates agent from config
          model=cfg["model"], instructions=cfg["instructions"],
      ),
  )
  ```
  - Priority chain in `_compile_subagent()`: `agent` > `agent_factory` > default `Agent()`
  - Enables frameworks like pydantic-deep to create full-featured agents as subagents
- **`default_agent_factory`** parameter on `create_agent_factory_toolset()` — overrides default `Agent()` creation for dynamically spawned agents
- **`SubAgentSpec`** — Pydantic model for declarative subagent configuration via YAML/JSON:
  ```yaml
  subagents:
    - name: researcher
      description: Research assistant
      instructions: You research topics thoroughly.
      model: openai:gpt-4.1-mini
  ```
  - `to_config()` / `from_config()` round-trip conversion
  - JSON/YAML serialization via Pydantic's `model_dump()` / `model_validate()`

- **Token usage tracking** (issue [#45](https://github.com/vstorm-co/pydantic-deepagents/issues/45)):
  - `TaskHandle.usage` — stores `RunUsage` from each subagent run
  - `check_task` displays token usage (input/output) for completed tasks
  - `get_total_usage()` on toolset — aggregates usage across all task handles
  - `TaskManager.list_handles()` — returns all task handles
- **Structured output serialization** (issue [#46](https://github.com/vstorm-co/pydantic-deepagents/issues/46)):
  - `_serialize_output()` uses `model_dump_json()` for Pydantic models and `json.dumps(asdict())` for dataclasses instead of `str()`, preserving JSON structure for the parent agent

### Changed

- `_compile_subagent()` now checks for custom `agent`/`agent_factory` before creating default `Agent()`
- Subagent results are now proper JSON when `output_type` is a Pydantic model (previously flattened to Python repr string)

## [0.1.0] - 2026-03-26

### Added

- **`SubAgentCapability`** — new pydantic-ai [capability](https://ai.pydantic.dev/capabilities/) that bundles subagent tools + dynamic system prompt into a single plug-and-play unit. This is now the recommended way to add subagent delegation:
  ```python
  from pydantic_ai import Agent
  from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

  agent = Agent("openai:gpt-4.1", capabilities=[SubAgentCapability(
      subagents=[SubAgentConfig(name="researcher", description="Researches topics", instructions="...")],
  )])
  ```
  - Registers all tools automatically (`task`, `check_task`, `answer_subagent`, `list_active_tasks`, `soft_cancel_task`, `hard_cancel_task`)
  - Injects dynamic system prompt listing available subagents
  - Exposes `task_manager` property for observability
  - Supports AgentSpec YAML serialization

### Changed

- **Minimum pydantic-ai version bumped to `>=1.71.0`** (capabilities API support)
- **Documentation rewritten for capabilities-first approach** — README and examples now lead with `SubAgentCapability`

## [0.0.8] - 2026-03-06

### Fixed

- **Accept `Model` objects in subagent configuration** — `create_subagent_toolset()`, `_compile_subagent()`, `create_agent_factory_toolset()`, and `SubAgentConfig.model` now accept `str | Model` instead of only `str`. Previously, passing a `Model` object (e.g. `TestModel()`, `AnthropicModel()`) as `default_model` would be silently discarded by the caller. ([#15](https://github.com/vstorm-co/subagents-pydantic-ai/pull/15), by [@ret2libc](https://github.com/ret2libc))
- **`ask_parent` tool broken in async mode** — `ask_parent()` checked `ctx._subagent_state` but pydantic-ai never sets custom attributes on `RunContext`. State is now injected via `deps._subagent_state` in `_run_async()`. Additionally, `answer_subagent` used `message_bus.send()` instead of resolving the future that `ask_parent` awaits, so answers were never delivered. Replaced message bus Q&A with direct `asyncio.Future` coordination via `TaskManager`. ([#14](https://github.com/vstorm-co/subagents-pydantic-ai/issues/14))

## [0.0.7] - 2026-02-26

### Added

- **Custom tool descriptions** — `create_subagent_toolset()` now accepts `descriptions: dict[str, str] | None` parameter to override any tool's built-in description

## [0.0.6] - 2026-02-24

### Changed

- **Expanded `TASK_TOOL_DESCRIPTION`** — From 8 lines to ~40 lines with "When to use" / "When NOT to use" sections, usage notes, and execution mode explanation. Follows the Claude Code / deepagents pattern of putting detailed guidance in tool descriptions rather than system prompt.
- **Added description constants for all secondary tools** — `CHECK_TASK_DESCRIPTION`, `ANSWER_SUBAGENT_DESCRIPTION`, `LIST_ACTIVE_TASKS_DESCRIPTION`, `WAIT_TASKS_DESCRIPTION`, `SOFT_CANCEL_TASK_DESCRIPTION`, `HARD_CANCEL_TASK_DESCRIPTION`. All wired via `@toolset.tool(description=CONSTANT)` and exported from the package.
- **Slimmed `get_subagent_system_prompt()`** — Changed from multi-line format with `DUAL_MODE_SYSTEM_PROMPT` injection to a compact `- **name**: description` listing. Dual-mode explanation moved into `TASK_TOOL_DESCRIPTION`.
- **Dynamic `task` tool description** — The task tool now builds its description by appending the available subagent list to `TASK_TOOL_DESCRIPTION` at toolset creation time, instead of using an f-string docstring.

## [0.0.5] - 2025-02-15

### Added

- **Dynamic registry lookup in `task()`**: `create_subagent_toolset()` now accepts an optional `registry` parameter. When a subagent type is not found in the static compiled list, the toolset falls back to the dynamic registry — enabling seamless delegation to agents created at runtime via `create_agent_factory_toolset()`.
- **`context_files` field in `SubAgentConfig`**: Per-subagent context file paths, loaded by consumer libraries (e.g., pydantic-deep's `ContextToolset`).
- **`extra` field in `SubAgentConfig`**: Generic extensibility dict for consumer libraries to attach metadata (e.g., `memory`, `team`, `cost_budget`) without subagents-pydantic-ai needing to know about them.
- **Documentation**: Expanded guides for dynamic agents, execution modes, message bus, and subagent questions.

## [0.0.4] - 2025-02-12

### Fixed

- **Compatibility**: Replaced all `agent._register_toolset()` calls with pydantic-ai public API ([#5](https://github.com/vstorm-co/subagents-pydantic-ai/issues/5), [#6](https://github.com/vstorm-co/subagents-pydantic-ai/pull/6) by [@pedroallenrevez](https://github.com/pedroallenrevez))
  - `_compile_subagent()`: toolsets passed to `Agent()` constructor via `toolsets=` parameter
  - `task()` runtime toolsets: passed to `agent.run(toolsets=...)` instead of registering on agent instance
  - `create_agent_factory_toolset()`: toolsets from factory/capabilities passed to `Agent()` constructor
  - Fixes `AttributeError: 'Agent' object has no attribute '_register_toolset'` with pydantic-ai >= 1.38

### Changed

- Bumped minimum `pydantic-ai-slim` dependency from `>=0.1.0` to `>=1.38`

## [0.0.3] - 2025-01-23

### Fixed

- **Documentation**: Fixed incorrect import `from pydantic_ai import Toolset` → `from pydantic_ai.toolsets import FunctionToolset`
- **Documentation**: Fixed typo `BuitinTools` → `WebSearchTool` from `pydantic_ai.builtin_tools`

## [0.0.2] - 2025-01-22

### Added

- Complete documentation site with Material for MkDocs
  - Core Concepts: subagents, toolset, types
  - Advanced Features: execution modes, questions, cancellation, dynamic agents, message bus
  - Examples: basic usage, sync/async, toolsets, questions, nesting, research team
  - API Reference with mkdocstrings
- CONTRIBUTING.md with development guidelines
- Use cases section in README

### Changed

- README rewritten with clearer structure
- Updated tagline to "Multi-Agent Orchestration for Pydantic AI"
- Improved "Why Choose This Library?" section

## [0.0.1] - 2025-01-15

### Added

- Initial release
- `create_subagent_toolset()` for creating subagent delegation tools
- `create_agent_factory_toolset()` for runtime agent creation
- Dual-mode execution (sync/async) with auto-mode selection
- Parent-child Q&A communication
- Soft and hard task cancellation
- Pluggable message bus architecture
- `SubAgentConfig`, `TaskHandle`, `TaskStatus`, `TaskPriority` types
- `DynamicAgentRegistry` for managing runtime-created agents
- 100% test coverage
