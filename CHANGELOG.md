# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.8] - 2026-06-26

### Added

- **Stateful subagent sessions.** A subagent run can now carry a `chat_trace_id` so its message history is persisted and a later task can resume the same conversation instead of starting cold.
- **Observability captured on task handles.** After a run, the task handle records the run's `usage`, full `message_history`, `run_id`, `conversation_id`, and `traceparent`. Capture is best-effort — a telemetry failure never flips a successful run to `FAILED`.

### Changed

- **Require `pydantic-ai-slim>=2.0`** (was `>=1.74.0`). The observability capture reads `AgentRunResult.usage`, which pydantic-ai 2.0 turned from a method into a property; the package now targets the 2.0 API.
- Renamed `session_id` to `chat_trace_id` so the identifier isn't redacted by Logfire.

### Fixed

- **pydantic-ai 2.0 compatibility: `'RunUsage' object is not callable`.** Usage was captured via `result.usage()`, but 2.0 made `usage` a property returning `RunUsage`. Calling it raised inside the post-run capture and flipped an otherwise-successful subagent run to `FAILED`, so every delegated task errored. Usage is now read as the `result.usage` property.
- Updated tests to register context-less tools via `FunctionToolset.tool_plain`; pydantic-ai 2.0 turned the context-less `.tool()` form from a deprecation warning into a hard error.

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
