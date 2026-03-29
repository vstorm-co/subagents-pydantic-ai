# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-29

### Added

- **`SubAgentSpec`** — Pydantic model for declarative subagent configuration via YAML/JSON. Enables defining subagents in spec files instead of Python code:
  ```yaml
  subagents:
    - name: researcher
      description: Research assistant
      instructions: You research topics thoroughly.
      model: openai:gpt-4.1-mini
  ```
  - `to_config()` — convert spec to `SubAgentConfig` dict
  - `from_config()` — create spec from `SubAgentConfig` dict
  - Full round-trip support: spec -> config -> spec and config -> spec -> config
  - JSON/YAML serialization via Pydantic's `model_dump()` / `model_validate()`

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
