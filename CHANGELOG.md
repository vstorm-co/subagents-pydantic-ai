# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
