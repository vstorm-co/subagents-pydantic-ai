# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
