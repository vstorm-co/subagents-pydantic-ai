# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-01-20

### Added

- Initial release of `subagents-pydantic-ai`
- **Dual-Mode Execution**: Support for synchronous (blocking) and asynchronous (background) task execution
- **Auto-Mode Selection**: Intelligent automatic mode selection based on task characteristics
  - `TaskPriority` enum with `LOW`, `NORMAL`, `HIGH`, `CRITICAL` levels
  - `TaskCharacteristics` dataclass for mode decision factors
  - `decide_execution_mode()` function for automatic sync/async selection
- **Subagent Configuration**: Flexible configuration via `SubAgentConfig`
  - Support for `preferred_mode`, `typical_complexity`, `typically_needs_context` fields
- **Message Bus**: In-memory message bus for agent communication
  - `InMemoryMessageBus` implementation
  - Support for ask/answer patterns between agents
- **Task Management**: Background task tracking and control
  - `TaskManager` for task lifecycle management
  - `TaskHandle` for task status monitoring
  - Soft and hard cancellation support
- **Dynamic Agent Registry**: Runtime agent creation and management
  - `DynamicAgentRegistry` for registering agents at runtime
  - `create_agent_factory_toolset()` for dynamic agent creation
- **Core Toolset**: Main subagent delegation toolset
  - `create_subagent_toolset()` factory function
  - `task` tool for delegating work to subagents
  - `check_task`, `list_active_tasks` for monitoring
  - `answer_subagent` for responding to subagent questions
  - `soft_cancel_task`, `hard_cancel_task` for task control
- **Protocols**: Type-safe interfaces
  - `SubAgentDepsProtocol` for dependency injection
  - `MessageBusProtocol` for message bus implementations

### Documentation

- Quick start guide
- Dual-mode execution guide
- API reference
- Usage examples

[0.0.1]: https://github.com/vstorm-co/subagents-pydantic-ai/releases/tag/v0.0.1
