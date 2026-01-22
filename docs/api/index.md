# API Reference

Complete API documentation for Subagents for Pydantic AI.

## Modules

### Toolset

The main entry point for creating subagent delegation capabilities.

- [`create_subagent_toolset()`](toolset.md#create_subagent_toolset) - Create a subagent toolset
- [`create_agent_factory_toolset()`](toolset.md#create_agent_factory_toolset) - Create dynamic agent factory
- [`SubAgentToolset`](toolset.md#subagenttoolset) - The toolset class
- [`get_subagent_system_prompt()`](toolset.md#get_subagent_system_prompt) - Generate system prompt

### Types

Data structures used throughout the library.

- [`SubAgentConfig`](types.md#subagentconfig) - Subagent configuration
- [`CompiledSubAgent`](types.md#compiledsubagent) - Pre-compiled subagent
- [`TaskHandle`](types.md#taskhandle) - Background task handle
- [`TaskStatus`](types.md#taskstatus) - Task status enum
- [`TaskPriority`](types.md#taskpriority) - Task priority enum
- [`ExecutionMode`](types.md#executionmode) - Execution mode type
- [`TaskCharacteristics`](types.md#taskcharacteristics) - Auto-mode characteristics
- [`AgentMessage`](types.md#agentmessage) - Inter-agent message
- [`MessageType`](types.md#messagetype) - Message type enum
- [`decide_execution_mode()`](types.md#decide_execution_mode) - Mode selection function

### Protocols

Interface definitions for extensibility.

- [`SubAgentDepsProtocol`](protocols.md#subagentdepsprotocol) - Dependencies protocol
- [`MessageBusProtocol`](protocols.md#messagebusprotocol) - Message bus protocol

### Message Bus

Communication layer components.

- [`InMemoryMessageBus`](protocols.md#inmemorymessagebus) - Default message bus
- [`TaskManager`](protocols.md#taskmanager) - Task coordination
- [`create_message_bus()`](protocols.md#create_message_bus) - Factory function

### Registry

Dynamic agent management.

- [`DynamicAgentRegistry`](protocols.md#dynamicagentregistry) - Agent registry

## Quick Reference

### Creating a Toolset

```python
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics",
        instructions="You are a research assistant.",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
```

### Implementing Dependencies

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())
```

### All Exports

```python
from subagents_pydantic_ai import (
    # Toolsets
    create_subagent_toolset,
    SubAgentToolset,
    create_agent_factory_toolset,
    # Types
    SubAgentConfig,
    CompiledSubAgent,
    TaskHandle,
    TaskStatus,
    TaskPriority,
    TaskCharacteristics,
    ExecutionMode,
    AgentMessage,
    MessageType,
    ToolsetFactory,
    # Functions
    decide_execution_mode,
    get_subagent_system_prompt,
    get_task_instructions_prompt,
    # Protocols
    SubAgentDepsProtocol,
    MessageBusProtocol,
    # Message Bus
    InMemoryMessageBus,
    create_message_bus,
    TaskManager,
    # Registry
    DynamicAgentRegistry,
    # Prompts
    SUBAGENT_SYSTEM_PROMPT,
    DUAL_MODE_SYSTEM_PROMPT,
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    TASK_TOOL_DESCRIPTION,
)
```
