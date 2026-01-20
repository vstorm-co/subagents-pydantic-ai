# API Reference

## Toolset Factories

### create_subagent_toolset

::: subagents_pydantic_ai.create_subagent_toolset

### create_agent_factory_toolset

::: subagents_pydantic_ai.create_agent_factory_toolset

## Types

### SubAgentConfig

::: subagents_pydantic_ai.SubAgentConfig

### TaskHandle

::: subagents_pydantic_ai.TaskHandle

### AgentMessage

::: subagents_pydantic_ai.AgentMessage

### ExecutionMode

::: subagents_pydantic_ai.ExecutionMode

### MessageType

::: subagents_pydantic_ai.MessageType

### TaskStatus

::: subagents_pydantic_ai.TaskStatus

### ToolsetFactory

::: subagents_pydantic_ai.ToolsetFactory

## Protocols

### SubAgentDepsProtocol

::: subagents_pydantic_ai.SubAgentDepsProtocol

### MessageBusProtocol

::: subagents_pydantic_ai.MessageBusProtocol

## Message Bus

### InMemoryMessageBus

::: subagents_pydantic_ai.InMemoryMessageBus

### create_message_bus

::: subagents_pydantic_ai.create_message_bus

### TaskManager

::: subagents_pydantic_ai.TaskManager

## Registry

### DynamicAgentRegistry

::: subagents_pydantic_ai.DynamicAgentRegistry

## Prompts

### get_subagent_system_prompt

::: subagents_pydantic_ai.get_subagent_system_prompt

### get_task_instructions_prompt

::: subagents_pydantic_ai.get_task_instructions_prompt

### Constants

- `SUBAGENT_SYSTEM_PROMPT` - Default system prompt for subagents
- `DUAL_MODE_SYSTEM_PROMPT` - Explanation of sync/async modes
- `DEFAULT_GENERAL_PURPOSE_DESCRIPTION` - Description for the general-purpose subagent
- `TASK_TOOL_DESCRIPTION` - Description for the task tool
