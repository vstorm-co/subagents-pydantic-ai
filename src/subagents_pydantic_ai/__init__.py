"""Subagent toolset for pydantic-ai with dual-mode execution.

This library provides a toolset for delegating tasks to specialized
subagents within pydantic-ai agents. It supports:

- **Dual-mode execution**: run a task synchronously (blocking) or in the
  background, or let `mode="auto"` choose from the task's characteristics
- **Chat traces**: continue a subagent's conversation across delegations
- **Questions**: a subagent can ask its parent for clarification mid-task
- **Steering**: redirect a running background task without losing its progress
- **Cancellation**: cooperative (stops at a clean boundary) or immediate
- **Streaming**: a delegation's events as they happen, labelled by task
- **Retries**: transient model/gateway failures resume from accumulated history
- **Observability**: per-task usage, cost, tool-call counts, and traceparent
- **Dynamic agents**: create specialists at runtime, reusable or one-shot

Basic usage:

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics",
        instructions="You are a research assistant.",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
agent = Agent("openai:gpt-4.1", toolsets=[toolset])
```
"""

from subagents_pydantic_ai.capability import (
    SubAgentCapability as SubAgentCapability,
)
from subagents_pydantic_ai.dynamic_agent import (
    AgentFactory as AgentFactory,
)
from subagents_pydantic_ai.factory import (
    create_agent_factory_toolset as create_agent_factory_toolset,
)
from subagents_pydantic_ai.message_bus import (
    DEFAULT_CANCEL_GRACE_SECONDS as DEFAULT_CANCEL_GRACE_SECONDS,
)
from subagents_pydantic_ai.message_bus import (
    InMemoryMessageBus as InMemoryMessageBus,
)
from subagents_pydantic_ai.message_bus import (
    TaskManager as TaskManager,
)
from subagents_pydantic_ai.message_bus import (
    create_message_bus as create_message_bus,
)
from subagents_pydantic_ai.prompts import (
    ANSWER_SUBAGENT_DESCRIPTION as ANSWER_SUBAGENT_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    CHECK_TASK_DESCRIPTION as CHECK_TASK_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION as DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    DELEGATE_TOOL_DESCRIPTION as DELEGATE_TOOL_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    DUAL_MODE_SYSTEM_PROMPT as DUAL_MODE_SYSTEM_PROMPT,
)
from subagents_pydantic_ai.prompts import (
    HARD_CANCEL_TASK_DESCRIPTION as HARD_CANCEL_TASK_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    LIST_ACTIVE_TASKS_DESCRIPTION as LIST_ACTIVE_TASKS_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    SOFT_CANCEL_TASK_DESCRIPTION as SOFT_CANCEL_TASK_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    SUBAGENT_SYSTEM_PROMPT as SUBAGENT_SYSTEM_PROMPT,
)
from subagents_pydantic_ai.prompts import (
    TASK_TOOL_DESCRIPTION as TASK_TOOL_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    WAIT_TASKS_DESCRIPTION as WAIT_TASKS_DESCRIPTION,
)
from subagents_pydantic_ai.prompts import (
    get_subagent_system_prompt as get_subagent_system_prompt,
)
from subagents_pydantic_ai.prompts import (
    get_task_instructions_prompt as get_task_instructions_prompt,
)
from subagents_pydantic_ai.protocols import (
    MessageBusProtocol as MessageBusProtocol,
)
from subagents_pydantic_ai.protocols import (
    SubAgentDepsProtocol as SubAgentDepsProtocol,
)
from subagents_pydantic_ai.registry import (
    DynamicAgentRegistry as DynamicAgentRegistry,
)
from subagents_pydantic_ai.retry import (
    RetryConfig as RetryConfig,
)
from subagents_pydantic_ai.retry import (
    compute_backoff_delay as compute_backoff_delay,
)
from subagents_pydantic_ai.retry import (
    is_transient_error as is_transient_error,
)
from subagents_pydantic_ai.retry import (
    run_with_retry as run_with_retry,
)
from subagents_pydantic_ai.spec import (
    SubAgentSpec as SubAgentSpec,
)
from subagents_pydantic_ai.toolset import (
    SubAgentToolset as SubAgentToolset,
)
from subagents_pydantic_ai.toolset import (
    create_subagent_toolset as create_subagent_toolset,
)
from subagents_pydantic_ai.types import (
    TERMINAL_STATUSES as TERMINAL_STATUSES,
)
from subagents_pydantic_ai.types import (
    AgentMessage as AgentMessage,
)
from subagents_pydantic_ai.types import (
    AskUserCallback as AskUserCallback,
)
from subagents_pydantic_ai.types import (
    CompiledSubAgent as CompiledSubAgent,
)
from subagents_pydantic_ai.types import (
    DelegationConfiguration as DelegationConfiguration,
)
from subagents_pydantic_ai.types import (
    EventStreamHandlerFactory as EventStreamHandlerFactory,
)
from subagents_pydantic_ai.types import (
    ExecutionMode as ExecutionMode,
)
from subagents_pydantic_ai.types import (
    MessageType as MessageType,
)
from subagents_pydantic_ai.types import (
    SubAgentConfig as SubAgentConfig,
)
from subagents_pydantic_ai.types import (
    TaskCharacteristics as TaskCharacteristics,
)
from subagents_pydantic_ai.types import (
    TaskHandle as TaskHandle,
)
from subagents_pydantic_ai.types import (
    TaskPriority as TaskPriority,
)
from subagents_pydantic_ai.types import (
    TaskStatus as TaskStatus,
)
from subagents_pydantic_ai.types import (
    ToolsetFactory as ToolsetFactory,
)
from subagents_pydantic_ai.types import (
    UsageLimitsFactory as UsageLimitsFactory,
)
from subagents_pydantic_ai.types import (
    decide_execution_mode as decide_execution_mode,
)
from subagents_pydantic_ai.types import (
    utcnow as utcnow,
)

__all__ = [
    # Capability
    "SubAgentCapability",
    # Protocols
    "SubAgentDepsProtocol",
    "MessageBusProtocol",
    # Types
    "SubAgentSpec",
    "SubAgentConfig",
    "TaskHandle",
    "AgentMessage",
    "MessageType",
    "TaskStatus",
    "TERMINAL_STATUSES",
    "TaskPriority",
    "TaskCharacteristics",
    "ExecutionMode",
    "DelegationConfiguration",
    "EventStreamHandlerFactory",
    "ToolsetFactory",
    "AgentFactory",
    "UsageLimitsFactory",
    "AskUserCallback",
    "CompiledSubAgent",
    # Functions
    "decide_execution_mode",
    "utcnow",
    # Toolsets
    "create_subagent_toolset",
    "SubAgentToolset",
    "create_agent_factory_toolset",
    # Message Bus
    "InMemoryMessageBus",
    "create_message_bus",
    "TaskManager",
    "DEFAULT_CANCEL_GRACE_SECONDS",
    # Registry
    "DynamicAgentRegistry",
    # Retry
    "RetryConfig",
    "run_with_retry",
    "is_transient_error",
    "compute_backoff_delay",
    # Utilities
    "get_subagent_system_prompt",
    "get_task_instructions_prompt",
    # Prompts & Tool Descriptions
    "SUBAGENT_SYSTEM_PROMPT",
    "DUAL_MODE_SYSTEM_PROMPT",
    "DEFAULT_GENERAL_PURPOSE_DESCRIPTION",
    "DELEGATE_TOOL_DESCRIPTION",
    "TASK_TOOL_DESCRIPTION",
    "CHECK_TASK_DESCRIPTION",
    "ANSWER_SUBAGENT_DESCRIPTION",
    "LIST_ACTIVE_TASKS_DESCRIPTION",
    "WAIT_TASKS_DESCRIPTION",
    "SOFT_CANCEL_TASK_DESCRIPTION",
    "HARD_CANCEL_TASK_DESCRIPTION",
    # Version
    "__version__",
]

from importlib.metadata import version

__version__ = version("subagents-pydantic-ai")
