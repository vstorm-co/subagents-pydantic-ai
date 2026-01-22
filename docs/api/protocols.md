# Protocols API

## SubAgentDepsProtocol

::: subagents_pydantic_ai.SubAgentDepsProtocol
    options:
      show_root_heading: true
      show_source: true

## MessageBusProtocol

::: subagents_pydantic_ai.MessageBusProtocol
    options:
      show_root_heading: true
      show_source: true

## InMemoryMessageBus

::: subagents_pydantic_ai.InMemoryMessageBus
    options:
      show_root_heading: true
      show_source: true

## TaskManager

::: subagents_pydantic_ai.TaskManager
    options:
      show_root_heading: true
      show_source: true

## create_message_bus

::: subagents_pydantic_ai.create_message_bus
    options:
      show_root_heading: true
      show_source: true

## DynamicAgentRegistry

::: subagents_pydantic_ai.DynamicAgentRegistry
    options:
      show_root_heading: true
      show_source: true

---

## Usage Examples

### Implementing SubAgentDepsProtocol

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MyDeps:
    """Custom dependencies implementing SubAgentDepsProtocol."""

    subagents: dict[str, Any] = field(default_factory=dict)
    database_url: str = ""
    api_key: str = ""

    def clone_for_subagent(self, max_depth: int = 0) -> "MyDeps":
        """Create isolated deps for subagent."""
        return MyDeps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            database_url=self.database_url,  # Share read-only config
            api_key=self.api_key,
        )
```

### Implementing Custom Message Bus

```python
from subagents_pydantic_ai import MessageBusProtocol, AgentMessage

class RedisMessageBus:
    """Redis-based message bus for distributed systems."""

    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)

    async def send(self, message: AgentMessage) -> None:
        channel = f"agent:{message.receiver}"
        await self.redis.publish(channel, message.json())

    async def receive(
        self,
        agent_id: str,
        timeout: float | None = None,
    ) -> AgentMessage | None:
        # Implementation...
        pass

    async def subscribe(self, agent_id: str) -> None:
        pass

    async def unsubscribe(self, agent_id: str) -> None:
        pass
```

### Using TaskManager

```python
from subagents_pydantic_ai import TaskManager, InMemoryMessageBus

bus = InMemoryMessageBus()
manager = TaskManager(message_bus=bus)

# Create a task
handle = await manager.create_task(
    subagent_name="researcher",
    description="Research Python async",
)

# Check status
status = await manager.get_task_status(handle.task_id)

# Answer a question
await manager.answer_question(handle.task_id, "Use asyncio")

# Cancel a task
await manager.cancel_task(handle.task_id, hard=False)
```

### Using DynamicAgentRegistry

```python
from subagents_pydantic_ai import DynamicAgentRegistry

registry = DynamicAgentRegistry()

# List registered agents
agents = registry.list_agents()

# Get a specific agent
agent = registry.get_agent("custom-analyst")

# Remove an agent
registry.remove_agent("custom-analyst")
```
