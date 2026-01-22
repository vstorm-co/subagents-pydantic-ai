# Message Bus

The message bus handles communication between parent agents and subagents. By default, an in-memory bus is used, but you can implement custom buses for distributed systems.

## Default In-Memory Bus

The library uses `InMemoryMessageBus` by default:

```python
from subagents_pydantic_ai import create_subagent_toolset

# Uses InMemoryMessageBus internally
toolset = create_subagent_toolset(subagents=subagents)
```

This is suitable for:

- Single-process applications
- Development and testing
- Simple deployments

## Message Types

The bus handles these message types:

| Type | Direction | Purpose |
|------|-----------|---------|
| `TASK_ASSIGNED` | Parent → Subagent | New task assignment |
| `TASK_UPDATE` | Subagent → Parent | Progress update |
| `TASK_COMPLETED` | Subagent → Parent | Task finished successfully |
| `TASK_FAILED` | Subagent → Parent | Task failed with error |
| `QUESTION` | Subagent → Parent | Asking for clarification |
| `ANSWER` | Parent → Subagent | Response to question |
| `CANCEL_REQUEST` | Parent → Subagent | Soft cancellation request |
| `CANCEL_FORCED` | Parent → Subagent | Hard cancellation |

## AgentMessage Structure

Messages passed through the bus:

```python
from subagents_pydantic_ai import AgentMessage, MessageType

message = AgentMessage(
    type=MessageType.QUESTION,
    sender="subagent-123",
    receiver="parent-456",
    payload={"question": "Which database should I use?"},
    task_id="task-789",
)
```

## Custom Message Bus

Implement `MessageBusProtocol` for custom buses:

```python
from subagents_pydantic_ai import MessageBusProtocol, AgentMessage

class RedisMessageBus:
    """Redis-based message bus for distributed systems."""

    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)

    async def send(self, message: AgentMessage) -> None:
        """Send a message to the bus."""
        channel = f"agent:{message.receiver}"
        await self.redis.publish(channel, message.json())

    async def receive(
        self,
        agent_id: str,
        timeout: float | None = None,
    ) -> AgentMessage | None:
        """Receive a message for an agent."""
        channel = f"agent:{agent_id}"
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)

        try:
            message = await asyncio.wait_for(
                pubsub.get_message(ignore_subscribe_messages=True),
                timeout=timeout,
            )
            if message:
                return AgentMessage.parse_raw(message["data"])
            return None
        except asyncio.TimeoutError:
            return None
        finally:
            await pubsub.unsubscribe(channel)

    async def subscribe(self, agent_id: str) -> None:
        """Subscribe to messages for an agent."""
        pass  # Handled in receive

    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe from messages for an agent."""
        pass  # Handled in receive
```

## TaskManager

The `TaskManager` coordinates tasks and message handling:

```python
from subagents_pydantic_ai import TaskManager, InMemoryMessageBus

bus = InMemoryMessageBus()
manager = TaskManager(message_bus=bus)

# Create a task
handle = await manager.create_task(
    subagent_name="researcher",
    description="Research Python async",
)

# Check task status
status = await manager.get_task_status(handle.task_id)

# Answer a question
await manager.answer_question(handle.task_id, "Use asyncio")

# Cancel a task
await manager.cancel_task(handle.task_id, hard=False)
```

## Use Cases for Custom Buses

### Distributed Systems

For multi-process or multi-machine deployments:

```python
# Process 1: Parent agent
bus = RedisMessageBus("redis://localhost")
toolset = create_subagent_toolset(
    subagents=subagents,
    message_bus=bus,
)

# Process 2: Subagent workers
bus = RedisMessageBus("redis://localhost")
worker = SubagentWorker(bus)
await worker.run()
```

### Persistent Queues

For reliable delivery with persistence:

```python
class RabbitMQBus:
    """RabbitMQ-based bus with message persistence."""
    ...
```

### Monitoring

For observability and debugging:

```python
class MonitoredBus:
    """Wrapper that logs all messages."""

    def __init__(self, inner_bus: MessageBusProtocol):
        self.inner = inner_bus

    async def send(self, message: AgentMessage) -> None:
        logger.info(f"Sending: {message.type} from {message.sender}")
        await self.inner.send(message)
        metrics.increment("messages_sent", tags={"type": message.type})
```

## Best Practices

### 1. Use In-Memory for Simple Cases

Don't over-engineer. In-memory is fine for most applications:

```python
# Simple and effective
toolset = create_subagent_toolset(subagents=subagents)
```

### 2. Consider Reliability

For production distributed systems, use reliable message queues:

- Redis with persistence
- RabbitMQ
- Amazon SQS

### 3. Handle Failures

Implement retry logic and dead letter queues:

```python
class ReliableBus:
    async def send(self, message: AgentMessage) -> None:
        for attempt in range(3):
            try:
                await self._send(message)
                return
            except Exception:
                if attempt == 2:
                    await self._send_to_dlq(message)
                    raise
                await asyncio.sleep(2 ** attempt)
```

## Next Steps

- [Examples](../examples/index.md) - Working examples
- [API Reference](../api/index.md) - Complete API documentation
