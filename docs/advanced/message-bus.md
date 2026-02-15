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

To build a custom message bus, implement the [`MessageBusProtocol`][subagents_pydantic_ai.protocols.MessageBusProtocol]. The protocol requires these methods:

| Method | Purpose |
|--------|---------|
| `send(message)` | Deliver a message to a specific agent |
| `ask(sender, receiver, question, task_id, timeout)` | Send a question and block until an answer arrives |
| `answer(original, answer)` | Reply to a previously received question |
| `register_agent(agent_id)` | Register an agent and return its message queue |
| `unregister_agent(agent_id)` | Remove an agent from the bus |
| `get_messages(agent_id, timeout)` | Retrieve pending messages for an agent |

### Step-by-Step: Building a Redis Message Bus

Below is a complete example of implementing a Redis-based message bus. This is useful when parent agents and subagent workers run in separate processes or machines.

```python
import asyncio
import json
from dataclasses import dataclass, field

from redis.asyncio import Redis

from subagents_pydantic_ai import (
    MessageBusProtocol,
    AgentMessage,
    MessageType,
)


@dataclass
class RedisMessageBus:
    """Redis-based message bus for distributed subagent systems.

    Uses Redis pub/sub for real-time message delivery and Redis lists
    as a fallback queue for messages sent before the receiver subscribes.
    """

    redis_url: str = "redis://localhost:6379"
    _redis: Redis | None = field(default=None, repr=False)
    _queues: dict[str, asyncio.Queue[AgentMessage]] = field(default_factory=dict)
    _pending_questions: dict[str, asyncio.Future[AgentMessage]] = field(
        default_factory=dict
    )
    _listeners: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    async def _get_redis(self) -> Redis:
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _channel(self, agent_id: str) -> str:
        return f"subagent:messages:{agent_id}"

    def _serialize(self, message: AgentMessage) -> str:
        return json.dumps({
            "type": message.type.value,
            "sender": message.sender,
            "receiver": message.receiver,
            "payload": message.payload,
            "task_id": message.task_id,
            "id": message.id,
            "correlation_id": message.correlation_id,
        })

    def _deserialize(self, data: str) -> AgentMessage:
        d = json.loads(data)
        return AgentMessage(
            type=MessageType(d["type"]),
            sender=d["sender"],
            receiver=d["receiver"],
            payload=d["payload"],
            task_id=d["task_id"],
            id=d.get("id", ""),
            correlation_id=d.get("correlation_id"),
        )

    # ---- MessageBusProtocol methods ----

    async def send(self, message: AgentMessage) -> None:
        """Publish a message to the receiver's Redis channel."""
        if message.receiver not in self._queues:
            raise KeyError(f"Agent '{message.receiver}' is not registered")

        redis = await self._get_redis()
        await redis.publish(
            self._channel(message.receiver),
            self._serialize(message),
        )

    async def ask(
        self,
        sender: str,
        receiver: str,
        question: object,
        task_id: str,
        timeout: float = 30.0,
    ) -> AgentMessage:
        """Send a question and wait for the correlated answer."""
        import uuid

        correlation_id = str(uuid.uuid4())

        loop = asyncio.get_event_loop()
        future: asyncio.Future[AgentMessage] = loop.create_future()
        self._pending_questions[correlation_id] = future

        try:
            msg = AgentMessage(
                type=MessageType.QUESTION,
                sender=sender,
                receiver=receiver,
                payload=question,
                task_id=task_id,
                correlation_id=correlation_id,
            )
            await self.send(msg)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_questions.pop(correlation_id, None)

    async def answer(self, original: AgentMessage, answer: object) -> None:
        """Answer a previously received question."""
        response = AgentMessage(
            type=MessageType.ANSWER,
            sender=original.receiver,
            receiver=original.sender,
            payload=answer,
            task_id=original.task_id,
            correlation_id=original.correlation_id,
        )

        if (
            original.correlation_id
            and original.correlation_id in self._pending_questions
        ):
            future = self._pending_questions[original.correlation_id]
            if not future.done():
                future.set_result(response)
        else:
            await self.send(response)

    def register_agent(self, agent_id: str) -> asyncio.Queue[AgentMessage]:
        """Register an agent and start listening on its Redis channel."""
        if agent_id in self._queues:
            raise ValueError(f"Agent '{agent_id}' is already registered")

        queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._queues[agent_id] = queue

        # Start a background listener for this agent's channel
        self._listeners[agent_id] = asyncio.create_task(
            self._listen(agent_id, queue)
        )
        return queue

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and stop its listener."""
        self._queues.pop(agent_id, None)
        listener = self._listeners.pop(agent_id, None)
        if listener and not listener.done():
            listener.cancel()

    async def get_messages(
        self,
        agent_id: str,
        timeout: float = 0.0,
    ) -> list[AgentMessage]:
        """Get pending messages from the local queue."""
        if agent_id not in self._queues:
            raise KeyError(f"Agent '{agent_id}' is not registered")

        queue = self._queues[agent_id]
        messages: list[AgentMessage] = []

        if timeout > 0 and queue.empty():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=timeout)
                messages.append(msg)
            except asyncio.TimeoutError:
                return messages

        while not queue.empty():
            messages.append(queue.get_nowait())

        return messages

    # ---- Internal helpers ----

    async def _listen(
        self,
        agent_id: str,
        queue: asyncio.Queue[AgentMessage],
    ) -> None:
        """Subscribe to a Redis channel and forward messages to the queue."""
        redis = await self._get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(self._channel(agent_id))

        try:
            async for raw_message in pubsub.listen():
                if raw_message["type"] != "message":
                    continue
                msg = self._deserialize(raw_message["data"])

                # Resolve pending question futures
                if (
                    msg.type == MessageType.ANSWER
                    and msg.correlation_id
                    and msg.correlation_id in self._pending_questions
                ):
                    future = self._pending_questions[msg.correlation_id]
                    if not future.done():
                        future.set_result(msg)
                    continue

                await queue.put(msg)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(self._channel(agent_id))

    async def close(self) -> None:
        """Clean up all listeners and the Redis connection."""
        for agent_id in list(self._listeners):
            self.unregister_agent(agent_id)
        if self._redis:
            await self._redis.close()
            self._redis = None
```

### Using the Custom Bus

Pass your custom bus to [`create_subagent_toolset()`][subagents_pydantic_ai.toolset.create_subagent_toolset] or inject it manually via [`TaskManager`][subagents_pydantic_ai.message_bus.TaskManager]:

```python
from subagents_pydantic_ai import TaskManager

# Create your custom bus
bus = RedisMessageBus(redis_url="redis://my-redis:6379")

# Use it with the task manager
manager = TaskManager(message_bus=bus)
```

!!! tip "Testing custom buses"
    Start by verifying that your bus passes the same patterns as `InMemoryMessageBus`: register two agents, send a message from one to the other, call `ask`/`answer`, and confirm that `get_messages` returns the expected results.

## TaskManager

The [`TaskManager`][subagents_pydantic_ai.message_bus.TaskManager] coordinates tasks and message handling:

```python
from subagents_pydantic_ai import TaskManager, InMemoryMessageBus

bus = InMemoryMessageBus()
manager = TaskManager(message_bus=bus)
```

The task manager provides methods for the full task lifecycle:

| Method | Description |
|--------|-------------|
| `create_task(task_id, coro, handle)` | Create and track a background asyncio task |
| `get_handle(task_id)` | Get the `TaskHandle` for a task |
| `get_cancel_event(task_id)` | Get the cancellation event for cooperative cancellation |
| `soft_cancel(task_id)` | Request cooperative cancellation |
| `hard_cancel(task_id)` | Force immediate cancellation |
| `cleanup_task(task_id)` | Remove task resources (handle is kept for status queries) |
| `list_active_tasks()` | Get IDs of all non-completed tasks |

## Use Cases for Custom Buses

### Distributed Systems

For multi-process or multi-machine deployments:

```python
# Process 1: Parent agent
bus = RedisMessageBus("redis://localhost")
manager = TaskManager(message_bus=bus)
```

### Persistent Queues

For reliable delivery with persistence:

```python
class RabbitMQBus:
    """RabbitMQ-based bus with message persistence."""
    ...
```

### Monitoring

For observability and debugging, wrap an existing bus:

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

Alternatively, use the `add_handler` method on `InMemoryMessageBus` to add lightweight monitoring without wrapping:

```python
bus = InMemoryMessageBus()

async def log_messages(message: AgentMessage) -> None:
    logger.info(f"[{message.type}] {message.sender} → {message.receiver}")

bus.add_handler(log_messages)
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

### 4. Match the Protocol Exactly

When implementing a custom bus, make sure every method from [`MessageBusProtocol`][subagents_pydantic_ai.protocols.MessageBusProtocol] is present with matching signatures. The protocol is `@runtime_checkable`, so you can verify at startup:

```python
assert isinstance(my_bus, MessageBusProtocol)
```

## Next Steps

- [Examples](../examples/index.md) - Working examples
- [API Reference](../api/index.md) - Complete API documentation
