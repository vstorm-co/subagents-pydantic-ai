# Core Concepts

This section covers the fundamental concepts you need to understand when working with Subagents for Pydantic AI.

## Overview

The library is built around three core concepts:

1. **Subagents** - Specialized agents that handle specific tasks
2. **Toolset** - The bridge that connects your parent agent to subagents
3. **Types** - Data structures for configuration and communication

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                     Parent Agent                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Subagent Toolset                   │    │
│  │  task() │ check_task() │ answer_subagent()      │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼               ▼               ▼               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ researcher │  │   writer   │  │   coder    │         │
│  │  (sync)    │  │  (async)   │  │  (auto)    │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│                                                         │
│              Message Bus (pluggable)                    │
└─────────────────────────────────────────────────────────┘
```

1. You define **SubAgentConfigs** with name, description, and instructions
2. The **toolset factory** creates a toolset with delegation tools
3. Your **parent agent** uses tools like `task()` to delegate work
4. **Subagents** execute the work and return results
5. The **message bus** handles communication (questions, answers, status)

## In This Section

- [Subagents](subagents.md) - Configuring specialized agents
- [Toolset](toolset.md) - The delegation toolset
- [Types](types.md) - Data structures and enums
