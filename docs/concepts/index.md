# Core Concepts

## Overview

The library provides two ways to add subagent delegation:

1. **SubAgentCapability** (recommended) — plug-and-play capability with auto-wired tools and instructions
2. **Toolset API** — lower-level control via `create_subagent_toolset()`

Both use the same underlying components: subagent configs, execution modes, and message bus.

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│                     Parent Agent                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │      SubAgentCapability (recommended)              │  │
│  │  tools + dynamic instructions — auto-configured    │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │              Subagent Toolset                      │  │
│  │  task() │ check_task() │ answer_subagent()         │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                │
│         ┌───────────────┼───────────────┐                │
│         ▼               ▼               ▼                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ researcher │  │   writer   │  │   coder    │          │
│  │  (sync)    │  │  (async)   │  │  (auto)    │          │
│  └────────────┘  └────────────┘  └────────────┘          │
│                                                          │
│              Message Bus (pluggable)                     │
└──────────────────────────────────────────────────────────┘
```

1. You define **SubAgentConfigs** with name, description, and instructions
2. **SubAgentCapability** (or the toolset factory) creates tools for delegation
3. Your **parent agent** uses tools like `task()` to delegate work
4. **Subagents** execute the work and return results
5. The **message bus** handles communication (questions, answers, status)

## In This Section

- [Capability](capability.md) - Recommended integration (plug-and-play)
- [Subagents](subagents.md) - Configuring specialized agents
- [Toolset](toolset.md) - Lower-level delegation toolset
- [Types](types.md) - Data structures and enums
