# Questions Example

This example demonstrates parent-child Q&A communication.

## Basic Questions

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


subagents = [
    SubAgentConfig(
        name="data-analyst",
        description="Analyzes data with clarifying questions",
        instructions="""You are a data analyst.

When analyzing data, ask clarifying questions if:
- The time period is unclear
- Metrics are ambiguous
- Comparison baselines aren't specified

Use ask_parent() to ask questions.
Wait for the answer before proceeding.
""",
        can_ask_questions=True,
        max_questions=3,
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You manage data analysis tasks.

When the data-analyst asks a question:
1. You'll see the question in the task status
2. Use answer_subagent(task_id, answer) to respond
3. The analyst will continue with your answer
""",
)


async def main():
    deps = Deps()

    # Start analysis - analyst may ask questions
    result = await agent.run(
        "Analyze our sales data",
        deps=deps,
    )

    print(result.output)
    # If analyst asked "Which quarter should I analyze?",
    # parent answered "Q4 2024",
    # then analyst completed the analysis


asyncio.run(main())
```

## Async Questions

With async mode, questions put the task in WAITING_FOR_ANSWER state:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


subagents = [
    SubAgentConfig(
        name="planner",
        description="Creates project plans with clarification",
        instructions="""You create detailed project plans.

Before creating a plan, clarify:
- Project scope and constraints
- Key stakeholders
- Timeline expectations

Ask questions using ask_parent().
""",
        can_ask_questions=True,
        max_questions=5,
        preferred_mode="async",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You manage project planning.

Workflow:
1. Start planning task (async)
2. Check task status periodically
3. If status is WAITING_FOR_ANSWER, answer the question
4. Continue until planning is complete
""",
)


async def main():
    deps = Deps()

    # Conversation 1: Start the task
    result1 = await agent.run(
        "Create a plan for building a mobile app",
        deps=deps,
    )
    print("Response 1:", result1.output)
    # "Started planning task. Task ID: plan-123"

    # Conversation 2: Check and answer
    result2 = await agent.run(
        "Check on the planning task",
        deps=deps,
        message_history=result1.all_messages(),
    )
    print("Response 2:", result2.output)
    # "The planner asks: What's the target platform (iOS, Android, or both)?
    #  I'll answer: Both platforms."

    # Conversation 3: Get final result
    result3 = await agent.run(
        "Check if planning is done",
        deps=deps,
        message_history=result2.all_messages(),
    )
    print("Response 3:", result3.output)
    # "Planning complete! Here's the project plan..."


asyncio.run(main())
```

## Multiple Questions

Handle a sequence of questions:

```python
SubAgentConfig(
    name="requirements-analyst",
    description="Gathers requirements through questions",
    instructions="""You gather software requirements.

Process:
1. Ask about the problem being solved
2. Ask about target users
3. Ask about key features
4. Ask about constraints
5. Compile requirements document

Ask one question at a time.
Wait for each answer before asking the next.
""",
    can_ask_questions=True,
    max_questions=10,
)
```

## Question Guidelines

Help subagents ask good questions:

```python
SubAgentConfig(
    name="analyst",
    instructions="""
## Asking Questions

When you need clarification:

1. Be specific
   ❌ "What should I do?"
   ✅ "Should I include Q4 data in the analysis?"

2. Provide context
   ❌ "Which one?"
   ✅ "I found 3 CSV files. Which should I use: sales_2023.csv, sales_2024.csv, or sales_all.csv?"

3. Offer options when possible
   ❌ "What format?"
   ✅ "Should I output as: (a) JSON, (b) Markdown, or (c) CSV?"

4. Ask one thing at a time
   ❌ "What time period, which metrics, and what format?"
   ✅ "What time period should I analyze?"
""",
    can_ask_questions=True,
)
```

## Handling No Answer

What if the parent doesn't answer?

```python
SubAgentConfig(
    name="analyst",
    instructions="""
## If No Answer

If you've asked a question and received no answer:
1. Wait a reasonable time
2. Proceed with the most sensible default
3. Clearly document your assumption
4. Note that results may need revision

Example:
"Note: No answer received for time period question.
Defaulting to full year 2024. Results can be re-run
for a different period if needed."
""",
    can_ask_questions=True,
)
```

## Next Steps

- [Nesting](nesting.md) - Subagents with their own subagents
- [Research Team](research-team.md) - Multi-agent workflow
