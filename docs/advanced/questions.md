# Parent-Child Questions

Subagents can ask the parent agent for clarification during task execution. This enables interactive workflows where subagents don't have to guess when information is ambiguous.

## Enabling Questions

Enable the `ask_parent` tool for a subagent:

```python
SubAgentConfig(
    name="analyst",
    description="Analyzes data with clarifying questions",
    instructions="""You analyze data thoroughly.

When data is ambiguous or you need clarification:
- Use the ask_parent tool to ask the parent
- Wait for the answer before proceeding
- Don't guess or assume
""",
    can_ask_questions=True,
    max_questions=3,  # Limit questions per task
)
```

## How It Works

1. **Subagent encounters ambiguity**:
   ```
   Subagent: "The data mentions 'revenue' but doesn't specify
   the time period. I need to ask for clarification."
   ```

2. **Subagent asks parent**:
   ```python
   # Subagent calls:
   ask_parent("Should I analyze Q1, Q2, or full year revenue?")
   ```

3. **Task enters WAITING_FOR_ANSWER state**

4. **Parent checks task** (if async) or sees question (if sync):
   ```python
   # Parent calls:
   check_task(task_id="abc123")
   # Returns: "Task waiting for answer: Should I analyze Q1, Q2, or full year revenue?"
   ```

5. **Parent answers**:
   ```python
   # Parent calls:
   answer_subagent(task_id="abc123", answer="Analyze full year revenue")
   ```

6. **Subagent continues** with the answer

## Question Limits

Prevent infinite question loops with `max_questions`:

```python
SubAgentConfig(
    name="analyst",
    ...
    can_ask_questions=True,
    max_questions=3,  # After 3 questions, must complete without asking more
)
```

If the limit is reached, the subagent should proceed with best judgment or return partial results.

## Sync vs Async Questions

### Sync Mode

In sync mode, questions block the parent immediately:

```python
# Parent calls:
task(description="Analyze the report", subagent_type="analyst", mode="sync")

# If subagent asks a question, the task pauses and parent sees:
# "Subagent question: What time period should I focus on?"

# Parent must answer to continue:
answer_subagent(task_id="...", answer="Focus on Q4 2024")

# Task continues and eventually returns result
```

### Async Mode

In async mode, questions put the task in WAITING_FOR_ANSWER state:

```python
# Parent calls:
task(description="Analyze the report", subagent_type="analyst", mode="async")
# Returns: "Task started with ID: abc123"

# Parent does other work...

# Later, parent checks:
check_task(task_id="abc123")
# Returns: "Task waiting for answer: What time period should I focus on?"

# Parent answers:
answer_subagent(task_id="abc123", answer="Focus on Q4 2024")

# Task continues in background
check_task(task_id="abc123")
# Eventually returns: "Task complete: [analysis results]"
```

## Writing Good Questions

Guide your subagents to ask effective questions:

```python
SubAgentConfig(
    name="analyst",
    instructions="""When asking questions:

1. Be specific about what you need
   ❌ "What should I do?"
   ✅ "Should I include outliers in the statistical analysis?"

2. Provide context
   ❌ "Which one?"
   ✅ "The data has two revenue columns: 'revenue_gross' and 'revenue_net'. Which should I use for the margin calculation?"

3. Offer options when possible
   ❌ "What format?"
   ✅ "Should I format the output as: (a) JSON, (b) Markdown table, or (c) plain text?"
""",
    can_ask_questions=True,
)
```

## Example: Interactive Analysis

```python
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

subagents = [
    SubAgentConfig(
        name="data-analyst",
        description="Analyzes data with clarifying questions",
        instructions="""You are a data analyst.

When analyzing data:
1. First understand what metrics are needed
2. Ask for clarification if requirements are ambiguous
3. Provide detailed analysis with visualizations

Always ask before making assumptions about:
- Time periods
- Metrics to include
- Comparison baselines
- Output format
""",
        can_ask_questions=True,
        max_questions=5,
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
```

## The `answer_subagent` Tool

The `answer_subagent` tool is how a parent agent responds to questions from subagents. It is part of the toolset created by [`create_subagent_toolset()`][subagents_pydantic_ai.toolset.create_subagent_toolset].

### How It Works

1. A subagent calls `ask_parent(question)` during task execution
2. For async tasks, the task status changes to `WAITING_FOR_ANSWER` and `handle.pending_question` is set to the question text
3. The parent discovers the question by calling `check_task(task_id)` (async) or sees it inline (sync)
4. The parent calls `answer_subagent(task_id, answer)` to provide the response
5. The tool creates an `ANSWER` message on the message bus and resets the task status to `RUNNING`
6. The subagent receives the answer and continues execution

### When the Parent Should Use It

The parent uses `answer_subagent` whenever `check_task` reports a `WAITING_FOR_ANSWER` status. In a typical workflow:

```python
# Parent checks on a background task
check_task(task_id="abc123")
# Response: "Task: abc123
#            Subagent: data-analyst
#            Status: waiting_for_answer
#            Question: Should I use gross or net revenue for the margin calculation?"

# Parent provides the answer
answer_subagent(task_id="abc123", answer="Use net revenue")
# Response: "Answer sent to task 'abc123'"
```

### Validation and Error Handling

The tool validates before sending:

- **Task not found**: Returns an error if the `task_id` does not exist
- **Wrong status**: Returns an error if the task is not in `WAITING_FOR_ANSWER` state (e.g., already completed or still running without a question)
- **Subagent unavailable**: Returns an error if the subagent's message bus channel is no longer registered

```python
# Attempting to answer a task that isn't waiting
answer_subagent(task_id="abc123", answer="Use net revenue")
# Response: "Error: Task 'abc123' is not waiting for an answer (status: running)"
```

### Complete Async Q&A Example

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
        name="auditor",
        description="Audits financial data with clarifying questions",
        instructions="""You audit financial records.

Before proceeding, clarify:
- Which fiscal year to audit
- Whether to include subsidiaries
- Which accounting standard (GAAP or IFRS)

Use ask_parent() for each clarification.
""",
        can_ask_questions=True,
        max_questions=3,
        preferred_mode="async",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You manage financial auditing tasks.

When an auditor asks a question:
1. Check task status with check_task()
2. If WAITING_FOR_ANSWER, use answer_subagent() to respond
3. Continue checking until the audit is complete
""",
)


async def main():
    deps = Deps()

    # Turn 1: Start audit
    result1 = await agent.run("Audit the company finances", deps=deps)
    print(result1.output)
    # "Started audit task. Task ID: audit-abc"

    # Turn 2: Check status, answer question
    result2 = await agent.run(
        "Check the audit task and answer any questions",
        deps=deps,
        message_history=result1.all_messages(),
    )
    print(result2.output)
    # "Auditor asks: Which fiscal year? I answered: FY 2025"

    # Turn 3: Continue checking, answer more questions or get results
    result3 = await agent.run(
        "Check audit progress",
        deps=deps,
        message_history=result2.all_messages(),
    )
    print(result3.output)


asyncio.run(main())
```

## Best Practices

### 1. Set Appropriate Limits

- Simple tasks: `max_questions=1-2`
- Complex analysis: `max_questions=3-5`
- Exploratory work: `max_questions=5-10`

### 2. Guide Question Behavior

Include instructions about when to ask vs. when to proceed:

```python
instructions="""
Ask questions when:
- Requirements are ambiguous
- Multiple valid interpretations exist
- Significant assumptions would be needed

Don't ask questions for:
- Minor details you can reasonably infer
- Standard conventions in the domain
- Things explicitly stated in the task
"""
```

### 3. Handle Timeouts

For async tasks, consider what happens if questions aren't answered:

```python
instructions="""
If you've asked a question and been waiting a long time:
- Proceed with the most reasonable assumption
- Clearly document your assumption in the output
- Note that results may need revision
"""
```

### 4. Coach the Parent on Answering

Include guidance in the parent agent's system prompt so it knows how to handle questions:

```python
system_prompt="""
When you see a subagent question via check_task():
- Read the question carefully
- Provide a clear, actionable answer using answer_subagent()
- If you don't know, say so explicitly rather than guessing
- Check back after answering to see if more questions arise
"""
```

## Next Steps

- [Cancellation](cancellation.md) - Managing task lifecycle
- [Examples](../examples/questions.md) - Working examples
