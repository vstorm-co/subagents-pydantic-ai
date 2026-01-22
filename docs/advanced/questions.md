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

## Next Steps

- [Cancellation](cancellation.md) - Managing task lifecycle
- [Examples](../examples/questions.md) - Working examples
