# Research Team Example

This example demonstrates a multi-agent research workflow with parallel execution.

## Overview

A research coordinator manages a team of specialists:

- **Researcher**: Gathers information on topics
- **Analyst**: Analyzes and synthesizes findings
- **Writer**: Produces the final report

## Complete Example

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)
    research_notes: list[str] = field(default_factory=list)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            research_notes=self.research_notes,  # Share notes
        )


# Define the research team
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches specific topics and gathers facts",
        instructions="""You are a thorough researcher.

When researching a topic:
1. Identify key aspects to investigate
2. Gather factual information
3. Note sources and credibility
4. Organize findings clearly

Output format:
## Topic: [topic name]

### Key Findings
- [finding 1]
- [finding 2]

### Sources
- [source 1]
- [source 2]
""",
        preferred_mode="async",  # Research can take time
        typical_complexity="complex",
    ),
    SubAgentConfig(
        name="analyst",
        description="Analyzes research findings and identifies patterns",
        instructions="""You are a critical analyst.

When analyzing findings:
1. Identify patterns and trends
2. Note contradictions or gaps
3. Draw evidence-based conclusions
4. Highlight areas needing more research

Be objective and cite your reasoning.
""",
        can_ask_questions=True,  # May need clarification
        max_questions=2,
    ),
    SubAgentConfig(
        name="writer",
        description="Writes clear, engaging reports from research",
        instructions="""You are an expert technical writer.

When writing:
1. Structure content logically
2. Use clear, accessible language
3. Include relevant examples
4. Cite sources appropriately

Produce professional-quality output.
""",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You are a research coordinator managing a team.

Your team:
- researcher: Gathers information (use async for parallel research)
- analyst: Analyzes findings (may ask clarifying questions)
- writer: Produces final reports

Workflow for research projects:
1. Break the topic into research areas
2. Assign researchers to each area (parallel, async)
3. Monitor progress with check_task()
4. Once research is complete, have the analyst synthesize
5. Finally, have the writer produce the report

Use async mode for research to enable parallel work.
Use sync mode for analysis and writing for immediate results.
""",
)


async def main():
    deps = Deps()

    # Single comprehensive request
    result = await agent.run(
        """Create a comprehensive report on "The Future of AI in Healthcare"

        Research these aspects:
        1. Current AI applications in healthcare
        2. Emerging technologies and trends
        3. Challenges and ethical considerations

        Then analyze the findings and write a professional report.""",
        deps=deps,
    )

    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.output)


asyncio.run(main())
```

## Workflow Breakdown

### Phase 1: Parallel Research

The coordinator starts multiple research tasks:

```python
# Coordinator's actions:
task(description="Research current AI applications in healthcare",
     subagent_type="researcher", mode="async")
# Returns: task_id_1

task(description="Research emerging AI healthcare technologies",
     subagent_type="researcher", mode="async")
# Returns: task_id_2

task(description="Research AI healthcare challenges and ethics",
     subagent_type="researcher", mode="async")
# Returns: task_id_3
```

### Phase 2: Monitor Progress

The coordinator checks on research:

```python
# Check each task
check_task(task_id=task_id_1)
check_task(task_id=task_id_2)
check_task(task_id=task_id_3)
```

### Phase 3: Analysis

Once research is complete, analyze findings:

```python
# Sync mode for immediate analysis
task(description="Analyze these research findings: [findings]",
     subagent_type="analyst", mode="sync")
```

### Phase 4: Report Writing

Finally, produce the report:

```python
# Sync mode for final output
task(description="Write a report based on this analysis: [analysis]",
     subagent_type="writer", mode="sync")
```

## Handling Analyst Questions

If the analyst needs clarification:

```python
# Analyst asks: "Should I focus more on technical or business aspects?"
# Coordinator answers:
answer_subagent(task_id=analysis_task_id,
                answer="Focus on both equally, but emphasize practical applications")
```

## Extended Example: Iterative Refinement

```python
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You coordinate research with iterative refinement.

Process:
1. Initial research (parallel, async)
2. Analysis of findings
3. Identify gaps
4. Additional targeted research if needed
5. Final analysis and report

Iterate until the research is comprehensive.
""",
)

# Conversation with iterative refinement
result1 = await agent.run(
    "Research AI in healthcare. Start with broad research.",
    deps=deps,
)

result2 = await agent.run(
    "What gaps exist in the research? Conduct follow-up if needed.",
    deps=deps,
    message_history=result1.all_messages(),
)

result3 = await agent.run(
    "Produce the final report.",
    deps=deps,
    message_history=result2.all_messages(),
)
```

## Best Practices

### 1. Use Async for Independent Research

```python
# Parallel research saves time
task(description="Research topic A", mode="async")
task(description="Research topic B", mode="async")
task(description="Research topic C", mode="async")
# Check all later
```

### 2. Use Sync for Sequential Steps

```python
# Analysis needs all research first
research_complete = all(check_task(t) == "completed" for t in task_ids)
if research_complete:
    task(description="Analyze findings", mode="sync")
```

### 3. Clear Handoffs

Ensure each agent has what it needs:

```python
# Bad: Vague handoff
task(description="Analyze the stuff", ...)

# Good: Clear handoff
task(description="""Analyze these findings:
1. Current applications: [details]
2. Emerging tech: [details]
3. Challenges: [details]

Focus on identifying trends and gaps.""", ...)
```

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Advanced Features](../advanced/execution-modes.md) - Deep dive into features
