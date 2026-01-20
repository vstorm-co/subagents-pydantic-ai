"""System prompts for subagent communication.

This module contains the system prompts used to configure subagents
and explain the task delegation system to the parent agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subagents_pydantic_ai.types import SubAgentConfig

SUBAGENT_SYSTEM_PROMPT = """You are a specialized subagent working on a delegated task.

## Your Role
You have been spawned by a parent agent to handle a specific task. Focus entirely
on completing the assigned task to the best of your ability.

## Communication
- If you need clarification, use the `ask_parent` tool to ask the parent agent
- Keep questions specific and actionable
- Do not ask unnecessary questions - use your judgment when possible
- If you cannot complete a task, explain why clearly

## Task Completion
- Complete the task thoroughly before returning
- Provide clear, structured results
- If the task cannot be completed, explain what was attempted and why it failed
"""

DUAL_MODE_SYSTEM_PROMPT = """## Subagent Execution Modes

You can delegate tasks to subagents in two modes:

### Sync Mode (Default)
- Use for simple, quick tasks
- Use when you need the result immediately
- Use when the task requires back-and-forth communication
- The task runs and you wait for the result

### Async Mode (Background)
- Use for complex, long-running tasks
- Use when you can continue with other work while waiting
- Use for tasks that can run independently
- Returns a task handle immediately - check status later
"""

DEFAULT_GENERAL_PURPOSE_DESCRIPTION = """A general-purpose agent that can handle a wide variety of tasks.
Use this when no specialized subagent matches the task requirements.
Capable of research, analysis, writing, and problem-solving."""

TASK_TOOL_DESCRIPTION = """Delegate a task to a specialized subagent.

Choose the appropriate subagent_type based on the task requirements.
The task will be executed according to the specified mode:
- "sync": Wait for completion (blocking) - default
- "async": Run in background (returns task handle)

Returns:
- In sync mode: The subagent's response as a string
- In async mode: A task handle with task_id for status checking
"""


def get_subagent_system_prompt(
    configs: list[SubAgentConfig],
    include_dual_mode: bool = True,
) -> str:
    """Generate system prompt section describing available subagents.

    Args:
        configs: List of subagent configurations.
        include_dual_mode: Whether to include dual-mode execution explanation.

    Returns:
        Formatted system prompt section.

    Example:
        ```python
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="...",
            ),
        ]
        prompt = get_subagent_system_prompt(configs)
        ```
    """
    lines = ["## Available Subagents", ""]
    lines.append("You can delegate tasks to the following specialized subagents:")
    lines.append("")

    for config in configs:
        name = config["name"]
        description = config["description"]
        lines.append(f"### {name}")
        lines.append(description)

        # Add hint if agent cannot ask questions
        if config.get("can_ask_questions") is False:
            lines.append("*Cannot ask clarifying questions*")
        lines.append("")

    if include_dual_mode:
        lines.append(DUAL_MODE_SYSTEM_PROMPT)

    return "\n".join(lines)


def get_task_instructions_prompt(
    task_description: str,
    can_ask_questions: bool = True,
    max_questions: int | None = None,
) -> str:
    """Generate the task instructions for a subagent.

    Args:
        task_description: The task to perform.
        can_ask_questions: Whether the subagent can ask the parent questions.
        max_questions: Maximum number of questions allowed.

    Returns:
        Formatted task instructions.
    """
    lines = ["## Your Task", "", task_description, ""]

    if can_ask_questions:
        lines.append("## Asking Questions")
        lines.append("If you need clarification, use the `ask_parent` tool.")
        if max_questions is not None:
            lines.append(f"You may ask up to {max_questions} questions.")
        lines.append("Keep questions specific and essential.")
    else:
        lines.append("## Note")
        lines.append("Complete this task using your best judgment.")
        lines.append("You cannot ask the parent for clarification.")

    return "\n".join(lines)
