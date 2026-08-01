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

DEFAULT_GENERAL_PURPOSE_DESCRIPTION = """A general-purpose agent for a wide variety of tasks.
Use this when no specialized subagent matches the task requirements.
Capable of research, analysis, writing, and problem-solving."""

DELEGATE_TOOL_DESCRIPTION = """\
Create an ephemeral specialist and delegate a task to it in a single call.

Use this instead of `create_agent` + `task` when you need a one-off specialist \
for a single job. The specialist is not registered and cannot be reused by name.

## When to use
- One-off tasks that need custom instructions or capabilities
- Ad-hoc specialists you will not delegate to again
- Quick delegation without polluting the agent registry

## When NOT to use
- Reusable specialists you will delegate to multiple times — use `task` with a \
configured subagent or create a persistent agent first
- Trivial tasks you can do faster yourself

## Parameters
- **description**: The task prompt for the specialist to execute
- **instructions**: The specialist's system prompt / role definition
- **name**: Label for logs and async task handles (letters, numbers, hyphens). \
Naming the specialist does not register it — it still cannot be reused via `task`
- **model**: Optional model override
- **capabilities**: Optional capability names to attach to the specialist
- **mode**: `"sync"` (default), `"async"`, or `"auto"`

Returns:
- In sync mode: The specialist's response as a string.
- In async mode: A task handle with task_id for status checking.
"""

TASK_TOOL_DESCRIPTION = """\
Delegate a task to a specialized subagent. The subagent runs independently \
with its own context and tools, and returns a result when done.

## When to use
- Complex multi-step tasks that can run independently from your main work
- Research or exploration tasks (e.g., "find all usages of function X", \
"understand how module Y works") — delegate so you can continue other work
- Multiple independent subtasks that can run in parallel — launch several \
subagents simultaneously for maximum efficiency
- Tasks that require deep focus on a single area while you handle the big picture

## When NOT to use
- Trivial tasks you can do faster yourself (single file read, simple grep)
- Tasks that require your full conversation context — subagents don't share \
your message history
- Tasks that need back-and-forth with the user — subagents work autonomously

## Usage notes
- **Be specific**: Subagents don't share your context. Include all necessary \
details in the description: file paths, function names, expected behavior, \
constraints. The more specific, the better the result.
- **Launch in parallel**: When you have multiple independent tasks, call \
`task()` multiple times in a single response. They run concurrently.
- **Synthesize results**: When subagents return, combine and analyze their \
results before presenting to the user. Don't just relay raw output.
- **Choose the right subagent**: Match the subagent_type to the task. \
Use "general-purpose" when no specialized subagent fits.
- **Continue intentionally**: When a result includes `Chat Trace ID: <id>`, pass \
that value as `chat_trace_id` only when you want the same subagent to resume \
that conversation. Omit `chat_trace_id` to start a new conversation. A trace \
can only be continued after its current task finishes, and only with the same \
subagent; continuing a busy or unknown trace returns an error.

## Execution modes
- **"sync"** (default): Blocks until the subagent completes. Use for quick \
tasks or when you need the result immediately.
- **"async"**: Returns a task handle immediately. Use for long-running tasks \
where you can continue other work. Check results with `check_task()` or \
wait with `wait_tasks()`.
- **"auto"**: Automatically picks sync or async based on task complexity.

Returns:
- In sync mode: The subagent's response as a string.
- In async mode: A task handle with task_id for status checking.
"""

CHECK_TASK_DESCRIPTION = """\
Check the status of a background (async) task and get its result if completed.

Use this after launching async tasks to see if they're done. Returns the \
task's current status, plus its result when completed, its error when failed, \
its pending question when waiting for an answer, and why it stopped when \
cancelled. The result is always returned in full, so call this when a \
`wait_tasks` listing showed a truncated one."""

ANSWER_SUBAGENT_DESCRIPTION = """\
Answer a question from a background subagent that is waiting for clarification.

When a task has status WAITING_FOR_ANSWER, the subagent needs information \
from you before it can continue. Provide a clear, specific answer."""

LIST_ACTIVE_TASKS_DESCRIPTION = """\
List all currently active background tasks with their status.

Use this to see what async tasks are running and their current state."""

WAIT_TASKS_DESCRIPTION = """\
Wait for one or more background tasks to finish before continuing.

A task is "finished" when it is completed, failed, or cancelled.

## Modes

- **mode="all"** (default): block until every task in `task_ids` is \
finished, or the timeout is reached. Use when you genuinely need every \
result together before the next step (e.g. final synthesis across all \
subagents).
- **mode="any"**: return as soon as ONE task finishes. Use when the \
subagents are independent and you can start acting on each finisher \
immediately — this avoids stalling on the slowest task. After reacting to \
the finisher, call `wait_tasks` again on the remaining ids (or use \
`check_task`) to handle the rest.

## When to prefer `mode="any"`

When you've dispatched several async tasks in parallel and any individual \
result is independently useful (e.g. routing decisions, progressive \
synthesis, fan-out research). Reactive orchestration is almost always \
faster than waiting on the slowest agent.

## Output

The result lists every requested task with its current state and a header \
showing `mode`, `<finished>/<total> finished`, and how many are still \
running. Unfinished tasks remain in the background — you can keep working \
or wait on them again later.

A long result may be cut here to save context, in which case it ends with an \
explicit truncation marker. That marker is a display limit on this listing, \
never an incomplete subagent answer: the full text is stored and \
`check_task` returns it. Never re-delegate a task to "finish" a result that \
carries the marker."""

SEND_MESSAGE_TO_SUBAGENT_DESCRIPTION = """\
Send a steering message to a running background (async) subagent without \
cancelling it.

Use this to redirect or refine a long-running task mid-flight when you learn \
something new — e.g. "narrow the search to packages/sparta/, it isn't in \
core/" or "stop after the first 5 matches". The subagent receives your message \
as an extra user instruction on its next step and adapts, keeping all partial \
progress (unlike cancel-and-respawn).

This is unprompted parent -> child steering — distinct from `answer_subagent`, \
which only replies to a question the subagent already asked. It applies to \
async tasks only; the target must still be running."""

SOFT_CANCEL_TASK_DESCRIPTION = """\
Request cooperative cancellation of a background task. The subagent will be \
notified and can clean up before stopping. Use this for graceful cancellation."""

HARD_CANCEL_TASK_DESCRIPTION = """\
Immediately cancel a background task. The task will be forcefully stopped. \
Use only when soft cancellation doesn't work or immediate stopping is required."""


def get_subagent_system_prompt(
    configs: list[SubAgentConfig],
    include_dual_mode: bool = False,
) -> str:
    """Generate the system prompt section describing available subagents.

    Args:
        configs: Subagent configurations to list.
        include_dual_mode: Append `DUAL_MODE_SYSTEM_PROMPT`, explaining sync
            versus background execution. Off by default because
            `TASK_TOOL_DESCRIPTION` already covers execution modes where the
            model needs them, and repeating it in the system prompt is wasted
            context. The parameter used to be accepted and ignored.

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
    lines = [
        "## Available Subagents",
        "",
        "Use the `task` tool to delegate work to these subagents:",
        "",
    ]

    for config in configs:
        line = f"- **{config['name']}**: {config['description']}"
        if config.get("can_ask_questions") is False:
            line += " *(cannot ask clarifying questions)*"
        lines.append(line)

    if include_dual_mode:
        lines.extend(["", DUAL_MODE_SYSTEM_PROMPT])

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
