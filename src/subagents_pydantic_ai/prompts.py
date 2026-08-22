"""System prompts for subagent communication.

This module contains the system prompts used to configure subagents
and explain the task delegation system to the parent agent.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ToolText:
    """What the model reads about one delegation tool.

    Held as parts rather than one string because they have different
    destinations: `summary` and `usage` become the tool's description, and
    `returns` is wrapped separately - which is how pydantic-ai renders a
    docstring that has a `Returns:` section, and so how every tool built from a
    docstring already reaches the model. Writing the return shape as a
    `Returns:` line inside the prose, which these descriptions used to do, puts
    two conventions in one tool list and tells the model nothing structural.
    """

    summary: str
    """One sentence: what the tool does."""

    usage: str = ""
    """When to use it, when not to, and what it will not do."""

    returns: str = ""
    """The shape of the answer, including what a failure looks like."""

    def render(self, extra: str = "") -> str:
        """The description handed to the model.

        Args:
            extra: Text appended to the prose before it is wrapped - the
                subagent list and the model list, which are known only once the
                toolset is built. It belongs inside the summary rather than
                after it, or the tags no longer bracket what they claim to.
        """
        parts = [self.summary]
        if self.usage:
            parts.append(self.usage)
        if extra:
            parts.append(extra)
        body = "\n\n".join(parts)
        if not self.returns:
            return body
        return (
            f"<summary>{body}</summary>\n"
            f"<returns>\n<description>{self.returns}</description>\n</returns>"
        )


TASK_TEXT = ToolText(
    summary=(
        "Delegate a task to a specialized subagent. The subagent runs "
        "independently with its own context and tools, and returns a result "
        "when done."
    ),
    usage="""\
## When to use
- Complex multi-step tasks that can run independently from your main work
- Research or exploration tasks (e.g., "find all usages of function X", \
"understand how module Y works") - delegate so you can continue other work
- Multiple independent subtasks that can run in parallel - launch several \
subagents simultaneously for maximum efficiency
- Tasks that require deep focus on a single area while you handle the big picture

## When NOT to use
- Trivial tasks you can do faster yourself (single file read, simple grep)
- Tasks that require your full conversation context - subagents don't share \
your message history
- Tasks that need back-and-forth with the user - subagents work autonomously

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
- **"auto"**: Automatically picks sync or async based on task complexity.""",
    returns=(
        "In sync mode the subagent's answer in full, sometimes with a "
        "`Chat Trace ID: <id>` line to continue that conversation with. In async "
        "mode a task handle carrying the `task_id` that `check_task` and "
        "`wait_tasks` take. A subagent type that does not exist comes back as "
        "`Error: Unknown subagent '<name>'` and the list of the ones that do."
    ),
)

DELEGATE_TEXT = ToolText(
    summary=("Create an ephemeral specialist and delegate a task to it in a single call."),
    usage="""\
Use this instead of `create_agent` + `task` when you need a one-off specialist \
for a single job. The specialist is not registered and cannot be reused by name.

## When to use
- One-off tasks that need custom instructions or capabilities
- Ad-hoc specialists you will not delegate to again
- Quick delegation without polluting the agent registry

## When NOT to use
- Reusable specialists you will delegate to multiple times - use `task` with a \
configured subagent or create a persistent agent first
- Trivial tasks you can do faster yourself""",
    returns=(
        "In sync mode the specialist's answer in full; in async mode a task "
        "handle carrying the `task_id` that `check_task` and `wait_tasks` take. "
        "A deployment with no model configured and no `model` argument answers "
        "with what it needs instead."
    ),
)

CREATE_AGENT_TEXT = ToolText(
    summary=(
        "Create a reusable specialized agent at runtime. The agent is stored in "
        "the registry and can be used repeatedly with the task tool."
    ),
    returns=(
        "A confirmation naming the agent, the model it will run on and the "
        "capabilities it was given. A name already in the registry comes back as "
        "`Error: Agent '<name>' already exists` - pick another, or delegate to "
        "the one that is there."
    ),
)

CHECK_TASK_TEXT = ToolText(
    summary="Check the status of a background (async) task and get its result if completed.",
    usage=(
        "Use this after launching async tasks to see if they're done. Call it "
        "whenever a `wait_tasks` listing showed a result cut short: the text is "
        "stored in full and comes back in full here."
    ),
    returns=(
        "The task's current status, and with it whichever applies: its result "
        "when completed, its error when failed, the question it is waiting on "
        "when it needs an answer, and why it stopped when cancelled. Never "
        "truncated. An id that is not running here answers "
        "`Error: Task '<id>' not found`."
    ),
)

ANSWER_SUBAGENT_TEXT = ToolText(
    summary="Answer a question from a background subagent that is waiting for clarification.",
    usage=(
        "When a task has status WAITING_FOR_ANSWER, the subagent needs "
        "information from you before it can continue. Provide a clear, specific "
        "answer."
    ),
    returns=(
        "A confirmation that the answer reached the task. A task that is not "
        "waiting for one says so with its current status instead, which means "
        "the question was already answered or the subagent moved on."
    ),
)

SEND_MESSAGE_TO_SUBAGENT_TEXT = ToolText(
    summary=(
        "Send a steering message to a running background (async) subagent without cancelling it."
    ),
    usage="""\
Use this to redirect or refine a long-running task mid-flight when you learn \
something new - e.g. "narrow the search to packages/sparta/, it isn't in \
core/" or "stop after the first 5 matches". The subagent receives your message \
as an extra user instruction on its next step and adapts, keeping all partial \
progress (unlike cancel-and-respawn).

This is unprompted parent -> child steering - distinct from `answer_subagent`, \
which only replies to a question the subagent already asked. It applies to \
async tasks only; the target must still be running.""",
    returns=(
        "A confirmation that the message was delivered, and that it applies on "
        "the subagent's next step rather than immediately. A task that has "
        "finished, or that is not running in the background, answers with its "
        "status instead."
    ),
)

LIST_ACTIVE_TASKS_TEXT = ToolText(
    summary="List all currently active background tasks with their status.",
    usage="Use this to see what async tasks are running and their current state.",
    returns=(
        "One line per task with its id and status, or `No active background "
        "tasks.` when none are running - which is an answer, not a failure."
    ),
)

WAIT_TASKS_TEXT = ToolText(
    summary="Wait for one or more background tasks to finish before continuing.",
    usage="""\
A task is "finished" when it is completed, failed, or cancelled.

## Modes

- **mode="all"** (default): block until every task in `task_ids` is \
finished, or the timeout is reached. Use when you genuinely need every \
result together before the next step (e.g. final synthesis across all \
subagents).
- **mode="any"**: return as soon as ONE task finishes. Use when the \
subagents are independent and you can start acting on each finisher \
immediately - this avoids stalling on the slowest task. After reacting to \
the finisher, call `wait_tasks` again on the remaining ids (or use \
`check_task`) to handle the rest.

## When to prefer `mode="any"`

When you've dispatched several async tasks in parallel and any individual \
result is independently useful (e.g. routing decisions, progressive \
synthesis, fan-out research). Reactive orchestration is almost always \
faster than waiting on the slowest agent.""",
    returns=(
        "Every requested task with its current state, under a header showing "
        "`mode`, `<finished>/<total> finished`, and how many are still running. "
        "Unfinished tasks stay in the background - keep working, or wait on them "
        "again later. A long result is cut here and ends with an explicit "
        "truncation marker; that marker is a display limit on this listing and "
        "never an incomplete subagent answer, so read the whole thing with "
        "`check_task` rather than re-delegating the task."
    ),
)

SOFT_CANCEL_TASK_TEXT = ToolText(
    summary="Request cooperative cancellation of a background task.",
    usage=(
        "The subagent is notified and can clean up before stopping. Use this "
        "for graceful cancellation."
    ),
    returns=(
        "A confirmation that cancellation was requested - the task stops at its "
        "next opportunity rather than at once. A task that had already finished "
        "says so, with the state it finished in."
    ),
)

HARD_CANCEL_TASK_TEXT = ToolText(
    summary="Immediately cancel a background task.",
    usage=(
        "The task is forcefully stopped. Use only when soft cancellation "
        "doesn't work or immediate stopping is required."
    ),
    returns=(
        "A confirmation that the task was cancelled. One that had already "
        "finished says so, with the state it finished in; there was nothing to "
        "stop."
    ),
)


TASK_TOOL_DESCRIPTION = TASK_TEXT.render()
DELEGATE_TOOL_DESCRIPTION = DELEGATE_TEXT.render()
CREATE_AGENT_DESCRIPTION = CREATE_AGENT_TEXT.render()
CHECK_TASK_DESCRIPTION = CHECK_TASK_TEXT.render()
ANSWER_SUBAGENT_DESCRIPTION = ANSWER_SUBAGENT_TEXT.render()
SEND_MESSAGE_TO_SUBAGENT_DESCRIPTION = SEND_MESSAGE_TO_SUBAGENT_TEXT.render()
LIST_ACTIVE_TASKS_DESCRIPTION = LIST_ACTIVE_TASKS_TEXT.render()
WAIT_TASKS_DESCRIPTION = WAIT_TASKS_TEXT.render()
SOFT_CANCEL_TASK_DESCRIPTION = SOFT_CANCEL_TASK_TEXT.render()
HARD_CANCEL_TASK_DESCRIPTION = HARD_CANCEL_TASK_TEXT.render()


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
