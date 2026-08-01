"""Per-delegation state that `ask_parent` needs to reach the parent agent.

The state used to be injected as an untyped `dict` onto the application's deps
object (`deps._subagent_state = {...}`). That fails outright for a deps dataclass
declared `frozen=True` or `slots=True`, both of which the protocol allows and the
docs never warned against, and it left the payload stringly-typed
(`state.get("ask_callback")`).

A `ContextVar` carries it instead: `asyncio.create_task` copies the current
context, so a background delegation sees the state set for it, and each task gets
its own copy rather than sharing a mutable attribute.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from subagents_pydantic_ai.types import AskUserCallback

if TYPE_CHECKING:
    from subagents_pydantic_ai.message_bus import TaskManager


@dataclass(slots=True)
class QuestionBudget:
    """How many more times this delegation may call `ask_parent`.

    `max_questions` used to reach the subagent only as a sentence in its prompt,
    which a model is free to ignore -- and every ignored question costs the task up
    to `ask_timeout_seconds`. The counter makes the documented limit real. It is
    mutable and per-delegation, so it lives here rather than on the config.

    Attributes:
        limit: Maximum questions allowed, or `None` for unlimited.
        asked: Questions already spent.
    """

    limit: int | None = None
    asked: int = 0

    def consume(self) -> bool:
        """Spend one question, returning whether it was within the limit."""
        if self.limit is not None and self.asked >= self.limit:
            return False
        self.asked += 1
        return True


@dataclass(frozen=True, slots=True)
class SubAgentState:
    """How the subagent currently being run can reach its parent.

    Exactly one of the two channels is set. In sync mode the parent's run loop is
    blocked inside the delegation, so a question can only be answered by
    `ask_callback`. In async mode the parent is still running and answers through
    `answer_subagent`, which resolves a future held by `task_manager`.

    Attributes:
        ask_callback: Sync mode. Called with the question, returns the answer.
        task_manager: Async mode. Holds the task handle and the answer future.
        task_id: Async mode. Identifies this delegation in `task_manager`.
        ask_timeout_seconds: How long `ask_parent` waits for the parent's answer
            before giving up and telling the subagent to proceed on its own.
        questions: This delegation's `max_questions` budget, when one is set.
    """

    ask_timeout_seconds: float
    ask_callback: AskUserCallback | None = None
    task_manager: TaskManager | None = None
    task_id: str | None = None
    questions: QuestionBudget | None = None


_SUBAGENT_STATE: ContextVar[SubAgentState | None] = ContextVar(
    "subagents_pydantic_ai_state", default=None
)


@contextmanager
def bind_subagent_state(state: SubAgentState) -> Iterator[None]:
    """Make `state` the current delegation's state for the duration of the block."""
    token = _SUBAGENT_STATE.set(state)
    try:
        yield
    finally:
        _SUBAGENT_STATE.reset(token)


def current_subagent_state() -> SubAgentState | None:
    """The state bound for the delegation running in this context, if any."""
    return _SUBAGENT_STATE.get()
