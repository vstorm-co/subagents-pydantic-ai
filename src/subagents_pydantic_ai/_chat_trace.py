"""Storage for subagent conversations that a `chat_trace_id` can resume.

A chat trace is one subagent's message history, kept so a later delegation can
continue that conversation instead of starting cold. Traces are held in memory and
bounded by an LRU, because a long-lived orchestrator would otherwise accumulate
every conversation it ever started.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

ChatTraceKey = tuple[str, str]
"""`(subagent_name, chat_trace_id)`. A trace belongs to the subagent that created
it: continuing it with a different subagent would replay one agent's history into
another."""


class ChatTraceStore:
    """LRU-bounded message histories, plus the set of traces currently running.

    A trace must finish before it can be continued. Two concurrent tasks on one
    trace would both save at the end, and the slower save would silently discard
    the faster one's history.

    A trace also belongs to the run that started it. One toolset instance serves
    every run of its agent, and a trace holds a whole subagent conversation, so an
    unscoped lookup let one run replay another run's history into its own subagent
    by passing an id it had seen. Ownership mirrors `_handle_for`: a trace claimed
    without a `run_id` stays continuable by everyone.
    """

    def __init__(self, max_traces: int) -> None:
        self.history: OrderedDict[ChatTraceKey, list[Any]] = OrderedDict()
        self._max_traces = max_traces
        self._active: set[ChatTraceKey] = set()
        self._owners: dict[ChatTraceKey, str] = {}

    def is_active(self, key: ChatTraceKey) -> bool:
        """Whether a task is currently running on this trace."""
        return key in self._active

    def owned_by(self, key: ChatTraceKey, run_id: str | None) -> bool:
        """Whether `run_id` may read this trace. Unclaimed traces are open."""
        owner = self._owners.get(key)
        return owner is None or owner == run_id

    def mark_active(self, key: ChatTraceKey, run_id: str | None = None) -> None:
        """Claim the trace for a task that is about to start.

        Ownership is recorded here rather than on `save`, so a trace whose first run
        is still in flight is already protected. Taking `run_id` on the same call
        that marks the trace busy is what stops a new caller forgetting one of them.
        """
        self._active.add(key)
        if run_id is not None:
            self._owners.setdefault(key, run_id)

    def release(self, key: ChatTraceKey) -> None:
        """Release the trace once its task has finished.

        A trace that never saved a history has nothing left to protect -- its first
        run failed -- so the ownership record goes with it rather than pinning an id
        no one can use.
        """
        self._active.discard(key)
        if key not in self.history:
            self._owners.pop(key, None)

    def __contains__(self, key: ChatTraceKey) -> bool:
        return key in self.history

    def history_for(self, key: ChatTraceKey) -> list[Any] | None:
        """The stored history for a trace, refreshing its LRU recency.

        Reading counts as use, so a trace the orchestrator keeps continuing does
        not get evicted underneath it.
        """
        messages = self.history.get(key)
        if messages is not None:
            self.history.move_to_end(key)
        return messages

    def save(self, key: ChatTraceKey, messages: list[Any]) -> None:
        """Store a finished run's history, evicting the least recently used traces."""
        self.history[key] = messages
        self.history.move_to_end(key)
        while len(self.history) > self._max_traces:
            evicted, _ = self.history.popitem(last=False)
            # An evicted trace is unreachable, so its owner record would only leak.
            # A trace evicted mid-run keeps its claim until `release` drops it.
            if evicted not in self._active:
                self._owners.pop(evicted, None)
