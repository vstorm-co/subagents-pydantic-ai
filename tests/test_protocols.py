"""Tests for protocols module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from subagents_pydantic_ai import InMemoryMessageBus
from subagents_pydantic_ai.protocols import MessageBusProtocol, SubAgentDepsProtocol


@dataclass
class ValidDeps:
    """A valid implementation of SubAgentDepsProtocol."""

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> ValidDeps:
        return ValidDeps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
        )


class TestSubAgentDepsProtocol:
    """Tests for SubAgentDepsProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """SubAgentDepsProtocol should be runtime checkable."""
        deps = ValidDeps()
        assert isinstance(deps, SubAgentDepsProtocol)

    def test_valid_implementation(self):
        """ValidDeps should satisfy the protocol."""
        deps = ValidDeps(subagents={"agent1": object()})
        assert hasattr(deps, "subagents")
        assert hasattr(deps, "clone_for_subagent")

    def test_clone_for_subagent_with_depth(self):
        """clone_for_subagent should respect max_depth."""
        deps = ValidDeps(subagents={"agent1": object()})

        # With depth 0, subagents should be empty
        cloned = deps.clone_for_subagent(0)
        assert cloned.subagents == {}

        # With depth > 0, subagents should be copied
        cloned = deps.clone_for_subagent(1)
        assert "agent1" in cloned.subagents

    def test_invalid_implementation(self):
        """Objects without required attributes should not satisfy protocol."""

        class InvalidDeps:
            pass

        deps = InvalidDeps()
        assert not isinstance(deps, SubAgentDepsProtocol)


class TestMessageBusProtocol:
    """Tests for MessageBusProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """MessageBusProtocol should be runtime checkable."""
        bus = InMemoryMessageBus()
        assert isinstance(bus, MessageBusProtocol)

    def test_inmemory_implements_protocol(self):
        """InMemoryMessageBus should implement all protocol methods."""
        bus = InMemoryMessageBus()
        assert hasattr(bus, "send")
        assert hasattr(bus, "ask")
        assert hasattr(bus, "answer")
        assert hasattr(bus, "register_agent")
        assert hasattr(bus, "unregister_agent")
        assert hasattr(bus, "get_messages")

    def test_invalid_implementation(self):
        """Objects without required methods should not satisfy protocol."""

        class InvalidBus:
            pass

        bus = InvalidBus()
        assert not isinstance(bus, MessageBusProtocol)
