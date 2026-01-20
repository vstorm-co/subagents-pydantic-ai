"""Pytest fixtures and configuration for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from subagents_pydantic_ai import (
    DynamicAgentRegistry,
    InMemoryMessageBus,
    SubAgentConfig,
    TaskManager,
)


@dataclass
class MockDeps:
    """Mock dependencies for testing."""

    subagents: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    clone_calls: list[int] = field(default_factory=list)

    def clone_for_subagent(self, max_depth: int = 0) -> "MockDeps":
        """Clone deps for subagent."""
        self.clone_calls.append(max_depth)
        return MockDeps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            data=self.data.copy(),
        )


@pytest.fixture
def mock_deps() -> MockDeps:
    """Create mock dependencies."""
    return MockDeps()


@pytest.fixture
def message_bus() -> InMemoryMessageBus:
    """Create an in-memory message bus."""
    return InMemoryMessageBus()


@pytest.fixture
def task_manager(message_bus: InMemoryMessageBus) -> TaskManager:
    """Create a task manager with message bus."""
    return TaskManager(message_bus=message_bus)


@pytest.fixture
def registry() -> DynamicAgentRegistry:
    """Create a dynamic agent registry."""
    return DynamicAgentRegistry(max_agents=10)


@pytest.fixture
def sample_config() -> SubAgentConfig:
    """Create a sample subagent configuration."""
    return SubAgentConfig(
        name="test-agent",
        description="A test agent for unit tests",
        instructions="You are a test agent. Return test results.",
        model="test-model",
        can_ask_questions=True,
        max_questions=3,
    )


@pytest.fixture
def researcher_config() -> SubAgentConfig:
    """Create a researcher subagent configuration."""
    return SubAgentConfig(
        name="researcher",
        description="Researches topics thoroughly",
        instructions="You are a research assistant. Investigate deeply.",
        can_ask_questions=True,
    )


@pytest.fixture
def simple_config() -> SubAgentConfig:
    """Create a simple subagent configuration."""
    return SubAgentConfig(
        name="simple-helper",
        description="Quick helper for simple tasks",
        instructions="You are a quick helper.",
        can_ask_questions=False,
    )
