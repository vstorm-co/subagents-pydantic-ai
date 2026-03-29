"""Declarative subagent specification for YAML/JSON configuration.

This module provides a Pydantic model for defining subagent configurations
in a serializable format, suitable for loading from YAML or JSON files.

Example YAML:
    ```yaml
    subagents:
      - name: researcher
        description: Research assistant
        instructions: You research topics thoroughly.
        model: openai:gpt-4.1-mini
    ```

Example usage:
    ```python
    import yaml
    from subagents_pydantic_ai.spec import SubAgentSpec

    with open("agents.yaml") as f:
        data = yaml.safe_load(f)

    specs = [SubAgentSpec(**s) for s in data["subagents"]]
    configs = [s.to_config() for s in specs]
    ```
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from subagents_pydantic_ai.types import SubAgentConfig


class SubAgentSpec(BaseModel):
    """Declarative subagent configuration for YAML/JSON specs.

    A Pydantic model that mirrors ``SubAgentConfig`` (a TypedDict) but
    provides validation, defaults, and serialization support. This makes
    it suitable for loading subagent definitions from YAML or JSON files.

    Attributes:
        name: Unique identifier for the subagent.
        description: Brief description shown to the parent agent.
        instructions: System prompt for the subagent.
        model: LLM model identifier (e.g. ``openai:gpt-4.1``).
            If None, the parent agent's default model is used.
        can_ask_questions: Whether the subagent can ask the parent questions.
        max_questions: Maximum number of questions per task.
        preferred_mode: Default execution mode preference.
        typical_complexity: Typical task complexity for this subagent.
        typically_needs_context: Whether this subagent typically needs user context.
        context_files: List of context file paths to inject into system prompt.
        extra: Generic extensibility dict for consumer libraries.
    """

    name: str
    description: str = ""
    instructions: str = ""
    model: str | None = None
    can_ask_questions: bool | None = None
    max_questions: int | None = None
    preferred_mode: Literal["sync", "async", "auto"] | None = None
    typical_complexity: Literal["simple", "moderate", "complex"] | None = None
    typically_needs_context: bool | None = None
    context_files: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_config(self) -> SubAgentConfig:
        """Convert to a SubAgentConfig TypedDict.

        Only includes fields that have been explicitly set (non-None values),
        except for ``extra`` which is included when non-empty.

        Returns:
            A SubAgentConfig dict suitable for ``create_subagent_toolset()``.
        """
        config = SubAgentConfig(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
        )

        if self.model is not None:
            config["model"] = self.model
        if self.can_ask_questions is not None:
            config["can_ask_questions"] = self.can_ask_questions
        if self.max_questions is not None:
            config["max_questions"] = self.max_questions
        if self.preferred_mode is not None:
            config["preferred_mode"] = self.preferred_mode
        if self.typical_complexity is not None:
            config["typical_complexity"] = self.typical_complexity
        if self.typically_needs_context is not None:
            config["typically_needs_context"] = self.typically_needs_context
        if self.context_files is not None:
            config["context_files"] = self.context_files
        if self.extra:
            config["extra"] = self.extra

        return config

    @classmethod
    def from_config(cls, config: SubAgentConfig) -> SubAgentSpec:
        """Create a SubAgentSpec from a SubAgentConfig dict.

        Args:
            config: A SubAgentConfig TypedDict.

        Returns:
            A new SubAgentSpec instance.
        """
        data: dict[str, Any] = {
            "name": config["name"],
            "description": config.get("description", ""),
            "instructions": config.get("instructions", ""),
        }

        # Map optional fields, converting Model objects to str for model field
        model_val = config.get("model")
        if model_val is not None:
            data["model"] = str(model_val)

        for field_name in (
            "can_ask_questions",
            "max_questions",
            "preferred_mode",
            "typical_complexity",
            "typically_needs_context",
            "context_files",
            "extra",
        ):
            if field_name in config:
                data[field_name] = config.get(field_name)

        return cls(**data)
