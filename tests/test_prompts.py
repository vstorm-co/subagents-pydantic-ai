"""Tests for prompts module."""

from __future__ import annotations

from subagents_pydantic_ai import SubAgentConfig
from subagents_pydantic_ai.prompts import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DUAL_MODE_SYSTEM_PROMPT,
    SUBAGENT_SYSTEM_PROMPT,
    TASK_TOOL_DESCRIPTION,
    get_subagent_system_prompt,
    get_task_instructions_prompt,
)


class TestSystemPrompts:
    """Tests for system prompt constants."""

    def test_subagent_system_prompt_content(self):
        """Test SUBAGENT_SYSTEM_PROMPT has expected content."""
        assert "specialized subagent" in SUBAGENT_SYSTEM_PROMPT
        assert "ask_parent" in SUBAGENT_SYSTEM_PROMPT
        assert "Communication" in SUBAGENT_SYSTEM_PROMPT

    def test_dual_mode_system_prompt_content(self):
        """Test DUAL_MODE_SYSTEM_PROMPT has expected content."""
        assert "Sync Mode" in DUAL_MODE_SYSTEM_PROMPT
        assert "Async Mode" in DUAL_MODE_SYSTEM_PROMPT
        assert "Background" in DUAL_MODE_SYSTEM_PROMPT

    def test_general_purpose_description(self):
        """Test DEFAULT_GENERAL_PURPOSE_DESCRIPTION has expected content."""
        assert "general-purpose" in DEFAULT_GENERAL_PURPOSE_DESCRIPTION
        assert "research" in DEFAULT_GENERAL_PURPOSE_DESCRIPTION.lower()

    def test_task_tool_description(self):
        """Test TASK_TOOL_DESCRIPTION has expected content."""
        assert "sync" in TASK_TOOL_DESCRIPTION
        assert "async" in TASK_TOOL_DESCRIPTION
        assert "subagent_type" in TASK_TOOL_DESCRIPTION


class TestGetSubagentSystemPrompt:
    """Tests for get_subagent_system_prompt function."""

    def test_empty_configs(self):
        """Test with empty config list."""
        result = get_subagent_system_prompt([])
        assert "Available Subagents" in result
        assert "delegate work" in result

    def test_single_config(self):
        """Test with single subagent config."""
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics thoroughly",
                instructions="Do research",
            )
        ]
        result = get_subagent_system_prompt(configs)

        assert "**researcher**" in result
        assert "Researches topics thoroughly" in result

    def test_multiple_configs(self):
        """Test with multiple subagent configs."""
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="Do research",
            ),
            SubAgentConfig(
                name="writer",
                description="Writes content",
                instructions="Write stuff",
            ),
        ]
        result = get_subagent_system_prompt(configs)

        assert "**researcher**" in result
        assert "**writer**" in result
        assert "Researches topics" in result
        assert "Writes content" in result

    def test_config_with_no_questions(self):
        """Test config with can_ask_questions=False."""
        configs = [
            SubAgentConfig(
                name="formatter",
                description="Formats text",
                instructions="Format things",
                can_ask_questions=False,
            )
        ]
        result = get_subagent_system_prompt(configs)

        assert "cannot ask clarifying questions" in result

    def test_config_with_questions_allowed(self):
        """Test config with can_ask_questions=True (default)."""
        configs = [
            SubAgentConfig(
                name="helper",
                description="Helps with tasks",
                instructions="Help out",
                can_ask_questions=True,
            )
        ]
        result = get_subagent_system_prompt(configs)

        # Should NOT have the "cannot ask" warning
        assert "cannot ask clarifying questions" not in result

    def test_include_dual_mode_true(self):
        """Test with dual mode prompt included — dual mode moved to tool description."""
        configs = [
            SubAgentConfig(
                name="worker",
                description="Does work",
                instructions="Work hard",
            )
        ]
        result = get_subagent_system_prompt(configs, include_dual_mode=True)

        # Dual mode explanation now lives in TASK_TOOL_DESCRIPTION, not system prompt
        assert "**worker**" in result
        assert "Does work" in result

    def test_include_dual_mode_false(self):
        """Test with dual mode prompt excluded."""
        configs = [
            SubAgentConfig(
                name="worker",
                description="Does work",
                instructions="Work hard",
            )
        ]
        result = get_subagent_system_prompt(configs, include_dual_mode=False)

        assert "Sync Mode" not in result
        assert "Async Mode" not in result


class TestGetTaskInstructionsPrompt:
    """Tests for get_task_instructions_prompt function."""

    def test_basic_task(self):
        """Test basic task instructions."""
        result = get_task_instructions_prompt("Analyze this data")

        assert "Your Task" in result
        assert "Analyze this data" in result

    def test_can_ask_questions_true(self):
        """Test with question asking enabled."""
        result = get_task_instructions_prompt(
            "Do the thing",
            can_ask_questions=True,
        )

        assert "Asking Questions" in result
        assert "ask_parent" in result

    def test_can_ask_questions_false(self):
        """Test with question asking disabled."""
        result = get_task_instructions_prompt(
            "Do the thing",
            can_ask_questions=False,
        )

        assert "Note" in result
        assert "cannot ask" in result.lower()
        assert "best judgment" in result

    def test_max_questions_specified(self):
        """Test with max_questions limit."""
        result = get_task_instructions_prompt(
            "Do the thing",
            can_ask_questions=True,
            max_questions=3,
        )

        assert "up to 3 questions" in result

    def test_max_questions_not_specified(self):
        """Test without max_questions limit."""
        result = get_task_instructions_prompt(
            "Do the thing",
            can_ask_questions=True,
            max_questions=None,
        )

        assert "up to" not in result
        assert "questions" not in result.lower() or "Asking Questions" in result

    def test_questions_disabled_ignores_max(self):
        """Test that max_questions is ignored when questions disabled."""
        result = get_task_instructions_prompt(
            "Do the thing",
            can_ask_questions=False,
            max_questions=5,
        )

        # Should not mention the limit since questions are disabled
        assert "up to 5" not in result
