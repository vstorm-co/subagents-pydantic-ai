"""Tests for the SubAgentSpec declarative configuration module."""

from __future__ import annotations

import json

from subagents_pydantic_ai.spec import SubAgentSpec
from subagents_pydantic_ai.types import SubAgentConfig


class TestSubAgentSpecDefaults:
    """Tests for SubAgentSpec default values."""

    def test_minimal_spec(self):
        """A spec with only a name should use sensible defaults."""
        spec = SubAgentSpec(name="worker")
        assert spec.name == "worker"
        assert spec.description == ""
        assert spec.instructions == ""
        assert spec.model is None
        assert spec.can_ask_questions is None
        assert spec.max_questions is None
        assert spec.preferred_mode is None
        assert spec.typical_complexity is None
        assert spec.typically_needs_context is None
        assert spec.context_files is None
        assert spec.extra == {}

    def test_full_spec(self):
        """A spec with all fields set should preserve them."""
        spec = SubAgentSpec(
            name="researcher",
            description="Research assistant",
            instructions="You research topics.",
            model="openai:gpt-4.1-mini",
            can_ask_questions=True,
            max_questions=3,
            preferred_mode="async",
            typical_complexity="complex",
            typically_needs_context=False,
            context_files=["/AGENTS.md"],
            extra={"memory": "project"},
        )
        assert spec.name == "researcher"
        assert spec.description == "Research assistant"
        assert spec.instructions == "You research topics."
        assert spec.model == "openai:gpt-4.1-mini"
        assert spec.can_ask_questions is True
        assert spec.max_questions == 3
        assert spec.preferred_mode == "async"
        assert spec.typical_complexity == "complex"
        assert spec.typically_needs_context is False
        assert spec.context_files == ["/AGENTS.md"]
        assert spec.extra == {"memory": "project"}


class TestSubAgentSpecToConfig:
    """Tests for SubAgentSpec.to_config() conversion."""

    def test_minimal_to_config(self):
        """Minimal spec should produce config with only required fields."""
        spec = SubAgentSpec(name="worker")
        config = spec.to_config()
        assert config["name"] == "worker"
        assert config["description"] == ""
        assert config["instructions"] == ""
        # Optional fields should not be present
        assert "model" not in config
        assert "can_ask_questions" not in config
        assert "max_questions" not in config
        assert "preferred_mode" not in config
        assert "typical_complexity" not in config
        assert "typically_needs_context" not in config
        assert "context_files" not in config
        assert "extra" not in config

    def test_full_to_config(self):
        """Full spec should produce config with all fields."""
        spec = SubAgentSpec(
            name="researcher",
            description="Research assistant",
            instructions="You research topics.",
            model="openai:gpt-4.1-mini",
            can_ask_questions=True,
            max_questions=5,
            preferred_mode="sync",
            typical_complexity="moderate",
            typically_needs_context=True,
            context_files=["/DEEP.md"],
            extra={"cost_budget": 100},
        )
        config = spec.to_config()
        assert config["name"] == "researcher"
        assert config["description"] == "Research assistant"
        assert config["instructions"] == "You research topics."
        assert config["model"] == "openai:gpt-4.1-mini"
        assert config["can_ask_questions"] is True
        assert config["max_questions"] == 5
        assert config["preferred_mode"] == "sync"
        assert config["typical_complexity"] == "moderate"
        assert config["typically_needs_context"] is True
        assert config["context_files"] == ["/DEEP.md"]
        assert config["extra"] == {"cost_budget": 100}

    def test_empty_extra_excluded(self):
        """An empty extra dict should not appear in the config."""
        spec = SubAgentSpec(name="worker", extra={})
        config = spec.to_config()
        assert "extra" not in config

    def test_false_values_included(self):
        """Boolean False values should be included in config."""
        spec = SubAgentSpec(
            name="worker",
            can_ask_questions=False,
            typically_needs_context=False,
        )
        config = spec.to_config()
        assert config["can_ask_questions"] is False
        assert config["typically_needs_context"] is False


class TestSubAgentSpecFromConfig:
    """Tests for SubAgentSpec.from_config() class method."""

    def test_minimal_from_config(self):
        """Minimal config should produce a valid spec."""
        config = SubAgentConfig(
            name="worker",
            description="A worker",
            instructions="Do work",
        )
        spec = SubAgentSpec.from_config(config)
        assert spec.name == "worker"
        assert spec.description == "A worker"
        assert spec.instructions == "Do work"
        assert spec.model is None

    def test_full_from_config(self):
        """Full config should round-trip through from_config."""
        config = SubAgentConfig(
            name="researcher",
            description="Research assistant",
            instructions="Research topics.",
            model="openai:gpt-4.1",
            can_ask_questions=True,
            max_questions=3,
            preferred_mode="async",
            typical_complexity="complex",
            typically_needs_context=False,
            context_files=["/AGENTS.md"],
            extra={"memory": "project"},
        )
        spec = SubAgentSpec.from_config(config)
        assert spec.name == "researcher"
        assert spec.description == "Research assistant"
        assert spec.instructions == "Research topics."
        assert spec.model == "openai:gpt-4.1"
        assert spec.can_ask_questions is True
        assert spec.max_questions == 3
        assert spec.preferred_mode == "async"
        assert spec.typical_complexity == "complex"
        assert spec.typically_needs_context is False
        assert spec.context_files == ["/AGENTS.md"]
        assert spec.extra == {"memory": "project"}


class TestSubAgentSpecRoundTrip:
    """Tests for SubAgentSpec to_config/from_config round-trip."""

    def test_roundtrip_minimal(self):
        """Minimal spec should survive a to_config -> from_config round-trip."""
        original = SubAgentSpec(name="worker", description="Worker", instructions="Work")
        config = original.to_config()
        restored = SubAgentSpec.from_config(config)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.instructions == original.instructions

    def test_roundtrip_full(self):
        """Full spec should survive a to_config -> from_config round-trip."""
        original = SubAgentSpec(
            name="researcher",
            description="Research assistant",
            instructions="You research topics.",
            model="openai:gpt-4.1-mini",
            can_ask_questions=True,
            max_questions=5,
            preferred_mode="sync",
            typical_complexity="moderate",
            typically_needs_context=True,
            context_files=["/DEEP.md", "/AGENTS.md"],
            extra={"cost_budget": 50},
        )
        config = original.to_config()
        restored = SubAgentSpec.from_config(config)
        assert restored == original

    def test_roundtrip_from_config_to_config(self):
        """Config -> from_config -> to_config should reproduce the config."""
        config = SubAgentConfig(
            name="coder",
            description="Code writer",
            instructions="Write code.",
            model="openai:gpt-4.1",
            preferred_mode="async",
            context_files=["/CODING.md"],
        )
        spec = SubAgentSpec.from_config(config)
        roundtripped = spec.to_config()
        assert roundtripped["name"] == config["name"]
        assert roundtripped["description"] == config["description"]
        assert roundtripped["instructions"] == config["instructions"]
        assert roundtripped["model"] == config["model"]
        assert roundtripped["preferred_mode"] == config["preferred_mode"]
        assert roundtripped["context_files"] == config["context_files"]


class TestSubAgentSpecSerialization:
    """Tests for JSON serialization of SubAgentSpec."""

    def test_json_serialization(self):
        """Spec should serialize to JSON and back."""
        spec = SubAgentSpec(
            name="researcher",
            description="Research assistant",
            instructions="You research topics.",
            model="openai:gpt-4.1-mini",
        )
        json_str = spec.model_dump_json()
        restored = SubAgentSpec.model_validate_json(json_str)
        assert restored == spec

    def test_dict_serialization(self):
        """Spec should serialize to dict and back."""
        spec = SubAgentSpec(
            name="worker",
            description="Worker agent",
            instructions="Do tasks.",
            can_ask_questions=False,
            extra={"tag": "v1"},
        )
        data = spec.model_dump()
        restored = SubAgentSpec.model_validate(data)
        assert restored == spec

    def test_json_list_serialization(self):
        """A list of specs should serialize as a JSON array."""
        specs = [
            SubAgentSpec(
                name="researcher",
                description="Research assistant",
                instructions="Research topics.",
                model="openai:gpt-4.1-mini",
            ),
            SubAgentSpec(
                name="coder",
                description="Code writer",
                instructions="Write code.",
                model="openai:gpt-4.1",
                preferred_mode="sync",
            ),
        ]
        json_str = json.dumps([s.model_dump() for s in specs])
        data = json.loads(json_str)
        restored = [SubAgentSpec.model_validate(d) for d in data]
        assert len(restored) == 2
        assert restored[0].name == "researcher"
        assert restored[1].name == "coder"
        assert restored[1].preferred_mode == "sync"

    def test_yaml_style_dict_loading(self):
        """Specs should load from plain dicts as YAML would produce."""
        yaml_data = [
            {
                "name": "researcher",
                "description": "Research assistant",
                "instructions": "You research topics.",
                "model": "openai:gpt-4.1-mini",
            },
            {
                "name": "coder",
                "description": "Code writer",
                "instructions": "Write Python code.",
            },
        ]
        specs = [SubAgentSpec.model_validate(d) for d in yaml_data]
        assert len(specs) == 2
        assert specs[0].name == "researcher"
        assert specs[0].model == "openai:gpt-4.1-mini"
        assert specs[1].name == "coder"
        assert specs[1].model is None

    def test_yaml_roundtrip(self):
        """Specs should round-trip through YAML serialization."""
        try:
            import yaml
        except ImportError:
            return  # Skip if pyyaml not available

        spec = SubAgentSpec(
            name="researcher",
            description="Research assistant",
            instructions="You research topics.",
            model="openai:gpt-4.1-mini",
            preferred_mode="async",
            context_files=["/DEEP.md"],
        )
        yaml_str = yaml.dump(
            {"subagents": [spec.model_dump(exclude_none=True)]},
            default_flow_style=False,
        )
        loaded = yaml.safe_load(yaml_str)
        restored = SubAgentSpec.model_validate(loaded["subagents"][0])
        assert restored.name == spec.name
        assert restored.model == spec.model
        assert restored.preferred_mode == spec.preferred_mode
        assert restored.context_files == spec.context_files


class TestSubAgentSpecImport:
    """Tests for SubAgentSpec importability."""

    def test_importable_from_package(self):
        """SubAgentSpec should be importable from the top-level package."""
        from subagents_pydantic_ai import SubAgentSpec as Imported

        assert Imported is SubAgentSpec

    def test_in_all(self):
        """SubAgentSpec should be listed in __all__."""
        import subagents_pydantic_ai

        assert "SubAgentSpec" in subagents_pydantic_ai.__all__
