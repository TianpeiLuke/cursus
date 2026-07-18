"""
Tests for bedrock_prompt_template_generation module (declarative meta-prompt assembler).

The step assembles the prompt-config bundle into ONE prompts.json prompt ruleset in the
{ruleset, rules} shape (the same contract the knowledge-routing producer emits), with the output schema
embedded in the prompt. These tests exercise the assembly functions + the end-to-end main() and
assert the downstream-contract invariants (ruleset/rules shape, enum-locked schema, embedded
output block, rule-shaped rules[] fields).
"""

import argparse
import json
import tempfile
from pathlib import Path

import pytest

from cursus.steps.scripts.bedrock_prompt_template_generation import (
    build_system_prompt,
    render_rule_block,
    render_rules_section,
    render_input_evidence,
    build_output_constraints_text,
    build_output_schema,
    build_rules_list,
    assemble_prompt_ruleset,
    load_config_from_json_file,
    load_category_definitions,
    main,
    DEFAULT_SYSTEM_PROMPT_CONFIG,
    DEFAULT_OUTPUT_FORMAT_CONFIG,
    DEFAULT_INSTRUCTION_CONFIG,
    META_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_categories():
    return [
        {
            "name": "TrueDNR",
            "description": "Delivered Not Received",
            "priority": 1,
            "conditions": ["Package marked delivered", "Buyer claims non-receipt"],
            "key_indicators": ["Tracking shows delivery"],
            "exceptions": ["Refund before delivery"],
            "validation_rules": ["Verify delivery timestamp"],
        },
        {
            "name": "Confirmed_Delay",
            "description": "Shipment delayed",
            "priority": 2,
            "conditions": ["Shipment delayed"],
            "key_indicators": ["Seller acknowledges delay"],
            "exceptions": ["Unconfirmed delays"],
        },
    ]


@pytest.fixture
def sample_system_config():
    return {
        "role_definition": "expert analyst",
        "expertise_areas": ["classification", "logistics"],
        "responsibilities": ["classify accurately"],
        "behavioral_guidelines": ["be precise"],
        "tone": "professional",
    }


@pytest.fixture
def sample_output_config():
    return {
        "format_type": "structured_json",
        "required_fields": ["category", "confidence", "reasoning"],
        "field_descriptions": {
            "category": "The category name",
            "confidence": "0.0-1.0",
            "reasoning": "Why",
        },
        "validation_requirements": ["category must match a rule name"],
    }


@pytest.fixture
def prompt_configs_dir(sample_categories, sample_system_config, sample_output_config):
    tmp = tempfile.mkdtemp()
    d = Path(tmp)
    (d / "category_definitions.json").write_text(json.dumps(sample_categories))
    (d / "system_prompt.json").write_text(json.dumps(sample_system_config))
    (d / "output_format.json").write_text(json.dumps(sample_output_config))
    (d / "instruction.json").write_text(json.dumps(DEFAULT_INSTRUCTION_CONFIG))
    return str(d)


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_includes_role_and_expertise(self, sample_system_config):
        s = build_system_prompt(sample_system_config, {})
        assert "expert analyst" in s
        assert "classification" in s and "logistics" in s

    def test_includes_responsibilities_and_guidelines(self, sample_system_config):
        s = build_system_prompt(sample_system_config, {})
        assert "classify accurately" in s
        assert "be precise" in s

    def test_no_tone_register_switching(self, sample_system_config):
        """The refactor removed the tone-adjustment table; a 'casual' tone changes nothing."""
        casual = dict(sample_system_config, tone="casual")
        assert build_system_prompt(casual, {}) == build_system_prompt(sample_system_config, {})

    def test_handles_missing_expertise(self):
        s = build_system_prompt({"role_definition": "analyst"}, {})
        assert "analyst" in s


# ---------------------------------------------------------------------------
# Rule rendering
# ---------------------------------------------------------------------------


class TestRenderRuleBlock:
    def test_renders_name_conditions_exceptions(self, sample_categories):
        block = render_rule_block(sample_categories[0], 1, include_examples=False)
        assert "<RULE 1: TrueDNR>" in block
        assert "</RULE 1>" in block
        assert "Conditions" in block
        assert "Buyer claims non-receipt" in block
        assert "Exceptions" in block

    def test_examples_gated_by_flag(self):
        rule = {"name": "R", "conditions": ["c"], "examples": ["ex1"]}
        with_ex = render_rule_block(rule, 1, include_examples=True)
        without_ex = render_rule_block(rule, 1, include_examples=False)
        assert "ex1" in with_ex
        assert "ex1" not in without_ex

    def test_rules_section_priority_order(self, sample_categories):
        # Reverse input priority; render must still order by priority ascending.
        section = render_rules_section(list(reversed(sample_categories)), include_examples=False)
        assert section.index("TrueDNR") < section.index("Confirmed_Delay")


class TestRenderInputEvidence:
    def test_emits_named_slots(self):
        ev = render_input_evidence(["dialogue", "shiptrack"])
        assert "{dialogue}" in ev and "{shiptrack}" in ev

    def test_default_placeholder(self):
        assert "{input_data}" in render_input_evidence([])


# ---------------------------------------------------------------------------
# Output schema (embedded + machine)
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_output_constraints_text_embeds_fields_and_enum(self, sample_output_config):
        text = build_output_constraints_text(sample_output_config, ["TrueDNR", "Confirmed_Delay"])
        assert "category" in text and "reasoning" in text
        # the enum (rule names) is spelled out in the prompt-visible constraints
        assert "TrueDNR" in text and "Confirmed_Delay" in text

    def test_build_output_schema_enum_locked_to_rule_names(self, sample_output_config):
        schema = build_output_schema(sample_output_config, ["TrueDNR", "Confirmed_Delay"])
        assert schema["properties"]["category"]["enum"] == ["TrueDNR", "Confirmed_Delay"]

    def test_build_output_schema_confidence_is_number(self, sample_output_config):
        schema = build_output_schema(sample_output_config, ["A"])
        assert schema["properties"]["confidence"]["type"] == "number"

    def test_explicit_json_schema_respected(self):
        cfg = {"json_schema": {"type": "object", "properties": {"category": {"type": "string"}}}}
        schema = build_output_schema(cfg, ["X"])
        assert schema["properties"]["category"]["enum"] == ["X"]


# ---------------------------------------------------------------------------
# rules[] mapping to the rule shape the consumer reads
# ---------------------------------------------------------------------------


class TestBuildRulesList:
    def test_maps_category_fields_to_rule_fields(self, sample_categories):
        rules = build_rules_list(sample_categories)
        r = rules[0]
        assert r["rule_name"] == "TrueDNR"
        assert r["conditions"] == ["Package marked delivered", "Buyer claims non-receipt"]
        assert r["exclusions"] == ["Refund before delivery"]  # exceptions -> exclusions
        assert r["key_elements"] == ["Tracking shows delivery"]  # key_indicators -> key_elements
        assert r["priority_tier"] == 1  # priority -> priority_tier

    def test_priority_order(self, sample_categories):
        rules = build_rules_list(list(reversed(sample_categories)))
        assert [r["rule_name"] for r in rules] == ["TrueDNR", "Confirmed_Delay"]


# ---------------------------------------------------------------------------
# Full assembly — the {ruleset, rules} contract
# ---------------------------------------------------------------------------


class TestAssemblePromptRuleset:
    def test_top_level_shape(self, sample_categories, sample_system_config, sample_output_config):
        out = assemble_prompt_ruleset(
            sample_categories, sample_system_config, sample_output_config,
            DEFAULT_INSTRUCTION_CONFIG, ["dialogue"], include_examples=True,
        )
        assert set(out) == {"ruleset", "rules"}
        assert set(["system_prompt", "input_placeholders", "user_prompt_template", "output_schema"]).issubset(out["ruleset"])

    def test_output_schema_embedded_in_user_prompt(self, sample_categories, sample_system_config, sample_output_config):
        out = assemble_prompt_ruleset(
            sample_categories, sample_system_config, sample_output_config,
            DEFAULT_INSTRUCTION_CONFIG, ["dialogue"], include_examples=True,
        )
        upt = out["ruleset"]["user_prompt_template"]
        assert "Required Output" in upt        # output block embedded in the prompt
        assert "{dialogue}" in upt             # input slot embedded
        assert "TrueDNR" in upt                # rules embedded

    def test_rules_are_rule_shaped(self, sample_categories, sample_system_config, sample_output_config):
        out = assemble_prompt_ruleset(
            sample_categories, sample_system_config, sample_output_config,
            DEFAULT_INSTRUCTION_CONFIG, ["dialogue"], include_examples=True,
        )
        assert out["rules"][0]["rule_name"] == "TrueDNR"
        assert "exclusions" in out["rules"][0]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_config_merges_over_defaults(self, prompt_configs_dir):
        cfg = load_config_from_json_file(
            prompt_configs_dir, "system_prompt", DEFAULT_SYSTEM_PROMPT_CONFIG, lambda s: None
        )
        assert cfg["role_definition"] == "expert analyst"

    def test_load_config_missing_returns_default(self):
        cfg = load_config_from_json_file(
            "/nonexistent", "system_prompt", DEFAULT_SYSTEM_PROMPT_CONFIG, lambda s: None
        )
        assert cfg == DEFAULT_SYSTEM_PROMPT_CONFIG

    def test_load_categories(self, prompt_configs_dir):
        cats = load_category_definitions(prompt_configs_dir, lambda s: None)
        assert len(cats) == 2
        assert cats[0]["name"] == "TrueDNR"


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    def test_emits_prompts_json_ruleset(self, prompt_configs_dir):
        tmp = tempfile.mkdtemp()
        out_paths = {
            "prompt_templates": tmp + "/t",
            "template_metadata": tmp + "/m",
            "validation_schema": tmp + "/s",
        }
        env = {
            "INPUT_PLACEHOLDERS": '["dialogue"]',
            "INCLUDE_EXAMPLES": "true",
            "GENERATE_VALIDATION_SCHEMA": "true",
            "TEMPLATE_VERSION": "2.0",
        }
        result = main({"prompt_configs": prompt_configs_dir}, out_paths, env, argparse.Namespace(), logger=lambda s: None)
        assert result["success"] is True
        assert result["rules_assembled"] == 2

        pj = json.loads(Path(tmp + "/t/prompts.json").read_text())
        assert set(pj) == {"ruleset", "rules"}
        assert len(pj["rules"]) == 2
        assert pj["ruleset"]["output_schema"]["properties"]["category"]["enum"] == ["TrueDNR", "Confirmed_Delay"]

    def test_missing_categories_raises(self):
        tmp = tempfile.mkdtemp()
        out_paths = {"prompt_templates": tmp + "/t", "template_metadata": tmp + "/m", "validation_schema": tmp + "/s"}
        with pytest.raises(ValueError, match="No category definitions"):
            main({"prompt_configs": tmp}, out_paths, {}, argparse.Namespace(), logger=lambda s: None)

    def test_no_prompt_configs_path_raises(self):
        with pytest.raises(ValueError, match="No prompt_configs input path"):
            main({}, {}, {}, argparse.Namespace(), logger=lambda s: None)
