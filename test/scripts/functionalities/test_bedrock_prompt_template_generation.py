"""
Tests for bedrock_prompt_template_generation module.

Following pytest best practices and common pitfalls prevention:
- Read source code first to understand implementation
- Mock at import locations, not definition locations
- Count method calls and match side_effect length
- Use MagicMock for Path operations
- Test actual behavior, not assumptions
- Use realistic fixtures with proper cleanup
"""

import pytest
import json
import tempfile
import argparse
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

# Import the module under test
from cursus.steps.scripts.bedrock_prompt_template_generation import (
    PlaceholderResolver,
    PromptTemplateGenerator,
    TemplateValidator,
    load_config_from_json_file,
    load_category_definitions,
    main,
    DEFAULT_SYSTEM_PROMPT_CONFIG,
    DEFAULT_OUTPUT_FORMAT_CONFIG,
    DEFAULT_INSTRUCTION_CONFIG,
)


class TestPlaceholderResolver:
    """Test the PlaceholderResolver class."""

    def test_resolve_placeholder_literal_text(self):
        """Test resolving literal text (no placeholder)."""
        resolver = PlaceholderResolver([], None)
        result = resolver.resolve_placeholder("literal text", "field")
        assert result == "literal text"

    def test_resolve_placeholder_schema_enum(self):
        """Test resolving placeholder from schema enum."""
        schema = {
            "properties": {
                "category": {"enum": ["A", "B", "C"]}
            }
        }
        resolver = PlaceholderResolver([], schema)

        result = resolver.resolve_placeholder("${category_enum}", "category", "schema_enum")
        assert "One of: A, B, C" in result

    def test_resolve_placeholder_schema_range(self):
        """Test resolving placeholder from schema numeric range."""
        schema = {
            "properties": {
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
        resolver = PlaceholderResolver([], schema)

        result = resolver.resolve_placeholder("${confidence_range}", "confidence", "schema_range")
        assert "Number between 0.0 and 1.0" in result

    def test_resolve_placeholder_from_categories(self):
        """Test resolving placeholder directly from categories."""
        categories = [{"name": "Category A"}, {"name": "Category B"}]
        resolver = PlaceholderResolver(categories, None)

        result = resolver.resolve_placeholder("${categories}", "category", "categories")
        assert "One of: Category A, Category B" in result

    def test_resolve_placeholder_fallback(self):
        """Test placeholder resolution fallback when strategies fail."""
        resolver = PlaceholderResolver([], None)

        result = resolver.resolve_placeholder("${unknown_placeholder}", "field")
        assert result == "[FIELD_UNRESOLVED]"

    def test_validate_all_resolved_success(self):
        """Test validation when all placeholders are resolved."""
        # Provide categories so placeholders can be resolved
        categories = [{"name": "Test Category"}]
        resolver = PlaceholderResolver(categories, None)

        # Resolve a placeholder that can actually be resolved
        resolver.resolve_placeholder("${categories}", "category", "categories")

        result = resolver.validate_all_resolved()
        assert result["all_resolved"] is True
        assert result["successful"] == 1
        assert result["failed"] == 0

    def test_validate_all_resolved_failure(self):
        """Test validation when placeholders fail to resolve."""
        resolver = PlaceholderResolver([], None)

        # Try to resolve a placeholder that will fail
        resolver.resolve_placeholder("${nonexistent}", "field")

        result = resolver.validate_all_resolved()
        assert result["all_resolved"] is False
        assert result["failed"] == 1


class TestPromptTemplateGenerator:
    """Test the PromptTemplateGenerator class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "category_definitions": json.dumps([
                {
                    "name": "Test Category",
                    "description": "A test category",
                    "conditions": ["test condition"],
                    "key_indicators": ["test indicator"]
                }
            ]),
            "system_prompt_config": DEFAULT_SYSTEM_PROMPT_CONFIG,
            "output_format_config": DEFAULT_OUTPUT_FORMAT_CONFIG,
            "instruction_config": DEFAULT_INSTRUCTION_CONFIG,
        }

    @pytest.fixture
    def sample_schema(self):
        """Create sample JSON schema for testing."""
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["Test Category"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["category", "confidence"]
        }

    def test_init_loads_categories(self, sample_config):
        """Test that generator loads categories correctly."""
        generator = PromptTemplateGenerator(sample_config)

        assert len(generator.categories) == 1
        assert generator.categories[0]["name"] == "Test Category"

    def test_init_validates_categories(self, sample_config):
        """Test that generator validates category structure."""
        # Modify config to have invalid category
        invalid_config = sample_config.copy()
        invalid_config["category_definitions"] = json.dumps([{"name": "Invalid"}])

        with pytest.raises(ValueError, match="missing required field"):
            PromptTemplateGenerator(invalid_config)

    def test_enrich_schema_with_categories(self, sample_config, sample_schema):
        """Test schema enrichment with category enum."""
        generator = PromptTemplateGenerator(sample_config, sample_schema)

        enriched = generator._enrich_schema_with_categories(sample_schema)
        assert "enum" in enriched["properties"]["category"]
        assert "Test Category" in enriched["properties"]["category"]["enum"]

    def test_generate_system_prompt(self, sample_config):
        """Test system prompt generation."""
        generator = PromptTemplateGenerator(sample_config)

        prompt = generator._generate_system_prompt()
        assert "expert analyst" in prompt
        assert "data analysis" in prompt

    def test_generate_system_prompt_tone_adjustments(self, sample_config):
        """Test system prompt tone adjustments."""
        # Test different tones
        config = sample_config.copy()
        config["system_prompt_config"] = DEFAULT_SYSTEM_PROMPT_CONFIG.copy()
        config["system_prompt_config"]["tone"] = "casual"

        generator = PromptTemplateGenerator(config)
        prompt = generator._generate_system_prompt()

        assert "Hey!" in prompt  # Casual tone opener

    def test_generate_category_definitions_section(self, sample_config):
        """Test category definitions section generation."""
        generator = PromptTemplateGenerator(sample_config)

        section = generator._generate_category_definitions_section()
        assert "Categories and their criteria:" in section
        assert "Test Category" in section
        assert "test condition" in section

    def test_generate_instructions_section(self, sample_config):
        """Test instructions section generation."""
        generator = PromptTemplateGenerator(sample_config)

        section = generator._generate_instructions_section()
        assert "Decision Criteria:" in section
        assert "Carefully review all provided data" in section

    def test_generate_output_format_section_json(self, sample_config, sample_schema):
        """Test JSON output format generation."""
        generator = PromptTemplateGenerator(sample_config, sample_schema)

        section = generator._generate_custom_output_format_from_schema()
        assert "Required Output Format" in section
        assert '"category":' in section
        assert '"confidence":' in section

    def test_generate_template_metadata(self, sample_config):
        """Test template metadata generation."""
        generator = PromptTemplateGenerator(sample_config)

        metadata = generator._generate_template_metadata()
        assert "template_version" in metadata
        assert "generation_timestamp" in metadata
        assert metadata["category_count"] == 1

    def test_generate_template_complete(self, sample_config, sample_schema):
        """Test complete template generation."""
        generator = PromptTemplateGenerator(sample_config, sample_schema)

        template = generator.generate_template()

        assert "system_prompt" in template
        assert "user_prompt_template" in template
        assert "metadata" in template
        assert template["metadata"]["category_count"] == 1


class TestTemplateValidator:
    """Test the TemplateValidator class."""

    def test_validate_template_valid(self):
        """Test validation of a valid template."""
        validator = TemplateValidator()

        valid_template = {
            "system_prompt": "You are an expert analyst with extensive knowledge in data analysis. Your task is to analyze and classify content systematically.",
            "user_prompt_template": "Categories and their criteria: analyze the data and provide structured output format with JSON response.",
            "metadata": {
                "template_version": "1.0",
                "generation_timestamp": "2024-01-01T00:00:00",
                "task_type": "classification",
                "category_count": 1
            }
        }

        result = validator.validate_template(valid_template)
        assert result["is_valid"] is True
        assert result["quality_score"] > 0.7

    def test_validate_template_invalid_system_prompt(self):
        """Test validation of template with invalid system prompt."""
        validator = TemplateValidator()

        invalid_template = {
            "system_prompt": "",  # Empty system prompt
            "user_prompt_template": "Valid user prompt with categories and format.",
            "metadata": {
                "template_version": "1.0",
                "generation_timestamp": "2024-01-01T00:00:00",
                "task_type": "classification",
                "category_count": 1
            }
        }

        result = validator.validate_template(invalid_template)
        assert result["is_valid"] is False
        assert result["quality_score"] < 0.7

    def test_validate_template_missing_metadata(self):
        """Test validation of template with missing metadata."""
        validator = TemplateValidator()

        template = {
            "system_prompt": "Valid system prompt.",
            "user_prompt_template": "Valid user prompt.",
            "metadata": {}  # Missing required fields
        }

        result = validator.validate_template(template)
        assert result["is_valid"] is False
        assert len(result["validation_details"]) == 3  # system, user, metadata


class TestHelperFunctions:
    """Test helper functions."""

    def test_load_config_from_json_file_exists(self, tmp_path):
        """Test loading config from existing JSON file."""
        # Create test config file
        config_data = {"test_key": "test_value", "number": 42}
        config_file = tmp_path / "test_config.json"

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        default_config = {"default_key": "default_value"}

        result = load_config_from_json_file(
            str(tmp_path), "test_config", default_config, lambda x: None
        )

        assert result["test_key"] == "test_value"
        assert result["number"] == 42
        assert result["default_key"] == "default_value"  # Merged with defaults

    def test_load_config_from_json_file_not_exists(self, tmp_path):
        """Test loading config when file doesn't exist."""
        default_config = {"default_key": "default_value"}

        result = load_config_from_json_file(
            str(tmp_path), "nonexistent", default_config, lambda x: None
        )

        assert result == default_config

    def test_load_config_from_json_file_invalid_json(self, tmp_path):
        """Test loading config with invalid JSON."""
        # Create invalid JSON file
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write("invalid json content")

        default_config = {"default_key": "default_value"}

        result = load_config_from_json_file(
            str(tmp_path), "invalid", default_config, lambda x: None
        )

        assert result == default_config

    def test_load_category_definitions_exists(self, tmp_path):
        """Test loading category definitions from existing file."""
        categories = [
            {"name": "Category A", "description": "Test category A"},
            {"name": "Category B", "description": "Test category B"}
        ]

        categories_file = tmp_path / "category_definitions.json"
        with open(categories_file, "w") as f:
            json.dump(categories, f)

        result = load_category_definitions(str(tmp_path), lambda x: None)

        assert len(result) == 2
        assert result[0]["name"] == "Category A"
        assert result[1]["name"] == "Category B"

    def test_load_category_definitions_not_exists(self, tmp_path):
        """Test loading category definitions when file doesn't exist."""
        result = load_category_definitions(str(tmp_path), lambda x: None)

        assert result == []

    def test_load_category_definitions_invalid_json(self, tmp_path):
        """Test loading category definitions with invalid JSON."""
        categories_file = tmp_path / "category_definitions.json"
        with open(categories_file, "w") as f:
            f.write("invalid json")

        result = load_category_definitions(str(tmp_path), lambda x: None)

        assert result == []


class TestMainFunction:
    """Test the main function."""

    @pytest.fixture
    def mock_input_paths(self, tmp_path):
        """Create mock input paths with test data."""
        # Create prompt configs directory
        prompt_dir = tmp_path / "prompt_configs"
        prompt_dir.mkdir()

        # Create category definitions
        categories = [
            {
                "name": "Test Category",
                "description": "A test category",
                "conditions": ["test condition"],
                "key_indicators": ["test indicator"]
            }
        ]

        with open(prompt_dir / "category_definitions.json", "w") as f:
            json.dump(categories, f)

        # Create config files
        with open(prompt_dir / "system_prompt.json", "w") as f:
            json.dump({"role_definition": "test analyst"}, f)

        with open(prompt_dir / "output_format.json", "w") as f:
            json.dump({"format_type": "structured_json"}, f)

        with open(prompt_dir / "instruction.json", "w") as f:
            json.dump({"include_analysis_steps": True}, f)

        return {"prompt_configs": str(prompt_dir)}

    @pytest.fixture
    def mock_output_paths(self, tmp_path):
        """Create mock output paths."""
        templates_dir = tmp_path / "templates"
        metadata_dir = tmp_path / "metadata"
        schema_dir = tmp_path / "schema"

        return {
            "prompt_templates": str(templates_dir),
            "template_metadata": str(metadata_dir),
            "validation_schema": str(schema_dir)
        }

    def test_main_success(self, mock_input_paths, mock_output_paths):
        """Test successful main function execution."""
        environ_vars = {
            "TEMPLATE_TASK_TYPE": "classification",
            "TEMPLATE_STYLE": "structured",
            "VALIDATION_LEVEL": "standard"
        }

        job_args = argparse.Namespace()

        result = main(
            input_paths=mock_input_paths,
            output_paths=mock_output_paths,
            environ_vars=environ_vars,
            job_args=job_args,
            logger=lambda x: None
        )

        assert result["success"] is True
        assert result["template_generated"] is True
        assert "category_count" in result
        assert result["category_count"] == 1

    def test_main_missing_prompt_configs(self, mock_output_paths):
        """Test main function with missing prompt configs."""
        input_paths = {}  # Missing prompt_configs

        environ_vars = {"TEMPLATE_TASK_TYPE": "classification"}
        job_args = argparse.Namespace()

        with pytest.raises(ValueError, match="No prompt_configs input path provided"):
            main(
                input_paths=input_paths,
                output_paths=mock_output_paths,
                environ_vars=environ_vars,
                job_args=job_args,
                logger=lambda x: None
            )

    def test_main_no_categories(self, tmp_path, mock_output_paths):
        """Test main function with no category definitions."""
        # Create empty prompt configs directory
        prompt_dir = tmp_path / "prompt_configs"
        prompt_dir.mkdir()

        input_paths = {"prompt_configs": str(prompt_dir)}
        environ_vars = {"TEMPLATE_TASK_TYPE": "classification"}
        job_args = argparse.Namespace()

        with pytest.raises(ValueError, match="No category definitions found"):
            main(
                input_paths=input_paths,
                output_paths=mock_output_paths,
                environ_vars=environ_vars,
                job_args=job_args,
                logger=lambda x: None
            )


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_full_template_generation_workflow(self, tmp_path):
        """Test complete template generation workflow."""
        # Create input directory structure
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create comprehensive category definitions
        categories = [
            {
                "name": "High Risk",
                "description": "High risk financial transactions",
                "conditions": ["amount > 10000", "international transfer"],
                "key_indicators": ["large amount", "cross-border"],
                "exceptions": ["verified customer", "internal transfer"]
            },
            {
                "name": "Medium Risk",
                "description": "Medium risk transactions",
                "conditions": ["amount > 1000", "unusual pattern"],
                "key_indicators": ["moderate amount", "pattern deviation"]
            }
        ]

        with open(input_dir / "category_definitions.json", "w") as f:
            json.dump(categories, f)

        # Create config files
        system_config = {
            "role_definition": "financial analyst",
            "expertise_areas": ["risk assessment", "financial analysis"],
            "tone": "professional"
        }

        with open(input_dir / "system_prompt.json", "w") as f:
            json.dump(system_config, f)

        output_config = {
            "format_type": "structured_json",
            "required_fields": ["category", "confidence", "key_evidence", "reasoning"]
        }

        with open(input_dir / "output_format.json", "w") as f:
            json.dump(output_config, f)

        # Create output directories
        output_dir = tmp_path / "output"
        templates_dir = output_dir / "templates"
        metadata_dir = output_dir / "metadata"
        schema_dir = output_dir / "schema"

        # Execute main function
        input_paths = {"prompt_configs": str(input_dir)}
        output_paths = {
            "prompt_templates": str(templates_dir),
            "template_metadata": str(metadata_dir),
            "validation_schema": str(schema_dir)
        }
        environ_vars = {
            "TEMPLATE_TASK_TYPE": "risk_classification",
            "TEMPLATE_STYLE": "structured",
            "VALIDATION_LEVEL": "standard"
        }
        job_args = argparse.Namespace()

        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=job_args,
            logger=lambda x: None
        )

        # Verify results
        assert result["success"] is True
        assert result["category_count"] == 2

        # Check output files were created
        assert (templates_dir / "prompts.json").exists()
        assert len(list(metadata_dir.glob("*.json"))) == 1
        assert len(list(schema_dir.glob("*.json"))) == 1

        # Verify template content
        with open(templates_dir / "prompts.json", "r") as f:
            template_data = json.load(f)

        assert "system_prompt" in template_data
        assert "user_prompt_template" in template_data
        assert "financial analyst" in template_data["system_prompt"]
        assert "High Risk" in template_data["user_prompt_template"]
        assert "Medium Risk" in template_data["user_prompt_template"]
