"""
Unit tests for RegistryValidator core validation methods.

Tests the main validation functionality including atomicity, connections,
metadata completeness, and tag consistency validation.
"""

import pytest
from unittest.mock import Mock, patch
from cursus.pipeline_catalog.utils.registry_validator import (
    RegistryValidator, ValidationSeverity, ValidationReport
)
from cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry

class TestRegistryValidatorCore:
    """Test suite for RegistryValidator core functionality."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry for testing."""
        registry = Mock(spec=CatalogRegistry)
        
        # Mock valid registry data
        valid_nodes = {
            "valid_pipeline": {
                "id": "valid_pipeline",
                "title": "Valid Pipeline",
                "description": "A valid pipeline for testing",
                "atomic_properties": {
                    "single_responsibility": "Test functionality",
                    "input_interface": ["data"],
                    "output_interface": ["model"],
                    "independence": "fully_self_contained",
                    "side_effects": "none",
                    "dependencies": ["sagemaker"]
                },
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training"],
                    "complexity_tags": ["simple"]
                },
                "connections": {
                    "alternatives": [],
                    "related": [{"id": "related_pipeline", "annotation": "Related functionality"}],
                    "used_in": []
                },
                "discovery_metadata": {
                    "estimated_runtime": "10 minutes",
                    "resource_requirements": "low",
                    "skill_level": "beginner"
                }
            },
            "related_pipeline": {
                "id": "related_pipeline",
                "title": "Related Pipeline",
                "description": "A related pipeline",
                "atomic_properties": {
                    "single_responsibility": "Related functionality",
                    "input_interface": ["data"],
                    "output_interface": ["result"],
                    "independence": "fully_self_contained",
                    "side_effects": "none"
                },
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "standard"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["pytorch"],
                    "task_tags": ["training"],
                    "complexity_tags": ["standard"]
                },
                "connections": {
                    "alternatives": [],
                    "related": [],
                    "used_in": []
                },
                "discovery_metadata": {
                    "estimated_runtime": "20 minutes",
                    "resource_requirements": "medium",
                    "skill_level": "intermediate"
                }
            }
        }
        
        registry_data = {
            "nodes": valid_nodes,
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["valid_pipeline"],
                    "pytorch": ["related_pipeline"]
                },
                "task_tags": {
                    "training": ["valid_pipeline", "related_pipeline"]
                },
                "complexity_tags": {
                    "simple": ["valid_pipeline"],
                    "standard": ["related_pipeline"]
                }
            }
        }
        
        registry.get_all_pipelines.return_value = list(valid_nodes.keys())
        registry.get_pipeline_node.side_effect = lambda pid: valid_nodes.get(pid)
        registry.get_pipeline_connections.side_effect = lambda pid: valid_nodes.get(pid, {}).get("connections", {})
        registry.load_registry.return_value = registry_data
        
        return registry
    
    @pytest.fixture
    def validator(self, mock_registry):
        """Create RegistryValidator instance with mock registry."""
        return RegistryValidator(mock_registry)
    
    def test_init(self, mock_registry):
        """Test RegistryValidator initialization."""
        validator = RegistryValidator(mock_registry)
        
        assert validator.registry == mock_registry
        assert validator._validation_cache == {}
        assert not validator._cache_valid
    
    def test_validate_atomicity_valid(self, validator):
        """Test atomicity validation with valid pipelines."""
        violations = validator.validate_atomicity()
        
        # Should have no violations for valid pipelines
        assert len(violations) == 0
    
    def test_validate_atomicity_missing_responsibility(self, validator):
        """Test atomicity validation with missing responsibility."""
        # Mock pipeline with missing responsibility
        invalid_node = {
            "id": "invalid_pipeline",
            "atomic_properties": {
                # Missing single_responsibility
                "input_interface": ["data"],
                "output_interface": ["model"]
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["invalid_pipeline"]
        validator.registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) > 0
        assert any(v.violation_type == "missing_responsibility" for v in violations)
    
    def test_validate_atomicity_verbose_responsibility(self, validator):
        """Test atomicity validation with verbose responsibility."""
        # Mock pipeline with verbose responsibility
        invalid_node = {
            "id": "verbose_pipeline",
            "atomic_properties": {
                "single_responsibility": "This is a very long and verbose single responsibility description that exceeds the recommended word limit",
                "input_interface": ["data"],
                "output_interface": ["model"]
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["verbose_pipeline"]
        validator.registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) > 0
        assert any(v.violation_type == "verbose_responsibility" for v in violations)
    
    def test_validate_atomicity_independence_contradiction(self, validator):
        """Test atomicity validation with independence contradiction."""
        # Mock pipeline claiming independence but having side effects
        invalid_node = {
            "id": "contradictory_pipeline",
            "atomic_properties": {
                "single_responsibility": "Test functionality",
                "input_interface": ["data"],
                "output_interface": ["model"],
                "independence": "fully_self_contained",
                "side_effects": "modifies_external_state"  # Contradicts independence
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["contradictory_pipeline"]
        validator.registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) > 0
        assert any(v.violation_type == "independence_contradiction" for v in violations)
    
    def test_validate_connections_valid(self, validator):
        """Test connection validation with valid connections."""
        errors = validator.validate_connections()
        
        # Should have no errors for valid connections
        assert len(errors) == 0
    
    def test_validate_connections_missing_target(self, validator):
        """Test connection validation with missing target."""
        # Mock pipeline with connection to non-existent target
        invalid_connections = {
            "alternatives": [],
            "related": [{"id": "nonexistent_pipeline", "annotation": "Missing target"}],
            "used_in": []
        }
        
        validator.registry.get_pipeline_connections.return_value = invalid_connections
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any(e.error_type == "missing_target" for e in errors)
    
    def test_validate_connections_missing_annotation(self, validator):
        """Test connection validation with missing annotation."""
        # Mock pipeline with connection missing annotation
        invalid_connections = {
            "alternatives": [],
            "related": [{"id": "related_pipeline", "annotation": ""}],  # Empty annotation
            "used_in": []
        }
        
        validator.registry.get_pipeline_connections.return_value = invalid_connections
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any(e.error_type == "missing_annotation" for e in errors)
    
    def test_validate_connections_self_reference(self, validator):
        """Test connection validation with self-reference."""
        # Mock pipeline connecting to itself
        invalid_connections = {
            "alternatives": [],
            "related": [{"id": "valid_pipeline", "annotation": "Self reference"}],  # Self-reference
            "used_in": []
        }
        
        validator.registry.get_pipeline_connections.return_value = invalid_connections
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any(e.error_type == "self_reference" for e in errors)
    
    def test_validate_connections_invalid_type(self, validator):
        """Test connection validation with invalid connection type."""
        # Mock registry data with invalid connection type
        registry_data = {
            "nodes": {
                "test_pipeline": {
                    "connections": {
                        "invalid_type": [{"id": "target", "annotation": "Invalid"}]
                    }
                }
            }
        }
        
        validator.registry.load_registry.return_value = registry_data
        validator.registry.get_all_pipelines.return_value = ["test_pipeline"]
        validator.registry.get_pipeline_connections.return_value = {
            "invalid_type": [{"id": "target", "annotation": "Invalid"}]
        }
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any(e.error_type == "invalid_type" for e in errors)
    
    def test_validate_metadata_completeness_valid(self, validator):
        """Test metadata completeness validation with valid metadata."""
        errors = validator.validate_metadata_completeness()
        
        # Should have no errors for complete metadata
        assert len(errors) == 0
    
    def test_validate_metadata_completeness_missing_fields(self, validator):
        """Test metadata completeness validation with missing fields."""
        # Mock pipeline with missing required fields
        incomplete_node = {
            "id": "incomplete_pipeline",
            "title": "Incomplete Pipeline"
            # Missing description, atomic_properties, zettelkasten_metadata
        }
        
        validator.registry.get_all_pipelines.return_value = ["incomplete_pipeline"]
        validator.registry.get_pipeline_node.return_value = incomplete_node
        
        errors = validator.validate_metadata_completeness()
        
        assert len(errors) > 0
        # Should have errors for missing required fields
        missing_fields = [e.field for e in errors]
        assert "description" in missing_fields
        assert "atomic_properties" in missing_fields
        assert "zettelkasten_metadata" in missing_fields
    
    def test_validate_metadata_completeness_missing_atomic_properties(self, validator):
        """Test metadata validation with missing atomic properties."""
        # Mock pipeline with incomplete atomic properties
        incomplete_node = {
            "id": "incomplete_pipeline",
            "title": "Incomplete Pipeline",
            "description": "Test description",
            "atomic_properties": {
                # Missing single_responsibility, input_interface, output_interface
            },
            "zettelkasten_metadata": {
                "framework": "xgboost",
                "complexity": "simple"
            },
            "multi_dimensional_tags": {
                "framework_tags": ["xgboost"]
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["incomplete_pipeline"]
        validator.registry.get_pipeline_node.return_value = incomplete_node
        
        errors = validator.validate_metadata_completeness()
        
        assert len(errors) > 0
        # Should have errors for missing atomic properties
        atomic_errors = [e for e in errors if "atomic_properties" in e.field]
        assert len(atomic_errors) > 0
    
    def test_validate_tag_consistency_valid(self, validator):
        """Test tag consistency validation with consistent tags."""
        errors = validator.validate_tag_consistency()
        
        # Should have no errors for consistent tags
        assert len(errors) == 0
    
    def test_validate_tag_consistency_framework_mismatch(self, validator):
        """Test tag consistency validation with framework mismatch."""
        # Mock pipeline with framework not in framework_tags
        inconsistent_node = {
            "id": "inconsistent_pipeline",
            "multi_dimensional_tags": {
                "framework_tags": ["pytorch"],  # Framework tags don't match metadata
                "task_tags": ["training"],
                "complexity_tags": ["simple"]
            },
            "zettelkasten_metadata": {
                "framework": "xgboost",  # Different from framework_tags
                "complexity": "simple"
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["inconsistent_pipeline"]
        validator.registry.get_pipeline_node.return_value = inconsistent_node
        
        errors = validator.validate_tag_consistency()
        
        assert len(errors) > 0
        assert any(e.tag_category == "framework_tags" for e in errors)
    
    def test_validate_tag_consistency_complexity_mismatch(self, validator):
        """Test tag consistency validation with complexity mismatch."""
        # Mock pipeline with complexity not in complexity_tags
        inconsistent_node = {
            "id": "inconsistent_pipeline",
            "multi_dimensional_tags": {
                "framework_tags": ["xgboost"],
                "task_tags": ["training"],
                "complexity_tags": ["simple"]  # Complexity tags don't match metadata
            },
            "zettelkasten_metadata": {
                "framework": "xgboost",
                "complexity": "advanced"  # Different from complexity_tags
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["inconsistent_pipeline"]
        validator.registry.get_pipeline_node.return_value = inconsistent_node
        
        errors = validator.validate_tag_consistency()
        
        assert len(errors) > 0
        assert any(e.tag_category == "complexity_tags" for e in errors)
    
    def test_validate_independence_claims_valid(self, validator):
        """Test independence claims validation with valid claims."""
        errors = validator.validate_independence_claims()
        
        # Should have no errors for valid independence claims
        assert len(errors) == 0
    
    def test_validate_independence_claims_side_effects_contradiction(self, validator):
        """Test independence validation with side effects contradiction."""
        # Mock pipeline claiming independence but having side effects
        contradictory_node = {
            "id": "contradictory_pipeline",
            "atomic_properties": {
                "independence": "fully_self_contained",
                "side_effects": "modifies_external_state",  # Contradicts independence
                "dependencies": ["sagemaker"],
                "input_interface": ["data"]
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["contradictory_pipeline"]
        validator.registry.get_pipeline_node.return_value = contradictory_node
        
        errors = validator.validate_independence_claims()
        
        assert len(errors) > 0
        assert any("side effects" in e.evidence for e in errors)
    
    def test_validate_independence_claims_many_dependencies(self, validator):
        """Test independence validation with many dependencies."""
        # Mock pipeline claiming independence but having many dependencies
        contradictory_node = {
            "id": "contradictory_pipeline",
            "atomic_properties": {
                "independence": "fully_self_contained",
                "side_effects": "none",
                "dependencies": ["dep1", "dep2", "dep3", "dep4", "dep5"],  # Too many dependencies
                "input_interface": ["data"]
            }
        }
        
        validator.registry.get_all_pipelines.return_value = ["contradictory_pipeline"]
        validator.registry.get_pipeline_node.return_value = contradictory_node
        
        errors = validator.validate_independence_claims()
        
        assert len(errors) > 0
        assert any("dependencies" in e.evidence for e in errors)
    
    def test_generate_validation_report_valid(self, validator):
        """Test generating validation report for valid registry."""
        report = validator.generate_validation_report()
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.total_issues == 0
        assert len(report.all_issues) == 0
    
    def test_generate_validation_report_invalid(self, validator):
        """Test generating validation report for invalid registry."""
        # Mock invalid pipeline
        invalid_node = {
            "id": "invalid_pipeline",
            "atomic_properties": {}  # Missing required fields
        }
        
        validator.registry.get_all_pipelines.return_value = ["invalid_pipeline"]
        validator.registry.get_pipeline_node.return_value = invalid_node
        
        report = validator.generate_validation_report()
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert report.total_issues > 0
        assert len(report.all_issues) > 0
        assert report.issues_by_severity[ValidationSeverity.ERROR] > 0
    
    def test_clear_cache(self, validator):
        """Test clearing validation cache."""
        # Populate cache
        validator._validation_cache = {"test": "data"}
        validator._cache_valid = True
        
        # Clear cache
        validator.clear_cache()
        
        assert validator._validation_cache == {}
        assert not validator._cache_valid
    
    @patch('cursus.pipeline_catalog.utils.registry_validator.logger')
    def test_error_handling(self, mock_logger, validator):
        """Test error handling and logging."""
        # Mock registry to raise exception
        validator.registry.get_all_pipelines.side_effect = Exception("Registry error")
        
        # Test method that should handle the error gracefully
        violations = validator.validate_atomicity()
        
        assert violations == []
        mock_logger.error.assert_called()
    
    def test_generate_validation_report_system_error(self, validator):
        """Test validation report generation with system error."""
        # Mock all validation methods to raise exceptions
        with patch.object(validator, 'validate_atomicity', side_effect=Exception("System error")):
            report = validator.generate_validation_report()
            
            assert report.is_valid is False
            assert report.total_issues == 1
            assert len(report.all_issues) == 1
            assert report.all_issues[0].severity == ValidationSeverity.ERROR
            assert report.all_issues[0].category == "system"
    
    def test_are_similar_tags(self, validator):
        """Test tag similarity detection."""
        # Test similar tags
        assert validator._are_similar_tags("training", "trainng")  # Missing 'i'
        assert validator._are_similar_tags("evaluation", "evalution")  # Missing 'a'
        assert validator._are_similar_tags("preprocessing", "preprocess")  # Common suffix
        
        # Test dissimilar tags
        assert not validator._are_similar_tags("training", "evaluation")
        assert not validator._are_similar_tags("xgboost", "pytorch")
        assert not validator._are_similar_tags("a", "xyz")
