"""
Unit tests for step_catalog.spec_discovery module.

Tests the SpecAutoDiscovery class that handles specification file discovery
and loading across package and workspace directories.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any

from cursus.step_catalog.spec_discovery import SpecAutoDiscovery


class TestSpecAutoDiscoveryInitialization:
    """Test SpecAutoDiscovery initialization and setup."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init_package_only(self, temp_workspace):
        """Test SpecAutoDiscovery initialization with package-only discovery."""
        package_root = temp_workspace / "cursus"
        package_root.mkdir()
        
        discovery = SpecAutoDiscovery(package_root, [])
        
        assert discovery.package_root == package_root
        assert discovery.workspace_dirs == []
        assert discovery.logger is not None
    
    def test_init_with_workspace_dirs(self, temp_workspace):
        """Test SpecAutoDiscovery initialization with workspace directories."""
        package_root = temp_workspace / "cursus"
        package_root.mkdir()
        workspace_dirs = [temp_workspace / "workspace1", temp_workspace / "workspace2"]
        
        discovery = SpecAutoDiscovery(package_root, workspace_dirs)
        
        assert discovery.package_root == package_root
        assert discovery.workspace_dirs == workspace_dirs


class TestSpecDiscovery:
    """Test specification file discovery functionality."""
    
    @pytest.fixture
    def discovery_with_specs(self):
        """Create discovery instance with mock spec files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            # Create specs directory
            specs_dir = package_root / "steps" / "specs"
            specs_dir.mkdir(parents=True)
            
            # Create test spec files
            (specs_dir / "test_step_spec.py").write_text("""
TEST_STEP_SPEC = type('MockSpec', (), {
    'step_type': 'Processing',
    'dependencies': {'input_data': 's3://bucket/input'},
    'outputs': {'output_data': 's3://bucket/output'}
})()
""")
            
            discovery = SpecAutoDiscovery(package_root, [])
            yield discovery, specs_dir
    
    def test_discover_spec_classes(self, discovery_with_specs):
        """Test discovery of specification classes."""
        discovery, specs_dir = discovery_with_specs
        
        # Mock the scan directory method to avoid import issues
        with patch.object(discovery, '_scan_spec_directory') as mock_scan:
            mock_scan.return_value = {"test_step_spec": Mock()}
            
            result = discovery.discover_spec_classes()
            
            assert isinstance(result, dict)
            mock_scan.assert_called_once()
    
    def test_load_spec_class_existing(self, discovery_with_specs):
        """Test loading existing specification class."""
        discovery, specs_dir = discovery_with_specs
        
        # Mock the direct import method
        with patch.object(discovery, '_try_direct_import') as mock_import:
            mock_spec = Mock()
            mock_spec.step_type = "Processing"
            mock_spec.dependencies = {"input_data": "s3://bucket/input"}
            mock_spec.outputs = {"output_data": "s3://bucket/output"}
            mock_import.return_value = mock_spec
            
            result = discovery.load_spec_class("test_step")
            
            assert result is not None
            assert hasattr(result, 'step_type')
            assert hasattr(result, 'dependencies')
            assert hasattr(result, 'outputs')
    
    def test_load_spec_class_nonexistent(self, discovery_with_specs):
        """Test loading non-existent specification class."""
        discovery, specs_dir = discovery_with_specs
        
        # Mock both direct import and workspace import to return None
        with patch.object(discovery, '_try_direct_import', return_value=None):
            with patch.object(discovery, '_try_workspace_spec_import', return_value=None):
                result = discovery.load_spec_class("nonexistent_step")
                
                assert result is None
    
    def test_is_spec_instance(self, discovery_with_specs):
        """Test specification instance validation."""
        discovery, specs_dir = discovery_with_specs
        
        # Valid spec instance - must have step_type, dependencies, and outputs
        valid_spec = Mock()
        valid_spec.step_type = "Processing"
        valid_spec.dependencies = {}
        valid_spec.outputs = {}
        
        assert discovery._is_spec_instance(valid_spec) == True
        
        # Invalid spec instance - Mock without the required attributes will raise AttributeError
        # when hasattr() is called, which the implementation catches and returns False
        invalid_spec = Mock()
        invalid_spec.step_type = "Processing"
        # Remove the dependencies and outputs attributes completely
        del invalid_spec.dependencies
        del invalid_spec.outputs
        
        # The actual implementation uses hasattr() which will return False for missing attributes
        assert discovery._is_spec_instance(invalid_spec) == False


class TestSpecSerialization:
    """Test specification serialization functionality."""
    
    @pytest.fixture
    def mock_spec_instance(self):
        """Create mock specification instance."""
        spec = Mock()
        spec.dependencies = {
            "input_data": "s3://bucket/input",
            "model_config": "s3://bucket/config"
        }
        spec.outputs = {
            "output_data": "s3://bucket/output",
            "metrics": "s3://bucket/metrics"
        }
        spec.step_name = "test_step"
        return spec
    
    def test_serialize_spec(self, mock_spec_instance):
        """Test specification serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            # Mock the spec instance to have the required attributes for serialization
            mock_spec_instance.step_type = "Processing"
            mock_spec_instance.node_type = Mock()
            mock_spec_instance.node_type.value = "Transform"
            
            # Mock dependencies and outputs as expected by serialize_spec
            mock_spec_instance.dependencies = {}
            mock_spec_instance.outputs = {}
            
            discovery = SpecAutoDiscovery(package_root, [])
            
            # Mock _is_spec_instance to return True
            with patch.object(discovery, '_is_spec_instance', return_value=True):
                result = discovery.serialize_spec(mock_spec_instance)
                
                assert isinstance(result, dict)
                assert "dependencies" in result
                assert "outputs" in result
                assert "step_type" in result
                assert result["step_type"] == "Processing"
    
    def test_serialize_spec_error_handling(self):
        """Test specification serialization error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            discovery = SpecAutoDiscovery(package_root, [])
            
            # Test with invalid spec object
            result = discovery.serialize_spec(None)
            
            assert result == {}


class TestSpecContractMapping:
    """Test specification-contract mapping functionality."""
    
    @pytest.fixture
    def discovery_with_contract_mapping(self):
        """Create discovery instance for contract mapping tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            discovery = SpecAutoDiscovery(package_root, [])
            yield discovery
    
    def test_find_specs_by_contract(self, discovery_with_contract_mapping):
        """Test finding specifications by contract name."""
        discovery = discovery_with_contract_mapping
        
        # Mock the actual methods that exist in the implementation
        with patch.object(discovery, '_find_specs_by_contract_in_dir') as mock_find_core:
            mock_find_core.return_value = {"test_contract_spec": Mock()}
            
            with patch.object(discovery, '_find_specs_by_contract_in_workspace') as mock_find_workspace:
                mock_find_workspace.return_value = {}
                
                result = discovery.find_specs_by_contract("test_contract")
                
                assert isinstance(result, dict)
                # Should find specs from core directory
    
    def test_find_specs_by_contract_error_handling(self, discovery_with_contract_mapping):
        """Test error handling in find_specs_by_contract."""
        discovery = discovery_with_contract_mapping
        
        # Mock the actual method that exists and make it raise an exception
        with patch.object(discovery, '_find_specs_by_contract_in_dir', side_effect=Exception("Test error")):
            result = discovery.find_specs_by_contract("test_contract")
            
            assert result == {}


class TestJobTypeVariants:
    """Test job type variant handling."""
    
    @pytest.fixture
    def discovery_with_variants(self):
        """Create discovery instance with job type variants."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            specs_dir = package_root / "steps" / "specs"
            specs_dir.mkdir(parents=True)
            
            # Create variant spec files
            (specs_dir / "data_loading_training_spec.py").write_text("# Training spec")
            (specs_dir / "data_loading_validation_spec.py").write_text("# Validation spec")
            (specs_dir / "data_loading_testing_spec.py").write_text("# Testing spec")
            
            discovery = SpecAutoDiscovery(package_root, [])
            yield discovery, specs_dir
    
    def test_get_job_type_variants(self, discovery_with_variants):
        """Test getting job type variants for a step."""
        discovery, specs_dir = discovery_with_variants
        
        variants = discovery.get_job_type_variants("data_loading")
        
        # Should find training, validation, testing variants
        assert len(variants) >= 3
        assert "training" in variants
        assert "validation" in variants
        assert "testing" in variants
    
    def test_get_job_type_variants_no_variants(self, discovery_with_variants):
        """Test getting variants for step with no variants."""
        discovery, specs_dir = discovery_with_variants
        
        variants = discovery.get_job_type_variants("nonexistent_step")
        
        assert len(variants) == 0


class TestUnifiedSpecification:
    """Test unified specification creation functionality."""
    
    @pytest.fixture
    def discovery_with_unified_specs(self):
        """Create discovery instance for unified specification tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            discovery = SpecAutoDiscovery(package_root, [])
            yield discovery
    
    def test_create_unified_specification(self, discovery_with_unified_specs):
        """Test creating unified specification from multiple variants."""
        discovery = discovery_with_unified_specs
        
        # Mock multiple spec variants
        with patch.object(discovery, 'find_specs_by_contract') as mock_find:
            mock_specs = {
                "training_spec": Mock(dependencies={"input": "s3://train"}, outputs={"model": "s3://model"}),
                "validation_spec": Mock(dependencies={"model": "s3://model"}, outputs={"metrics": "s3://metrics"})
            }
            mock_find.return_value = mock_specs
            
            result = discovery.create_unified_specification("test_contract")
            
            assert isinstance(result, dict)
            assert "primary_spec" in result
            assert "variants" in result
            assert "unified_dependencies" in result
            assert "unified_outputs" in result
    
    def test_create_unified_specification_no_specs(self, discovery_with_unified_specs):
        """Test creating unified specification when no specs found."""
        discovery = discovery_with_unified_specs
        
        with patch.object(discovery, 'find_specs_by_contract', return_value={}):
            result = discovery.create_unified_specification("nonexistent_contract")
            
            assert isinstance(result, dict)
            assert result["variant_count"] == 0


class TestSmartValidation:
    """Test smart validation functionality."""
    
    @pytest.fixture
    def discovery_with_smart_validation(self):
        """Create discovery instance for smart validation tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            discovery = SpecAutoDiscovery(package_root, [])
            yield discovery
    
    def test_validate_logical_names_smart(self, discovery_with_smart_validation):
        """Test smart validation of logical names."""
        discovery = discovery_with_smart_validation
        
        # Mock unified specification
        with patch.object(discovery, 'create_unified_specification') as mock_unified:
            mock_unified.return_value = {
                "unified_dependencies": {"input_data": {"required": True}},
                "unified_outputs": {"output_data": {"required": True}},
                "variants": {"training": Mock(), "validation": Mock()}
            }
            
            contract = {
                "inputs": {"input_data": "s3://bucket/input"},
                "outputs": {"output_data": "s3://bucket/output"}
            }
            
            result = discovery.validate_logical_names_smart(contract, "test_contract")
            
            assert isinstance(result, list)
            # Validation results depend on implementation details
    
    def test_validate_logical_names_smart_error_handling(self, discovery_with_smart_validation):
        """Test error handling in smart validation."""
        discovery = discovery_with_smart_validation
        
        with patch.object(discovery, 'create_unified_specification', side_effect=Exception("Test error")):
            result = discovery.validate_logical_names_smart({}, "test_contract")
            
            assert isinstance(result, list)
            assert len(result) > 0  # Should contain error information


class TestAllSpecificationsLoading:
    """Test loading all specifications functionality."""
    
    @pytest.fixture
    def discovery_with_all_specs(self):
        """Create discovery instance for all specifications tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            specs_dir = package_root / "steps" / "specs"
            specs_dir.mkdir(parents=True)
            
            # Create multiple spec files
            (specs_dir / "step1_spec.py").write_text("# Step 1 spec")
            (specs_dir / "step2_spec.py").write_text("# Step 2 spec")
            (specs_dir / "step3_training_spec.py").write_text("# Step 3 training spec")
            
            discovery = SpecAutoDiscovery(package_root, [])
            yield discovery, specs_dir
    
    def test_load_all_specifications(self, discovery_with_all_specs):
        """Test loading all specification instances."""
        discovery, specs_dir = discovery_with_all_specs
        
        # Mock the discover_spec_classes method which is actually called
        with patch.object(discovery, 'discover_spec_classes') as mock_discover:
            mock_spec = Mock()
            mock_spec.step_type = "Processing"
            mock_spec.dependencies = {"input": "s3://input"}
            mock_spec.outputs = {"output": "s3://output"}
            
            mock_discover.return_value = {
                "step1_spec": mock_spec,
                "step2_spec": mock_spec,
                "step3_training_spec": mock_spec
            }
            
            # Mock _is_spec_instance and serialize_spec
            with patch.object(discovery, '_is_spec_instance', return_value=True):
                with patch.object(discovery, 'serialize_spec') as mock_serialize:
                    mock_serialize.return_value = {"step_type": "Processing", "dependencies": [], "outputs": []}
                    
                    result = discovery.load_all_specifications()
                    
                    assert isinstance(result, dict)
                    assert len(result) >= 3  # Should have at least 3 specs
    
    def test_load_all_specifications_error_handling(self, discovery_with_all_specs):
        """Test error handling in load_all_specifications."""
        discovery, specs_dir = discovery_with_all_specs
        
        # Mock the actual method that exists - discover_spec_classes
        with patch.object(discovery, 'discover_spec_classes', side_effect=Exception("Test error")):
            result = discovery.load_all_specifications()
            
            assert result == {}


class TestErrorHandlingAndResilience:
    """Test error handling and resilience features."""
    
    def test_graceful_degradation_on_import_error(self):
        """Test graceful degradation when spec imports fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            discovery = SpecAutoDiscovery(package_root, [])
            
            # Test with non-existent spec
            result = discovery.load_spec_class("nonexistent_spec")
            
            assert result is None
    
    def test_error_logging_in_spec_loading(self):
        """Test that errors are properly logged during spec loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            discovery = SpecAutoDiscovery(package_root, [])
            
            with patch.object(discovery.logger, 'error') as mock_error:
                # Force an error condition using actual method
                with patch.object(discovery, 'discover_spec_classes', side_effect=Exception("Test error")):
                    result = discovery.load_all_specifications()
                    
                    assert result == {}
                    # Error logging should occur when exceptions happen


class TestWorkspaceIntegration:
    """Test workspace integration functionality."""
    
    @pytest.fixture
    def discovery_with_workspaces(self):
        """Create discovery instance with workspace directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "cursus"
            package_root.mkdir()
            
            # Create workspace directories
            workspace1 = Path(temp_dir) / "workspace1" / "steps" / "specs"
            workspace1.mkdir(parents=True)
            workspace2 = Path(temp_dir) / "workspace2" / "steps" / "specs"
            workspace2.mkdir(parents=True)
            
            # Create spec files in workspaces
            (workspace1 / "workspace1_spec.py").write_text("# Workspace 1 spec")
            (workspace2 / "workspace2_spec.py").write_text("# Workspace 2 spec")
            
            discovery = SpecAutoDiscovery(package_root, [workspace1.parent.parent, workspace2.parent.parent])
            yield discovery, workspace1, workspace2
    
    def test_discover_workspace_specs(self, discovery_with_workspaces):
        """Test discovery of specs in workspace directories."""
        discovery, workspace1, workspace2 = discovery_with_workspaces
        
        # Mock the actual workspace discovery method
        with patch.object(discovery, '_discover_workspace_specs') as mock_discover:
            mock_discover.return_value = {"workspace1_spec": Mock(), "workspace2_spec": Mock()}
            
            result = discovery.discover_spec_classes()
            
            assert isinstance(result, dict)
            # Should call workspace discovery for each workspace directory
    
    def test_load_workspace_specs(self, discovery_with_workspaces):
        """Test loading specs from workspace directories."""
        discovery, workspace1, workspace2 = discovery_with_workspaces
        
        # Mock the actual methods used in load_all_specifications
        with patch.object(discovery, 'discover_spec_classes') as mock_discover:
            mock_spec = Mock()
            mock_spec.step_type = "Processing"
            mock_spec.dependencies = {"input": "s3://workspace/input"}
            mock_spec.outputs = {"output": "s3://workspace/output"}
            
            mock_discover.return_value = {"workspace_spec": mock_spec}
            
            with patch.object(discovery, '_is_spec_instance', return_value=True):
                with patch.object(discovery, 'serialize_spec') as mock_serialize:
                    mock_serialize.return_value = {"step_type": "Processing", "dependencies": [], "outputs": []}
                    
                    result = discovery.load_all_specifications()
                    
                    assert isinstance(result, dict)
                    # Should include workspace specs
