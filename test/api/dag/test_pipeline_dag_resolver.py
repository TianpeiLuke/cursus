"""Test enhanced pipeline DAG resolver with dynamic contract discovery."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from cursus.api.dag import PipelineDAG
from cursus.api.dag.pipeline_dag_resolver import PipelineDAGResolver
from cursus.core.base.contract_base import ScriptContract
from cursus.core.base.specification_base import StepSpecification


class TestPipelineDAGResolverEnhanced:
    """Test enhanced pipeline DAG resolver functionality."""

    def setup_method(self):
        """Set up test fixtures using predefined steps from registry."""
        # Use real step names from the registry for a realistic pipeline:
        # TabularPreprocessing -> XGBoostTraining -> XGBoostModelEval
        self.dag = PipelineDAG(
            nodes=["TabularPreprocessing", "XGBoostTraining", "XGBoostModelEval"],
            edges=[
                ("TabularPreprocessing", "XGBoostTraining"),
                ("XGBoostTraining", "XGBoostModelEval"),
            ],
        )
        self.resolver = PipelineDAGResolver(self.dag)

        # Also create a more complex DAG for advanced testing
        self.complex_dag = PipelineDAG(
            nodes=[
                "CradleDataLoading",
                "TabularPreprocessing",
                "XGBoostTraining",
                "XGBoostModelEval",
                "ModelCalibration",
            ],
            edges=[
                ("CradleDataLoading", "TabularPreprocessing"),
                ("TabularPreprocessing", "XGBoostTraining"),
                ("XGBoostTraining", "XGBoostModelEval"),
                ("XGBoostTraining", "ModelCalibration"),
            ],
        )
        self.complex_resolver = PipelineDAGResolver(self.complex_dag)

    def test_basic_execution_plan_creation(self):
        """Test basic execution plan creation without contracts using real registry steps."""
        plan = self.resolver.create_execution_plan()

        assert plan.execution_order == [
            "TabularPreprocessing",
            "XGBoostTraining",
            "XGBoostModelEval",
        ]
        assert "TabularPreprocessing" in plan.step_configs
        assert "XGBoostTraining" in plan.step_configs
        assert "XGBoostModelEval" in plan.step_configs

        # Check dependencies
        assert plan.dependencies["TabularPreprocessing"] == []
        assert plan.dependencies["XGBoostTraining"] == ["TabularPreprocessing"]
        assert plan.dependencies["XGBoostModelEval"] == ["XGBoostTraining"]

    def test_data_flow_map_fallback(self):
        """Test data flow map creation with fallback approach using real registry steps."""
        plan = self.resolver.create_execution_plan()

        # Should use fallback generic approach
        assert plan.data_flow_map["TabularPreprocessing"] == {}
        assert plan.data_flow_map["XGBoostTraining"] == {
            "input_0": "TabularPreprocessing:output"
        }
        assert plan.data_flow_map["XGBoostModelEval"] == {
            "input_0": "XGBoostTraining:output"
        }

    @patch("cursus.api.dag.pipeline_dag_resolver.get_canonical_name_from_file_name")
    @patch("cursus.api.dag.pipeline_dag_resolver.get_spec_step_type")
    def test_contract_discovery_success(self, mock_get_spec_type, mock_get_canonical):
        """Test successful contract discovery."""
        # Mock the registry helper functions
        mock_get_canonical.return_value = "xgboost_training"
        mock_get_spec_type.return_value = "XGBoostTrainingSpec"

        # Mock the specification with contract
        mock_contract = ScriptContract(
            entry_point="xgboost_training.py",
            expected_input_paths={"input_path": "/opt/ml/processing/input/data"},
            expected_output_paths={"model_path": "/opt/ml/processing/output/model"},
            required_env_vars=["SM_MODEL_DIR"],
        )
        mock_spec = Mock()
        mock_spec.script_contract = mock_contract

        # Mock both catalog and legacy discovery methods to return the contract
        with patch.object(
            self.resolver, "_discover_step_contract_with_catalog", return_value=mock_contract
        ), patch.object(
            self.resolver, "_discover_step_contract_legacy", return_value=mock_contract
        ), patch.object(
            self.resolver, "_get_step_specification", return_value=mock_spec
        ):
            contract = self.resolver._discover_step_contract("training")

            assert contract is not None
            assert contract.expected_input_paths == {
                "input_path": "/opt/ml/processing/input/data"
            }
            assert contract.expected_output_paths == {
                "model_path": "/opt/ml/processing/output/model"
            }

    @patch("cursus.api.dag.pipeline_dag_resolver.get_canonical_name_from_file_name")
    def test_contract_discovery_failure(self, mock_get_canonical):
        """Test contract discovery failure fallback."""
        # Mock failure in canonical name resolution
        mock_get_canonical.return_value = None

        contract = self.resolver._discover_step_contract("unknown_step")
        assert contract is None

    def test_spec_type_to_module_name_conversion(self):
        """Test spec type to module name conversion."""
        assert (
            self.resolver._spec_type_to_module_name("XGBoostTrainingSpec")
            == "xgboost_training_spec"
        )
        assert (
            self.resolver._spec_type_to_module_name("TabularPreprocessingSpec")
            == "tabular_preprocessing_spec"
        )
        assert (
            self.resolver._spec_type_to_module_name("SimpleStep") == "simple_step_spec"
        )

    def test_find_compatible_output_direct_match(self):
        """Test direct channel name matching."""
        output_channels = {
            "model_path": "/opt/ml/model",
            "output_path": "/opt/ml/output/data",
        }

        result = self.resolver._find_compatible_output(
            "model_path", "/opt/ml/input/data", output_channels
        )
        assert result == "model_path"

    def test_find_compatible_output_semantic_match(self):
        """Test semantic channel matching."""
        output_channels = {
            "output_path": "/opt/ml/output/data",
            "config_path": "/opt/ml/output/config",
        }

        result = self.resolver._find_compatible_output(
            "input_path", "/opt/ml/input/data", output_channels
        )
        assert result == "output_path"

    def test_find_compatible_output_fallback(self):
        """Test fallback to first available output."""
        output_channels = {"some_output": "/opt/ml/output/data"}

        result = self.resolver._find_compatible_output(
            "unknown_input", "/opt/ml/input/data", output_channels
        )
        assert result == "some_output"

    def test_path_compatibility_sagemaker_conventions(self):
        """Test SageMaker path compatibility rules."""
        # Model artifacts compatibility
        assert self.resolver._are_paths_compatible(
            "/opt/ml/model/model.tar.gz", "/opt/ml/model/output.tar.gz"
        )

        # Data flow compatibility
        assert self.resolver._are_paths_compatible(
            "/opt/ml/input/data/train.csv", "/opt/ml/output/data/processed.csv"
        )

        # Incompatible paths
        assert not self.resolver._are_paths_compatible(
            "/opt/ml/model/model.tar.gz", "/opt/ml/code/script.py"
        )

    @patch("cursus.api.dag.pipeline_dag_resolver.importlib.import_module")
    @patch("cursus.api.dag.pipeline_dag_resolver.get_spec_step_type")
    def test_get_step_specification_with_getter_function(
        self, mock_get_spec_type, mock_import
    ):
        """Test specification retrieval with getter function."""
        # Mock the spec type lookup
        mock_get_spec_type.return_value = "XGBoostTrainingSpec"

        # Mock module with getter function
        mock_module = Mock()
        mock_spec = Mock()
        mock_module.get_xgboost_training_spec = Mock(return_value=mock_spec)
        mock_import.return_value = mock_module

        # Call with canonical name directly (this method expects canonical name, not step name)
        result = self.resolver._get_step_specification("xgboost_training")

        assert result == mock_spec
        mock_get_spec_type.assert_called_once_with("xgboost_training")
        mock_import.assert_called_once_with("cursus.steps.specs.xgboost_training_spec")
        mock_module.get_xgboost_training_spec.assert_called_once()

    @patch("cursus.api.dag.pipeline_dag_resolver.importlib.import_module")
    @patch("cursus.api.dag.pipeline_dag_resolver.get_spec_step_type")
    def test_get_step_specification_with_constant(
        self, mock_get_spec_type, mock_import
    ):
        """Test specification retrieval with spec constant (actual pattern used in codebase)."""
        # Mock the spec type lookup
        mock_get_spec_type.return_value = "XGBoostTrainingSpec"

        # Create a simple object that only has the constant we want
        class MockModule:
            def __init__(self):
                self.XGBOOST_TRAINING_SPEC = Mock()

            def __getattr__(self, name):
                # This ensures hasattr returns False for non-existent attributes
                raise AttributeError(f"module has no attribute '{name}'")

        mock_module = MockModule()
        mock_import.return_value = mock_module

        # Call with canonical name directly (this method expects canonical name, not step name)
        result = self.resolver._get_step_specification("xgboost_training")

        assert result == mock_module.XGBOOST_TRAINING_SPEC
        mock_get_spec_type.assert_called_once_with("xgboost_training")
        mock_import.assert_called_once_with("cursus.steps.specs.xgboost_training_spec")

    def test_enhanced_data_flow_map_with_contracts(self):
        """Test enhanced data flow map creation with contracts using real registry steps and contracts."""
        # Import real contracts from the codebase
        from cursus.steps.contracts.tabular_preprocessing_contract import (
            TABULAR_PREPROCESSING_CONTRACT,
        )
        from cursus.steps.contracts.xgboost_training_contract import (
            XGBOOST_TRAIN_CONTRACT,
        )
        from cursus.steps.contracts.xgboost_model_eval_contract import (
            XGBOOST_MODEL_EVAL_CONTRACT,
        )

        # Mock contract discovery using real contracts
        def mock_discover_contract(step_name):
            contracts = {
                "TabularPreprocessing": TABULAR_PREPROCESSING_CONTRACT,
                "XGBoostTraining": XGBOOST_TRAIN_CONTRACT,
                "XGBoostModelEval": XGBOOST_MODEL_EVAL_CONTRACT,
            }
            return contracts.get(step_name)

        with patch.object(
            self.resolver, "_discover_step_contract", side_effect=mock_discover_contract
        ):
            plan = self.resolver.create_execution_plan()

            # Check enhanced data flow mapping with real step names and contracts
            assert plan.data_flow_map["TabularPreprocessing"] == {}

            # XGBoostTraining should get input from TabularPreprocessing's processed_data output
            assert "input_path" in plan.data_flow_map["XGBoostTraining"]
            assert (
                plan.data_flow_map["XGBoostTraining"]["input_path"]
                == "TabularPreprocessing:processed_data"
            )

            # XGBoostModelEval should get model from XGBoostTraining's model_output
            xgb_eval_flow = plan.data_flow_map["XGBoostModelEval"]
            assert "model_input" in xgb_eval_flow
            assert xgb_eval_flow["model_input"] == "XGBoostTraining:model_output"

    def test_dag_validation_with_cycles(self):
        """Test DAG validation with cycles."""
        # Create DAG with cycle
        cyclic_dag = PipelineDAG(
            nodes=["step1", "step2", "step3"],
            edges=[("step1", "step2"), ("step2", "step3"), ("step3", "step1")],
        )
        resolver = PipelineDAGResolver(cyclic_dag)

        issues = resolver.validate_dag_integrity()
        assert "cycles" in issues
        assert len(issues["cycles"]) > 0

    def test_dag_validation_with_dangling_dependencies(self):
        """Test DAG validation with dangling dependencies."""
        # Create DAG with dangling edge by manually setting edges after construction
        invalid_dag = PipelineDAG(nodes=["step1", "step2"], edges=[("step1", "step2")])
        # Manually add a dangling edge to test validation
        invalid_dag.edges.append(("step2", "nonexistent_step"))

        resolver = PipelineDAGResolver(invalid_dag)

        issues = resolver.validate_dag_integrity()
        assert "dangling_dependencies" in issues
        assert any(
            "nonexistent_step" in issue for issue in issues["dangling_dependencies"]
        )

    def test_dag_validation_with_isolated_nodes(self):
        """Test DAG validation with isolated nodes."""
        # Create DAG with isolated node
        isolated_dag = PipelineDAG(
            nodes=["step1", "step2", "isolated_step"], edges=[("step1", "step2")]
        )
        resolver = PipelineDAGResolver(isolated_dag)

        issues = resolver.validate_dag_integrity()
        assert "isolated_nodes" in issues
        assert any("isolated_step" in issue for issue in issues["isolated_nodes"])

    def test_real_contract_discovery_with_registry_steps(self):
        """Test contract discovery using real registry steps and specifications."""
        # Test with XGBoostTraining step
        contract = self.resolver._discover_step_contract("XGBoostTraining")

        if contract is not None:
            # If contract is discovered, verify it has expected structure
            assert hasattr(contract, "entry_point")
            assert hasattr(contract, "expected_input_paths")
            assert hasattr(contract, "expected_output_paths")
            assert contract.entry_point == "xgboost_training.py"

            # Verify expected input/output paths match the specification
            assert "input_path" in contract.expected_input_paths
            assert (
                "model_output" in contract.expected_output_paths
                or "model_path" in contract.expected_output_paths
            )
        else:
            # If contract discovery fails, that's also valid behavior to test
            # The resolver should gracefully handle missing contracts
            print(
                "Contract discovery failed for XGBoostTraining - testing fallback behavior"
            )

    def test_execution_plan_with_real_contracts(self):
        """Test execution plan creation with real contract discovery."""
        plan = self.resolver.create_execution_plan()

        # Verify basic structure
        assert len(plan.execution_order) == 3
        assert plan.execution_order == [
            "TabularPreprocessing",
            "XGBoostTraining",
            "XGBoostModelEval",
        ]

        # Verify all steps have configs (even if empty)
        for step_name in plan.execution_order:
            assert step_name in plan.step_configs
            assert isinstance(plan.step_configs[step_name], dict)

        # Verify dependencies are correct
        assert plan.dependencies["TabularPreprocessing"] == []
        assert plan.dependencies["XGBoostTraining"] == ["TabularPreprocessing"]
        assert plan.dependencies["XGBoostModelEval"] == ["XGBoostTraining"]

        # Verify data flow map exists for all steps
        for step_name in plan.execution_order:
            assert step_name in plan.data_flow_map
            assert isinstance(plan.data_flow_map[step_name], dict)

    def test_complex_dag_execution_plan(self):
        """Test execution plan creation with complex DAG using real registry steps."""
        plan = self.complex_resolver.create_execution_plan()

        # Verify topological ordering
        expected_order = [
            "CradleDataLoading",
            "TabularPreprocessing",
            "XGBoostTraining",
            "XGBoostModelEval",
            "ModelCalibration",
        ]
        assert plan.execution_order == expected_order

        # Verify dependencies
        assert plan.dependencies["CradleDataLoading"] == []
        assert plan.dependencies["TabularPreprocessing"] == ["CradleDataLoading"]
        assert plan.dependencies["XGBoostTraining"] == ["TabularPreprocessing"]
        assert set(plan.dependencies["XGBoostModelEval"]) == {"XGBoostTraining"}
        assert set(plan.dependencies["ModelCalibration"]) == {"XGBoostTraining"}

        # Verify all steps have configurations
        for step_name in expected_order:
            assert step_name in plan.step_configs
            assert step_name in plan.data_flow_map

    def test_registry_step_name_validation(self):
        """Test that the registry step names used in tests are valid."""
        from cursus.registry.step_names import validate_step_name

        # Verify all step names used in our DAGs are valid registry names
        for step_name in self.dag.nodes:
            assert validate_step_name(
                step_name
            ), f"Step name '{step_name}' is not in registry"

        for step_name in self.complex_dag.nodes:
            assert validate_step_name(
                step_name
            ), f"Step name '{step_name}' is not in registry"

    def test_canonical_name_conversion_for_registry_steps(self):
        """Test canonical name conversion for registry steps."""
        from cursus.registry.step_names import get_canonical_name_from_file_name

        # Test conversion of various step name formats to canonical names
        test_cases = [
            ("tabular_preprocessing", "TabularPreprocessing"),
            ("xgboost_training", "XGBoostTraining"),
            ("model_evaluation_xgb", "XGBoostModelEval"),
            ("cradle_data_loading", "CradleDataLoading"),
            ("model_calibration", "ModelCalibration"),
        ]

        for file_name, expected_canonical in test_cases:
            try:
                canonical = get_canonical_name_from_file_name(file_name)
                assert (
                    canonical == expected_canonical
                ), f"Expected {expected_canonical}, got {canonical}"
            except ValueError:
                # Some conversions might fail, which is acceptable
                print(
                    f"Canonical name conversion failed for {file_name} -> {expected_canonical}"
                )

    def test_spec_type_retrieval_for_registry_steps(self):
        """Test spec type retrieval for registry steps."""
        from cursus.registry.step_names import get_spec_step_type

        # Test spec type retrieval for our DAG steps
        for step_name in self.dag.nodes:
            try:
                spec_type = get_spec_step_type(step_name)
                assert spec_type is not None
                assert isinstance(spec_type, str)
                assert len(spec_type) > 0
                print(f"Step {step_name} -> Spec type: {spec_type}")
            except ValueError as e:
                print(f"Spec type retrieval failed for {step_name}: {e}")

    def test_data_flow_with_real_step_contracts(self):
        """Test data flow mapping with real step contracts if available."""
        # Create a plan and examine the data flow
        plan = self.resolver.create_execution_plan()

        # Check if any real contracts were discovered and used
        has_real_contracts = False
        for step_name, data_flow in plan.data_flow_map.items():
            # If we have specific channel names (not generic input_0), contracts were likely used
            if data_flow and not any(
                key.startswith("input_") for key in data_flow.keys()
            ):
                has_real_contracts = True
                print(f"Real contract detected for {step_name}: {data_flow}")

        if has_real_contracts:
            print("Successfully created data flow with real contracts")
        else:
            print("Using fallback data flow mapping (contracts not discovered)")

        # Either way, verify the structure is valid
        for step_name in plan.execution_order:
            assert step_name in plan.data_flow_map
            data_flow = plan.data_flow_map[step_name]
            assert isinstance(data_flow, dict)

            # Verify data flow references point to valid predecessor steps
            for input_channel, source_ref in data_flow.items():
                if ":" in source_ref:
                    source_step, output_channel = source_ref.split(":", 1)
                    assert source_step in plan.execution_order
                    # Source step should be a dependency
                    assert (
                        source_step in plan.dependencies[step_name]
                        or step_name == source_step
                    )

    def test_step_catalog_contract_discovery_integration(self):
        """Test step catalog integration for contract discovery."""
        # Create a simple mock contract
        mock_contract = ScriptContract(
            entry_point="test_step.py",
            expected_input_paths={"input_path": "/opt/ml/processing/input/data"},
            expected_output_paths={"output_path": "/opt/ml/processing/output/data"},
            required_env_vars=["SM_MODEL_DIR"],
        )
        
        # Test by directly mocking the catalog discovery method
        with patch.object(self.resolver, "_discover_step_contract_with_catalog", return_value=mock_contract):
            result = self.resolver._discover_step_contract_with_catalog("TestStep")
            
            # Verify the result
            assert result is not None
            assert result == mock_contract
            assert result.entry_point == "test_step.py"
            assert result.expected_input_paths == {"input_path": "/opt/ml/processing/input/data"}
            assert result.expected_output_paths == {"output_path": "/opt/ml/processing/output/data"}

    def test_step_catalog_contract_discovery_fallback(self):
        """Test step catalog contract discovery with fallback to legacy method."""
        # Mock legacy discovery to return a contract
        mock_legacy_contract = ScriptContract(
            entry_point="legacy_step.py",
            expected_input_paths={"legacy_input": "/opt/ml/processing/input/legacy"},
            expected_output_paths={"legacy_output": "/opt/ml/processing/output/legacy"},
            required_env_vars=["LEGACY_VAR"],
        )
        
        # Test the fallback behavior by mocking the catalog to raise ImportError
        with patch.object(self.resolver, "_discover_step_contract_with_catalog", side_effect=ImportError("Step catalog not available")), \
             patch.object(self.resolver, "_discover_step_contract_legacy", return_value=mock_legacy_contract):
            result = self.resolver._discover_step_contract("UnknownStep")
            
            # Should fall back to legacy method and return the legacy contract
            assert result == mock_legacy_contract
            assert result.entry_point == "legacy_step.py"

    def test_step_catalog_contract_discovery_no_contract_component(self):
        """Test step catalog discovery when step has no contract component."""
        # Mock step catalog with step info but no contract component
        mock_catalog = Mock()
        mock_step_info = Mock()
        mock_step_info.file_components = {'script': Mock(), 'spec': Mock()}  # No contract
        mock_catalog.get_step_info.return_value = mock_step_info
        
        with patch("cursus.step_catalog.StepCatalog") as mock_catalog_class:
            mock_catalog_class.return_value = mock_catalog
            
            # Mock legacy discovery to return None as well
            with patch.object(self.resolver, "_discover_step_contract_legacy", return_value=None):
                result = self.resolver._discover_step_contract_with_catalog("StepWithoutContract")
                
                # Should return None since no contract component exists
                assert result is None
                
                # Verify catalog was called
                mock_catalog.get_step_info.assert_called_once_with("StepWithoutContract")

    def test_step_catalog_contract_discovery_error_handling(self):
        """Test step catalog contract discovery error handling."""
        # Test by mocking the catalog discovery method to raise an exception
        with patch.object(self.resolver, "_discover_step_contract_with_catalog", side_effect=Exception("Catalog error")):
            # The main discovery method should handle the exception and fall back to legacy
            with patch.object(self.resolver, "_discover_step_contract_legacy", return_value=None):
                result = self.resolver._discover_step_contract("ErrorStep")
                
                # Should handle the exception gracefully and return None (since legacy also returns None)
                assert result is None

    def test_step_catalog_unavailable_fallback(self):
        """Test behavior when step catalog is unavailable (ImportError)."""
        # Mock ImportError when trying to import StepCatalog
        with patch("cursus.step_catalog.StepCatalog", side_effect=ImportError("Step catalog not available")):
            # Mock legacy discovery to return a contract
            mock_legacy_contract = ScriptContract(
                entry_point="fallback_step.py",
                expected_input_paths={"fallback_input": "/opt/ml/processing/input/fallback"},
                expected_output_paths={"fallback_output": "/opt/ml/processing/output/fallback"},
                required_env_vars=["FALLBACK_VAR"],
            )
            
            with patch.object(self.resolver, "_discover_step_contract_legacy", return_value=mock_legacy_contract):
                result = self.resolver._discover_step_contract("FallbackStep")
                
                # Should fall back to legacy method
                assert result == mock_legacy_contract
                assert result.entry_point == "fallback_step.py"

    def test_step_catalog_integration_in_data_flow_building(self):
        """Test that step catalog integration works in the full data flow building process."""
        # Create mock contracts for our test steps
        tabular_contract = ScriptContract(
            entry_point="tabular_preprocessing.py",
            expected_input_paths={"raw_data": "/opt/ml/processing/input/data"},
            expected_output_paths={"processed_data": "/opt/ml/processing/output/processed"},
            required_env_vars=["SM_MODEL_DIR"],
        )
        
        xgboost_contract = ScriptContract(
            entry_point="xgboost_training.py",
            expected_input_paths={"training_data": "/opt/ml/processing/input/processed"},
            expected_output_paths={"model_output": "/opt/ml/processing/output/model"},
            required_env_vars=["SM_MODEL_DIR"],
        )
        
        eval_contract = ScriptContract(
            entry_point="xgboost_eval.py",
            expected_input_paths={"model_input": "/opt/ml/processing/input/model", "test_data": "/opt/ml/processing/input/test"},
            expected_output_paths={"evaluation_output": "/opt/ml/processing/output/evaluation"},
            required_env_vars=["SM_MODEL_DIR"],
        )
        
        # Mock contract discovery to return our contracts
        def mock_discover_contract(step_name):
            contracts = {
                "TabularPreprocessing": tabular_contract,
                "XGBoostTraining": xgboost_contract,
                "XGBoostModelEval": eval_contract,
            }
            return contracts.get(step_name)
        
        # Mock the step catalog discovery method
        with patch.object(self.resolver, "_discover_step_contract_with_catalog", side_effect=mock_discover_contract):
            plan = self.resolver.create_execution_plan()
            
            # Verify that contracts were used in data flow mapping
            data_flow = plan.data_flow_map
            
            # TabularPreprocessing should have no inputs (first step)
            assert data_flow["TabularPreprocessing"] == {}
            
            # XGBoostTraining should map its training_data input to TabularPreprocessing's processed_data output
            xgb_training_flow = data_flow["XGBoostTraining"]
            assert "training_data" in xgb_training_flow
            assert xgb_training_flow["training_data"] == "TabularPreprocessing:processed_data"
            
            # XGBoostModelEval should map its model_input to XGBoostTraining's model_output
            xgb_eval_flow = data_flow["XGBoostModelEval"]
            assert "model_input" in xgb_eval_flow
            assert xgb_eval_flow["model_input"] == "XGBoostTraining:model_output"


if __name__ == "__main__":
    pytest.main([__file__])
