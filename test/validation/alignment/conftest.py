"""
Test configuration for alignment validation tests.

This module provides fixtures and configuration for alignment validation tests,
ensuring proper workspace root configuration for the unified step catalog system.
"""

import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch

# Import the adapter classes (new unified system)
from cursus.step_catalog.adapters import (
    FlexibleFileResolverAdapter,
    ContractDiscoveryEngineAdapter,
    ContractDiscoveryManagerAdapter,
)


@pytest.fixture
def workspace_root():
    """Provide correct workspace root for tests."""
    # Get the project root (3 levels up from test/validation/alignment)
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent.parent
    return project_root


@pytest.fixture
def resolver(workspace_root):
    """Create FlexibleFileResolverAdapter with correct workspace root."""
    return FlexibleFileResolverAdapter(workspace_root)


@pytest.fixture
def contract_discovery_engine(workspace_root):
    """Create ContractDiscoveryEngineAdapter with correct workspace root."""
    return ContractDiscoveryEngineAdapter(workspace_root)


@pytest.fixture
def contract_discovery_manager(workspace_root):
    """Create ContractDiscoveryManagerAdapter with correct workspace root."""
    return ContractDiscoveryManagerAdapter(workspace_root)


@pytest.fixture
def temp_step_structure():
    """Create temporary step directory structure for isolated tests."""
    temp_dir = tempfile.mkdtemp()
    
    # Create step directory structure
    step_dirs = ["scripts", "contracts", "specs", "builders", "configs"]
    for dir_name in step_dirs:
        os.makedirs(os.path.join(temp_dir, "src", "cursus", "steps", dir_name), exist_ok=True)
    
    # Create sample files with correct naming patterns
    test_files = {
        "src/cursus/steps/scripts/xgboost_training.py": "# XGBoost training script",
        "src/cursus/steps/contracts/xgboost_training_contract.py": "# XGBoost training contract",
        "src/cursus/steps/specs/xgboost_training_spec.py": "# XGBoost training spec",
        "src/cursus/steps/builders/builder_xgboost_training_step.py": "# XGBoost training builder",
        "src/cursus/steps/configs/config_xgboost_training_step.py": "# XGBoost training config",
        
        "src/cursus/steps/scripts/dummy_training.py": "# Dummy training script",
        "src/cursus/steps/contracts/dummy_training_contract.py": "# Dummy training contract",
        "src/cursus/steps/specs/dummy_training_spec.py": "# Dummy training spec",
        "src/cursus/steps/builders/builder_dummy_training_step.py": "# Dummy training builder",
        "src/cursus/steps/configs/config_dummy_training_step.py": "# Dummy training config",
    }
    
    for file_path, content in test_files.items():
        full_path = os.path.join(temp_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    
    yield Path(temp_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_resolver(temp_step_structure):
    """Create resolver with temporary step structure."""
    return FlexibleFileResolverAdapter(temp_step_structure)


# Step name mapping for tests (old test names -> actual step names)
STEP_NAME_MAPPING = {
    "train": "xgboost_training",
    "training": "xgboost_training", 
    "dummy_train": "dummy_training",
    "preprocessing": "tabular_preprocessing",
    "preprocess": "tabular_preprocessing",
    "evaluation": "xgboost_model_eval",
    "eval": "xgboost_model_eval",
    "model_evaluation": "xgboost_model_eval",
    "data_load": "cradle_data_loading",
    "data_loading": "cradle_data_loading",
}


@pytest.fixture
def step_name_mapper():
    """Provide step name mapping for tests."""
    def map_step_name(test_name):
        """Map test step name to actual step name."""
        return STEP_NAME_MAPPING.get(test_name, test_name)
    
    return map_step_name


# Mock fixtures for legacy compatibility
@pytest.fixture
def mock_step_names():
    """Mock STEP_NAMES for tests that need registry data."""
    return {
        "XGBoostTraining": {
            "config_class": "XGBoostTrainingConfig",
            "builder_step_name": "XGBoostTrainingStepBuilder",
            "spec_type": "XGBoostTraining",
            "sagemaker_step_type": "Training",
            "description": "XGBoost model training step",
        },
        "DummyTraining": {
            "config_class": "DummyTrainingConfig", 
            "builder_step_name": "DummyTrainingStepBuilder",
            "spec_type": "DummyTraining",
            "sagemaker_step_type": "Training",
            "description": "Dummy training step for testing",
        },
        "XGBoostModelEval": {
            "config_class": "XGBoostModelEvalConfig",
            "builder_step_name": "XGBoostModelEvalStepBuilder", 
            "spec_type": "XGBoostModelEval",
            "sagemaker_step_type": "Processing",
            "description": "XGBoost model evaluation step",
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(workspace_root):
    """Auto-setup test environment with correct paths."""
    # Ensure we're using the correct workspace root for all tests
    os.environ["CURSUS_TEST_WORKSPACE_ROOT"] = str(workspace_root)
    yield
    # Cleanup
    os.environ.pop("CURSUS_TEST_WORKSPACE_ROOT", None)
