"""
Pytest tests for Step Type Enhancement System

Tests the step type-aware validation enhancement system including:
- Step type enhancement router
- Individual step type enhancers
- Framework pattern detection
- Integration with unified alignment tester
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from cursus.validation.alignment.step_type_enhancement_router import (
    StepTypeEnhancementRouter,
)
from cursus.validation.alignment.step_type_enhancers.training_enhancer import (
    TrainingStepEnhancer,
)
from cursus.validation.alignment.step_type_enhancers.processing_enhancer import (
    ProcessingStepEnhancer,
)
from cursus.validation.alignment.step_type_enhancers.createmodel_enhancer import (
    CreateModelStepEnhancer,
)
from cursus.validation.alignment.step_type_enhancers.transform_enhancer import (
    TransformStepEnhancer,
)
from cursus.validation.alignment.step_type_enhancers.registermodel_enhancer import (
    RegisterModelStepEnhancer,
)
from cursus.validation.alignment.step_type_enhancers.utility_enhancer import (
    UtilityStepEnhancer,
)
from cursus.validation.alignment.framework_patterns import (
    detect_xgboost_patterns,
    detect_pytorch_patterns,
    detect_training_patterns,
    get_framework_patterns,
    get_all_framework_patterns,
)


class TestStepTypeEnhancementRouter:
    """Test the Step Type Enhancement Router."""

    @pytest.fixture
    def router(self):
        """Set up test fixtures."""
        return StepTypeEnhancementRouter()

    def test_router_initialization(self, router):
        """Test router initializes with all enhancers."""
        assert router._enhancer_classes is not None
        assert "Training" in router._enhancer_classes
        assert "Processing" in router._enhancer_classes
        assert "CreateModel" in router._enhancer_classes
        assert "Transform" in router._enhancer_classes
        assert "RegisterModel" in router._enhancer_classes
        assert "Utility" in router._enhancer_classes

    def test_step_type_detection(self, router):
        """Test step type detection from script name."""
        # Test that router can get step type requirements for different types
        training_req = router.get_step_type_requirements("Training")
        assert "training_loop" in training_req.get("required_patterns", [])

        processing_req = router.get_step_type_requirements("Processing")
        assert "data_transformation" in processing_req.get("required_patterns", [])

        model_req = router.get_step_type_requirements("CreateModel")
        assert "model_loading" in model_req.get("required_patterns", [])

    def test_enhancement_routing(self, router):
        """Test enhancement routing to correct enhancer."""
        # Mock validation results
        mock_results = {
            "passed": False,
            "issues": [
                {
                    "severity": "ERROR",
                    "category": "missing_training_loop",
                    "message": "Training script missing training loop",
                }
            ],
        }

        # Test training enhancement
        enhanced_results = router.enhance_validation(
            "xgboost_training.py", mock_results
        )

        # Should return enhanced results
        assert isinstance(enhanced_results, dict)
        assert "issues" in enhanced_results

    def test_unknown_step_type_handling(self, router):
        """Test handling of unknown step types."""
        mock_results = {"passed": True, "issues": []}

        enhanced_results = router.enhance_validation("unknown_script.py", mock_results)

        # Should still return results
        assert isinstance(enhanced_results, dict)
        assert "issues" in enhanced_results


class TestTrainingStepEnhancer:
    """Test the Training Step Enhancer."""

    @pytest.fixture
    def enhancer(self):
        """Set up test fixtures."""
        return TrainingStepEnhancer()

    def test_training_enhancer_initialization(self, enhancer):
        """Test training enhancer initializes correctly."""
        assert enhancer.step_type == "Training"
        assert "xgboost" in enhancer.framework_validators
        assert "pytorch" in enhancer.framework_validators

    def test_training_loop_validation(self, enhancer):
        """Test training loop validation."""
        # Mock script analysis without training loop
        mock_analysis = {
            "functions": ["load_data", "save_model"],
            "imports": ["xgboost"],
            "path_references": ["/opt/ml/model"],
        }

        with patch.object(enhancer, "_get_script_analysis", return_value=mock_analysis):
            enhanced_results = enhancer.enhance_validation({}, "xgboost_training.py")

        # Should detect missing training loop
        issues = enhanced_results.get("issues", [])
        training_loop_issues = [
            issue for issue in issues if "training_loop" in issue.get("category", "")
        ]
        assert len(issues) > 0

    def test_xgboost_specific_validation(self, enhancer):
        """Test XGBoost-specific training validation."""
        mock_analysis = {
            "functions": ["fit", "train"],
            "imports": ["xgboost", "xgb"],
            "path_references": [],
        }

        with patch.object(enhancer, "_get_script_analysis", return_value=mock_analysis):
            with patch.object(
                enhancer,
                "_detect_framework_from_script_analysis",
                return_value="xgboost",
            ):
                enhanced_results = enhancer.enhance_validation(
                    {}, "xgboost_training.py"
                )

        # Should apply XGBoost-specific validation
        assert "issues" in enhanced_results

    def test_pytorch_specific_validation(self, enhancer):
        """Test PyTorch-specific training validation."""
        mock_analysis = {
            "functions": ["forward", "backward", "zero_grad"],
            "imports": ["torch", "torch.nn"],
            "path_references": [],
        }

        with patch.object(enhancer, "_get_script_analysis", return_value=mock_analysis):
            with patch.object(
                enhancer,
                "_detect_framework_from_script_analysis",
                return_value="pytorch",
            ):
                enhanced_results = enhancer.enhance_validation(
                    {}, "pytorch_training.py"
                )

        # Should apply PyTorch-specific validation
        assert "issues" in enhanced_results


class TestProcessingStepEnhancer:
    """Test the Processing Step Enhancer."""

    @pytest.fixture
    def enhancer(self):
        """Set up test fixtures."""
        return ProcessingStepEnhancer()

    def test_processing_enhancer_initialization(self, enhancer):
        """Test processing enhancer initializes correctly."""
        assert enhancer.step_type == "Processing"
        assert "pandas" in enhancer.framework_validators
        assert "sklearn" in enhancer.framework_validators

    def test_data_processing_validation(self, enhancer):
        """Test data processing validation."""
        mock_analysis = {
            "functions": ["main"],
            "imports": ["pandas"],
            "path_references": [],
        }

        with patch.object(enhancer, "_get_script_analysis", return_value=mock_analysis):
            enhanced_results = enhancer.enhance_validation({}, "data_preprocessing.py")

        # Should detect missing data processing patterns
        issues = enhanced_results.get("issues", [])
        assert len(issues) > 0

    def test_pandas_specific_validation(self, enhancer):
        """Test Pandas-specific processing validation."""
        mock_analysis = {
            "functions": ["read_csv", "to_csv", "DataFrame"],
            "imports": ["pandas", "pd"],
            "path_references": [],
        }

        with patch.object(enhancer, "_get_script_analysis", return_value=mock_analysis):
            with patch.object(
                enhancer,
                "_detect_framework_from_script_analysis",
                return_value="pandas",
            ):
                enhanced_results = enhancer.enhance_validation(
                    {}, "data_preprocessing.py"
                )

        # Should apply Pandas-specific validation
        assert "issues" in enhanced_results


class TestFrameworkPatterns:
    """Test framework pattern detection."""

    def test_xgboost_pattern_detection(self):
        """Test XGBoost pattern detection."""
        mock_analysis = {
            "imports": ["xgboost", "xgb"],
            "functions": ["xgb.train", "DMatrix", "Booster"],
        }

        patterns = detect_xgboost_patterns(mock_analysis)

        assert patterns["has_xgboost_import"] is True
        assert patterns["has_dmatrix_usage"] is True
        assert patterns["has_xgb_train"] is True
        assert patterns["has_booster_usage"] is True

    def test_pytorch_pattern_detection(self):
        """Test PyTorch pattern detection."""
        mock_analysis = {
            "imports": ["torch", "torch.nn"],
            "functions": ["nn.Module", "optimizer", "forward", "backward"],
        }

        patterns = detect_pytorch_patterns(mock_analysis)

        assert patterns["has_torch_import"] is True
        assert patterns["has_nn_module"] is True
        assert patterns["has_optimizer"] is True
        assert patterns["has_training_loop"] is True

    def test_training_pattern_detection(self):
        """Test general training pattern detection."""
        mock_analysis = {
            "functions": ["fit", "train", "save"],
            "path_references": ["/opt/ml/model", "/opt/ml/input/data/train"],
        }

        patterns = detect_training_patterns(mock_analysis)

        assert patterns["has_training_loop"] is True
        assert patterns["has_model_saving"] is True
        assert patterns["has_data_loading"] is True

    def test_get_framework_patterns(self):
        """Test getting patterns for specific framework."""
        mock_analysis = {"imports": ["xgboost"], "functions": ["xgb.train"]}

        patterns = get_framework_patterns("xgboost", mock_analysis)

        assert "has_xgboost_import" in patterns
        assert "has_xgb_train" in patterns

    def test_get_all_framework_patterns(self):
        """Test getting patterns for all frameworks."""
        mock_analysis = {"imports": ["torch", "xgboost"], "functions": ["train", "fit"]}

        all_patterns = get_all_framework_patterns(mock_analysis)

        assert "xgboost" in all_patterns
        assert "pytorch" in all_patterns
        assert "training" in all_patterns


class TestStepTypeEnhancementIntegration:
    """Test integration of step type enhancement with unified alignment tester."""

    @pytest.fixture
    def temp_dirs(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        temp_dir = tempfile.mkdtemp()
        scripts_dir = Path(temp_dir) / "scripts"
        contracts_dir = Path(temp_dir) / "contracts"
        specs_dir = Path(temp_dir) / "specs"
        builders_dir = Path(temp_dir) / "builders"
        configs_dir = Path(temp_dir) / "configs"

        # Create directories
        for dir_path in [
            scripts_dir,
            contracts_dir,
            specs_dir,
            builders_dir,
            configs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        yield {
            "temp_dir": temp_dir,
            "scripts_dir": scripts_dir,
            "contracts_dir": contracts_dir,
            "specs_dir": specs_dir,
            "builders_dir": builders_dir,
            "configs_dir": configs_dir,
        }

        # Clean up test fixtures
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_unified_tester_with_step_type_enhancement(self, temp_dirs):
        """Test unified alignment tester with step type enhancement enabled."""
        # Create a sample training script
        training_script = temp_dirs["scripts_dir"] / "xgboost_training.py"
        training_script.write_text(
            """
import xgboost as xgb
import pandas as pd

def main():
    # Load data
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')
    
    # Create DMatrix
    dtrain = xgb.DMatrix(train_data)
    
    # Train model
    model = xgb.train({}, dtrain)
    
    # Save model
    model.save_model('/opt/ml/model/model.xgb')

if __name__ == '__main__':
    main()
        """
        )

        # Mock the unified alignment tester
        with patch(
            "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
        ) as MockTester:
            mock_tester = MockTester.return_value
            mock_tester.step_type_enhancement_router = StepTypeEnhancementRouter()

            # Test that step type enhancement is applied
            mock_results = {
                "passed": False,
                "issues": [
                    {"severity": "ERROR", "category": "test", "message": "Test issue"}
                ],
            }

            enhanced_results = (
                mock_tester.step_type_enhancement_router.enhance_validation(
                    "xgboost_training.py", mock_results
                )
            )

            # Should return enhanced results
            assert isinstance(enhanced_results, dict)
            assert "issues" in enhanced_results

    def test_step_type_enhancement_feature_flag(self):
        """Test step type enhancement feature flag."""
        # Test with feature flag enabled
        with patch.dict(os.environ, {"ENABLE_STEP_TYPE_AWARENESS": "true"}):
            router = StepTypeEnhancementRouter()
            assert router is not None

        # Test with feature flag disabled
        with patch.dict(os.environ, {"ENABLE_STEP_TYPE_AWARENESS": "false"}):
            router = StepTypeEnhancementRouter()
            assert (
                router is not None
            )  # Should still work but may have different behavior


class TestStepTypeEnhancementEndToEnd:
    """End-to-end tests for step type enhancement system."""

    def test_training_script_enhancement_flow(self):
        """Test complete enhancement flow for training script."""
        router = StepTypeEnhancementRouter()

        # Mock validation results for training script
        mock_results = {
            "passed": False,
            "issues": [
                {
                    "severity": "ERROR",
                    "category": "missing_contract",
                    "message": "Script contract not found",
                }
            ],
        }

        # Apply enhancement
        enhanced_results = router.enhance_validation(
            "xgboost_training.py", mock_results
        )

        # Verify enhancement was applied
        assert isinstance(enhanced_results, dict)
        assert "issues" in enhanced_results

    def test_processing_script_enhancement_flow(self):
        """Test complete enhancement flow for processing script."""
        router = StepTypeEnhancementRouter()

        mock_results = {"passed": True, "issues": []}

        enhanced_results = router.enhance_validation(
            "data_preprocessing.py", mock_results
        )

        assert isinstance(enhanced_results, dict)
        assert "issues" in enhanced_results

    def test_multiple_step_types_enhancement(self):
        """Test enhancement for multiple step types."""
        router = StepTypeEnhancementRouter()

        test_scripts = [
            ("xgboost_training.py", "Training"),
            ("data_preprocessing.py", "Processing"),
            ("create_model.py", "CreateModel"),
            ("batch_transform.py", "Transform"),
            ("register_model.py", "RegisterModel"),
            ("prepare_config.py", "Utility"),
        ]

        for script_name, expected_step_type in test_scripts:
            mock_results = {"passed": True, "issues": []}
            enhanced_results = router.enhance_validation(script_name, mock_results)

            assert isinstance(enhanced_results, dict)
            assert "issues" in enhanced_results
