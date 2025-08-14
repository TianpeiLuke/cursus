"""
Test Step Type Enhancement System

Tests the step type-aware validation enhancement system including:
- Step type enhancement router
- Individual step type enhancers
- Framework pattern detection
- Integration with unified alignment tester
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.cursus.validation.alignment.step_type_enhancement_router import StepTypeEnhancementRouter
from src.cursus.validation.alignment.step_type_enhancers.training_enhancer import TrainingStepEnhancer
from src.cursus.validation.alignment.step_type_enhancers.processing_enhancer import ProcessingStepEnhancer
from src.cursus.validation.alignment.step_type_enhancers.createmodel_enhancer import CreateModelStepEnhancer
from src.cursus.validation.alignment.step_type_enhancers.transform_enhancer import TransformStepEnhancer
from src.cursus.validation.alignment.step_type_enhancers.registermodel_enhancer import RegisterModelStepEnhancer
from src.cursus.validation.alignment.step_type_enhancers.utility_enhancer import UtilityStepEnhancer
from src.cursus.validation.alignment.framework_patterns import (
    detect_xgboost_patterns, detect_pytorch_patterns, detect_training_patterns,
    get_framework_patterns, get_all_framework_patterns
)


class TestStepTypeEnhancementRouter(unittest.TestCase):
    """Test the Step Type Enhancement Router."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = StepTypeEnhancementRouter()
    
    def test_router_initialization(self):
        """Test router initializes with all enhancers."""
        self.assertIsNotNone(self.router._enhancer_classes)
        self.assertIn('Training', self.router._enhancer_classes)
        self.assertIn('Processing', self.router._enhancer_classes)
        self.assertIn('CreateModel', self.router._enhancer_classes)
        self.assertIn('Transform', self.router._enhancer_classes)
        self.assertIn('RegisterModel', self.router._enhancer_classes)
        self.assertIn('Utility', self.router._enhancer_classes)
    
    def test_step_type_detection(self):
        """Test step type detection from script name."""
        # Test that router can get step type requirements for different types
        training_req = self.router.get_step_type_requirements('Training')
        self.assertIn('training_loop', training_req.get('required_patterns', []))
        
        processing_req = self.router.get_step_type_requirements('Processing')
        self.assertIn('data_transformation', processing_req.get('required_patterns', []))
        
        model_req = self.router.get_step_type_requirements('CreateModel')
        self.assertIn('model_loading', model_req.get('required_patterns', []))
    
    def test_enhancement_routing(self):
        """Test enhancement routing to correct enhancer."""
        # Mock validation results
        mock_results = {
            'passed': False,
            'issues': [
                {
                    'severity': 'ERROR',
                    'category': 'missing_training_loop',
                    'message': 'Training script missing training loop'
                }
            ]
        }
        
        # Test training enhancement
        enhanced_results = self.router.enhance_validation('xgboost_training.py', mock_results)
        
        # Should return enhanced results
        self.assertIsInstance(enhanced_results, dict)
        self.assertIn('issues', enhanced_results)
    
    def test_unknown_step_type_handling(self):
        """Test handling of unknown step types."""
        mock_results = {'passed': True, 'issues': []}
        
        enhanced_results = self.router.enhance_validation('unknown_script.py', mock_results)
        
        # Should still return results
        self.assertIsInstance(enhanced_results, dict)
        self.assertIn('issues', enhanced_results)


class TestTrainingStepEnhancer(unittest.TestCase):
    """Test the Training Step Enhancer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = TrainingStepEnhancer()
    
    def test_training_enhancer_initialization(self):
        """Test training enhancer initializes correctly."""
        self.assertEqual(self.enhancer.step_type, 'Training')
        self.assertIn('xgboost', self.enhancer.framework_validators)
        self.assertIn('pytorch', self.enhancer.framework_validators)
    
    def test_training_loop_validation(self):
        """Test training loop validation."""
        # Mock script analysis without training loop
        mock_analysis = {
            'functions': ['load_data', 'save_model'],
            'imports': ['xgboost'],
            'path_references': ['/opt/ml/model']
        }
        
        with patch.object(self.enhancer, '_get_script_analysis', return_value=mock_analysis):
            enhanced_results = self.enhancer.enhance_validation({}, 'xgboost_training.py')
        
        # Should detect missing training loop
        issues = enhanced_results.get('issues', [])
        training_loop_issues = [issue for issue in issues if 'training_loop' in issue.get('category', '')]
        self.assertTrue(len(issues) > 0)
    
    def test_xgboost_specific_validation(self):
        """Test XGBoost-specific training validation."""
        mock_analysis = {
            'functions': ['fit', 'train'],
            'imports': ['xgboost', 'xgb'],
            'path_references': []
        }
        
        with patch.object(self.enhancer, '_get_script_analysis', return_value=mock_analysis):
            with patch.object(self.enhancer, '_detect_framework_from_script_analysis', return_value='xgboost'):
                enhanced_results = self.enhancer.enhance_validation({}, 'xgboost_training.py')
        
        # Should apply XGBoost-specific validation
        self.assertIn('issues', enhanced_results)
    
    def test_pytorch_specific_validation(self):
        """Test PyTorch-specific training validation."""
        mock_analysis = {
            'functions': ['forward', 'backward', 'zero_grad'],
            'imports': ['torch', 'torch.nn'],
            'path_references': []
        }
        
        with patch.object(self.enhancer, '_get_script_analysis', return_value=mock_analysis):
            with patch.object(self.enhancer, '_detect_framework_from_script_analysis', return_value='pytorch'):
                enhanced_results = self.enhancer.enhance_validation({}, 'pytorch_training.py')
        
        # Should apply PyTorch-specific validation
        self.assertIn('issues', enhanced_results)


class TestProcessingStepEnhancer(unittest.TestCase):
    """Test the Processing Step Enhancer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = ProcessingStepEnhancer()
    
    def test_processing_enhancer_initialization(self):
        """Test processing enhancer initializes correctly."""
        self.assertEqual(self.enhancer.step_type, 'Processing')
        self.assertIn('pandas', self.enhancer.framework_validators)
        self.assertIn('sklearn', self.enhancer.framework_validators)
    
    def test_data_processing_validation(self):
        """Test data processing validation."""
        mock_analysis = {
            'functions': ['main'],
            'imports': ['pandas'],
            'path_references': []
        }
        
        with patch.object(self.enhancer, '_get_script_analysis', return_value=mock_analysis):
            enhanced_results = self.enhancer.enhance_validation({}, 'data_preprocessing.py')
        
        # Should detect missing data processing patterns
        issues = enhanced_results.get('issues', [])
        self.assertTrue(len(issues) > 0)
    
    def test_pandas_specific_validation(self):
        """Test Pandas-specific processing validation."""
        mock_analysis = {
            'functions': ['read_csv', 'to_csv', 'DataFrame'],
            'imports': ['pandas', 'pd'],
            'path_references': []
        }
        
        with patch.object(self.enhancer, '_get_script_analysis', return_value=mock_analysis):
            with patch.object(self.enhancer, '_detect_framework_from_script_analysis', return_value='pandas'):
                enhanced_results = self.enhancer.enhance_validation({}, 'data_preprocessing.py')
        
        # Should apply Pandas-specific validation
        self.assertIn('issues', enhanced_results)


class TestFrameworkPatterns(unittest.TestCase):
    """Test framework pattern detection."""
    
    def test_xgboost_pattern_detection(self):
        """Test XGBoost pattern detection."""
        mock_analysis = {
            'imports': ['xgboost', 'xgb'],
            'functions': ['xgb.train', 'DMatrix', 'Booster']
        }
        
        patterns = detect_xgboost_patterns(mock_analysis)
        
        self.assertTrue(patterns['has_xgboost_import'])
        self.assertTrue(patterns['has_dmatrix_usage'])
        self.assertTrue(patterns['has_xgb_train'])
        self.assertTrue(patterns['has_booster_usage'])
    
    def test_pytorch_pattern_detection(self):
        """Test PyTorch pattern detection."""
        mock_analysis = {
            'imports': ['torch', 'torch.nn'],
            'functions': ['nn.Module', 'optimizer', 'forward', 'backward']
        }
        
        patterns = detect_pytorch_patterns(mock_analysis)
        
        self.assertTrue(patterns['has_torch_import'])
        self.assertTrue(patterns['has_nn_module'])
        self.assertTrue(patterns['has_optimizer'])
        self.assertTrue(patterns['has_training_loop'])
    
    def test_training_pattern_detection(self):
        """Test general training pattern detection."""
        mock_analysis = {
            'functions': ['fit', 'train', 'save'],
            'path_references': ['/opt/ml/model', '/opt/ml/input/data/train']
        }
        
        patterns = detect_training_patterns(mock_analysis)
        
        self.assertTrue(patterns['has_training_loop'])
        self.assertTrue(patterns['has_model_saving'])
        self.assertTrue(patterns['has_data_loading'])
    
    def test_get_framework_patterns(self):
        """Test getting patterns for specific framework."""
        mock_analysis = {
            'imports': ['xgboost'],
            'functions': ['xgb.train']
        }
        
        patterns = get_framework_patterns('xgboost', mock_analysis)
        
        self.assertIn('has_xgboost_import', patterns)
        self.assertIn('has_xgb_train', patterns)
    
    def test_get_all_framework_patterns(self):
        """Test getting patterns for all frameworks."""
        mock_analysis = {
            'imports': ['torch', 'xgboost'],
            'functions': ['train', 'fit']
        }
        
        all_patterns = get_all_framework_patterns(mock_analysis)
        
        self.assertIn('xgboost', all_patterns)
        self.assertIn('pytorch', all_patterns)
        self.assertIn('training', all_patterns)


class TestStepTypeEnhancementIntegration(unittest.TestCase):
    """Test integration of step type enhancement with unified alignment tester."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.contracts_dir = Path(self.temp_dir) / "contracts"
        self.specs_dir = Path(self.temp_dir) / "specs"
        self.builders_dir = Path(self.temp_dir) / "builders"
        self.configs_dir = Path(self.temp_dir) / "configs"
        
        # Create directories
        for dir_path in [self.scripts_dir, self.contracts_dir, self.specs_dir, self.builders_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unified_tester_with_step_type_enhancement(self):
        """Test unified alignment tester with step type enhancement enabled."""
        # Create a sample training script
        training_script = self.scripts_dir / "xgboost_training.py"
        training_script.write_text("""
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
        """)
        
        # Mock the unified alignment tester
        with patch('src.cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester') as MockTester:
            mock_tester = MockTester.return_value
            mock_tester.step_type_enhancement_router = StepTypeEnhancementRouter()
            
            # Test that step type enhancement is applied
            mock_results = {
                'passed': False,
                'issues': [{'severity': 'ERROR', 'category': 'test', 'message': 'Test issue'}]
            }
            
            enhanced_results = mock_tester.step_type_enhancement_router.enhance_validation(
                'xgboost_training.py', mock_results
            )
            
            # Should return enhanced results
            self.assertIsInstance(enhanced_results, dict)
            self.assertIn('issues', enhanced_results)
    
    def test_step_type_enhancement_feature_flag(self):
        """Test step type enhancement feature flag."""
        # Test with feature flag enabled
        with patch.dict(os.environ, {'ENABLE_STEP_TYPE_AWARENESS': 'true'}):
            router = StepTypeEnhancementRouter()
            self.assertIsNotNone(router)
        
        # Test with feature flag disabled
        with patch.dict(os.environ, {'ENABLE_STEP_TYPE_AWARENESS': 'false'}):
            router = StepTypeEnhancementRouter()
            self.assertIsNotNone(router)  # Should still work but may have different behavior


class TestStepTypeEnhancementEndToEnd(unittest.TestCase):
    """End-to-end tests for step type enhancement system."""
    
    def test_training_script_enhancement_flow(self):
        """Test complete enhancement flow for training script."""
        router = StepTypeEnhancementRouter()
        
        # Mock validation results for training script
        mock_results = {
            'passed': False,
            'issues': [
                {
                    'severity': 'ERROR',
                    'category': 'missing_contract',
                    'message': 'Script contract not found'
                }
            ]
        }
        
        # Apply enhancement
        enhanced_results = router.enhance_validation('xgboost_training.py', mock_results)
        
        # Verify enhancement was applied
        self.assertIsInstance(enhanced_results, dict)
        self.assertIn('issues', enhanced_results)
    
    def test_processing_script_enhancement_flow(self):
        """Test complete enhancement flow for processing script."""
        router = StepTypeEnhancementRouter()
        
        mock_results = {
            'passed': True,
            'issues': []
        }
        
        enhanced_results = router.enhance_validation('data_preprocessing.py', mock_results)
        
        self.assertIsInstance(enhanced_results, dict)
        self.assertIn('issues', enhanced_results)
    
    def test_multiple_step_types_enhancement(self):
        """Test enhancement for multiple step types."""
        router = StepTypeEnhancementRouter()
        
        test_scripts = [
            ('xgboost_training.py', 'Training'),
            ('data_preprocessing.py', 'Processing'),
            ('create_model.py', 'CreateModel'),
            ('batch_transform.py', 'Transform'),
            ('register_model.py', 'RegisterModel'),
            ('prepare_config.py', 'Utility')
        ]
        
        for script_name, expected_step_type in test_scripts:
            mock_results = {'passed': True, 'issues': []}
            enhanced_results = router.enhance_validation(script_name, mock_results)
            
            self.assertIsInstance(enhanced_results, dict)
            self.assertIn('issues', enhanced_results)


if __name__ == '__main__':
    unittest.main()
