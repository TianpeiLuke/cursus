#!/usr/bin/env python3
"""
XGBoost Pipeline Individual Step Testing Script

This script contains all the code for testing individual pipeline steps.
It can be run directly or converted to a Jupyter notebook.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup environment and imports."""
    print("=== SETUP AND IMPORTS ===")
    
    # Add cursus to path
    sys.path.append(str(Path.cwd().parent.parent.parent / 'src'))
    
    # Import Cursus components
    try:
        from cursus.validation.runtime.jupyter.notebook_interface import NotebookInterface
        from cursus.validation.runtime.core.data_flow_manager import DataFlowManager
        from cursus.steps.registry.step_names import STEP_NAMES
        print("✓ Successfully imported Cursus components")
        cursus_available = True
    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print("Using mock implementations for testing...")
        cursus_available = False
    
    print(f"Individual step testing started at {datetime.now()}")
    return cursus_available

def load_configuration():
    """Load pipeline configuration and validate environment."""
    print("\n=== LOAD CONFIGURATION AND VALIDATE ENVIRONMENT ===")
    
    # Define directory structure
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / 'data'
    CONFIG_DIR = BASE_DIR / 'configs'
    OUTPUTS_DIR = BASE_DIR / 'outputs'
    WORKSPACE_DIR = OUTPUTS_DIR / 'workspace'
    LOGS_DIR = OUTPUTS_DIR / 'logs'
    RESULTS_DIR = OUTPUTS_DIR / 'results'
    
    # Load pipeline configuration
    pipeline_config_path = CONFIG_DIR / 'pipeline_config.json'
    if pipeline_config_path.exists():
        with open(pipeline_config_path, 'r') as f:
            pipeline_config = json.load(f)
        print(f"✓ Loaded pipeline configuration: {pipeline_config_path}")
        
        # Extract step configurations
        step_configs = pipeline_config['step_configurations']
        pipeline_metadata = pipeline_config['pipeline_metadata']
        
        print(f"Pipeline: {pipeline_metadata['name']}")
        print(f"Steps loaded: {list(step_configs.keys())}")
    else:
        print(f"⚠ Pipeline configuration not found: {pipeline_config_path}")
        print("Please run 02_pipeline_configuration.ipynb first!")
        step_configs = {}
        pipeline_metadata = {}
    
    # Validate required files exist
    required_files = [
        DATA_DIR / 'train_data.csv',
        DATA_DIR / 'eval_data.csv',
        DATA_DIR / 'dataset_metadata.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ Required file exists: {file_path.name}")
        else:
            print(f"⚠ Required file missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print("\nPlease run previous notebooks to generate required files!")
    else:
        print("\n✓ All required files are available for testing")
    
    return {
        'step_configs': step_configs,
        'pipeline_metadata': pipeline_metadata,
        'directories': {
            'BASE_DIR': BASE_DIR,
            'DATA_DIR': DATA_DIR,
            'CONFIG_DIR': CONFIG_DIR,
            'OUTPUTS_DIR': OUTPUTS_DIR,
            'WORKSPACE_DIR': WORKSPACE_DIR,
            'LOGS_DIR': LOGS_DIR,
            'RESULTS_DIR': RESULTS_DIR
        },
        'missing_files': missing_files
    }

class MockStepTester:
    """Mock implementation for testing individual pipeline steps."""
    
    def __init__(self, workspace_dir):
        self.workspace_dir = Path(workspace_dir)
        self.execution_times = {}
        self.step_results = {}
        
        # Ensure workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        print(f"MockStepTester initialized with workspace: {self.workspace_dir}")
    
    def test_xgboost_training(self, config):
        """Test XGBoost training step."""
        print("\n" + "=" * 50)
        print("TESTING XGBOOST TRAINING STEP")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Load and validate training data
            input_path = Path(config['input_data_path'])
            if not input_path.exists():
                raise FileNotFoundError(f"Training data not found: {input_path}")
            
            train_data = pd.read_csv(input_path)
            print(f"✓ Loaded training data: {train_data.shape}")
            
            # Validate required columns
            target_col = config['target_column']
            feature_cols = config['feature_columns']
            
            if target_col not in train_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            missing_features = [col for col in feature_cols if col not in train_data.columns]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")
            
            print(f"✓ Validated columns: {len(feature_cols)} features, target='{target_col}'")
            
            # Simulate training process
            print("Training XGBoost model...")
            time.sleep(1.0)  # Simulate training time
            
            # Create mock model output
            model_path = Path(config['output_model_path'])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save mock model metadata
            model_info = {
                'model_type': 'XGBoost',
                'hyperparameters': config['hyperparameters'],
                'training_samples': len(train_data),
                'features': feature_cols,
                'target_column': target_col,
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Save model (mock)
            with open(model_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            execution_time = time.time() - start_time
            self.execution_times['XGBoostTraining'] = execution_time
            self.step_results['XGBoostTraining'] = {
                'status': 'success',
                'model_path': str(model_path),
                'training_samples': len(train_data),
                'execution_time': execution_time
            }
            
            print(f"✓ XGBoost Training completed successfully in {execution_time:.2f}s")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"✗ XGBoost Training failed: {e}")
            self.execution_times['XGBoostTraining'] = execution_time
            self.step_results['XGBoostTraining'] = {
                'status': 'failed', 
                'error': str(e),
                'execution_time': execution_time
            }
            return False
    
    def test_xgboost_eval(self, config):
        """Test XGBoost model evaluation step."""
        print("\n" + "=" * 50)
        print("TESTING XGBOOST MODEL EVALUATION STEP")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Check if model exists (from previous step)
            model_path = Path(config['model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load evaluation data
            eval_path = Path(config['eval_data_path'])
            eval_data = pd.read_csv(eval_path)
            print(f"✓ Loaded evaluation data: {eval_data.shape}")
            
            # Simulate model evaluation
            print("Evaluating model on test data...")
            time.sleep(0.5)  # Simulate evaluation time
            
            # Generate mock predictions
            n_samples = len(eval_data)
            np.random.seed(42)
            predictions_proba = np.random.beta(2, 2, n_samples)
            predictions_class = (predictions_proba > 0.5).astype(int)
            
            # Save predictions
            pred_df = pd.DataFrame({
                'prediction_proba': predictions_proba,
                'prediction_class': predictions_class
            })
            
            pred_path = Path(config['output_predictions_path'])
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(pred_path, index=False)
            
            # Generate mock metrics
            metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'auc_roc': 0.91
            }
            
            # Save metrics
            metrics_path = Path(config['output_metrics_path'])
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            execution_time = time.time() - start_time
            self.execution_times['XGBoostModelEval'] = execution_time
            self.step_results['XGBoostModelEval'] = {
                'status': 'success',
                'metrics': metrics,
                'eval_samples': n_samples,
                'execution_time': execution_time
            }
            
            print(f"✓ XGBoost Evaluation completed successfully in {execution_time:.2f}s")
            print(f"  Accuracy: {metrics['accuracy']:.3f}, AUC-ROC: {metrics['auc_roc']:.3f}")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"✗ XGBoost Evaluation failed: {e}")
            self.execution_times['XGBoostModelEval'] = execution_time
            self.step_results['XGBoostModelEval'] = {
                'status': 'failed', 
                'error': str(e),
                'execution_time': execution_time
            }
            return False
    
    def test_model_calibration(self, config):
        """Test model calibration step."""
        print("\n" + "=" * 50)
        print("TESTING MODEL CALIBRATION STEP")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Load predictions from previous step
            pred_path = Path(config['predictions_path'])
            predictions = pd.read_csv(pred_path)
            print(f"✓ Loaded predictions: {predictions.shape}")
            
            # Simulate calibration process
            print(f"Calibrating predictions using {config['calibration_method']} method...")
            time.sleep(0.3)  # Simulate calibration time
            
            # Generate mock calibrated predictions
            original_proba = predictions['prediction_proba'].values
            calibrated_proba = 0.1 + 0.8 * original_proba  # Simple transformation
            calibrated_class = (calibrated_proba > 0.5).astype(int)
            
            # Create calibrated predictions dataframe
            calibrated_df = pd.DataFrame({
                'original_proba': original_proba,
                'calibrated_proba': calibrated_proba,
                'calibrated_class': calibrated_class
            })
            
            # Save calibrated predictions
            calib_pred_path = Path(config['output_calibrated_predictions_path'])
            calib_pred_path.parent.mkdir(parents=True, exist_ok=True)
            calibrated_df.to_csv(calib_pred_path, index=False)
            
            # Save mock calibrated model
            calibrated_model_info = {
                'calibration_method': config['calibration_method'],
                'calibrated_samples': len(predictions),
                'calibration_improvement': 0.03
            }
            
            calib_model_path = Path(config['output_calibrated_model_path'])
            with open(calib_model_path, 'w') as f:
                json.dump(calibrated_model_info, f, indent=2)
            
            execution_time = time.time() - start_time
            self.execution_times['ModelCalibration'] = execution_time
            self.step_results['ModelCalibration'] = {
                'status': 'success',
                'calibration_method': config['calibration_method'],
                'calibrated_samples': len(predictions),
                'execution_time': execution_time
            }
            
            print(f"✓ Model Calibration completed successfully in {execution_time:.2f}s")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"✗ Model Calibration failed: {e}")
            self.execution_times['ModelCalibration'] = execution_time
            self.step_results['ModelCalibration'] = {
                'status': 'failed', 
                'error': str(e),
                'execution_time': execution_time
            }
            return False

def run_individual_step_tests(config_data):
    """Run individual step tests."""
    print("\n=== RUN INDIVIDUAL STEP TESTS ===")
    
    step_configs = config_data['step_configs']
    directories = config_data['directories']
    
    if not step_configs:
        print("Cannot run tests without step configurations!")
        return None
    
    # Initialize step tester
    step_tester = MockStepTester(directories['WORKSPACE_DIR'])
    
    print("RUNNING INDIVIDUAL STEP TESTS")
    print("=" * 60)
    
    # Test Step 1: XGBoost Training
    if 'XGBoostTraining' in step_configs:
        training_success = step_tester.test_xgboost_training(step_configs['XGBoostTraining'])
    else:
        print("⚠ XGBoostTraining configuration not found")
        training_success = False
    
    # Test Step 2: XGBoost Model Evaluation (depends on Step 1)
    if 'XGBoostModelEval' in step_configs and training_success:
        eval_success = step_tester.test_xgboost_eval(step_configs['XGBoostModelEval'])
    else:
        if not training_success:
            print("\n⚠ Skipping XGBoost Evaluation due to training failure")
        else:
            print("\n⚠ XGBoostModelEval configuration not found")
        eval_success = False
    
    # Test Step 3: Model Calibration (depends on Step 2)
    if 'ModelCalibration' in step_configs and eval_success:
        calibration_success = step_tester.test_model_calibration(step_configs['ModelCalibration'])
    else:
        if not eval_success:
            print("\n⚠ Skipping Model Calibration due to evaluation failure")
        else:
            print("\n⚠ ModelCalibration configuration not found")
        calibration_success = False
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL STEP TESTS COMPLETED")
    print("=" * 60)
    
    return step_tester

def generate_test_summary(step_tester, config_data):
    """Generate test results summary."""
    print("\n=== TEST RESULTS SUMMARY ===")
    
    if not step_tester:
        print("Cannot generate summary without test results!")
        return
    
    step_configs = config_data['step_configs']
    pipeline_metadata = config_data['pipeline_metadata']
    directories = config_data['directories']
    
    print("INDIVIDUAL STEP TEST SUMMARY")
    print("=" * 50)
    
    total_steps = len(step_configs)
    successful_steps = sum(1 for result in step_tester.step_results.values() 
                          if result['status'] == 'success')
    failed_steps = total_steps - successful_steps
    success_rate = successful_steps / total_steps * 100 if total_steps > 0 else 0
    total_time = sum(step_tester.execution_times.values())
    
    print(f"Total Steps: {total_steps}")
    print(f"Successful Steps: {successful_steps}")
    print(f"Failed Steps: {failed_steps}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Execution Time: {total_time:.2f}s")
    
    print("\nStep-by-Step Results:")
    for step_name in ['XGBoostTraining', 'XGBoostModelEval', 'ModelCalibration']:
        if step_name in step_tester.step_results:
            result = step_tester.step_results[step_name]
            exec_time = step_tester.execution_times.get(step_name, 0)
            status_icon = "✓" if result['status'] == 'success' else "✗"
            print(f"  {status_icon} {step_name}: {result['status']} ({exec_time:.2f}s)")
            
            if result['status'] == 'failed' and 'error' in result:
                print(f"    Error: {result['error']}")
            elif result['status'] == 'success':
                if step_name == 'XGBoostTraining':
                    print(f"    Training samples: {result.get('training_samples', 'N/A')}")
                elif step_name == 'XGBoostModelEval':
                    metrics = result.get('metrics', {})
                    print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                    print(f"    AUC-ROC: {metrics.get('auc_roc', 'N/A'):.3f}")
                elif step_name == 'ModelCalibration':
                    print(f"    Method: {result.get('calibration_method', 'N/A')}")
                    print(f"    Samples: {result.get('calibrated_samples', 'N/A')}")
        else:
            print(f"  - {step_name}: not tested")
    
    # Save test results
    test_results = {
        'test_timestamp': datetime.now().isoformat(),
        'test_type': 'individual_step_testing',
        'pipeline_name': pipeline_metadata.get('name', 'Unknown'),
        'total_steps': total_steps,
        'successful_steps': successful_steps,
        'failed_steps': failed_steps,
        'success_rate': success_rate,
        'total_execution_time': total_time,
        'step_results': step_tester.step_results,
        'execution_times': step_tester.execution_times
    }
    
    results_path = directories['RESULTS_DIR'] / 'individual_step_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Test results saved: {results_path}")
    
    if success_rate == 100:
        print("\n🎉 All individual step tests passed!")
        print("Ready for end-to-end pipeline testing.")
        print("Next: Run 04_end_to_end_pipeline_test.ipynb")
    else:
        print(f"\n⚠ {failed_steps} step(s) failed. Please review and fix issues before proceeding.")

def verify_output_files(directories):
    """Verify that expected output files were created."""
    print("\n=== OUTPUT FILE VERIFICATION ===")
    
    WORKSPACE_DIR = directories['WORKSPACE_DIR']
    
    print("OUTPUT FILE VERIFICATION")
    print("=" * 40)
    
    expected_outputs = [
        WORKSPACE_DIR / 'xgboost_model.pkl',
        WORKSPACE_DIR / 'predictions.csv',
        WORKSPACE_DIR / 'eval_metrics.json',
        WORKSPACE_DIR / 'calibrated_model.pkl',
        WORKSPACE_DIR / 'calibrated_predictions.csv'
    ]
    
    created_files = []
    missing_files = []
    
    for file_path in expected_outputs:
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✓ {file_path.name} ({file_size} bytes)")
            created_files.append(file_path)
        else:
            print(f"✗ {file_path.name} (missing)")
            missing_files.append(file_path)
    
    print(f"\nFiles created: {len(created_files)}/{len(expected_outputs)}")
    
    if missing_files:
        print(f"Missing files: {[f.name for f in missing_files]}")
    else:
        print("All expected output files were created successfully!")
    
    # Show workspace contents
    print(f"\nWorkspace contents ({WORKSPACE_DIR}):")
    if WORKSPACE_DIR.exists():
        workspace_files = list(WORKSPACE_DIR.glob('*'))
        for file_path in sorted(workspace_files):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  {file_path.name} ({size} bytes)")
    else:
        print("  Workspace directory does not exist")

def main():
    """Main function to run all tests."""
    print("XGBoost Pipeline Individual Step Testing")
    print("=" * 60)
    
    # Setup environment
    cursus_available = setup_environment()
    
    # Load configuration
    config_data = load_configuration()
    
    if config_data['missing_files']:
        print("Cannot proceed with missing required files!")
        return
    
    # Run individual step tests
    step_tester = run_individual_step_tests(config_data)
    
    # Generate test summary
    generate_test_summary(step_tester, config_data)
    
    # Verify output files
    verify_output_files(config_data['directories'])
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL STEP TESTING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
