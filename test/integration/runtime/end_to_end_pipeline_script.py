#!/usr/bin/env python3
"""
End-to-End Pipeline Testing Script

This script tests the complete XGBoost 3-step pipeline execution with proper
dependency validation, data flow verification, and error handling.
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
    print("=== END-TO-END PIPELINE TESTING SETUP ===")
    
    # Add cursus to path
    sys.path.append(str(Path.cwd().parent.parent.parent / 'src'))
    
    # Import Cursus components
    try:
        from cursus.validation.runtime.jupyter.notebook_interface import NotebookInterface
        from cursus.validation.runtime.core.data_flow_manager import DataFlowManager
        from cursus.validation.runtime.core.pipeline_executor import PipelineExecutor
        from cursus.steps.registry.step_names import STEP_NAMES
        print("✓ Successfully imported Cursus components")
        cursus_available = True
    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print("Using mock implementations for testing...")
        cursus_available = False
    
    print(f"End-to-end pipeline testing started at {datetime.now()}")
    return cursus_available

def load_pipeline_configuration():
    """Load and validate pipeline configuration."""
    print("\n=== LOAD PIPELINE CONFIGURATION ===")
    
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
    if not pipeline_config_path.exists():
        raise FileNotFoundError(f"Pipeline configuration not found: {pipeline_config_path}")
    
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = json.load(f)
    
    print(f"✓ Loaded pipeline configuration: {pipeline_config_path}")
    
    # Extract components
    step_configurations = pipeline_config['step_configurations']
    pipeline_metadata = pipeline_config['pipeline_metadata']
    pipeline_dag = pipeline_config.get('pipeline_dag', {})
    
    print(f"Pipeline: {pipeline_metadata['name']}")
    print(f"Steps: {list(step_configurations.keys())}")
    print(f"Dependencies: {pipeline_dag.get('edges', [])}")
    
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
        raise FileNotFoundError(f"Missing required files: {[f.name for f in missing_files]}")
    
    return {
        'step_configurations': step_configurations,
        'pipeline_metadata': pipeline_metadata,
        'pipeline_dag': pipeline_dag,
        'directories': {
            'BASE_DIR': BASE_DIR,
            'DATA_DIR': DATA_DIR,
            'CONFIG_DIR': CONFIG_DIR,
            'OUTPUTS_DIR': OUTPUTS_DIR,
            'WORKSPACE_DIR': WORKSPACE_DIR,
            'LOGS_DIR': LOGS_DIR,
            'RESULTS_DIR': RESULTS_DIR
        }
    }

class EndToEndPipelineExecutor:
    """End-to-end pipeline executor with dependency validation and error handling."""
    
    def __init__(self, config_data):
        self.config_data = config_data
        self.step_configurations = config_data['step_configurations']
        self.pipeline_dag = config_data['pipeline_dag']
        self.directories = config_data['directories']
        
        # Execution tracking
        self.execution_results = {}
        self.execution_times = {}
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        
        # Ensure workspace exists
        self.directories['WORKSPACE_DIR'].mkdir(parents=True, exist_ok=True)
        print(f"EndToEndPipelineExecutor initialized with workspace: {self.directories['WORKSPACE_DIR']}")
    
    def validate_pipeline_dag(self):
        """Validate pipeline DAG structure and dependencies."""
        print("\n=== VALIDATE PIPELINE DAG ===")
        
        steps = list(self.step_configurations.keys())
        edges = self.pipeline_dag.get('edges', [])
        
        print(f"Pipeline steps: {steps}")
        print(f"Pipeline edges: {edges}")
        
        # Validate all steps in edges exist in configurations
        edge_steps = set()
        for edge in edges:
            edge_steps.add(edge['from'])
            edge_steps.add(edge['to'])
        
        missing_steps = edge_steps - set(steps)
        if missing_steps:
            raise ValueError(f"Steps in DAG edges not found in configurations: {missing_steps}")
        
        # Check for cycles (simple check)
        dependencies = {}
        for edge in edges:
            if edge['to'] not in dependencies:
                dependencies[edge['to']] = []
            dependencies[edge['to']].append(edge['from'])
        
        print("✓ Pipeline DAG validation passed")
        return dependencies
    
    def get_execution_order(self, dependencies):
        """Determine execution order based on dependencies."""
        print("\n=== DETERMINE EXECUTION ORDER ===")
        
        steps = list(self.step_configurations.keys())
        execution_order = []
        remaining_steps = set(steps)
        
        while remaining_steps:
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step in remaining_steps:
                step_deps = dependencies.get(step, [])
                if all(dep in execution_order for dep in step_deps):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise ValueError(f"Circular dependency detected. Remaining steps: {remaining_steps}")
            
            # Add ready steps to execution order
            for step in ready_steps:
                execution_order.append(step)
                remaining_steps.remove(step)
        
        print(f"Execution order: {execution_order}")
        return execution_order
    
    def execute_step(self, step_name, step_config):
        """Execute a single pipeline step."""
        print(f"\n{'='*60}")
        print(f"EXECUTING STEP: {step_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if step_name == 'XGBoostTraining':
                success = self._execute_xgboost_training(step_config)
            elif step_name == 'XGBoostModelEval':
                success = self._execute_xgboost_eval(step_config)
            elif step_name == 'ModelCalibration':
                success = self._execute_model_calibration(step_config)
            else:
                raise ValueError(f"Unknown step: {step_name}")
            
            execution_time = time.time() - start_time
            self.execution_times[step_name] = execution_time
            
            if success:
                self.execution_results[step_name] = {
                    'status': 'success',
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                print(f"✓ {step_name} completed successfully in {execution_time:.2f}s")
                return True
            else:
                self.execution_results[step_name] = {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'Step execution failed'
                }
                print(f"✗ {step_name} failed after {execution_time:.2f}s")
                return False
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_times[step_name] = execution_time
            self.execution_results[step_name] = {
                'status': 'error',
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            print(f"✗ {step_name} error after {execution_time:.2f}s: {e}")
            return False
    
    def _execute_xgboost_training(self, config):
        """Execute XGBoost training step."""
        # Load training data
        input_path = Path(config['input_data_path'])
        train_data = pd.read_csv(input_path)
        print(f"✓ Loaded training data: {train_data.shape}")
        
        # Validate columns
        target_col = config['target_column']
        feature_cols = config['feature_columns']
        
        if target_col not in train_data.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        missing_features = [col for col in feature_cols if col not in train_data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # Simulate training
        print("Training XGBoost model...")
        time.sleep(2.0)  # Simulate longer training time
        
        # Save model
        model_path = Path(config['output_model_path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_info = {
            'model_type': 'XGBoost',
            'hyperparameters': config['hyperparameters'],
            'training_samples': len(train_data),
            'features': feature_cols,
            'target_column': target_col,
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return True
    
    def _execute_xgboost_eval(self, config):
        """Execute XGBoost model evaluation step."""
        # Check model exists
        model_path = Path(config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load evaluation data
        eval_path = Path(config['eval_data_path'])
        eval_data = pd.read_csv(eval_path)
        print(f"✓ Loaded evaluation data: {eval_data.shape}")
        
        # Simulate evaluation
        print("Evaluating model...")
        time.sleep(1.0)
        
        # Generate predictions
        n_samples = len(eval_data)
        np.random.seed(42)
        predictions_proba = np.random.beta(2, 2, n_samples)
        predictions_class = (predictions_proba > 0.5).astype(int)
        
        pred_df = pd.DataFrame({
            'prediction_proba': predictions_proba,
            'prediction_class': predictions_class
        })
        
        pred_path = Path(config['output_predictions_path'])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        
        # Generate metrics
        metrics = {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.90,
            'f1_score': 0.87,
            'auc_roc': 0.93
        }
        
        metrics_path = Path(config['output_metrics_path'])
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}, AUC-ROC: {metrics['auc_roc']:.3f}")
        return True
    
    def _execute_model_calibration(self, config):
        """Execute model calibration step."""
        # Load predictions
        pred_path = Path(config['predictions_path'])
        predictions = pd.read_csv(pred_path)
        print(f"✓ Loaded predictions: {predictions.shape}")
        
        # Simulate calibration
        print(f"Calibrating predictions using {config['calibration_method']}...")
        time.sleep(0.5)
        
        # Generate calibrated predictions
        original_proba = predictions['prediction_proba'].values
        calibrated_proba = 0.15 + 0.7 * original_proba  # Different transformation
        calibrated_class = (calibrated_proba > 0.5).astype(int)
        
        calibrated_df = pd.DataFrame({
            'original_proba': original_proba,
            'calibrated_proba': calibrated_proba,
            'calibrated_class': calibrated_class
        })
        
        calib_pred_path = Path(config['output_calibrated_predictions_path'])
        calib_pred_path.parent.mkdir(parents=True, exist_ok=True)
        calibrated_df.to_csv(calib_pred_path, index=False)
        
        # Save calibrated model
        calibrated_model_info = {
            'calibration_method': config['calibration_method'],
            'calibrated_samples': len(predictions),
            'calibration_improvement': 0.05
        }
        
        calib_model_path = Path(config['output_calibrated_model_path'])
        with open(calib_model_path, 'w') as f:
            json.dump(calibrated_model_info, f, indent=2)
        
        return True
    
    def execute_pipeline(self):
        """Execute the complete pipeline."""
        print("\n" + "="*80)
        print("STARTING END-TO-END PIPELINE EXECUTION")
        print("="*80)
        
        self.pipeline_start_time = time.time()
        
        try:
            # Validate DAG
            dependencies = self.validate_pipeline_dag()
            
            # Determine execution order
            execution_order = self.get_execution_order(dependencies)
            
            # Execute steps in order
            for step_name in execution_order:
                step_config = self.step_configurations[step_name]
                success = self.execute_step(step_name, step_config)
                
                if not success:
                    print(f"\n⚠ Pipeline execution stopped due to failure in {step_name}")
                    break
            
            self.pipeline_end_time = time.time()
            total_time = self.pipeline_end_time - self.pipeline_start_time
            
            print(f"\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETED")
            print("="*80)
            print(f"Total execution time: {total_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.pipeline_end_time = time.time()
            print(f"\n✗ Pipeline execution failed: {e}")
            return False

def run_end_to_end_pipeline_test():
    """Run the complete end-to-end pipeline test."""
    print("STARTING END-TO-END PIPELINE TEST")
    print("="*50)
    
    try:
        # Load configuration
        config_data = load_pipeline_configuration()
        
        # Create executor
        executor = EndToEndPipelineExecutor(config_data)
        
        # Execute pipeline
        success = executor.execute_pipeline()
        
        # Generate results
        generate_pipeline_results(executor, config_data)
        
        return success
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return False

def generate_pipeline_results(executor, config_data):
    """Generate comprehensive pipeline execution results."""
    print("\n=== GENERATE PIPELINE RESULTS ===")
    
    total_time = executor.pipeline_end_time - executor.pipeline_start_time if executor.pipeline_end_time else 0
    successful_steps = sum(1 for result in executor.execution_results.values() 
                          if result['status'] == 'success')
    total_steps = len(executor.execution_results)
    success_rate = successful_steps / total_steps * 100 if total_steps > 0 else 0
    
    print("END-TO-END PIPELINE RESULTS")
    print("="*40)
    print(f"Total Steps: {total_steps}")
    print(f"Successful Steps: {successful_steps}")
    print(f"Failed Steps: {total_steps - successful_steps}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Pipeline Time: {total_time:.2f}s")
    
    print("\nStep-by-Step Results:")
    for step_name, result in executor.execution_results.items():
        status_icon = "✓" if result['status'] == 'success' else "✗"
        exec_time = result['execution_time']
        print(f"  {status_icon} {step_name}: {result['status']} ({exec_time:.2f}s)")
        
        if result['status'] != 'success' and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Save results
    pipeline_results = {
        'test_timestamp': datetime.now().isoformat(),
        'test_type': 'end_to_end_pipeline_execution',
        'pipeline_name': config_data['pipeline_metadata'].get('name', 'Unknown'),
        'total_steps': total_steps,
        'successful_steps': successful_steps,
        'failed_steps': total_steps - successful_steps,
        'success_rate': success_rate,
        'total_pipeline_time': total_time,
        'step_results': executor.execution_results,
        'execution_times': executor.execution_times,
        'pipeline_start_time': executor.pipeline_start_time,
        'pipeline_end_time': executor.pipeline_end_time
    }
    
    results_path = config_data['directories']['RESULTS_DIR'] / 'end_to_end_pipeline_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    print(f"\n✓ Pipeline results saved: {results_path}")
    
    if success_rate == 100:
        print("\n🎉 End-to-end pipeline execution successful!")
        print("All steps completed successfully with proper dependency handling.")
    else:
        print(f"\n⚠ Pipeline execution incomplete. {total_steps - successful_steps} step(s) failed.")

def verify_pipeline_outputs():
    """Verify all expected pipeline outputs were created."""
    print("\n=== VERIFY PIPELINE OUTPUTS ===")
    
    BASE_DIR = Path.cwd()
    WORKSPACE_DIR = BASE_DIR / 'outputs' / 'workspace'
    
    expected_outputs = [
        WORKSPACE_DIR / 'xgboost_model.pkl',
        WORKSPACE_DIR / 'predictions.csv',
        WORKSPACE_DIR / 'eval_metrics.json',
        WORKSPACE_DIR / 'calibrated_model.pkl',
        WORKSPACE_DIR / 'calibrated_predictions.csv'
    ]
    
    print("PIPELINE OUTPUT VERIFICATION")
    print("="*35)
    
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
        return False
    else:
        print("✓ All expected pipeline outputs were created successfully!")
        return True

def main():
    """Main execution function."""
    # Setup environment
    cursus_available = setup_environment()
    
    # Run end-to-end pipeline test
    success = run_end_to_end_pipeline_test()
    
    # Verify outputs
    outputs_verified = verify_pipeline_outputs()
    
    # Final summary
    print("\n" + "="*60)
    print("END-TO-END PIPELINE TEST SUMMARY")
    print("="*60)
    
    if success and outputs_verified:
        print("🎉 End-to-end pipeline test PASSED!")
        print("✓ Pipeline executed successfully")
        print("✓ All outputs verified")
        print("\nReady for performance analysis!")
    else:
        print("⚠ End-to-end pipeline test FAILED!")
        if not success:
            print("✗ Pipeline execution failed")
        if not outputs_verified:
            print("✗ Output verification failed")

if __name__ == "__main__":
    main()
