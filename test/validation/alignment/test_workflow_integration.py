#!/usr/bin/env python3
"""
Test script for the complete alignment validation visualization integration workflow.

This script demonstrates the full end-to-end workflow including scoring, visualization,
enhanced reporting, and workflow integration capabilities.
"""

import sys
import os
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta


from cursus.validation.alignment.workflow_integration import (
    AlignmentValidationWorkflow, run_alignment_validation_workflow
)
from cursus.validation.alignment.alignment_reporter import AlignmentIssue, SeverityLevel

class TestWorkflowIntegration(unittest.TestCase):
    """Test the complete alignment validation workflow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflow = AlignmentValidationWorkflow(
            output_dir=self.temp_dir,
            enable_charts=True,
            enable_trends=True,
            enable_comparisons=True
        )
        
        # Create sample validation results
        self.sample_validation_results = {
            'script_contract_path_alignment': {
                'passed': True,
                'details': {'script_path': '/opt/ml/code/train.py', 'contract_path': '/opt/ml/code/train.py'}
            },
            'script_contract_environment_vars': {
                'passed': False,
                'issues': [
                    {
                        'level': 'error',
                        'category': 'environment_variables',
                        'message': 'Script accesses undeclared environment variable CUSTOM_VAR',
                        'recommendation': 'Add CUSTOM_VAR to contract environment variables section'
                    }
                ],
                'details': {'undeclared_vars': ['CUSTOM_VAR']}
            },
            'contract_spec_logical_names': {
                'passed': True,
                'details': {'aligned_names': ['training_data', 'model_output']}
            },
            'contract_spec_input_output_mapping': {
                'passed': False,
                'issues': [
                    {
                        'level': 'warning',
                        'category': 'logical_names',
                        'message': 'Contract output model_artifacts not found in specification outputs',
                        'recommendation': 'Update specification to include model_artifacts output'
                    }
                ],
                'details': {'missing_outputs': ['model_artifacts']}
            },
            'spec_dependency_resolution': {
                'passed': True,
                'details': {'resolved_dependencies': ['preprocessing_step', 'feature_engineering_step']}
            },
            'builder_config_environment_vars': {
                'passed': False,
                'issues': [
                    {
                        'level': 'critical',
                        'category': 'configuration',
                        'message': 'Builder does not set required environment variable MODEL_TYPE',
                        'recommendation': 'Update builder to set MODEL_TYPE from configuration'
                    }
                ],
                'details': {'missing_env_vars': ['MODEL_TYPE']}
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow(self):
        """Test the complete alignment validation workflow."""
        # Run the complete workflow
        results = self.workflow.run_validation_workflow(
            validation_results=self.sample_validation_results,
            script_name="test_script",
            load_historical=False,
            save_results=True
        )
        
        # Verify workflow results structure
        self.assertIn('script_name', results)
        self.assertIn('timestamp', results)
        self.assertIn('workflow_config', results)
        self.assertIn('report_created', results)
        self.assertIn('scoring', results)
        self.assertIn('charts_generated', results)
        self.assertIn('reports_generated', results)
        self.assertIn('action_plan', results)
        
        # Verify workflow configuration
        config = results['workflow_config']
        self.assertTrue(config['enable_charts'])
        self.assertTrue(config['enable_trends'])
        self.assertTrue(config['enable_comparisons'])
        
        # Verify scoring results
        scoring = results['scoring']
        self.assertIn('overall_score', scoring)
        self.assertIn('level_scores', scoring)
        self.assertIn('quality_rating', scoring)
        self.assertIn('scoring_report', scoring)
        
        # Verify score ranges
        self.assertGreaterEqual(scoring['overall_score'], 0.0)
        self.assertLessEqual(scoring['overall_score'], 100.0)
        
        # Verify level scores
        level_scores = scoring['level_scores']
        for level, score in level_scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
        
        # Verify action plan
        action_plan = results['action_plan']
        self.assertIn('total_action_items', action_plan)
        self.assertIn('high_priority_items', action_plan)
        self.assertIn('medium_priority_items', action_plan)
        self.assertIn('action_items', action_plan)
        self.assertIn('next_steps', action_plan)
        
        # Should have action items due to critical and error issues
        self.assertGreater(action_plan['total_action_items'], 0)
        self.assertGreater(action_plan['high_priority_items'], 0)
    
    def test_historical_data_integration(self):
        """Test historical data integration for trend analysis."""
        # Create some historical data
        historical_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(5):
            date = base_date + timedelta(days=i*7)
            historical_data.append({
                'timestamp': date.isoformat(),
                'scoring': {
                    'overall_score': 70.0 + i * 5,  # Improving trend
                    'level_scores': {
                        'level1_script_contract': 75.0 + i * 3,
                        'level2_contract_specification': 65.0 + i * 4,
                        'level3_specification_dependencies': 80.0 + i * 2,
                        'level4_builder_configuration': 60.0 + i * 6
                    }
                },
                'summary': {
                    'total_tests': 6,
                    'passed_tests': 3 + i,
                    'failed_tests': 3 - i,
                    'pass_rate': (3 + i) / 6 * 100
                }
            })
        
        # Save historical data
        import json
        historical_file = os.path.join(self.temp_dir, "test_script_historical.json")
        with open(historical_file, 'w') as f:
            json.dump(historical_data, f)
        
        # Run workflow with historical data
        results = self.workflow.run_validation_workflow(
            validation_results=self.sample_validation_results,
            script_name="test_script",
            load_historical=True,
            save_results=True
        )
        
        # Verify historical data was loaded
        self.assertEqual(results['historical_data_loaded'], 5)
        
        # Verify enhanced report has trend analysis
        self.assertIsNotNone(self.workflow.current_report)
        self.assertIn('trends', self.workflow.current_report.quality_metrics)
        
        trends = self.workflow.current_report.quality_metrics['trends']
        self.assertIn('overall_trend', trends)
        self.assertIn('level_trends', trends)
        
        # Should detect improving trend
        overall_trend = trends['overall_trend']
        self.assertEqual(overall_trend['direction'], 'improving')
        self.assertGreater(overall_trend['improvement'], 0)
    
    def test_comparison_data_integration(self):
        """Test comparison data integration."""
        # Create comparison report data
        comparison_data = {
            'baseline_script': {
                'scoring': {
                    'overall_score': 75.0,
                    'level_scores': {
                        'level1_script_contract': 80.0,
                        'level2_contract_specification': 70.0,
                        'level3_specification_dependencies': 85.0,
                        'level4_builder_configuration': 65.0
                    }
                }
            },
            'reference_script': {
                'scoring': {
                    'overall_score': 85.0,
                    'level_scores': {
                        'level1_script_contract': 90.0,
                        'level2_contract_specification': 80.0,
                        'level3_specification_dependencies': 90.0,
                        'level4_builder_configuration': 80.0
                    }
                }
            }
        }
        
        # Save comparison data files
        import json
        for name, data in comparison_data.items():
            comp_file = os.path.join(self.temp_dir, f"{name}_enhanced_report.json")
            with open(comp_file, 'w') as f:
                json.dump(data, f)
        
        # Run workflow
        results = self.workflow.run_validation_workflow(
            validation_results=self.sample_validation_results,
            script_name="test_script",
            load_historical=False,
            save_results=True
        )
        
        # Verify comparison data was loaded
        self.assertEqual(results['comparison_data_loaded'], 2)
        
        # Verify enhanced report has comparison analysis
        self.assertIn('comparisons', self.workflow.current_report.quality_metrics)
        
        comparisons = self.workflow.current_report.quality_metrics['comparisons']
        self.assertIn('baseline_script', comparisons)
        self.assertIn('reference_script', comparisons)
        
        # Verify comparison calculations
        baseline_comp = comparisons['baseline_script']
        self.assertIn('overall_difference', baseline_comp)
        self.assertIn('level_differences', baseline_comp)
        self.assertIn('performance', baseline_comp)
    
    def test_batch_validation(self):
        """Test batch validation workflow."""
        # Create multiple validation configurations
        validation_configs = [
            {
                'script_name': 'script_1',
                'validation_results': self.sample_validation_results,
                'save_results': True
            },
            {
                'script_name': 'script_2',
                'validation_results': {
                    'script_contract_test': {'passed': True},
                    'spec_dependency_test': {'passed': True},
                    'builder_config_test': {'passed': False, 'issues': [
                        {'level': 'error', 'category': 'configuration', 'message': 'Config error'}
                    ]}
                },
                'save_results': True
            },
            {
                'script_name': 'script_3',
                'validation_results': {
                    'all_tests_pass': {'passed': True}
                },
                'save_results': True
            }
        ]
        
        # Run batch validation
        batch_results = self.workflow.run_batch_validation(validation_configs)
        
        # Verify batch results structure
        self.assertEqual(batch_results['total_validations'], 3)
        self.assertEqual(batch_results['successful_validations'], 3)
        self.assertEqual(batch_results['failed_validations'], 0)
        
        # Verify individual validation results
        self.assertIn('validation_results', batch_results)
        self.assertIn('script_1', batch_results['validation_results'])
        self.assertIn('script_2', batch_results['validation_results'])
        self.assertIn('script_3', batch_results['validation_results'])
        
        # Verify batch summary
        self.assertIn('batch_summary', batch_results)
        summary = batch_results['batch_summary']
        self.assertIn('average_score', summary)
        self.assertIn('highest_score', summary)
        self.assertIn('lowest_score', summary)
        self.assertIn('score_distribution', summary)
        
        # Verify score distribution
        distribution = summary['score_distribution']
        total_scores = sum(distribution.values())
        self.assertEqual(total_scores, 3)  # Should have 3 scores
    
    def test_convenience_function(self):
        """Test the convenience function for running workflows."""
        results = run_alignment_validation_workflow(
            validation_results=self.sample_validation_results,
            script_name="convenience_test",
            output_dir=self.temp_dir,
            enable_charts=True,
            enable_trends=False,
            enable_comparisons=False,
            save_results=True
        )
        
        # Verify results structure
        self.assertIn('script_name', results)
        self.assertIn('scoring', results)
        self.assertIn('action_plan', results)
        
        # Verify workflow configuration was applied
        config = results['workflow_config']
        self.assertTrue(config['enable_charts'])
        self.assertFalse(config['enable_trends'])
        self.assertFalse(config['enable_comparisons'])
    
    def test_action_plan_generation(self):
        """Test action plan generation based on validation results."""
        # Create validation results with various issue types
        validation_results = {
            'critical_test': {
                'passed': False,
                'issues': [
                    {
                        'level': 'critical',
                        'category': 'configuration',
                        'message': 'Critical configuration error',
                        'recommendation': 'Fix critical config'
                    }
                ]
            },
            'error_test': {
                'passed': False,
                'issues': [
                    {
                        'level': 'error',
                        'category': 'environment_variables',
                        'message': 'Environment variable error',
                        'recommendation': 'Fix env vars'
                    }
                ]
            },
            'warning_test': {
                'passed': False,
                'issues': [
                    {
                        'level': 'warning',
                        'category': 'logical_names',
                        'message': 'Logical name warning',
                        'recommendation': 'Fix logical names'
                    }
                ]
            },
            'passing_test': {
                'passed': True
            }
        }
        
        results = self.workflow.run_validation_workflow(
            validation_results=validation_results,
            script_name="action_plan_test",
            load_historical=False,
            save_results=False
        )
        
        action_plan = results['action_plan']
        
        # Should have high priority items due to critical and error issues
        self.assertGreater(action_plan['high_priority_items'], 0)
        
        # Should have medium priority items due to warning issues
        self.assertGreater(action_plan['medium_priority_items'], 0)
        
        # Verify action items structure
        action_items = action_plan['action_items']
        self.assertGreater(len(action_items), 0)
        
        for item in action_items:
            self.assertIn('priority', item)
            self.assertIn('title', item)
            self.assertIn('description', item)
            self.assertIn('category', item)
            self.assertIn('estimated_effort', item)
            self.assertIn('impact', item)
            self.assertIn('recommendations', item)
        
        # Verify next steps
        self.assertIn('next_steps', action_plan)
        self.assertGreater(len(action_plan['next_steps']), 0)

def run_workflow_demo():
    """Run a comprehensive demonstration of the workflow integration."""
    print("=" * 80)
    print("ALIGNMENT VALIDATION WORKFLOW INTEGRATION DEMO")
    print("=" * 80)
    
    # Create test instance
    test_instance = TestWorkflowIntegration()
    test_instance.setUp()
    
    try:
        print("\n1. Testing complete workflow...")
        test_instance.test_complete_workflow()
        print("✅ Complete workflow test passed")
        
        print("\n2. Testing historical data integration...")
        test_instance.test_historical_data_integration()
        print("✅ Historical data integration test passed")
        
        print("\n3. Testing comparison data integration...")
        test_instance.test_comparison_data_integration()
        print("✅ Comparison data integration test passed")
        
        print("\n4. Testing batch validation...")
        test_instance.test_batch_validation()
        print("✅ Batch validation test passed")
        
        print("\n5. Testing convenience function...")
        test_instance.test_convenience_function()
        print("✅ Convenience function test passed")
        
        print("\n6. Testing action plan generation...")
        test_instance.test_action_plan_generation()
        print("✅ Action plan generation test passed")
        
        print("\n" + "=" * 80)
        print("WORKFLOW INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nAll 4 phases of the alignment validation visualization integration plan have been implemented:")
        print("✅ Phase 1: AlignmentScorer class with 4-level scoring system")
        print("✅ Phase 2: Chart Generation Infrastructure")
        print("✅ Phase 3: Enhanced Report Generation")
        print("✅ Phase 4: Workflow Integration")
        print("\nKey features implemented:")
        print("📊 Comprehensive scoring system with weighted levels")
        print("📈 Trend analysis with historical data tracking")
        print("🔄 Comparison analysis across validation runs")
        print("📋 Enhanced reporting with actionable recommendations")
        print("🎯 Complete workflow integration with batch processing")
        print("📊 Advanced chart generation and visualization")
        print("🔧 Action plan generation for improvement guidance")
        
    finally:
        test_instance.tearDown()

if __name__ == "__main__":
    # Run demo first
    run_workflow_demo()
    
    # Then run unit tests
    print("\n" + "=" * 80)
    print("RUNNING UNIT TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)
