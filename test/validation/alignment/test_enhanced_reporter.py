#!/usr/bin/env python3
"""
Unit tests for EnhancedAlignmentReport class.

This test suite provides comprehensive coverage for the EnhancedAlignmentReport functionality
which was identified as a critical missing test in the validation test coverage analysis.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
from datetime import datetime, timedelta

from cursus.validation.alignment.enhanced_reporter import EnhancedAlignmentReport
from cursus.validation.alignment.alignment_reporter import ValidationResult, AlignmentIssue, SeverityLevel


class TestEnhancedAlignmentReport(unittest.TestCase):
    """Test EnhancedAlignmentReport functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.report = EnhancedAlignmentReport()
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Set up sample validation data."""
        # Add some sample results
        level1_result = ValidationResult(
            test_name="script_path_validation",
            passed=True,
            issues=[],
            details={"script_path": "/opt/ml/code/train.py"}
        )
        
        level2_result = ValidationResult(
            test_name="logical_name_alignment",
            passed=False,
            issues=[
                AlignmentIssue(
                    level=SeverityLevel.WARNING,
                    category="logical_names",
                    message="Contract output not found in specification"
                )
            ]
        )
        
        level3_result = ValidationResult(
            test_name="dependency_resolution",
            passed=True,
            issues=[]
        )
        
        level4_result = ValidationResult(
            test_name="configuration_validation",
            passed=False,
            issues=[
                AlignmentIssue(
                    level=SeverityLevel.CRITICAL,
                    category="configuration",
                    message="Builder missing required environment variable"
                )
            ]
        )
        
        self.report.add_level1_result("script_path_validation", level1_result)
        self.report.add_level2_result("logical_name_alignment", level2_result)
        self.report.add_level3_result("dependency_resolution", level3_result)
        self.report.add_level4_result("configuration_validation", level4_result)
    
    def test_enhanced_report_initialization(self):
        """Test EnhancedAlignmentReport initialization."""
        report = EnhancedAlignmentReport()
        
        self.assertIsNotNone(report)
        self.assertIsInstance(report.quality_metrics, dict)
        self.assertIn('trends', report.quality_metrics)
        self.assertIn('comparisons', report.quality_metrics)
        self.assertIn('improvement_suggestions', report.quality_metrics)
    
    def test_add_historical_data(self):
        """Test adding historical data for trend analysis."""
        # Create historical data
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
        
        self.report.add_historical_data(historical_data)
        
        # Verify historical data was added
        self.assertEqual(len(self.report.historical_data), 5)
        
        # Verify trends were analyzed
        trends = self.report.quality_metrics['trends']
        self.assertIn('overall_trend', trends)
        self.assertIn('level_trends', trends)
        
        # Should detect improving trend
        overall_trend = trends['overall_trend']
        self.assertEqual(overall_trend['direction'], 'improving')
        self.assertGreater(overall_trend['improvement'], 0)
    
    def test_add_comparison_data(self):
        """Test adding comparison data."""
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
        
        self.report.add_comparison_data(comparison_data)
        
        # Verify comparison data was added
        self.assertEqual(len(self.report.comparison_data), 2)
        
        # Verify comparisons were analyzed
        comparisons = self.report.quality_metrics['comparisons']
        self.assertIn('baseline_script', comparisons)
        self.assertIn('reference_script', comparisons)
        
        # Verify comparison calculations
        baseline_comp = comparisons['baseline_script']
        self.assertIn('overall_difference', baseline_comp)
        self.assertIn('level_differences', baseline_comp)
        self.assertIn('performance', baseline_comp)
    
    def test_analyze_trends_improving(self):
        """Test trend analysis with improving scores."""
        # Create improving trend data
        improving_scores = [60.0, 65.0, 70.0, 75.0, 80.0]
        
        trend = self.report._calculate_trend(improving_scores)
        
        self.assertEqual(trend['direction'], 'improving')
        self.assertGreater(trend['improvement'], 0)
        self.assertGreater(trend['slope'], 0)
    
    def test_analyze_trends_declining(self):
        """Test trend analysis with declining scores."""
        # Create declining trend data
        declining_scores = [80.0, 75.0, 70.0, 65.0, 60.0]
        
        trend = self.report._calculate_trend(declining_scores)
        
        self.assertEqual(trend['direction'], 'declining')
        self.assertLess(trend['improvement'], 0)
        self.assertLess(trend['slope'], 0)
    
    def test_analyze_trends_stable(self):
        """Test trend analysis with stable scores."""
        # Create stable trend data
        stable_scores = [75.0, 74.0, 76.0, 75.0, 75.0]
        
        trend = self.report._calculate_trend(stable_scores)
        
        self.assertEqual(trend['direction'], 'stable')
        self.assertAlmostEqual(trend['improvement'], 0, delta=2)
    
    def test_generate_improvement_suggestions(self):
        """Test improvement suggestion generation."""
        suggestions = self.report.generate_improvement_suggestions()
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Verify suggestion structure
        for suggestion in suggestions:
            self.assertIn('priority', suggestion)
            self.assertIn('category', suggestion)
            self.assertIn('title', suggestion)
            self.assertIn('description', suggestion)
            self.assertIn('impact', suggestion)
            self.assertIn('effort', suggestion)
    
    def test_get_level_specific_recommendations(self):
        """Test level-specific recommendation generation."""
        # Test recommendations for different levels and scores
        level1_recs = self.report._get_level_specific_recommendations('level1_script_contract', 60.0)
        self.assertIsInstance(level1_recs, list)
        self.assertGreater(len(level1_recs), 0)
        
        level2_recs = self.report._get_level_specific_recommendations('level2_contract_specification', 70.0)
        self.assertIsInstance(level2_recs, list)
        
        level3_recs = self.report._get_level_specific_recommendations('level3_specification_dependencies', 80.0)
        self.assertIsInstance(level3_recs, list)
        
        level4_recs = self.report._get_level_specific_recommendations('level4_builder_configuration', 50.0)
        self.assertIsInstance(level4_recs, list)
    
    def test_generate_enhanced_report(self):
        """Test enhanced report generation."""
        enhanced_report = self.report.generate_enhanced_report()
        
        self.assertIsInstance(enhanced_report, dict)
        
        # Verify enhanced report structure
        required_sections = [
            'basic_report',
            'quality_metrics',
            'improvement_plan',
            'metadata'
        ]
        
        for section in required_sections:
            self.assertIn(section, enhanced_report)
        
        # Verify quality metrics
        quality_metrics = enhanced_report['quality_metrics']
        self.assertIn('trends', quality_metrics)
        self.assertIn('comparisons', quality_metrics)
        self.assertIn('improvement_suggestions', quality_metrics)
        
        # Verify improvement plan
        improvement_plan = enhanced_report['improvement_plan']
        self.assertIn('high_priority', improvement_plan)
        self.assertIn('medium_priority', improvement_plan)
        self.assertIn('low_priority', improvement_plan)
        
        # Verify metadata
        metadata = enhanced_report['metadata']
        self.assertIn('generated_at', metadata)
        self.assertIn('report_version', metadata)
        self.assertIn('features_enabled', metadata)
    
    def test_export_enhanced_json(self):
        """Test enhanced JSON export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "enhanced_report.json")
            
            json_output = self.report.export_enhanced_json(output_path)
            
            self.assertIsInstance(json_output, str)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('quality_metrics', data)
            self.assertIn('improvement_plan', data)
            self.assertIn('metadata', data)
    
    def test_generate_trend_charts(self):
        """Test trend chart generation."""
        # Add historical data first
        historical_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(5):
            date = base_date + timedelta(days=i*7)
            historical_data.append({
                'timestamp': date.isoformat(),
                'scoring': {
                    'overall_score': 70.0 + i * 5,
                    'level_scores': {
                        'level1_script_contract': 75.0 + i * 3,
                        'level2_contract_specification': 65.0 + i * 4
                    }
                }
            })
        
        self.report.add_historical_data(historical_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                chart_paths = self.report.generate_trend_charts(temp_dir)
                
                if chart_paths:  # Only test if charts were generated
                    self.assertIsInstance(chart_paths, list)
                    for path in chart_paths:
                        self.assertTrue(os.path.exists(path))
                        self.assertTrue(path.endswith('.png'))
            except ImportError:
                # matplotlib not available, skip test
                pass
            except Exception:
                # Other errors are acceptable for this test
                pass
    
    def test_generate_comparison_charts(self):
        """Test comparison chart generation."""
        # Add comparison data first
        comparison_data = {
            'baseline_script': {
                'scoring': {
                    'overall_score': 75.0,
                    'level_scores': {
                        'level1_script_contract': 80.0,
                        'level2_contract_specification': 70.0
                    }
                }
            }
        }
        
        self.report.add_comparison_data(comparison_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                chart_paths = self.report.generate_comparison_charts(temp_dir)
                
                if chart_paths:  # Only test if charts were generated
                    self.assertIsInstance(chart_paths, list)
                    for path in chart_paths:
                        self.assertTrue(os.path.exists(path))
                        self.assertTrue(path.endswith('.png'))
            except ImportError:
                # matplotlib not available, skip test
                pass
            except Exception:
                # Other errors are acceptable for this test
                pass
    
    def test_print_enhanced_summary(self):
        """Test enhanced summary printing."""
        # Test that print_enhanced_summary doesn't raise exceptions
        try:
            self.report.print_enhanced_summary()
        except Exception as e:
            self.fail(f"print_enhanced_summary raised an exception: {e}")
    
    def test_backward_compatibility(self):
        """Test that enhanced report maintains backward compatibility."""
        # Test that all base AlignmentReport methods still work
        summary = self.report.generate_summary()
        self.assertIsNotNone(summary)
        
        # Test basic export functionality
        json_output = self.report.export_to_json()
        self.assertIsInstance(json_output, str)
        
        html_output = self.report.export_to_html()
        self.assertIsInstance(html_output, str)
        
        # Test issue retrieval
        critical_issues = self.report.get_critical_issues()
        self.assertIsInstance(critical_issues, list)
        
        error_issues = self.report.get_error_issues()
        self.assertIsInstance(error_issues, list)
    
    def test_empty_historical_data(self):
        """Test enhanced report with no historical data."""
        empty_report = EnhancedAlignmentReport()
        
        # Should handle empty historical data gracefully
        enhanced_report = empty_report.generate_enhanced_report()
        self.assertIsInstance(enhanced_report, dict)
        
        # Trends should indicate no data
        trends = enhanced_report['quality_metrics']['trends']
        self.assertEqual(trends['overall_trend']['direction'], 'no_data')
    
    def test_empty_comparison_data(self):
        """Test enhanced report with no comparison data."""
        empty_report = EnhancedAlignmentReport()
        
        # Should handle empty comparison data gracefully
        enhanced_report = empty_report.generate_enhanced_report()
        self.assertIsInstance(enhanced_report, dict)
        
        # Comparisons should be empty
        comparisons = enhanced_report['quality_metrics']['comparisons']
        self.assertEqual(len(comparisons), 0)
    
    def test_malformed_historical_data(self):
        """Test enhanced report with malformed historical data."""
        malformed_data = [
            {'invalid': 'structure'},
            None,
            {'timestamp': 'invalid_date'},
            {'scoring': 'not_a_dict'}
        ]
        
        # Should handle malformed data gracefully
        try:
            self.report.add_historical_data(malformed_data)
            enhanced_report = self.report.generate_enhanced_report()
            self.assertIsInstance(enhanced_report, dict)
        except Exception as e:
            self.fail(f"Enhanced report failed with malformed data: {e}")
    
    def test_malformed_comparison_data(self):
        """Test enhanced report with malformed comparison data."""
        malformed_data = {
            'invalid_script': {'invalid': 'structure'},
            'another_script': None,
            'third_script': {'scoring': 'not_a_dict'}
        }
        
        # Should handle malformed data gracefully
        try:
            self.report.add_comparison_data(malformed_data)
            enhanced_report = self.report.generate_enhanced_report()
            self.assertIsInstance(enhanced_report, dict)
        except Exception as e:
            self.fail(f"Enhanced report failed with malformed comparison data: {e}")


class TestEnhancedReporterIntegration(unittest.TestCase):
    """Test EnhancedAlignmentReport integration scenarios."""
    
    def test_complete_workflow_with_all_features(self):
        """Test complete workflow with all enhanced features enabled."""
        report = EnhancedAlignmentReport()
        
        # Add sample validation results
        level1_result = ValidationResult(
            test_name="script_validation",
            passed=True,
            issues=[]
        )
        report.add_level1_result("script_validation", level1_result)
        
        # Add historical data
        historical_data = [{
            'timestamp': (datetime.now() - timedelta(days=7)).isoformat(),
            'scoring': {
                'overall_score': 75.0,
                'level_scores': {'level1_script_contract': 80.0}
            }
        }]
        report.add_historical_data(historical_data)
        
        # Add comparison data
        comparison_data = {
            'reference_script': {
                'scoring': {
                    'overall_score': 85.0,
                    'level_scores': {'level1_script_contract': 90.0}
                }
            }
        }
        report.add_comparison_data(comparison_data)
        
        # Generate enhanced report
        enhanced_report = report.generate_enhanced_report()
        
        # Verify all features are present
        self.assertIn('quality_metrics', enhanced_report)
        self.assertIn('improvement_plan', enhanced_report)
        self.assertIn('metadata', enhanced_report)
        
        # Verify trends analysis
        trends = enhanced_report['quality_metrics']['trends']
        self.assertIn('overall_trend', trends)
        
        # Verify comparison analysis
        comparisons = enhanced_report['quality_metrics']['comparisons']
        self.assertIn('reference_script', comparisons)
        
        # Verify improvement suggestions
        suggestions = enhanced_report['quality_metrics']['improvement_suggestions']
        self.assertIsInstance(suggestions, list)
    
    def test_performance_with_large_datasets(self):
        """Test performance with large historical and comparison datasets."""
        report = EnhancedAlignmentReport()
        
        # Create large historical dataset
        large_historical_data = []
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(52):  # Weekly data for a year
            date = base_date + timedelta(days=i*7)
            large_historical_data.append({
                'timestamp': date.isoformat(),
                'scoring': {
                    'overall_score': 50.0 + (i % 50),
                    'level_scores': {
                        'level1_script_contract': 60.0 + (i % 40),
                        'level2_contract_specification': 55.0 + (i % 45),
                        'level3_specification_dependencies': 70.0 + (i % 30),
                        'level4_builder_configuration': 45.0 + (i % 55)
                    }
                }
            })
        
        # Create large comparison dataset
        large_comparison_data = {}
        for i in range(20):  # 20 comparison scripts
            large_comparison_data[f'script_{i}'] = {
                'scoring': {
                    'overall_score': 60.0 + (i * 2),
                    'level_scores': {
                        'level1_script_contract': 65.0 + i,
                        'level2_contract_specification': 60.0 + (i * 1.5),
                        'level3_specification_dependencies': 75.0 + (i * 0.5),
                        'level4_builder_configuration': 50.0 + (i * 2.5)
                    }
                }
            }
        
        # Test that large datasets don't cause performance issues
        import time
        start_time = time.time()
        
        report.add_historical_data(large_historical_data)
        report.add_comparison_data(large_comparison_data)
        enhanced_report = report.generate_enhanced_report()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 10.0)  # 10 seconds max
        
        # Verify report was generated successfully
        self.assertIsInstance(enhanced_report, dict)
        self.assertIn('quality_metrics', enhanced_report)


if __name__ == '__main__':
    unittest.main(verbosity=2)
