"""
Alignment Validation Workflow Integration.

This module provides workflow integration for the enhanced alignment validation
system, demonstrating how to use scoring, visualization, and reporting features
together in a complete validation workflow.
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .alignment_reporter import AlignmentReport, ValidationResult, AlignmentIssue, SeverityLevel
from .enhanced_reporter import EnhancedAlignmentReport
from .alignment_scorer import score_alignment_results
from ..shared.chart_utils import create_score_bar_chart, create_quality_distribution_chart


class AlignmentValidationWorkflow:
    """
    Complete workflow for alignment validation with scoring and visualization.
    
    This class orchestrates the entire alignment validation process including:
    - Running validation tests
    - Generating scores and ratings
    - Creating visualizations
    - Producing comprehensive reports
    - Managing historical data
    """
    
    def __init__(self, 
                 output_dir: str = "alignment_validation_results",
                 enable_charts: bool = True,
                 enable_trends: bool = True,
                 enable_comparisons: bool = True):
        """
        Initialize the alignment validation workflow.
        
        Args:
            output_dir: Directory to save all outputs
            enable_charts: Whether to generate charts
            enable_trends: Whether to perform trend analysis
            enable_comparisons: Whether to perform comparison analysis
        """
        self.output_dir = Path(output_dir)
        self.enable_charts = enable_charts
        self.enable_trends = enable_trends
        self.enable_comparisons = enable_comparisons
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize report storage
        self.current_report: Optional[EnhancedAlignmentReport] = None
        self.historical_reports: List[Dict[str, Any]] = []
        self.comparison_reports: Dict[str, Dict[str, Any]] = {}
    
    def run_validation_workflow(self, 
                               validation_results: Dict[str, Any],
                               script_name: str = "alignment_validation",
                               load_historical: bool = True,
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete alignment validation workflow.
        
        Args:
            validation_results: Raw validation results from alignment tests
            script_name: Name identifier for this validation run
            load_historical: Whether to load historical data for trends
            save_results: Whether to save results to disk
            
        Returns:
            Complete workflow results including scores, charts, and reports
        """
        workflow_results = {
            'script_name': script_name,
            'timestamp': datetime.now().isoformat(),
            'workflow_config': {
                'enable_charts': self.enable_charts,
                'enable_trends': self.enable_trends,
                'enable_comparisons': self.enable_comparisons
            }
        }
        
        print(f"\n{'='*80}")
        print(f"ALIGNMENT VALIDATION WORKFLOW: {script_name}")
        print(f"{'='*80}")
        
        # Step 1: Create enhanced alignment report
        print("\n1. Creating enhanced alignment report...")
        self.current_report = self._create_enhanced_report(validation_results)
        workflow_results['report_created'] = True
        
        # Step 2: Load historical data for trends
        if self.enable_trends and load_historical:
            print("\n2. Loading historical data for trend analysis...")
            historical_data = self._load_historical_data(script_name)
            if historical_data:
                self.current_report.add_historical_data(historical_data)
                workflow_results['historical_data_loaded'] = len(historical_data)
            else:
                workflow_results['historical_data_loaded'] = 0
        
        # Step 3: Load comparison data
        if self.enable_comparisons:
            print("\n3. Loading comparison data...")
            comparison_data = self._load_comparison_data(script_name)
            if comparison_data:
                self.current_report.add_comparison_data(comparison_data)
                workflow_results['comparison_data_loaded'] = len(comparison_data)
            else:
                workflow_results['comparison_data_loaded'] = 0
        
        # Step 4: Generate comprehensive scoring
        print("\n4. Generating comprehensive scoring...")
        scoring_results = self._generate_scoring(script_name)
        workflow_results['scoring'] = scoring_results
        
        # Step 5: Generate visualizations
        chart_paths = []
        if self.enable_charts:
            print("\n5. Generating visualizations...")
            chart_paths = self._generate_visualizations(script_name)
            workflow_results['charts_generated'] = chart_paths
        
        # Step 6: Generate comprehensive reports
        print("\n6. Generating comprehensive reports...")
        report_paths = self._generate_reports(script_name, save_results)
        workflow_results['reports_generated'] = report_paths
        
        # Step 7: Save current results as historical data
        if save_results:
            print("\n7. Saving results for future trend analysis...")
            self._save_as_historical(script_name)
            workflow_results['saved_as_historical'] = True
        
        # Step 8: Print comprehensive summary
        print("\n8. Generating comprehensive summary...")
        self.current_report.print_enhanced_summary()
        
        # Step 9: Generate improvement action plan
        print("\n9. Generating improvement action plan...")
        action_plan = self._generate_action_plan()
        workflow_results['action_plan'] = action_plan
        
        print(f"\n{'='*80}")
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Charts generated: {len(chart_paths)}")
        print(f"📋 Reports generated: {len(report_paths)}")
        print(f"🎯 Action items: {len(action_plan.get('action_items', []))}")
        
        return workflow_results
    
    def _create_enhanced_report(self, validation_results: Dict[str, Any]) -> EnhancedAlignmentReport:
        """Create enhanced alignment report from validation results."""
        report = EnhancedAlignmentReport()
        
        # Convert validation results to report format
        # This is a simplified conversion - in practice, this would be more sophisticated
        for test_name, result in validation_results.items():
            validation_result = ValidationResult(
                test_name=test_name,
                passed=result.get('passed', False),
                issues=[],
                details=result.get('details', {})
            )
            
            # Add issues if present
            if 'issues' in result:
                for issue_data in result['issues']:
                    issue = AlignmentIssue(
                        level=SeverityLevel(issue_data.get('level', 'info')),
                        category=issue_data.get('category', 'general'),
                        message=issue_data.get('message', 'Unknown issue'),
                        recommendation=issue_data.get('recommendation', '')
                    )
                    validation_result.add_issue(issue)
            
            # Determine which level this test belongs to
            level = self._determine_test_level(test_name)
            if level == 1:
                report.add_level1_result(test_name, validation_result)
            elif level == 2:
                report.add_level2_result(test_name, validation_result)
            elif level == 3:
                report.add_level3_result(test_name, validation_result)
            elif level == 4:
                report.add_level4_result(test_name, validation_result)
        
        return report
    
    def _determine_test_level(self, test_name: str) -> int:
        """Determine which alignment level a test belongs to."""
        test_lower = test_name.lower()
        
        if any(keyword in test_lower for keyword in ['script', 'contract', 'path_alignment']):
            return 1
        elif any(keyword in test_lower for keyword in ['spec', 'specification', 'logical_names']):
            return 2
        elif any(keyword in test_lower for keyword in ['dependency', 'dependencies', 'property_paths']):
            return 3
        elif any(keyword in test_lower for keyword in ['builder', 'configuration', 'config']):
            return 4
        else:
            return 1  # Default to level 1
    
    def _load_historical_data(self, script_name: str) -> List[Dict[str, Any]]:
        """Load historical validation data for trend analysis."""
        historical_file = self.output_dir / f"{script_name}_historical.json"
        
        if historical_file.exists():
            try:
                with open(historical_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load historical data: {e}")
        
        return []
    
    def _load_comparison_data(self, script_name: str) -> Dict[str, Dict[str, Any]]:
        """Load comparison data from other validation runs."""
        comparison_data = {}
        
        # Look for other validation reports in the same directory
        for report_file in self.output_dir.glob("*_enhanced_report.json"):
            if script_name not in report_file.name:
                try:
                    with open(report_file, 'r') as f:
                        data = json.load(f)
                        comparison_name = report_file.stem.replace('_enhanced_report', '')
                        comparison_data[comparison_name] = data
                except Exception as e:
                    print(f"Warning: Could not load comparison data from {report_file}: {e}")
        
        return comparison_data
    
    def _generate_scoring(self, script_name: str) -> Dict[str, Any]:
        """Generate comprehensive scoring results."""
        scoring_results = {
            'overall_score': self.current_report.get_alignment_score(),
            'level_scores': self.current_report.get_level_scores(),
            'quality_rating': self.current_report.get_scorer().get_rating(
                self.current_report.get_alignment_score()
            ),
            'scoring_report': self.current_report.get_scoring_report()
        }
        
        return scoring_results
    
    def _generate_visualizations(self, script_name: str) -> List[str]:
        """Generate all visualization charts."""
        chart_paths = []
        
        try:
            # Generate main alignment score chart
            main_chart = self.current_report.generate_alignment_chart(
                str(self.output_dir / f"{script_name}_alignment_scores.png")
            )
            if main_chart:
                chart_paths.append(main_chart)
            
            # Generate trend charts if historical data available
            if self.enable_trends and hasattr(self.current_report, 'quality_metrics'):
                trend_charts = self.current_report.generate_trend_charts(str(self.output_dir))
                chart_paths.extend(trend_charts)
            
            # Generate comparison charts if comparison data available
            if self.enable_comparisons and hasattr(self.current_report, 'quality_metrics'):
                comparison_charts = self.current_report.generate_comparison_charts(str(self.output_dir))
                chart_paths.extend(comparison_charts)
            
            # Generate quality distribution chart
            all_scores = []
            level_scores = self.current_report.get_level_scores()
            all_scores.extend(level_scores.values())
            all_scores.append(self.current_report.get_alignment_score())
            
            if all_scores:
                dist_chart = create_quality_distribution_chart(
                    scores=all_scores,
                    title=f"Alignment Quality Distribution - {script_name}",
                    output_path=str(self.output_dir / f"{script_name}_quality_distribution.png")
                )
                if dist_chart:
                    chart_paths.append(dist_chart)
            
        except Exception as e:
            print(f"Warning: Some charts could not be generated: {e}")
        
        return chart_paths
    
    def _generate_reports(self, script_name: str, save_results: bool) -> List[str]:
        """Generate comprehensive reports."""
        report_paths = []
        
        if save_results:
            # Generate enhanced JSON report
            json_path = self.output_dir / f"{script_name}_enhanced_report.json"
            self.current_report.export_enhanced_json(str(json_path))
            report_paths.append(str(json_path))
            
            # Generate enhanced HTML report
            html_path = self.output_dir / f"{script_name}_enhanced_report.html"
            html_content = self.current_report.export_to_html()
            with open(html_path, 'w') as f:
                f.write(html_content)
            report_paths.append(str(html_path))
            
            # Generate basic JSON report for compatibility
            basic_json_path = self.output_dir / f"{script_name}_basic_report.json"
            basic_content = self.current_report.export_to_json()
            with open(basic_json_path, 'w') as f:
                f.write(basic_content)
            report_paths.append(str(basic_json_path))
        
        return report_paths
    
    def _save_as_historical(self, script_name: str):
        """Save current results as historical data for future trend analysis."""
        historical_file = self.output_dir / f"{script_name}_historical.json"
        
        # Load existing historical data
        historical_data = []
        if historical_file.exists():
            try:
                with open(historical_file, 'r') as f:
                    historical_data = json.load(f)
            except Exception:
                historical_data = []
        
        # Add current results
        current_data = {
            'timestamp': datetime.now().isoformat(),
            'scoring': {
                'overall_score': self.current_report.get_alignment_score(),
                'level_scores': self.current_report.get_level_scores()
            },
            'summary': json.loads(self.current_report.export_to_json())['summary']
        }
        
        historical_data.append(current_data)
        
        # Keep only last 50 entries to prevent file from growing too large
        if len(historical_data) > 50:
            historical_data = historical_data[-50:]
        
        # Save updated historical data
        with open(historical_file, 'w') as f:
            json.dump(historical_data, f, indent=2, default=str)
    
    def _generate_action_plan(self) -> Dict[str, Any]:
        """Generate actionable improvement plan."""
        suggestions = self.current_report.generate_improvement_suggestions()
        
        # Organize suggestions by priority
        high_priority = [s for s in suggestions if s['priority'] == 'high']
        medium_priority = [s for s in suggestions if s['priority'] == 'medium']
        
        # Create action items
        action_items = []
        
        for suggestion in high_priority:
            action_items.append({
                'priority': 'HIGH',
                'title': suggestion['title'],
                'description': suggestion['description'],
                'category': suggestion['category'],
                'estimated_effort': 'Medium to High',
                'impact': 'High',
                'recommendations': suggestion['recommendations']
            })
        
        for suggestion in medium_priority:
            action_items.append({
                'priority': 'MEDIUM',
                'title': suggestion['title'],
                'description': suggestion['description'],
                'category': suggestion['category'],
                'estimated_effort': 'Low to Medium',
                'impact': 'Medium',
                'recommendations': suggestion['recommendations']
            })
        
        action_plan = {
            'total_action_items': len(action_items),
            'high_priority_items': len(high_priority),
            'medium_priority_items': len(medium_priority),
            'action_items': action_items,
            'next_steps': [
                'Review high priority action items with development team',
                'Create implementation timeline for critical fixes',
                'Set up regular alignment validation monitoring',
                'Schedule follow-up validation after improvements'
            ]
        }
        
        return action_plan
    
    def run_batch_validation(self, 
                           validation_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run validation workflow for multiple configurations.
        
        Args:
            validation_configs: List of validation configurations
            
        Returns:
            Batch validation results
        """
        batch_results = {
            'total_validations': len(validation_configs),
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_results': {},
            'batch_summary': {}
        }
        
        print(f"\n{'='*80}")
        print(f"BATCH ALIGNMENT VALIDATION - {len(validation_configs)} configurations")
        print(f"{'='*80}")
        
        all_scores = []
        
        for i, config in enumerate(validation_configs, 1):
            script_name = config.get('script_name', f'validation_{i}')
            validation_results = config.get('validation_results', {})
            
            print(f"\n[{i}/{len(validation_configs)}] Processing {script_name}...")
            
            try:
                workflow_result = self.run_validation_workflow(
                    validation_results=validation_results,
                    script_name=script_name,
                    load_historical=config.get('load_historical', True),
                    save_results=config.get('save_results', True)
                )
                
                batch_results['validation_results'][script_name] = workflow_result
                batch_results['successful_validations'] += 1
                
                # Collect score for batch analysis
                if 'scoring' in workflow_result:
                    all_scores.append(workflow_result['scoring']['overall_score'])
                
            except Exception as e:
                print(f"❌ Failed to process {script_name}: {e}")
                batch_results['failed_validations'] += 1
                batch_results['validation_results'][script_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generate batch summary
        if all_scores:
            batch_results['batch_summary'] = {
                'average_score': sum(all_scores) / len(all_scores),
                'highest_score': max(all_scores),
                'lowest_score': min(all_scores),
                'score_distribution': {
                    'excellent': len([s for s in all_scores if s >= 90]),
                    'good': len([s for s in all_scores if 80 <= s < 90]),
                    'satisfactory': len([s for s in all_scores if 70 <= s < 80]),
                    'needs_work': len([s for s in all_scores if 60 <= s < 70]),
                    'poor': len([s for s in all_scores if s < 60])
                }
            }
            
            # Generate batch quality distribution chart
            if self.enable_charts:
                batch_chart = create_quality_distribution_chart(
                    scores=all_scores,
                    title="Batch Alignment Quality Distribution",
                    output_path=str(self.output_dir / "batch_quality_distribution.png")
                )
                if batch_chart:
                    batch_results['batch_chart'] = batch_chart
        
        print(f"\n{'='*80}")
        print("BATCH VALIDATION COMPLETED")
        print(f"{'='*80}")
        print(f"✅ Successful: {batch_results['successful_validations']}")
        print(f"❌ Failed: {batch_results['failed_validations']}")
        if all_scores:
            print(f"📊 Average Score: {batch_results['batch_summary']['average_score']:.1f}")
        
        return batch_results


def run_alignment_validation_workflow(validation_results: Dict[str, Any],
                                    script_name: str = "alignment_validation",
                                    output_dir: str = "alignment_validation_results",
                                    **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run the complete alignment validation workflow.
    
    Args:
        validation_results: Raw validation results from alignment tests
        script_name: Name identifier for this validation run
        output_dir: Directory to save all outputs
        **kwargs: Additional workflow configuration options
        
    Returns:
        Complete workflow results
    """
    workflow = AlignmentValidationWorkflow(
        output_dir=output_dir,
        enable_charts=kwargs.get('enable_charts', True),
        enable_trends=kwargs.get('enable_trends', True),
        enable_comparisons=kwargs.get('enable_comparisons', True)
    )
    
    return workflow.run_validation_workflow(
        validation_results=validation_results,
        script_name=script_name,
        load_historical=kwargs.get('load_historical', True),
        save_results=kwargs.get('save_results', True)
    )
