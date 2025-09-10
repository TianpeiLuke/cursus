"""
Enhanced Alignment Validation Reporting System.

This module extends the basic alignment reporting with advanced features including
detailed scoring metadata, trend analysis, comparative reporting, and enhanced
visualization capabilities.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import statistics

from .alignment_reporter import AlignmentReport, ValidationResult, AlignmentSummary
from .alignment_scorer import AlignmentScorer
from ..shared.chart_utils import (
    create_score_bar_chart, create_comparison_chart, create_trend_chart,
    create_quality_distribution_chart, get_quality_rating
)


class EnhancedAlignmentReport(AlignmentReport):
    """
    Enhanced alignment report with advanced scoring, trending, and comparison capabilities.
    
    Extends the base AlignmentReport with:
    - Historical trend tracking
    - Comparative analysis across multiple validation runs
    - Advanced scoring metadata
    - Enhanced visualization options
    - Quality improvement recommendations
    """
    
    def __init__(self):
        super().__init__()
        self.historical_data: List[Dict[str, Any]] = []
        self.comparison_data: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, Any] = {}
        self.improvement_suggestions: List[Dict[str, Any]] = []
        
    def add_historical_data(self, historical_reports: List[Dict[str, Any]]):
        """
        Add historical validation data for trend analysis.
        
        Args:
            historical_reports: List of previous validation report data
        """
        self.historical_data = historical_reports
        self._analyze_trends()
    
    def add_comparison_data(self, comparison_reports: Dict[str, Dict[str, Any]]):
        """
        Add comparison data from other validation runs.
        
        Args:
            comparison_reports: Dictionary mapping comparison names to report data
        """
        self.comparison_data = comparison_reports
        self._analyze_comparisons()
    
    def _analyze_trends(self):
        """Analyze trends from historical data."""
        if not self.historical_data:
            # Initialize empty trends structure even when no data
            self.quality_metrics['trends'] = {
                'timestamps': [],
                'overall_scores': [],
                'level_scores': {
                    'level1_script_contract': [],
                    'level2_contract_specification': [],
                    'level3_specification_dependencies': [],
                    'level4_builder_configuration': []
                },
                'overall_trend': {
                    'direction': 'no_data',
                    'slope': 0.0,
                    'improvement': 0.0,
                    'volatility': 0.0
                },
                'level_trends': {}
            }
            return
            
        # Extract scores over time
        timestamps = []
        overall_scores = []
        level_scores = {
            'level1_script_contract': [],
            'level2_contract_specification': [],
            'level3_specification_dependencies': [],
            'level4_builder_configuration': []
        }
        
        for report_data in self.historical_data:
            # Handle None or malformed data gracefully
            if report_data is None:
                continue
                
            if 'timestamp' in report_data:
                timestamps.append(report_data['timestamp'])
            else:
                timestamps.append(datetime.now().isoformat())
                
            # Extract overall score
            if 'scoring' in report_data and report_data['scoring'] is not None and 'overall_score' in report_data['scoring']:
                overall_scores.append(report_data['scoring']['overall_score'])
            else:
                overall_scores.append(0.0)
            
            # Extract level scores
            if 'scoring' in report_data and report_data['scoring'] is not None and 'level_scores' in report_data['scoring']:
                level_data = report_data['scoring']['level_scores']
                if level_data is not None:
                    for level in level_scores.keys():
                        if level in level_data:
                            level_scores[level].append(level_data[level])
                        else:
                            level_scores[level].append(0.0)
                else:
                    for level in level_scores.keys():
                        level_scores[level].append(0.0)
            else:
                for level in level_scores.keys():
                    level_scores[level].append(0.0)
        
        # Calculate trend metrics
        self.quality_metrics['trends'] = {
            'timestamps': timestamps,
            'overall_scores': overall_scores,
            'level_scores': level_scores,
            'overall_trend': self._calculate_trend(overall_scores),
            'level_trends': {
                level: self._calculate_trend(scores)
                for level, scores in level_scores.items()
            }
        }
    
    def _calculate_trend(self, scores: List[float]) -> Dict[str, Any]:
        """
        Calculate trend metrics for a series of scores.
        
        Args:
            scores: List of scores over time
            
        Returns:
            Dictionary containing trend analysis
        """
        if len(scores) < 2:
            return {
                'direction': 'insufficient_data',
                'slope': 0.0,
                'improvement': 0.0,
                'volatility': 0.0
            }
        
        # Calculate simple linear trend
        x = list(range(len(scores)))
        n = len(scores)
        
        # Linear regression slope
        sum_x = sum(x)
        sum_y = sum(scores)
        sum_xy = sum(x[i] * scores[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        # Determine trend direction
        if slope > 0.5:
            direction = 'improving'
        elif slope < -0.5:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # Calculate improvement (latest vs first)
        improvement = scores[-1] - scores[0] if scores else 0.0
        
        # Calculate volatility (standard deviation)
        volatility = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        return {
            'direction': direction,
            'slope': slope,
            'improvement': improvement,
            'volatility': volatility,
            'latest_score': scores[-1] if scores else 0.0,
            'best_score': max(scores) if scores else 0.0,
            'worst_score': min(scores) if scores else 0.0
        }
    
    def _analyze_comparisons(self):
        """Analyze comparison data against other validation runs."""
        if not self.comparison_data:
            # Initialize empty comparisons structure even when no data
            self.quality_metrics['comparisons'] = {}
            return
        
        current_score = self.get_alignment_score()
        current_level_scores = self.get_level_scores()
        
        comparisons = {}
        
        for comparison_name, comparison_report in self.comparison_data.items():
            # Handle None or malformed data gracefully
            if comparison_report is None:
                continue
                
            # Extract comparison scores
            comp_overall = 0.0
            comp_levels = {}
            
            if 'scoring' in comparison_report and comparison_report['scoring'] is not None:
                scoring_data = comparison_report['scoring']
                if isinstance(scoring_data, dict):
                    comp_overall = scoring_data.get('overall_score', 0.0)
                    comp_levels = scoring_data.get('level_scores', {})
                else:
                    # Handle malformed scoring data
                    comp_overall = 0.0
                    comp_levels = {}
            
            # Calculate differences
            overall_diff = current_score - comp_overall
            level_diffs = {}
            
            for level, current_level_score in current_level_scores.items():
                comp_level_score = comp_levels.get(level, 0.0)
                level_diffs[level] = current_level_score - comp_level_score
            
            comparisons[comparison_name] = {
                'overall_difference': overall_diff,
                'level_differences': level_diffs,
                'comparison_overall_score': comp_overall,
                'comparison_level_scores': comp_levels,
                'performance': 'better' if overall_diff > 0 else 'worse' if overall_diff < 0 else 'equal'
            }
        
        self.quality_metrics['comparisons'] = comparisons
    
    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate specific improvement suggestions based on scoring analysis.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Get current scores
        overall_score = self.get_alignment_score()
        level_scores = self.get_level_scores()
        
        # Overall score suggestions
        if overall_score < 70:
            suggestions.append({
                'category': 'overall',
                'priority': 'high',
                'title': 'Critical Alignment Issues',
                'description': f'Overall alignment score ({overall_score:.1f}) is below acceptable threshold',
                'recommendations': [
                    'Focus on fixing critical and error-level alignment issues',
                    'Review and update script contracts for better alignment',
                    'Ensure specification dependencies are properly resolved',
                    'Validate builder configuration matches specifications'
                ]
            })
        elif overall_score < 85:
            suggestions.append({
                'category': 'overall',
                'priority': 'medium',
                'title': 'Alignment Optimization Opportunities',
                'description': f'Overall alignment score ({overall_score:.1f}) has room for improvement',
                'recommendations': [
                    'Address warning-level alignment issues',
                    'Optimize path mappings and property references',
                    'Improve environment variable handling consistency',
                    'Enhance dependency resolution accuracy'
                ]
            })
        
        # Level-specific suggestions
        level_names = {
            'level1_script_contract': 'Script ↔ Contract Alignment',
            'level2_contract_specification': 'Contract ↔ Specification Alignment',
            'level3_specification_dependencies': 'Specification ↔ Dependencies Alignment',
            'level4_builder_configuration': 'Builder ↔ Configuration Alignment'
        }
        
        for level, score in level_scores.items():
            level_name = level_names.get(level, level)
            
            if score < 70:
                suggestions.append({
                    'category': 'level_specific',
                    'level': level,
                    'priority': 'high',
                    'title': f'Critical Issues in {level_name}',
                    'description': f'{level_name} score ({score:.1f}) requires immediate attention',
                    'recommendations': self._get_level_specific_recommendations(level, score)
                })
            elif score < 85:
                suggestions.append({
                    'category': 'level_specific',
                    'level': level,
                    'priority': 'medium',
                    'title': f'Optimization Opportunities in {level_name}',
                    'description': f'{level_name} score ({score:.1f}) can be improved',
                    'recommendations': self._get_level_specific_recommendations(level, score)
                })
        
        # Trend-based suggestions
        if 'trends' in self.quality_metrics:
            trends = self.quality_metrics['trends']
            overall_trend = trends.get('overall_trend', {})
            
            if overall_trend.get('direction') == 'declining':
                suggestions.append({
                    'category': 'trend',
                    'priority': 'high',
                    'title': 'Declining Alignment Quality Trend',
                    'description': f'Alignment scores are trending downward (slope: {overall_trend.get("slope", 0):.2f})',
                    'recommendations': [
                        'Review recent changes that may have impacted alignment',
                        'Implement stricter alignment validation in CI/CD pipeline',
                        'Conduct alignment quality review with development team',
                        'Consider reverting recent changes if alignment degradation is severe'
                    ]
                })
            elif overall_trend.get('volatility', 0) > 10:
                suggestions.append({
                    'category': 'trend',
                    'priority': 'medium',
                    'title': 'High Alignment Score Volatility',
                    'description': f'Alignment scores show high volatility (std dev: {overall_trend.get("volatility", 0):.1f})',
                    'recommendations': [
                        'Standardize alignment validation processes',
                        'Implement consistent testing practices',
                        'Review and stabilize frequently changing components',
                        'Add more comprehensive alignment test coverage'
                    ]
                })
        
        self.improvement_suggestions = suggestions
        return suggestions
    
    def _get_level_specific_recommendations(self, level: str, score: float) -> List[str]:
        """
        Get specific recommendations for a particular alignment level.
        
        Args:
            level: Alignment level identifier
            score: Current score for the level
            
        Returns:
            List of specific recommendations
        """
        recommendations = []
        
        if level == 'level1_script_contract':
            recommendations.extend([
                'Review script entry points and ensure they match contract specifications',
                'Validate environment variable usage against contract declarations',
                'Check input/output path mappings between script and contract',
                'Ensure script dependencies are properly declared in contract'
            ])
        elif level == 'level2_contract_specification':
            recommendations.extend([
                'Align logical names between contract outputs and specification inputs',
                'Verify contract input/output types match specification expectations',
                'Update contract metadata to match specification requirements',
                'Ensure contract job type specifications are consistent'
            ])
        elif level == 'level3_specification_dependencies':
            recommendations.extend([
                'Review dependency resolution logic and compatible sources',
                'Validate property path references in dependency specifications',
                'Check for circular dependencies in specification chain',
                'Ensure all required dependencies have valid sources'
            ])
        elif level == 'level4_builder_configuration':
            recommendations.extend([
                'Verify builder sets all required environment variables from configuration',
                'Check configuration parameter mapping to builder properties',
                'Ensure builder specification attachment is correct',
                'Validate step creation parameters match configuration'
            ])
        
        # Add score-specific recommendations
        if score < 50:
            recommendations.append('Consider major refactoring of this alignment level')
        elif score < 70:
            recommendations.append('Focus on fixing critical issues in this alignment level')
        
        return recommendations
    
    def generate_enhanced_report(self) -> Dict[str, Any]:
        """
        Generate an enhanced report with all advanced features.
        
        Returns:
            Comprehensive enhanced report dictionary
        """
        # Get base report
        base_report = super().export_to_json()
        base_data = json.loads(base_report)
        
        # Add enhanced features
        improvement_suggestions = self.generate_improvement_suggestions()
        
        # Ensure quality_metrics has all required keys
        if 'trends' not in self.quality_metrics:
            self._analyze_trends()  # This will initialize the trends structure
        if 'comparisons' not in self.quality_metrics:
            self._analyze_comparisons()  # This will initialize the comparisons structure
        
        # Add improvement_suggestions to quality_metrics for test compatibility
        self.quality_metrics['improvement_suggestions'] = improvement_suggestions
        
        enhanced_report = {
            **base_data,
            'quality_metrics': self.quality_metrics,
            'improvement_plan': {
                'suggestions': improvement_suggestions,
                'priority_actions': [s for s in improvement_suggestions if s.get('priority') == 'high'],
                'optimization_opportunities': [s for s in improvement_suggestions if s.get('priority') == 'medium'],
                'total_suggestions': len(improvement_suggestions)
            },
            'metadata': {
                'report_generation_timestamp': datetime.now().isoformat(),
                'enhancement_version': '1.0',
                'quality_rating': get_quality_rating(self.get_alignment_score()),
                'has_historical_data': len(self.historical_data) > 0,
                'has_comparison_data': len(self.comparison_data) > 0,
                'trend_direction': self.quality_metrics.get('trends', {}).get('overall_trend', {}).get('direction', 'unknown')
            },
            'enhanced_features': {
                'quality_metrics': self.quality_metrics,
                'improvement_suggestions': improvement_suggestions,
                'trend_analysis': self.quality_metrics.get('trends', {}),
                'comparison_analysis': self.quality_metrics.get('comparisons', {}),
                'quality_rating': get_quality_rating(self.get_alignment_score()),
                'report_generation_timestamp': datetime.now().isoformat(),
                'enhancement_version': '1.0'
            }
        }
        
        return enhanced_report
    
    def export_enhanced_json(self, output_path: str = None) -> str:
        """
        Export enhanced report to JSON format.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            JSON string of the enhanced report
        """
        enhanced_report = self.generate_enhanced_report()
        json_str = json.dumps(enhanced_report, indent=2, default=str)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def generate_trend_charts(self, output_dir: str = "alignment_reports") -> List[str]:
        """
        Generate trend analysis charts.
        
        Args:
            output_dir: Directory to save charts
            
        Returns:
            List of paths to generated charts
        """
        chart_paths = []
        
        if 'trends' not in self.quality_metrics:
            return chart_paths
        
        trends = self.quality_metrics['trends']
        
        # Overall score trend chart
        if trends.get('overall_scores'):
            timestamps = trends.get('timestamps', [])
            scores = trends['overall_scores']
            
            # Convert timestamps to readable format
            x_labels = []
            for ts in timestamps:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    x_labels.append(dt.strftime('%Y-%m-%d'))
                except:
                    x_labels.append(str(ts))
            
            chart_path = create_trend_chart(
                x_values=x_labels,
                y_values=scores,
                title="Overall Alignment Score Trend",
                x_label="Date",
                y_label="Alignment Score (%)",
                output_path=f"{output_dir}/alignment_trend_overall.png"
            )
            
            if chart_path:
                chart_paths.append(chart_path)
        
        # Level-specific trend charts
        level_scores = trends.get('level_scores', {})
        level_names = {
            'level1_script_contract': 'Script ↔ Contract',
            'level2_contract_specification': 'Contract ↔ Specification',
            'level3_specification_dependencies': 'Specification ↔ Dependencies',
            'level4_builder_configuration': 'Builder ↔ Configuration'
        }
        
        for level, scores in level_scores.items():
            if scores:
                level_name = level_names.get(level, level)
                timestamps = trends.get('timestamps', [])
                
                x_labels = []
                for ts in timestamps:
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        x_labels.append(dt.strftime('%Y-%m-%d'))
                    except:
                        x_labels.append(str(ts))
                
                chart_path = create_trend_chart(
                    x_values=x_labels,
                    y_values=scores,
                    title=f"{level_name} Alignment Score Trend",
                    x_label="Date",
                    y_label="Alignment Score (%)",
                    output_path=f"{output_dir}/alignment_trend_{level}.png"
                )
                
                if chart_path:
                    chart_paths.append(chart_path)
        
        return chart_paths
    
    def generate_comparison_charts(self, output_dir: str = "alignment_reports") -> List[str]:
        """
        Generate comparison analysis charts.
        
        Args:
            output_dir: Directory to save charts
            
        Returns:
            List of paths to generated charts
        """
        chart_paths = []
        
        if 'comparisons' not in self.quality_metrics:
            return chart_paths
        
        comparisons = self.quality_metrics['comparisons']
        
        # Overall score comparison
        categories = ['Current'] + list(comparisons.keys())
        current_score = self.get_alignment_score()
        comparison_scores = [current_score] + [
            comp_data['comparison_overall_score'] 
            for comp_data in comparisons.values()
        ]
        
        chart_path = create_score_bar_chart(
            levels=categories,
            scores=comparison_scores,
            title="Overall Alignment Score Comparison",
            output_path=f"{output_dir}/alignment_comparison_overall.png"
        )
        
        if chart_path:
            chart_paths.append(chart_path)
        
        # Level-specific comparison
        current_level_scores = self.get_level_scores()
        level_names = {
            'level1_script_contract': 'Script ↔ Contract',
            'level2_contract_specification': 'Contract ↔ Specification', 
            'level3_specification_dependencies': 'Specification ↔ Dependencies',
            'level4_builder_configuration': 'Builder ↔ Configuration'
        }
        
        # Create series data for comparison chart
        series_data = {}
        
        # Add current scores
        current_scores = []
        level_labels = []
        for level, score in current_level_scores.items():
            level_labels.append(level_names.get(level, level))
            current_scores.append(score)
        
        series_data['Current'] = current_scores
        
        # Add comparison scores
        for comp_name, comp_data in comparisons.items():
            comp_level_scores = comp_data.get('comparison_level_scores', {})
            comp_scores = []
            for level in current_level_scores.keys():
                comp_scores.append(comp_level_scores.get(level, 0.0))
            series_data[comp_name] = comp_scores
        
        chart_path = create_comparison_chart(
            categories=level_labels,
            series_data=series_data,
            title="Alignment Score Comparison by Level",
            output_path=f"{output_dir}/alignment_comparison_levels.png"
        )
        
        if chart_path:
            chart_paths.append(chart_path)
        
        return chart_paths
    
    def print_enhanced_summary(self):
        """Print an enhanced summary with all advanced features."""
        # Print base summary first
        super().print_summary()
        
        # Print enhanced features
        print("\n" + "=" * 80)
        print("ENHANCED ALIGNMENT ANALYSIS")
        print("=" * 80)
        
        # Quality rating
        overall_score = self.get_alignment_score()
        quality_rating = get_quality_rating(overall_score)
        print(f"\nQuality Rating: {quality_rating}")
        
        # Trend analysis
        if 'trends' in self.quality_metrics:
            trends = self.quality_metrics['trends']
            overall_trend = trends.get('overall_trend', {})
            
            print(f"\nTrend Analysis:")
            print(f"  Direction: {overall_trend.get('direction', 'unknown').title()}")
            print(f"  Improvement: {overall_trend.get('improvement', 0):.1f} points")
            print(f"  Volatility: {overall_trend.get('volatility', 0):.1f}")
            
            if overall_trend.get('direction') == 'improving':
                print("  📈 Alignment quality is improving over time")
            elif overall_trend.get('direction') == 'declining':
                print("  📉 Alignment quality is declining - attention needed")
            else:
                print("  📊 Alignment quality is stable")
        
        # Comparison analysis
        if 'comparisons' in self.quality_metrics:
            comparisons = self.quality_metrics['comparisons']
            print(f"\nComparison Analysis:")
            
            for comp_name, comp_data in comparisons.items():
                performance = comp_data['performance']
                diff = comp_data['overall_difference']
                
                if performance == 'better':
                    print(f"  ✅ {comp_name}: {diff:+.1f} points better")
                elif performance == 'worse':
                    print(f"  ❌ {comp_name}: {diff:+.1f} points worse")
                else:
                    print(f"  ➖ {comp_name}: Equal performance")
        
        # Improvement suggestions
        suggestions = self.generate_improvement_suggestions()
        if suggestions:
            print(f"\nImprovement Suggestions ({len(suggestions)}):")
            
            high_priority = [s for s in suggestions if s['priority'] == 'high']
            medium_priority = [s for s in suggestions if s['priority'] == 'medium']
            
            if high_priority:
                print("  🚨 High Priority:")
                for suggestion in high_priority:
                    print(f"    • {suggestion['title']}")
            
            if medium_priority:
                print("  ⚠️  Medium Priority:")
                for suggestion in medium_priority:
                    print(f"    • {suggestion['title']}")
        
        print("\n" + "=" * 80)
