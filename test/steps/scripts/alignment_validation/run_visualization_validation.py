#!/usr/bin/env python3
"""
Enhanced Alignment Validation with Visualization Integration

This script runs alignment validation for all scripts and generates enhanced reports
with scoring, quality ratings, and visualization charts using the new AlignmentScorer
functionality implemented in Phase 1-4 of the visualization integration plan.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester


class VisualizedAlignmentValidator:
    """Enhanced validator with visualization and scoring capabilities."""
    
    def __init__(self):
        """Initialize the validator with proper directory paths."""
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.scripts_dir = self.project_root / "src" / "cursus" / "steps" / "scripts"
        self.contracts_dir = self.project_root / "src" / "cursus" / "steps" / "contracts"
        self.specs_dir = self.project_root / "src" / "cursus" / "steps" / "specs"
        self.builders_dir = self.project_root / "src" / "cursus" / "steps" / "builders"
        self.configs_dir = self.project_root / "src" / "cursus" / "steps" / "configs"
        
        # Output directories
        self.output_dir = Path(__file__).parent / "reports"
        self.json_reports_dir = self.output_dir / "json"
        self.html_reports_dir = self.output_dir / "html"
        self.charts_dir = self.output_dir / "charts"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.json_reports_dir.mkdir(exist_ok=True)
        self.html_reports_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        
        # Initialize the unified alignment tester
        self.tester = UnifiedAlignmentTester(
            scripts_dir=str(self.scripts_dir),
            contracts_dir=str(self.contracts_dir),
            specs_dir=str(self.specs_dir),
            builders_dir=str(self.builders_dir),
            configs_dir=str(self.configs_dir)
        )
        
        # Script names to validate
        self.script_names = [
            "currency_conversion",
            "dummy_training", 
            "model_calibration",
            "package",
            "payload",
            "risk_table_mapping",
            "tabular_preprocessing",
            "xgboost_model_evaluation",
            "xgboost_training"
        ]
    
    def validate_script_with_visualization(self, script_name: str) -> Dict[str, Any]:
        """
        Run comprehensive validation for a single script with visualization.
        
        Args:
            script_name: Name of the script to validate
            
        Returns:
            Validation results with scoring information
        """
        print(f"\n{'='*60}")
        print(f"🔍 VALIDATING SCRIPT WITH VISUALIZATION: {script_name}")
        print(f"{'='*60}")
        
        try:
            # Run full validation for this specific script to populate the report
            self.tester.run_full_validation(target_scripts=[script_name])
            
            # Get the validation results for status determination
            results = self.tester.get_validation_summary()
            results['script_name'] = script_name
            
            # Export JSON report with scoring using the populated report
            json_output_path = self.json_reports_dir / f"{script_name}_alignment_report.json"
            json_content = self.tester.export_report(
                format='json',
                output_path=str(json_output_path),
                generate_chart=False,  # We'll generate chart separately to control location
                script_name=script_name
            )
            
            # Export HTML report with scoring using the populated report
            html_output_path = self.html_reports_dir / f"{script_name}_alignment_report.html"
            html_content = self.tester.export_report(
                format='html',
                output_path=str(html_output_path),
                generate_chart=False,  # We'll generate chart separately to control location
                script_name=script_name
            )
            
            # Generate alignment score chart in the correct charts directory
            try:
                scorer = self.tester.report.get_scorer()
                chart_path = scorer.generate_chart(script_name, str(self.charts_dir))
                if chart_path:
                    print(f"📊 Alignment score chart generated: {chart_path}")
                else:
                    print("⚠️  Chart generation skipped (matplotlib not available)")
            except Exception as e:
                print(f"⚠️  Chart generation failed: {e}")
            
            # Get scoring information from the populated report
            try:
                scorer = self.tester.report.get_scorer()
                overall_score = scorer.calculate_overall_score()
                overall_rating = scorer.get_rating(overall_score)
                
                # Print scoring summary
                print(f"📈 Alignment Quality Scoring:")
                print(f"   Overall Score: {overall_score:.1f}/100 ({overall_rating})")
                
                # Print level scores
                level_names = {
                    "level1_script_contract": "L1 Script↔Contract",
                    "level2_contract_spec": "L2 Contract↔Spec", 
                    "level3_spec_dependencies": "L3 Spec↔Dependencies",
                    "level4_builder_config": "L4 Builder↔Config"
                }
                
                for level_key, level_name in level_names.items():
                    score, passed, total = scorer.calculate_level_score(level_key)
                    print(f"   {level_name}: {score:.1f}/100 ({passed}/{total} tests)")
                
                # Add scoring information to results
                results['scoring'] = {
                    'overall_score': overall_score,
                    'overall_rating': overall_rating,
                    'level_scores': {}
                }
                
                for level_key, level_name in level_names.items():
                    score, passed, total = scorer.calculate_level_score(level_key)
                    results['scoring']['level_scores'][level_key] = {
                        'score': score,
                        'passed': passed,
                        'total': total,
                        'name': level_name
                    }
                    
            except Exception as e:
                print(f"   ⚠️  Scoring calculation failed: {e}")
                results['scoring_error'] = str(e)
            
            # Add metadata
            results['metadata'] = {
                'script_path': str(self.scripts_dir / f"{script_name}.py"),
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '2.0.0-visualization',
                'visualization_enabled': True,
                'reports': {
                    'json': str(json_output_path),
                    'html': str(html_output_path),
                    'chart': str(self.charts_dir / f"{script_name}_alignment_score_chart.png")
                }
            }
            
            print(f"✅ Validation completed with visualization for {script_name}")
            return results
            
        except Exception as e:
            error_result = {
                'script_name': script_name,
                'overall_status': 'ERROR',
                'error': str(e),
                'metadata': {
                    'script_path': str(self.scripts_dir / f"{script_name}.py"),
                    'validation_timestamp': datetime.now().isoformat(),
                    'validator_version': '2.0.0-visualization',
                    'visualization_enabled': True
                }
            }
            
            print(f"❌ ERROR validating {script_name}: {e}")
            return error_result
    
    def run_all_visualized_validations(self):
        """Run alignment validation with visualization for all scripts."""
        print("🚀 Starting Enhanced Alignment Validation with Visualization")
        print(f"📁 Scripts Directory: {self.scripts_dir}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📊 Charts Directory: {self.charts_dir}")
        
        # Validation results summary
        validation_summary = {
            'total_scripts': len(self.script_names),
            'passed_scripts': 0,
            'failed_scripts': 0,
            'error_scripts': 0,
            'validation_timestamp': datetime.now().isoformat(),
            'visualization_enabled': True,
            'script_results': {},
            'scoring_summary': {
                'average_score': 0.0,
                'score_distribution': {},
                'quality_ratings': {}
            }
        }
        
        all_scores = []
        quality_ratings = {}
        
        # Validate each script with visualization
        for script_name in self.script_names:
            try:
                results = self.validate_script_with_visualization(script_name)
                
                # Update summary
                status = results.get('overall_status', 'UNKNOWN')
                validation_summary['script_results'][script_name] = {
                    'status': status,
                    'timestamp': results.get('metadata', {}).get('validation_timestamp'),
                    'scoring': results.get('scoring', {}),
                    'reports': results.get('metadata', {}).get('reports', {})
                }
                
                # Collect scoring data
                if 'scoring' in results:
                    score = results['scoring'].get('overall_score', 0.0)
                    rating = results['scoring'].get('overall_rating', 'Unknown')
                    all_scores.append(score)
                    quality_ratings[rating] = quality_ratings.get(rating, 0) + 1
                
                if status == 'PASSING':
                    validation_summary['passed_scripts'] += 1
                elif status == 'FAILING':
                    validation_summary['failed_scripts'] += 1
                else:
                    validation_summary['error_scripts'] += 1
                    
            except Exception as e:
                print(f"❌ Failed to validate {script_name}: {e}")
                validation_summary['error_scripts'] += 1
                validation_summary['script_results'][script_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Calculate scoring summary
        if all_scores:
            validation_summary['scoring_summary']['average_score'] = sum(all_scores) / len(all_scores)
            validation_summary['scoring_summary']['score_distribution'] = {
                'min': min(all_scores),
                'max': max(all_scores),
                'scores': all_scores
            }
            validation_summary['scoring_summary']['quality_ratings'] = quality_ratings
        
        # Save enhanced summary
        self._save_enhanced_validation_summary(validation_summary)
        self._print_enhanced_final_summary(validation_summary)
    
    def _save_enhanced_validation_summary(self, summary: Dict[str, Any]):
        """Save the enhanced validation summary with scoring information."""
        summary_file = self.output_dir / "enhanced_validation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 Enhanced validation summary saved: {summary_file}")
    
    def _print_enhanced_final_summary(self, summary: Dict[str, Any]):
        """Print the enhanced final validation summary with scoring."""
        print(f"\n{'='*80}")
        print("🎯 ENHANCED VALIDATION SUMMARY WITH VISUALIZATION")
        print(f"{'='*80}")
        
        total = summary['total_scripts']
        passed = summary['passed_scripts']
        failed = summary['failed_scripts']
        errors = summary['error_scripts']
        
        print(f"📊 Total Scripts: {total}")
        print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"⚠️  Errors: {errors} ({errors/total*100:.1f}%)")
        
        # Print scoring summary
        scoring_summary = summary.get('scoring_summary', {})
        if scoring_summary.get('average_score'):
            print(f"\n📈 SCORING SUMMARY:")
            print(f"   Average Score: {scoring_summary['average_score']:.1f}/100")
            
            score_dist = scoring_summary.get('score_distribution', {})
            if score_dist:
                print(f"   Score Range: {score_dist.get('min', 0):.1f} - {score_dist.get('max', 0):.1f}")
            
            quality_ratings = scoring_summary.get('quality_ratings', {})
            if quality_ratings:
                print(f"   Quality Ratings:")
                for rating, count in quality_ratings.items():
                    print(f"     {rating}: {count} scripts")
        
        print(f"\n📁 Reports saved in: {self.output_dir}")
        print(f"   📄 JSON reports: {self.json_reports_dir}")
        print(f"   🌐 HTML reports: {self.html_reports_dir}")
        print(f"   📊 Charts: {self.charts_dir}")
        
        # List scripts by status with scores
        if passed > 0:
            passing_scripts = [(name, result) for name, result in summary['script_results'].items() 
                             if result['status'] == 'PASSING']
            print(f"\n✅ PASSING SCRIPTS WITH SCORES ({len(passing_scripts)}):")
            for script, result in passing_scripts:
                scoring = result.get('scoring', {})
                if scoring:
                    score = scoring.get('overall_score', 0)
                    rating = scoring.get('overall_rating', 'Unknown')
                    print(f"   • {script}: {score:.1f}/100 ({rating})")
                else:
                    print(f"   • {script}: No scoring data")
        
        print(f"\n{'='*80}")


def main():
    """Main entry point for the enhanced alignment validation program."""
    try:
        validator = VisualizedAlignmentValidator()
        validator.run_all_visualized_validations()
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Fatal error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
