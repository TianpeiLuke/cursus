#!/usr/bin/env python3
"""
Regenerate alignment validation visualizations with corrected scoring system.
This script processes existing alignment reports and generates updated charts and scoring.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

)

# Import the fixed AlignmentScorer directly from its module file to avoid pydantic dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    'alignment_scorer', 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'cursus', 'validation', 'alignment', 'alignment_scorer.py')
)
alignment_scorer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment_scorer_module)
AlignmentScorer = alignment_scorer_module.AlignmentScorer

def regenerate_visualizations():
    """Regenerate all alignment validation visualizations with corrected scoring."""
    
    # Define the reports directory
    reports_dir = Path("test/steps/scripts/alignment_validation/reports/json")
    
    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        return
    
    # Find all alignment report JSON files
    report_files = list(reports_dir.glob("*_alignment_report.json"))
    
    if not report_files:
        print("No alignment report files found")
        return
    
    print(f"Found {len(report_files)} alignment reports to process")
    print("=" * 80)
    
    success_count = 0
    total_count = len(report_files)
    
    for report_file in sorted(report_files):
        try:
            # Extract script name from filename
            script_name = report_file.stem.replace("_alignment_report", "")
            
            print(f"\nProcessing: {script_name}")
            print("-" * 40)
            
            # Load the existing report
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Use the fixed AlignmentScorer to generate updated scoring and charts
            scorer = AlignmentScorer(report_data)
            
            # Generate comprehensive report with scoring
            scoring_report = scorer.generate_report()
            
            # Print the corrected scoring results
            overall_score = scoring_report["overall"]["score"]
            overall_rating = scoring_report["overall"]["rating"]
            pass_rate = scoring_report["overall"]["pass_rate"]
            
            print(f"Overall Score: {overall_score:.1f}/100 - {overall_rating}")
            print(f"Pass Rate: {pass_rate:.1f}% ({scoring_report['overall']['passed']}/{scoring_report['overall']['total']} tests)")
            
            # Show level scores
            level_names = {
                "level1_script_contract": "Level 1 (Script ↔ Contract)",
                "level2_contract_spec": "Level 2 (Contract ↔ Specification)",
                "level3_spec_dependencies": "Level 3 (Specification ↔ Dependencies)",
                "level4_builder_config": "Level 4 (Builder ↔ Configuration)"
            }
            
            for level, data in scoring_report["levels"].items():
                display_name = level_names.get(level, level)
                print(f"  {display_name}: {data['score']:.1f}/100 ({data['passed']}/{data['total']} tests)")
            
            # Generate updated chart
            chart_path = scorer.generate_chart(script_name, str(reports_dir))
            if chart_path:
                print(f"✅ Updated chart: {chart_path}")
            else:
                print("⚠️  Chart generation skipped (matplotlib not available)")
            
            # Save updated scoring report
            scoring_report_path = reports_dir / f"{script_name}_alignment_score_report.json"
            with open(scoring_report_path, 'w') as f:
                json.dump(scoring_report, f, indent=2)
            print(f"✅ Updated scoring report: {scoring_report_path}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {report_file}: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print(f"REGENERATION COMPLETE")
    print(f"Successfully processed: {success_count}/{total_count} reports")
    print("=" * 80)
    
    if success_count == total_count:
        print("🎉 All alignment visualizations have been successfully regenerated with corrected scoring!")
    else:
        print(f"⚠️  {total_count - success_count} reports had issues during processing")

def main():
    """Main function to run the visualization regeneration."""
    print("Alignment Validation Visualization Regeneration")
    print("=" * 80)
    print("This script regenerates all alignment validation visualizations")
    print("with the corrected scoring system that was recently fixed.")
    print("=" * 80)
    
    regenerate_visualizations()

if __name__ == "__main__":
    main()
