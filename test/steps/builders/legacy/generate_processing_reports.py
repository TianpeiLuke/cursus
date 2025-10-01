#!/usr/bin/env python3
"""
Generate comprehensive test reports for Processing step builders.

This script uses the BuilderTestReporter to generate detailed JSON reports
in the same format as alignment validation reports.
"""

import sys
from pathlib import Path

)

from cursus.validation.builders.builder_reporter import BuilderTestReporter

def main():
    """Generate reports for all Processing step builders."""
    print("Generating Processing Step Builder Test Reports...")
    print("=" * 60)
    
    # Create reporter with output directory
    output_dir = Path(__file__).parent / "reports"
    reporter = BuilderTestReporter(output_dir)
    
    # Generate reports for all Processing step builders
    reports = reporter.test_step_type_builders("Processing")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING STEP BUILDERS REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    
    total_builders = len(reports)
    passing_builders = sum(1 for r in reports.values() if r.is_passing())
    
    print(f"\nTotal Processing Builders Tested: {total_builders}")
    print(f"Passing Builders: {passing_builders}")
    print(f"Success Rate: {(passing_builders/total_builders*100):.1f}%" if total_builders > 0 else "N/A")
    
    print(f"\nReports saved in: {output_dir}")
    print("  📁 Individual reports: reports/individual/")
    print("  📊 Summary report: reports/json/processing_builder_test_summary.json")
    
    # Show individual builder status
    print(f"\nIndividual Builder Status:")
    for name, report in reports.items():
        status_icon = "✅" if report.is_passing() else "❌"
        status = report.summary.overall_status if report.summary else "UNKNOWN"
        pass_rate = f"{report.summary.pass_rate:.1f}%" if report.summary else "N/A"
        print(f"  {status_icon} {name}: {status} ({pass_rate})")

if __name__ == "__main__":
    main()
