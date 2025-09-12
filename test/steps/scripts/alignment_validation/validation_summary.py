#!/usr/bin/env python3
"""
Validation Summary and Analysis

This script provides a comprehensive summary of the alignment validation results
and analysis of the current state of script alignment in the Cursus system.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def load_validation_results() -> Dict[str, Any]:
    """Load the validation summary and individual reports."""
    reports_dir = Path(__file__).parent / "reports"

    # Load overall summary
    summary_file = reports_dir / "validation_summary.json"
    if not summary_file.exists():
        print("❌ Validation summary not found. Run validation first.")
        sys.exit(1)

    with open(summary_file, "r") as f:
        summary = json.load(f)

    # Load individual reports
    json_dir = reports_dir / "json"
    individual_reports = {}

    if json_dir.exists():
        for report_file in json_dir.glob("*_alignment_report.json"):
            script_name = report_file.stem.replace("_alignment_report", "")
            with open(report_file, "r") as f:
                individual_reports[script_name] = json.load(f)

    return {"summary": summary, "individual_reports": individual_reports}


def analyze_validation_results(results: Dict[str, Any]):
    """Analyze and display validation results."""
    summary = results["summary"]
    individual_reports = results["individual_reports"]

    print("🔍 CURSUS SCRIPT ALIGNMENT VALIDATION ANALYSIS")
    print("=" * 80)

    # Overall statistics
    total = summary["total_scripts"]
    passed = summary["passed_scripts"]
    failed = summary["failed_scripts"]
    errors = summary["error_scripts"]

    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Scripts Analyzed: {total}")
    print(f"   ✅ Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"   ❌ Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"   ⚠️  Errors: {errors} ({errors/total*100:.1f}%)")

    # Issue analysis
    print(f"\n🔍 DETAILED ISSUE ANALYSIS")
    print("-" * 50)

    issue_categories = {}
    severity_counts = {"CRITICAL": 0, "ERROR": 0, "WARNING": 0, "INFO": 0}
    level_failures = {"level1": 0, "level2": 0, "level3": 0, "level4": 0}

    for script_name, report in individual_reports.items():
        print(f"\n📄 {script_name.upper()}")

        for level_num in range(1, 5):
            level_key = f"level{level_num}"
            level_data = report.get(level_key, {})

            if not level_data.get("passed", True):
                level_failures[level_key] += 1

                level_names = {
                    "level1": "Script ↔ Contract",
                    "level2": "Contract ↔ Specification",
                    "level3": "Specification ↔ Dependencies",
                    "level4": "Builder ↔ Configuration",
                }

                print(f"   ❌ {level_names[level_key]}")

                for issue in level_data.get("issues", []):
                    severity = issue.get("severity", "ERROR")
                    category = issue.get("category", "unknown")
                    message = issue.get("message", "No message")

                    # Count issues
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    issue_categories[category] = issue_categories.get(category, 0) + 1

                    print(f"      • {severity} [{category}]: {message}")

    # Summary of common issues
    print(f"\n📈 ISSUE PATTERNS")
    print("-" * 30)

    print(f"\n🔥 By Severity:")
    for severity, count in sorted(
        severity_counts.items(), key=lambda x: x[1], reverse=True
    ):
        if count > 0:
            print(f"   {severity}: {count}")

    print(f"\n📂 By Category:")
    for category, count in sorted(
        issue_categories.items(), key=lambda x: x[1], reverse=True
    ):
        if count > 0:
            print(f"   {category}: {count}")

    print(f"\n🎯 By Validation Level:")
    level_names = {
        "level1": "Script ↔ Contract",
        "level2": "Contract ↔ Specification",
        "level3": "Specification ↔ Dependencies",
        "level4": "Builder ↔ Configuration",
    }

    for level_key, count in sorted(
        level_failures.items(), key=lambda x: x[1], reverse=True
    ):
        if count > 0:
            print(f"   {level_names[level_key]}: {count}/{total} scripts failing")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 25)

    if issue_categories.get("missing_contract", 0) > 0:
        print("   1. ⚠️  Missing contract files detected")
        print("      → The validation system expects JSON contract files")
        print("      → Current contracts are Python files (.py)")
        print("      → Update validation system to handle Python contracts")

    if issue_categories.get("missing_file", 0) > 0:
        print("   2. 📁 Missing specification/builder files")
        print("      → Some scripts lack corresponding spec or builder files")
        print("      → Review file naming conventions and create missing files")

    if level_failures.get("level1", 0) == total:
        print("   3. 🔧 Contract alignment issues")
        print("      → All scripts failing Level 1 validation")
        print("      → Review contract file format expectations")
        print("      → Ensure proper script-contract mapping")

    print(f"\n🎯 NEXT STEPS")
    print("-" * 15)
    print("   1. Fix validation system to handle Python contract files")
    print("   2. Review and update file naming conventions")
    print("   3. Create missing specification and builder files")
    print("   4. Re-run validation after fixes")
    print("   5. Address remaining alignment issues")

    # File locations
    print(f"\n📁 GENERATED REPORTS")
    print("-" * 20)
    reports_dir = Path(__file__).parent / "reports"
    print(f"   📊 Summary: {reports_dir / 'validation_summary.json'}")
    print(f"   📄 JSON Reports: {reports_dir / 'json'}")
    print(f"   🌐 HTML Reports: {reports_dir / 'html'}")

    print(f"\n{'=' * 80}")


def main():
    """Main entry point."""
    try:
        results = load_validation_results()
        analyze_validation_results(results)

    except Exception as e:
        print(f"❌ Error analyzing validation results: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
