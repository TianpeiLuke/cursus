#!/usr/bin/env python3
"""
Detailed Test Coverage Report for Cursus Core Module

This script generates a comprehensive script-by-script coverage report
for the cursus/core module based on pytest coverage analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def load_coverage_data():
    """Load the coverage data from the JSON file."""
    coverage_file = Path(__file__).parent / "coverage_core_tests.json"
    
    if not coverage_file.exists():
        print("❌ Coverage data file not found. Please run pytest with coverage on test/core first.")
        print("Run: cd test && python -m pytest core/ --cov=cursus.core --cov-report=json:coverage_core_tests.json")
        sys.exit(1)
        
    with open(coverage_file, 'r') as f:
        return json.load(f)

def filter_core_files(coverage_data: Dict) -> Dict:
    """Filter coverage data to only include core module files."""
    core_files = {}
    
    for file_path, file_data in coverage_data["files"].items():
        if "/cursus/core/" in file_path and not file_path.endswith("__pycache__"):
            core_files[file_path] = file_data
            
    return core_files

def analyze_script_coverage(file_path: str, file_data: Dict) -> Dict:
    """Analyze coverage for a single script file."""
    summary = file_data["summary"]
    
    # Calculate basic metrics
    total_statements = summary["num_statements"]
    covered_lines = summary["covered_lines"]
    missing_lines = summary["missing_lines"]
    excluded_lines = summary["excluded_lines"]
    coverage_percent = summary["percent_covered"]
    
    # Analyze functions
    functions = file_data.get("functions", {})
    function_coverage = []
    
    for func_name, func_data in functions.items():
        if func_name == "":  # Module-level code
            continue
            
        func_summary = func_data["summary"]
        function_coverage.append({
            "name": func_name,
            "statements": func_summary["num_statements"],
            "covered": func_summary["covered_lines"],
            "missing": func_summary["missing_lines"],
            "coverage_percent": func_summary["percent_covered"]
        })
    
    # Analyze classes
    classes = file_data.get("classes", {})
    class_coverage = []
    
    for class_name, class_data in classes.items():
        if class_name == "":  # Module-level code
            continue
            
        class_summary = class_data["summary"]
        class_coverage.append({
            "name": class_name,
            "statements": class_summary["num_statements"],
            "covered": class_summary["covered_lines"],
            "missing": class_summary["missing_lines"],
            "coverage_percent": class_summary["percent_covered"]
        })
    
    return {
        "file_path": file_path,
        "total_statements": total_statements,
        "covered_lines": covered_lines,
        "missing_lines": missing_lines,
        "excluded_lines": excluded_lines,
        "coverage_percent": coverage_percent,
        "functions": function_coverage,
        "classes": class_coverage
    }

def get_coverage_status(coverage_percent: float) -> Tuple[str, str]:
    """Get coverage status and emoji based on percentage."""
    if coverage_percent >= 80:
        return "Excellent", "🟢"
    elif coverage_percent >= 60:
        return "Good", "🟡"
    elif coverage_percent >= 40:
        return "Fair", "🟠"
    elif coverage_percent > 0:
        return "Poor", "🔴"
    else:
        return "No Coverage", "⚫"

def print_detailed_report(core_files_analysis: List[Dict]):
    """Print a detailed coverage report for all core files."""
    
    print("=" * 100)
    print("📊 DETAILED TEST COVERAGE REPORT - CURSUS CORE MODULE")
    print("=" * 100)
    
    # Sort files by coverage percentage (lowest first to highlight issues)
    sorted_files = sorted(core_files_analysis, key=lambda x: x["coverage_percent"])
    
    total_statements = sum(f["total_statements"] for f in core_files_analysis)
    total_covered = sum(f["covered_lines"] for f in core_files_analysis)
    overall_coverage = (total_covered / total_statements * 100) if total_statements > 0 else 0
    
    print(f"\n📈 OVERALL CORE MODULE COVERAGE: {overall_coverage:.1f}%")
    print(f"   Total Statements: {total_statements:,}")
    print(f"   Covered Lines: {total_covered:,}")
    print(f"   Missing Lines: {total_statements - total_covered:,}")
    
    print(f"\n📁 ANALYZED FILES: {len(core_files_analysis)}")
    
    # Group files by subdirectory
    subdirs = {}
    for file_analysis in sorted_files:
        file_path = file_analysis["file_path"]
        # Extract subdirectory from the path
        path_parts = file_path.split("/")
        
        # Find cursus/core in the path and extract subdirectory
        try:
            core_index = next(i for i, part in enumerate(path_parts) if part == "core")
            if core_index + 1 < len(path_parts):
                subdir = path_parts[core_index + 1]
            else:
                subdir = "root"
        except StopIteration:
            subdir = "unknown"
        
        if subdir not in subdirs:
            subdirs[subdir] = []
        subdirs[subdir].append(file_analysis)
    
    # Print coverage by subdirectory
    print(f"\n📂 COVERAGE BY SUBDIRECTORY:")
    print("-" * 100)
    
    for subdir, files in subdirs.items():
        subdir_statements = sum(f["total_statements"] for f in files)
        subdir_covered = sum(f["covered_lines"] for f in files)
        subdir_coverage = (subdir_covered / subdir_statements * 100) if subdir_statements > 0 else 0
        status, emoji = get_coverage_status(subdir_coverage)
        
        print(f"{emoji} {subdir:<20} {subdir_coverage:>6.1f}% ({subdir_covered:>4}/{subdir_statements:<4}) {status}")
    
    print(f"\n📄 DETAILED FILE-BY-FILE ANALYSIS:")
    print("=" * 100)
    
    for file_analysis in sorted_files:
        file_path = file_analysis["file_path"]
        filename = file_path.split("/")[-1]
        coverage_percent = file_analysis["coverage_percent"]
        status, emoji = get_coverage_status(coverage_percent)
        
        print(f"\n{emoji} {filename}")
        print(f"   Path: {file_path}")
        print(f"   Coverage: {coverage_percent:.1f}% ({file_analysis['covered_lines']}/{file_analysis['total_statements']} statements)")
        print(f"   Status: {status}")
        
        if file_analysis["excluded_lines"] > 0:
            print(f"   Excluded Lines: {file_analysis['excluded_lines']}")
        
        # Show function coverage if any functions exist
        if file_analysis["functions"]:
            print(f"   📋 Functions ({len(file_analysis['functions'])}):")
            for func in file_analysis["functions"]:
                func_status, func_emoji = get_coverage_status(func["coverage_percent"])
                print(f"      {func_emoji} {func['name']:<30} {func['coverage_percent']:>6.1f}% ({func['covered']}/{func['statements']})")
        
        # Show class coverage if any classes exist
        if file_analysis["classes"]:
            print(f"   🏗️  Classes ({len(file_analysis['classes'])}):")
            for cls in file_analysis["classes"]:
                cls_status, cls_emoji = get_coverage_status(cls["coverage_percent"])
                print(f"      {cls_emoji} {cls['name']:<30} {cls['coverage_percent']:>6.1f}% ({cls['covered']}/{cls['statements']})")
        
        print("-" * 80)

def print_summary_statistics(core_files_analysis: List[Dict]):
    """Print summary statistics."""
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print("=" * 50)
    
    # Coverage distribution
    no_coverage = len([f for f in core_files_analysis if f["coverage_percent"] == 0])
    poor_coverage = len([f for f in core_files_analysis if 0 < f["coverage_percent"] < 40])
    fair_coverage = len([f for f in core_files_analysis if 40 <= f["coverage_percent"] < 60])
    good_coverage = len([f for f in core_files_analysis if 60 <= f["coverage_percent"] < 80])
    excellent_coverage = len([f for f in core_files_analysis if f["coverage_percent"] >= 80])
    
    total_files = len(core_files_analysis)
    
    print(f"⚫ No Coverage (0%):        {no_coverage:>3} files ({no_coverage/total_files*100:>5.1f}%)")
    print(f"🔴 Poor Coverage (1-39%):   {poor_coverage:>3} files ({poor_coverage/total_files*100:>5.1f}%)")
    print(f"🟠 Fair Coverage (40-59%):  {fair_coverage:>3} files ({fair_coverage/total_files*100:>5.1f}%)")
    print(f"🟡 Good Coverage (60-79%):  {good_coverage:>3} files ({good_coverage/total_files*100:>5.1f}%)")
    print(f"🟢 Excellent Coverage (80%+): {excellent_coverage:>3} files ({excellent_coverage/total_files*100:>5.1f}%)")
    
    # Top uncovered files
    uncovered_files = [f for f in core_files_analysis if f["coverage_percent"] == 0]
    if uncovered_files:
        print(f"\n🚨 FILES WITH NO COVERAGE ({len(uncovered_files)} files):")
        for file_analysis in uncovered_files[:10]:  # Show top 10
            filename = file_analysis["file_path"].split("/")[-1]
            statements = file_analysis["total_statements"]
            print(f"   • {filename:<40} ({statements:>3} statements)")
        
        if len(uncovered_files) > 10:
            print(f"   ... and {len(uncovered_files) - 10} more files")

def print_recommendations(core_files_analysis: List[Dict]):
    """Print recommendations for improving coverage."""
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    uncovered_files = [f for f in core_files_analysis if f["coverage_percent"] == 0]
    low_coverage_files = [f for f in core_files_analysis if 0 < f["coverage_percent"] < 40]
    
    if uncovered_files:
        print(f"🎯 Priority 1: Add tests for {len(uncovered_files)} files with no coverage")
        print("   Focus on core functionality files first:")
        
        # Prioritize by number of statements
        priority_files = sorted(uncovered_files, key=lambda x: x["total_statements"], reverse=True)[:5]
        for file_analysis in priority_files:
            filename = file_analysis["file_path"].split("/")[-1]
            statements = file_analysis["total_statements"]
            print(f"   • {filename} ({statements} statements)")
    
    if low_coverage_files:
        print(f"\n🎯 Priority 2: Improve coverage for {len(low_coverage_files)} files with low coverage")
        priority_files = sorted(low_coverage_files, key=lambda x: x["total_statements"], reverse=True)[:3]
        for file_analysis in priority_files:
            filename = file_analysis["file_path"].split("/")[-1]
            coverage = file_analysis["coverage_percent"]
            statements = file_analysis["total_statements"]
            print(f"   • {filename} ({coverage:.1f}% coverage, {statements} statements)")
    
    # Calculate potential impact
    total_uncovered_statements = sum(f["total_statements"] for f in uncovered_files)
    total_statements = sum(f["total_statements"] for f in core_files_analysis)
    potential_improvement = (total_uncovered_statements / total_statements * 100) if total_statements > 0 else 0
    
    print(f"\n📈 POTENTIAL IMPACT:")
    print(f"   Adding tests for all uncovered files could improve overall coverage by {potential_improvement:.1f}%")
    
    # Test file suggestions
    print(f"\n📝 SUGGESTED TEST FILES TO CREATE:")
    for subdir, files in {"assembler": [], "base": [], "compiler": [], "config_fields": [], "utils": []}.items():
        subdir_files = [f for f in uncovered_files if f"/{subdir}/" in f["file_path"]]
        if subdir_files:
            print(f"   • test/core/{subdir}/test_{subdir}_comprehensive.py")

def main():
    """Main function to generate the detailed coverage report."""
    
    print("🚀 Loading coverage data...")
    coverage_data = load_coverage_data()
    
    print("🔍 Filtering core module files...")
    core_files = filter_core_files(coverage_data)
    
    if not core_files:
        print("❌ No core module files found in coverage data.")
        sys.exit(1)
    
    print(f"📊 Analyzing {len(core_files)} core module files...")
    
    # Analyze each file
    core_files_analysis = []
    for file_path, file_data in core_files.items():
        analysis = analyze_script_coverage(file_path, file_data)
        core_files_analysis.append(analysis)
    
    # Generate reports
    print_detailed_report(core_files_analysis)
    print_summary_statistics(core_files_analysis)
    print_recommendations(core_files_analysis)
    
    # Save detailed analysis to JSON
    output_file = Path(__file__).parent / "core_detailed_coverage_analysis.json"
    
    detailed_report = {
        "timestamp": coverage_data["meta"]["timestamp"],
        "total_files_analyzed": len(core_files_analysis),
        "overall_coverage": {
            "total_statements": sum(f["total_statements"] for f in core_files_analysis),
            "covered_lines": sum(f["covered_lines"] for f in core_files_analysis),
            "coverage_percent": (sum(f["covered_lines"] for f in core_files_analysis) / 
                              sum(f["total_statements"] for f in core_files_analysis) * 100) 
                              if sum(f["total_statements"] for f in core_files_analysis) > 0 else 0
        },
        "files": core_files_analysis
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    print(f"\n✅ Coverage analysis complete!")

if __name__ == "__main__":
    main()
