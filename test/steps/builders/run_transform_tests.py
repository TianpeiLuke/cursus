#!/usr/bin/env python3
"""
Runner script for Transform step builder tests.

This script runs the comprehensive test suite for all Transform step builders
and generates detailed reports with scoring and charts.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_transform_step_builders import TransformStepBuilderTestSuite
from cursus.validation.builders.universal_test import UniversalStepBuilderTest


def generate_transform_reports():
    """Generate comprehensive reports for Transform step builders."""
    print("=" * 80)
    print("TRANSFORM STEP BUILDERS COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    # Initialize test suite
    test_suite = TransformStepBuilderTestSuite()
    available_builders = test_suite.get_available_transform_builders()
    
    if not available_builders:
        print("❌ No Transform step builders available for testing")
        return
    
    print(f"Found {len(available_builders)} Transform step builders:")
    for step_name, builder_class in available_builders:
        print(f"  - {step_name}: {builder_class.__name__}")
    
    # Run tests and collect results
    all_results = {}
    detailed_scores = {}
    
    for step_name, builder_class in available_builders:
        print(f"\n{'='*60}")
        print(f"Testing {step_name} ({builder_class.__name__})")
        print(f"{'='*60}")
        
        try:
            # Run universal tests with scoring enabled
            tester = UniversalStepBuilderTest(
                builder_class, 
                verbose=True,
                enable_scoring=True,
                enable_structured_reporting=True
            )
            universal_results = tester.run_all_tests()
            
            # Run Transform-specific tests
            transform_results = test_suite.run_transform_specific_tests(step_name, builder_class)
            
            # Combine results
            if isinstance(universal_results, dict) and 'test_results' in universal_results:
                combined_results = {**universal_results['test_results'], **transform_results}
                detailed_scores[step_name] = universal_results.get('scoring', {})
            else:
                combined_results = {**universal_results, **transform_results}
                detailed_scores[step_name] = {}
            
            all_results[step_name] = combined_results
            
            # Report individual results
            passed_tests = sum(1 for result in combined_results.values() 
                             if isinstance(result, dict) and result.get("passed", False))
            total_tests = len([r for r in combined_results.values() if isinstance(r, dict)])
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            print(f"\n{step_name} Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
            
        except Exception as e:
            print(f"❌ Failed to test {step_name}: {str(e)}")
            all_results[step_name] = {"error": str(e)}
            detailed_scores[step_name] = {}
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create reports directory structure
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Create transform-specific subdirectories
    for step_name, builder_class in available_builders:
        if step_name in all_results and "error" not in all_results[step_name]:
            step_dir = Path(__file__).parent / "transform" / builder_class.__name__
            step_dir.mkdir(parents=True, exist_ok=True)
            scoring_dir = step_dir / "scoring_reports"
            scoring_dir.mkdir(exist_ok=True)
    
    # Generate JSON report
    json_report = {
        "timestamp": timestamp,
        "test_type": "Transform Step Builders",
        "summary": generate_summary(all_results),
        "detailed_results": all_results,
        "scoring": detailed_scores
    }
    
    json_file = reports_dir / f"transform_step_builders_report_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    
    print(f"\n📊 JSON report saved to: {json_file}")
    
    # Generate score charts for each builder
    for step_name, builder_class in available_builders:
        if step_name in all_results and "error" not in all_results[step_name]:
            generate_score_chart(step_name, all_results[step_name], detailed_scores.get(step_name, {}), builder_class.__name__)
    
    # Generate overall summary chart
    generate_overall_chart(all_results, "transform")
    
    print(f"\n✅ Transform step builders testing completed!")
    print(f"📁 Reports saved in: {reports_dir}")


def generate_summary(all_results):
    """Generate summary statistics."""
    total_builders = len(all_results)
    successful_builders = 0
    total_tests = 0
    total_passed = 0
    
    for step_name, results in all_results.items():
        if "error" in results:
            continue
            
        builder_tests = len([r for r in results.values() if isinstance(r, dict)])
        builder_passed = sum(1 for result in results.values() 
                           if isinstance(result, dict) and result.get("passed", False))
        
        total_tests += builder_tests
        total_passed += builder_passed
        
        if builder_passed == builder_tests:
            successful_builders += 1
    
    return {
        "total_builders": total_builders,
        "successful_builders": successful_builders,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_pass_rate": (total_passed / total_tests) * 100 if total_tests > 0 else 0,
        "builder_success_rate": (successful_builders / total_builders) * 100 if total_builders > 0 else 0
    }


def generate_score_chart(step_name, results, scoring_data, builder_class_name):
    """Generate score chart for a specific step builder."""
    # Calculate test scores
    passed_tests = sum(1 for result in results.values() 
                      if isinstance(result, dict) and result.get("passed", False))
    total_tests = len([r for r in results.values() if isinstance(r, dict)])
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{step_name} Transform Step Builder Test Results', fontsize=16, fontweight='bold')
    
    # Pie chart for pass/fail
    labels = ['Passed', 'Failed']
    sizes = [passed_tests, total_tests - passed_tests]
    colors = ['#2ecc71', '#e74c3c']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Test Results Overview\n({passed_tests}/{total_tests} tests passed)')
    
    # Bar chart for test categories
    test_categories = {}
    for test_name, result in results.items():
        if isinstance(result, dict):
            category = test_name.split('_')[1] if '_' in test_name else 'other'
            if category not in test_categories:
                test_categories[category] = {'passed': 0, 'total': 0}
            test_categories[category]['total'] += 1
            if result.get("passed", False):
                test_categories[category]['passed'] += 1
    
    categories = list(test_categories.keys())
    passed_counts = [test_categories[cat]['passed'] for cat in categories]
    failed_counts = [test_categories[cat]['total'] - test_categories[cat]['passed'] for cat in categories]
    
    x = range(len(categories))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], passed_counts, width, label='Passed', color='#2ecc71')
    ax2.bar([i + width/2 for i in x], failed_counts, width, label='Failed', color='#e74c3c')
    
    ax2.set_xlabel('Test Categories')
    ax2.set_ylabel('Number of Tests')
    ax2.set_title('Test Results by Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = Path(__file__).parent / "transform" / builder_class_name / "scoring_reports"
    chart_file = chart_dir / f"{builder_class_name}_score_chart.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save score report JSON
    score_report = {
        "step_name": step_name,
        "builder_type": "Transform",
        "timestamp": datetime.now().isoformat(),
        "test_summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate
        },
        "test_categories": test_categories,
        "detailed_results": results,
        "scoring_data": scoring_data
    }
    
    score_file = chart_dir / f"{builder_class_name}_score_report.json"
    with open(score_file, 'w') as f:
        json.dump(score_report, f, indent=2, default=str)
    
    print(f"📊 Score chart and report saved for {step_name}")


def generate_overall_chart(all_results, test_type):
    """Generate overall summary chart."""
    # Calculate overall statistics
    builder_stats = {}
    for step_name, results in all_results.items():
        if "error" in results:
            builder_stats[step_name] = {"pass_rate": 0, "status": "error"}
            continue
            
        passed_tests = sum(1 for result in results.values() 
                          if isinstance(result, dict) and result.get("passed", False))
        total_tests = len([r for r in results.values() if isinstance(r, dict)])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        builder_stats[step_name] = {
            "pass_rate": pass_rate,
            "status": "success" if pass_rate == 100 else "partial"
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    builders = list(builder_stats.keys())
    pass_rates = [builder_stats[b]["pass_rate"] for b in builders]
    colors = ['#2ecc71' if builder_stats[b]["status"] == "success" 
              else '#f39c12' if builder_stats[b]["status"] == "partial"
              else '#e74c3c' for b in builders]
    
    bars = ax.bar(builders, pass_rates, color=colors)
    
    # Add value labels on bars
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    ax.set_xlabel('Step Builders')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title(f'{test_type.title()} Step Builders - Overall Test Results')
    ax.set_ylim(0, 105)
    
    # Add legend
    success_patch = mpatches.Patch(color='#2ecc71', label='All Tests Passed')
    partial_patch = mpatches.Patch(color='#f39c12', label='Some Tests Failed')
    error_patch = mpatches.Patch(color='#e74c3c', label='Test Execution Failed')
    ax.legend(handles=[success_patch, partial_patch, error_patch])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    chart_file = Path(__file__).parent / "reports" / f"{test_type}_overall_summary.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Overall summary chart saved: {chart_file}")


if __name__ == '__main__':
    generate_transform_reports()
