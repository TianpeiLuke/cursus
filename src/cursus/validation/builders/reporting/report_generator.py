"""
Enhanced report generator for step builder testing.

This module provides comprehensive reporting capabilities including visual charts,
metadata tracking, and structured reporting inspired by legacy report generators.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

# Optional matplotlib import for chart generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EnhancedReportGenerator:
    """
    Enhanced report generator with visual charts and comprehensive metadata.
    
    Provides structured reporting capabilities inspired by the legacy report
    generation scripts, including visual score charts, comprehensive metadata
    tracking, and summary reports across step types.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the enhanced report generator."""
        if base_path is None:
            base_path = Path("test/steps/builders/results")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def generate_score_chart(self, results: Dict[str, Any], output_path: Path) -> bool:
        """
        Generate a visual score chart for test results.
        
        Args:
            results: Test results dictionary
            output_path: Path to save the chart
            
        Returns:
            True if chart was generated successfully, False otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping chart generation")
            return False
        
        try:
            # Extract test results
            test_results = results.get("test_results", {})
            if not test_results:
                # Fallback for older format
                test_results = {
                    k: v for k, v in results.items() 
                    if isinstance(v, dict) and "passed" in v
                }
            
            if not test_results:
                print("Warning: No test results found for chart generation")
                return False
            
            # Calculate scores
            total_tests = len(test_results)
            passed_tests = sum(1 for result in test_results.values() if result.get("passed", False))
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Pie chart for pass/fail ratio
            labels = ["Passed", "Failed"]
            sizes = [passed_tests, total_tests - passed_tests]
            colors = ["#2ECC71", "#E74C3C"]  # Green for pass, red for fail
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax1.set_title(f'Test Results Overview\n{results.get("step_name", "Unknown")}')
            
            # Bar chart for individual test results
            test_names = list(test_results.keys())
            test_scores = [1 if result.get("passed", False) else 0 for result in test_results.values()]
            
            # Truncate long test names for display
            display_names = [
                name.replace("test_", "").replace("_", " ").title()[:20] 
                for name in test_names
            ]
            
            bars = ax2.bar(
                range(len(test_names)),
                test_scores,
                color=["#2ECC71" if score == 1 else "#E74C3C" for score in test_scores],
            )
            
            ax2.set_xlabel("Tests")
            ax2.set_ylabel("Pass (1) / Fail (0)")
            ax2.set_title(f"Individual Test Results\nPass Rate: {pass_rate:.1f}%")
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels(display_names, rotation=45, ha="right")
            ax2.set_ylim(0, 1.2)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, test_scores)):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    "✓" if score == 1 else "✗",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )
            
            # Add metadata
            step_name = results.get("step_name", "Unknown")
            builder_class = results.get("builder_class_name", "Unknown")
            timestamp = results.get("timestamp", "Unknown")
            
            fig.suptitle(f"{builder_class} Test Results", fontsize=16, fontweight="bold")
            fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error generating score chart: {e}")
            return False
    
    def enhance_results_with_metadata(self, results: Dict[str, Any], 
                                    canonical_name: str, step_type: str = None) -> Dict[str, Any]:
        """
        Enhance test results with comprehensive metadata.
        
        Args:
            results: Original test results
            canonical_name: Canonical builder name
            step_type: Optional step type
            
        Returns:
            Enhanced results with metadata
        """
        enhanced = results.copy()
        
        # Add comprehensive metadata
        enhanced.update({
            "canonical_name": canonical_name,
            "step_type": step_type,
            "timestamp": datetime.now().isoformat(),
            "generator_version": "2.0.0",
            "enhanced_features": True,
            "chart_generation_available": MATPLOTLIB_AVAILABLE
        })
        
        # Calculate summary statistics
        if 'test_results' in results:
            test_results = results['test_results']
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results.values() if r.get('passed', False))
            pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            enhanced["summary_statistics"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": pass_rate,
                "status": "PASSING" if pass_rate >= 80 else "WARNING" if pass_rate >= 60 else "FAILING"
            }
        
        return enhanced
    
    def generate_comprehensive_report(self, all_results: Dict[str, Dict[str, Any]], 
                                    report_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate comprehensive summary report across all builders.
        
        Args:
            all_results: Dictionary of all test results by builder name
            report_type: Type of report to generate
            
        Returns:
            Comprehensive summary report
        """
        summary = {
            "report_type": report_type,
            "timestamp": datetime.now().isoformat(),
            "total_builders": len(all_results),
            "generator_version": "2.0.0",
            "builders": {},
            "overall_statistics": {},
            "step_type_breakdown": {}
        }
        
        # Process individual builder results
        total_tests = 0
        total_passed = 0
        step_types = {}
        
        for canonical_name, results in all_results.items():
            if 'test_results' in results:
                test_results = results['test_results']
                builder_total = len(test_results)
                builder_passed = sum(1 for r in test_results.values() if r.get('passed', False))
                builder_pass_rate = (builder_passed / builder_total * 100) if builder_total > 0 else 0
                
                total_tests += builder_total
                total_passed += builder_passed
                
                # Track by step type
                step_type = results.get('step_type', 'Unknown')
                if step_type not in step_types:
                    step_types[step_type] = {"builders": 0, "total_tests": 0, "passed_tests": 0}
                
                step_types[step_type]["builders"] += 1
                step_types[step_type]["total_tests"] += builder_total
                step_types[step_type]["passed_tests"] += builder_passed
                
                summary["builders"][canonical_name] = {
                    "total_tests": builder_total,
                    "passed_tests": builder_passed,
                    "pass_rate": builder_pass_rate,
                    "status": "PASSING" if builder_pass_rate >= 80 else "WARNING" if builder_pass_rate >= 60 else "FAILING",
                    "step_type": step_type
                }
        
        # Calculate overall statistics
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        summary["overall_statistics"] = {
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "failed_tests": total_tests - total_passed,
            "pass_rate": overall_pass_rate,
            "status": "PASSING" if overall_pass_rate >= 80 else "WARNING" if overall_pass_rate >= 60 else "FAILING"
        }
        
        # Calculate step type breakdown
        for step_type, stats in step_types.items():
            pass_rate = (stats["passed_tests"] / stats["total_tests"] * 100) if stats["total_tests"] > 0 else 0
            summary["step_type_breakdown"][step_type] = {
                "builders": stats["builders"],
                "total_tests": stats["total_tests"],
                "passed_tests": stats["passed_tests"],
                "pass_rate": pass_rate,
                "status": "PASSING" if pass_rate >= 80 else "WARNING" if pass_rate >= 60 else "FAILING"
            }
        
        return summary
    
    def create_step_subfolder_structure(self, step_type: str, canonical_name: str) -> Path:
        """
        Create organized subfolder structure for a specific step.
        
        Args:
            step_type: Type of step (Processing, Training, etc.)
            canonical_name: Canonical builder name
            
        Returns:
            Path to the scoring_reports directory
        """
        # Create directory structure: {step_type}/{canonical_name}/scoring_reports/
        step_dir = self.base_path / step_type.lower() / canonical_name
        scoring_dir = step_dir / "scoring_reports"
        
        # Create directories
        step_dir.mkdir(parents=True, exist_ok=True)
        scoring_dir.mkdir(exist_ok=True)
        
        # Create README for the step directory
        readme_path = step_dir / "README.md"
        if not readme_path.exists():
            self._create_step_readme(readme_path, canonical_name, step_type)
        
        return scoring_dir
    
    def _create_step_readme(self, readme_path: Path, canonical_name: str, step_type: str):
        """Create README file for a step directory."""
        content = f"""# {canonical_name} Test Reports

This directory contains test reports and scoring charts for the {canonical_name} step builder.

## Directory Structure

- `scoring_reports/` - Contains detailed test scoring reports and charts
  - `{canonical_name}_score_report.json` - Detailed test results in JSON format
  - `{canonical_name}_score_chart.png` - Visual score chart (if matplotlib available)

## Step Information

- **Builder Class**: {canonical_name}
- **Step Type**: {step_type}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

The scoring reports are generated automatically by the dynamic testing system:

```bash
# Test all builders with enhanced reporting
cursus builder-test test-all-discovered --scoring

# Test individual builder with enhanced reporting
cursus builder-test test-single {canonical_name} --scoring

# Run pytest with enhanced reporting
pytest test/steps/builders/test_dynamic_universal.py -v
```

## Features

- **Visual Charts**: Pie charts and bar charts showing test results (requires matplotlib)
- **Comprehensive Metadata**: Detailed information about test execution
- **Summary Statistics**: Pass rates, test counts, and status indicators
- **Step Type Classification**: Organized by SageMaker step type
"""
        readme_path.write_text(content)
    
    def save_enhanced_report(self, results: Dict[str, Any], canonical_name: str, 
                           step_type: str = None, generate_chart: bool = True) -> Dict[str, str]:
        """
        Save enhanced report with optional chart generation.
        
        Args:
            results: Test results to save
            canonical_name: Canonical builder name
            step_type: Optional step type for organization
            generate_chart: Whether to generate visual chart
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        # Enhance results with metadata
        enhanced_results = self.enhance_results_with_metadata(results, canonical_name, step_type)
        
        # Create organized directory structure if step_type provided
        if step_type:
            scoring_dir = self.create_step_subfolder_structure(step_type, canonical_name)
        else:
            scoring_dir = self.base_path / "individual"
            scoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = scoring_dir / f"{canonical_name}_enhanced_report.json"
        with open(json_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        saved_files['json_report'] = str(json_path)
        
        # Generate and save chart if requested and available
        if generate_chart and MATPLOTLIB_AVAILABLE:
            chart_path = scoring_dir / f"{canonical_name}_score_chart.png"
            if self.generate_score_chart(enhanced_results, chart_path):
                saved_files['score_chart'] = str(chart_path)
        
        return saved_files


def create_enhanced_report_generator(base_path: Optional[Path] = None) -> EnhancedReportGenerator:
    """
    Factory function to create an enhanced report generator.
    
    Args:
        base_path: Optional base path for reports
        
    Returns:
        Configured EnhancedReportGenerator instance
    """
    return EnhancedReportGenerator(base_path)
