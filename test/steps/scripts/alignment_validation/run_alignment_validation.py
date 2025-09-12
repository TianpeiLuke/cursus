#!/usr/bin/env python3
"""
Comprehensive Alignment Validation for Cursus Scripts

This program runs the unified alignment tester for each script under src/cursus/steps/scripts
and generates detailed reports for each script's alignment across all four levels:

1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment  
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment

The results are saved as both JSON and HTML reports in separate directories.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Define workspace directory structure
# workspace_dir points to src/cursus (the main workspace)
current_file = Path(__file__).resolve()
workspace_dir = (
    current_file.parent.parent.parent.parent.parent / "src" / "cursus" / "steps"
)

# Define component directories within the workspace
scripts_dir = str(workspace_dir / "scripts")
contracts_dir = str(workspace_dir / "contracts")
specs_dir = str(workspace_dir / "specs")
builders_dir = str(workspace_dir / "builders")
configs_dir = str(workspace_dir / "configs")

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester


class ScriptAlignmentValidator:
    """Comprehensive alignment validator for cursus scripts."""

    def __init__(self):
        """Initialize the script alignment validator."""
        # Set up directory paths
        self.scripts_dir = Path(scripts_dir)
        self.contracts_dir = Path(contracts_dir)
        self.specs_dir = Path(specs_dir)
        self.builders_dir = Path(builders_dir)
        self.configs_dir = Path(configs_dir)

        # Set up output directories
        self.output_dir = Path(__file__).parent / "reports"
        self.json_reports_dir = self.output_dir / "json"
        self.html_reports_dir = self.output_dir / "html"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.json_reports_dir.mkdir(exist_ok=True)
        self.html_reports_dir.mkdir(exist_ok=True)

        # Initialize the unified alignment tester with configs directory
        self.tester = UnifiedAlignmentTester(
            scripts_dir=str(self.scripts_dir),
            contracts_dir=str(self.contracts_dir),
            specs_dir=str(self.specs_dir),
            builders_dir=str(self.builders_dir),
            configs_dir=str(self.configs_dir),
        )

        # Script to contract mapping
        self.script_mappings = {
            "currency_conversion": "currency_conversion_contract",
            "dummy_training": "dummy_training_contract",
            "model_calibration": "model_calibration_contract",
            "package": "package_contract",
            "payload": "payload_contract",
            "risk_table_mapping": "risk_table_mapping_contract",
            "tabular_preprocessing": "tabular_preprocessing_contract",
            "xgboost_model_evaluation": "xgboost_model_eval_contract",
            "xgboost_training": "xgboost_training_contract",
        }

    def discover_scripts(self) -> List[str]:
        """Discover all Python scripts in the scripts directory."""
        scripts = []

        if self.scripts_dir.exists():
            for script_file in self.scripts_dir.glob("*.py"):
                if not script_file.name.startswith("__"):
                    script_name = script_file.stem
                    scripts.append(script_name)

        return sorted(scripts)

    def validate_single_script(self, script_name: str) -> Dict[str, Any]:
        """
        Run comprehensive validation for a single script.

        Args:
            script_name: Name of the script to validate

        Returns:
            Validation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"🔍 VALIDATING SCRIPT: {script_name}")
        print(f"{'='*60}")

        try:
            # Run validation across all levels
            results = self.tester.validate_specific_script(script_name)

            # Add metadata
            results["metadata"] = {
                "script_path": str(self.scripts_dir / f"{script_name}.py"),
                "contract_mapping": self.script_mappings.get(
                    script_name, f"{script_name}_contract"
                ),
                "validation_timestamp": datetime.now().isoformat(),
                "validator_version": "1.0.0",
            }

            # Print summary
            self._print_script_summary(script_name, results)

            return results

        except Exception as e:
            error_result = {
                "script_name": script_name,
                "overall_status": "ERROR",
                "error": str(e),
                "metadata": {
                    "script_path": str(self.scripts_dir / f"{script_name}.py"),
                    "validation_timestamp": datetime.now().isoformat(),
                    "validator_version": "1.0.0",
                },
            }

            print(f"❌ ERROR validating {script_name}: {e}")
            return error_result

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object to a JSON-serializable representation.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation
        """
        # Handle None
        if obj is None:
            return None

        # Handle basic JSON types
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]

        # Handle dictionaries
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Ensure keys are strings
                str_key = str(key)
                result[str_key] = self._make_json_serializable(value)
            return result

        # Handle sets - convert to sorted lists
        if isinstance(obj, set):
            return sorted([self._make_json_serializable(item) for item in obj])

        # Handle Path objects
        if hasattr(obj, "__fspath__"):  # Path-like objects
            return str(obj)

        # Handle datetime objects
        if hasattr(obj, "isoformat"):
            return obj.isoformat()

        # Handle Enum objects
        if hasattr(obj, "value"):
            return obj.value

        # Handle type objects
        if isinstance(obj, type):
            return str(obj.__name__)

        # For everything else, try string conversion
        try:
            str_value = str(obj)
            # Avoid storing string representations of complex objects
            if "<" in str_value and ">" in str_value and "object at" in str_value:
                return f"<{type(obj).__name__}>"
            return str_value
        except Exception:
            return f"<{type(obj).__name__}>"

    def _print_script_summary(self, script_name: str, results: Dict[str, Any]):
        """Print a summary of validation results for a script."""
        status = results.get("overall_status", "UNKNOWN")
        status_emoji = (
            "✅" if status == "PASSING" else "❌" if status == "FAILING" else "⚠️"
        )

        print(f"\n{status_emoji} Overall Status: {status}")

        # Print level-by-level results
        for level_num, level_name in enumerate(
            [
                "Script ↔ Contract",
                "Contract ↔ Specification",
                "Specification ↔ Dependencies",
                "Builder ↔ Configuration",
            ],
            1,
        ):
            level_key = f"level{level_num}"
            level_result = results.get(level_key, {})
            level_passed = level_result.get("passed", False)
            level_emoji = "✅" if level_passed else "❌"

            issues_count = len(level_result.get("issues", []))
            issues_text = f" ({issues_count} issues)" if issues_count > 0 else ""

            print(
                f"  {level_emoji} Level {level_num} ({level_name}): {'PASS' if level_passed else 'FAIL'}{issues_text}"
            )

    def save_reports(self, script_name: str, results: Dict[str, Any]):
        """
        Save validation results as both JSON and HTML reports.

        Args:
            script_name: Name of the script
            results: Validation results dictionary
        """
        # Save JSON report with robust serialization
        json_file = self.json_reports_dir / f"{script_name}_alignment_report.json"
        try:
            # Clean the results to ensure JSON serializability
            cleaned_results = self._make_json_serializable(results)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            print(f"📄 JSON report saved: {json_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save JSON report for {script_name}: {e}")
            # Try to save a simplified version
            try:
                simplified_results = {
                    "script_name": script_name,
                    "overall_status": results.get("overall_status", "ERROR"),
                    "error": f"JSON serialization failed: {str(e)}",
                    "metadata": results.get("metadata", {}),
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(simplified_results, f, indent=2, default=str)
                print(f"📄 Simplified JSON report saved: {json_file}")
            except Exception as e2:
                print(
                    f"❌ Failed to save even simplified JSON report for {script_name}: {e2}"
                )

        # Generate and save HTML report
        try:
            html_content = self._generate_html_report(script_name, results)
            html_file = self.html_reports_dir / f"{script_name}_alignment_report.html"

            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"🌐 HTML report saved: {html_file}")

        except Exception as e:
            print(f"⚠️  Warning: Could not generate HTML report for {script_name}: {e}")

    def _generate_html_report(self, script_name: str, results: Dict[str, Any]) -> str:
        """Generate HTML report for a script's validation results."""
        status = results.get("overall_status", "UNKNOWN")
        status_class = "passing" if status == "PASSING" else "failing"
        timestamp = results.get("metadata", {}).get("validation_timestamp", "Unknown")

        # Count issues by severity
        total_issues = 0
        critical_issues = 0
        error_issues = 0
        warning_issues = 0

        for level_num in range(1, 5):
            level_key = f"level{level_num}"
            level_result = results.get(level_key, {})
            issues = level_result.get("issues", [])
            total_issues += len(issues)

            for issue in issues:
                severity = issue.get("severity", "ERROR")
                if severity == "CRITICAL":
                    critical_issues += 1
                elif severity == "ERROR":
                    error_issues += 1
                elif severity == "WARNING":
                    warning_issues += 1

        # Generate level sections
        level_sections = ""
        for level_num, level_name in enumerate(
            [
                "Level 1: Script ↔ Contract",
                "Level 2: Contract ↔ Specification",
                "Level 3: Specification ↔ Dependencies",
                "Level 4: Builder ↔ Configuration",
            ],
            1,
        ):
            level_key = f"level{level_num}"
            level_result = results.get(level_key, {})
            level_passed = level_result.get("passed", False)
            level_issues = level_result.get("issues", [])

            result_class = "test-passed" if level_passed else "test-failed"
            status_text = "PASSED" if level_passed else "FAILED"

            issues_html = ""
            for issue in level_issues:
                severity = issue.get("severity", "ERROR").lower()
                message = issue.get("message", "No message")
                recommendation = issue.get("recommendation", "")

                issues_html += f"""
                <div class="issue {severity}">
                    <strong>{issue.get('severity', 'ERROR')}:</strong> {message}
                    {f'<br><em>Recommendation: {recommendation}</em>' if recommendation else ''}
                </div>
                """

            level_sections += f"""
            <div class="test-result {result_class}">
                <h4>{level_name}</h4>
                <p>Status: {status_text}</p>
                {issues_html}
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Alignment Validation Report - {script_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; }}
        .metric h3 {{ margin: 0; font-size: 2em; }}
        .metric p {{ margin: 5px 0; color: #666; }}
        .passing {{ color: #28a745; }}
        .failing {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .level-section {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .level-header {{ background-color: #e9ecef; padding: 10px; font-weight: bold; }}
        .test-result {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .test-passed {{ border-left: 4px solid #28a745; }}
        .test-failed {{ border-left: 4px solid #dc3545; }}
        .issue {{ margin: 5px 0; padding: 5px; background-color: #f8f9fa; border-radius: 3px; }}
        .critical {{ border-left: 4px solid #dc3545; }}
        .error {{ border-left: 4px solid #fd7e14; }}
        .warning {{ border-left: 4px solid #ffc107; }}
        .info {{ border-left: 4px solid #17a2b8; }}
        .metadata {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Alignment Validation Report</h1>
        <h2>Script: {script_name}</h2>
        <p>Generated: {timestamp}</p>
        <p>Overall Status: <span class="{status_class}">{status}</span></p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{total_issues}</h3>
            <p>Total Issues</p>
        </div>
        <div class="metric">
            <h3>{critical_issues}</h3>
            <p>Critical Issues</p>
        </div>
        <div class="metric">
            <h3>{error_issues}</h3>
            <p>Error Issues</p>
        </div>
        <div class="metric">
            <h3>{warning_issues}</h3>
            <p>Warning Issues</p>
        </div>
    </div>
    
    <div class="level-section">
        <div class="level-header">Alignment Validation Results</div>
        {level_sections}
    </div>
    
    <div class="metadata">
        <h3>Metadata</h3>
        <p><strong>Script Path:</strong> {results.get('metadata', {}).get('script_path', 'Unknown')}</p>
        <p><strong>Contract Mapping:</strong> {results.get('metadata', {}).get('contract_mapping', 'Unknown')}</p>
        <p><strong>Validation Timestamp:</strong> {timestamp}</p>
        <p><strong>Validator Version:</strong> {results.get('metadata', {}).get('validator_version', 'Unknown')}</p>
    </div>
</body>
</html>"""

        return html_template

    def run_all_validations(self):
        """Run alignment validation for all discovered scripts."""
        print("🚀 Starting Comprehensive Script Alignment Validation")
        print(f"📁 Scripts Directory: {self.scripts_dir}")
        print(f"📁 Contracts Directory: {self.contracts_dir}")
        print(f"📁 Specifications Directory: {self.specs_dir}")
        print(f"📁 Builders Directory: {self.builders_dir}")
        print(f"📁 Configs Directory: {self.configs_dir}")
        print(f"📁 Output Directory: {self.output_dir}")

        # Discover all scripts
        scripts = self.discover_scripts()
        print(f"\n📋 Discovered {len(scripts)} scripts: {', '.join(scripts)}")

        # Validation results summary
        validation_summary = {
            "total_scripts": len(scripts),
            "passed_scripts": 0,
            "failed_scripts": 0,
            "error_scripts": 0,
            "validation_timestamp": datetime.now().isoformat(),
            "script_results": {},
        }

        # Validate each script
        for script_name in scripts:
            try:
                results = self.validate_single_script(script_name)
                self.save_reports(script_name, results)

                # Update summary
                status = results.get("overall_status", "UNKNOWN")
                validation_summary["script_results"][script_name] = {
                    "status": status,
                    "timestamp": results.get("metadata", {}).get(
                        "validation_timestamp"
                    ),
                }

                if status == "PASSING":
                    validation_summary["passed_scripts"] += 1
                elif status == "FAILING":
                    validation_summary["failed_scripts"] += 1
                else:
                    validation_summary["error_scripts"] += 1

            except Exception as e:
                print(f"❌ Failed to validate {script_name}: {e}")
                validation_summary["error_scripts"] += 1
                validation_summary["script_results"][script_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        # Save overall summary
        self._save_validation_summary(validation_summary)
        self._print_final_summary(validation_summary)

    def _save_validation_summary(self, summary: Dict[str, Any]):
        """Save the overall validation summary."""
        summary_file = self.output_dir / "validation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n📊 Validation summary saved: {summary_file}")

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print the final validation summary."""
        print(f"\n{'='*80}")
        print("🎯 FINAL VALIDATION SUMMARY")
        print(f"{'='*80}")

        total = summary["total_scripts"]
        passed = summary["passed_scripts"]
        failed = summary["failed_scripts"]
        errors = summary["error_scripts"]

        print(f"📊 Total Scripts: {total}")
        print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"⚠️  Errors: {errors} ({errors/total*100:.1f}%)")

        print(f"\n📁 Reports saved in: {self.output_dir}")
        print(f"   📄 JSON reports: {self.json_reports_dir}")
        print(f"   🌐 HTML reports: {self.html_reports_dir}")

        # List scripts by status
        if passed > 0:
            passing_scripts = [
                name
                for name, result in summary["script_results"].items()
                if result["status"] == "PASSING"
            ]
            print(f"\n✅ PASSING SCRIPTS ({len(passing_scripts)}):")
            for script in passing_scripts:
                print(f"   • {script}")

        if failed > 0:
            failing_scripts = [
                name
                for name, result in summary["script_results"].items()
                if result["status"] == "FAILING"
            ]
            print(f"\n❌ FAILING SCRIPTS ({len(failing_scripts)}):")
            for script in failing_scripts:
                print(f"   • {script}")

        if errors > 0:
            error_scripts = [
                name
                for name, result in summary["script_results"].items()
                if result["status"] == "ERROR"
            ]
            print(f"\n⚠️  ERROR SCRIPTS ({len(error_scripts)}):")
            for script in error_scripts:
                print(f"   • {script}")

        print(f"\n{'='*80}")


def main():
    """Main entry point for the alignment validation program."""
    try:
        validator = ScriptAlignmentValidator()
        validator.run_all_validations()

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
