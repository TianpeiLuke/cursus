"""
Script Contract Validation Tool

Provides utilities to validate script implementations against their contracts
and generate compliance reports.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path
import logging

from ...core.base.contract_base import ScriptContract, ValidationResult
from ...step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class ContractValidationReport(BaseModel):
    """Report of contract validation results"""

    script_name: str
    contract_name: str
    is_compliant: bool
    errors: List[str] = []
    warnings: List[str] = []
    missing_inputs: List[str] = []
    missing_outputs: List[str] = []
    missing_env_vars: List[str] = []
    unexpected_inputs: List[str] = []

    @property
    def summary(self) -> str:
        """Generate a summary of the validation report"""
        status = "✅ COMPLIANT" if self.is_compliant else "❌ NON-COMPLIANT"
        return f"{self.script_name} vs {self.contract_name}: {status}"


class ScriptContractValidator:
    """Validates script implementations against their contracts using StepCatalog"""

    def __init__(self, scripts_directory: str = "src/pipeline_scripts"):
        self.scripts_directory = Path(scripts_directory)
        self.step_catalog = StepCatalog()

    def validate_script(self, script_name: str) -> ContractValidationReport:
        """
        Validate a single script against its contract

        Args:
            script_name: Name of the script file (e.g., "tabular_preprocess.py")

        Returns:
            ContractValidationReport with validation results
        """
        if script_name not in self.CONTRACTS:
            return ContractValidationReport(
                script_name=script_name,
                contract_name="UNKNOWN",
                is_compliant=False,
                errors=[f"No contract defined for script: {script_name}"],
            )

        contract = self.CONTRACTS[script_name]
        script_path = self.scripts_directory / script_name

        # Validate script exists
        if not script_path.exists():
            return ContractValidationReport(
                script_name=script_name,
                contract_name=contract.__class__.__name__,
                is_compliant=False,
                errors=[f"Script file not found: {script_path}"],
            )

        # Run contract validation
        validation_result = contract.validate_implementation(str(script_path))

        # Create detailed report
        report = ContractValidationReport(
            script_name=script_name,
            contract_name=contract.__class__.__name__,
            is_compliant=validation_result.is_valid,
            errors=validation_result.errors,
            warnings=validation_result.warnings,
        )

        # Analyze specific gaps
        self._analyze_io_gaps(contract, validation_result, report)

        return report

    def _analyze_io_gaps(
        self,
        contract: ScriptContract,
        validation_result: ValidationResult,
        report: ContractValidationReport,
    ):
        """Analyze I/O and environment variable gaps"""
        # Parse errors to extract specific gap information
        for error in validation_result.errors:
            if "doesn't use expected input path" in error:
                # Extract the path from error message
                path_start = error.find(": ") + 2
                path_end = error.find(" (for ")
                if path_start > 1 and path_end > path_start:
                    missing_path = error[path_start:path_end]
                    report.missing_inputs.append(missing_path)

            elif "doesn't use expected output path" in error:
                # Extract the path from error message
                path_start = error.find(": ") + 2
                path_end = error.find(" (for ")
                if path_start > 1 and path_end > path_start:
                    missing_path = error[path_start:path_end]
                    report.missing_outputs.append(missing_path)

            elif "missing required environment variables" in error:
                # Extract environment variables from error message
                vars_start = error.find("[") + 1
                vars_end = error.find("]")
                if vars_start > 0 and vars_end > vars_start:
                    vars_str = error[vars_start:vars_end]
                    # Parse the list string
                    missing_vars = [v.strip().strip("'\"") for v in vars_str.split(",")]
                    report.missing_env_vars.extend(missing_vars)

        # Parse warnings for unexpected inputs
        for warning in validation_result.warnings:
            if "uses undeclared input path" in warning:
                path_start = warning.find(": ") + 2
                if path_start > 1:
                    unexpected_path = warning[path_start:]
                    report.unexpected_inputs.append(unexpected_path)

    def validate_all_scripts(self) -> List[ContractValidationReport]:
        """
        Validate all scripts against their contracts

        Returns:
            List of ContractValidationReport for all scripts
        """
        reports = []
        for script_name in self.CONTRACTS.keys():
            report = self.validate_script(script_name)
            reports.append(report)
        return reports

    def generate_compliance_summary(
        self, reports: Optional[List[ContractValidationReport]] = None
    ) -> str:
        """
        Generate a human-readable compliance summary

        Args:
            reports: List of validation reports. If None, validates all scripts.

        Returns:
            Formatted compliance summary string
        """
        if reports is None:
            reports = self.validate_all_scripts()

        compliant_count = sum(1 for r in reports if r.is_compliant)
        total_count = len(reports)

        summary_lines = [
            "=" * 60,
            "SCRIPT CONTRACT COMPLIANCE REPORT",
            "=" * 60,
            f"Overall Compliance: {compliant_count}/{total_count} scripts compliant",
            "",
        ]

        # Group by compliance status
        compliant_scripts = [r for r in reports if r.is_compliant]
        non_compliant_scripts = [r for r in reports if not r.is_compliant]

        if compliant_scripts:
            summary_lines.extend(["✅ COMPLIANT SCRIPTS:", "-" * 20])
            for report in compliant_scripts:
                summary_lines.append(f"  • {report.script_name}")
                if report.warnings:
                    summary_lines.append(f"    Warnings: {len(report.warnings)}")
            summary_lines.append("")

        if non_compliant_scripts:
            summary_lines.extend(["❌ NON-COMPLIANT SCRIPTS:", "-" * 25])
            for report in non_compliant_scripts:
                summary_lines.append(f"  • {report.script_name}")
                summary_lines.append(f"    Errors: {len(report.errors)}")
                if report.missing_inputs:
                    summary_lines.append(f"    Missing Inputs: {report.missing_inputs}")
                if report.missing_outputs:
                    summary_lines.append(
                        f"    Missing Outputs: {report.missing_outputs}"
                    )
                if report.missing_env_vars:
                    summary_lines.append(
                        f"    Missing Env Vars: {report.missing_env_vars}"
                    )
                if report.unexpected_inputs:
                    summary_lines.append(
                        f"    Unexpected Inputs: {report.unexpected_inputs}"
                    )
                summary_lines.append("")

        summary_lines.extend(
            [
                "=" * 60,
                "RECOMMENDATIONS:",
                "=" * 60,
                "1. Address missing I/O paths in non-compliant scripts",
                "2. Add required environment variable handling",
                "3. Document any intentional deviations from contracts",
                "4. Update contracts if script requirements have changed",
                "",
            ]
        )

        return "\n".join(summary_lines)


def main():
    """CLI entry point for contract validation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate pipeline scripts against their contracts"
    )
    parser.add_argument(
        "--script", help="Validate specific script (e.g., tabular_preprocess.py)"
    )
    parser.add_argument(
        "--scripts-dir",
        default="src/pipeline_scripts",
        help="Directory containing scripts",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed validation results"
    )

    args = parser.parse_args()

    validator = ScriptContractValidator(args.scripts_dir)

    if args.script:
        # Validate single script
        report = validator.validate_script(args.script)
        print(report.summary)
        if args.verbose:
            if report.errors:
                print(f"Errors: {report.errors}")
            if report.warnings:
                print(f"Warnings: {report.warnings}")
    else:
        # Validate all scripts
        reports = validator.validate_all_scripts()
        summary = validator.generate_compliance_summary(reports)
        print(summary)


if __name__ == "__main__":
    main()
