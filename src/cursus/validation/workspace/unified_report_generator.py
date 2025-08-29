"""
Unified Report Generator

Single report generator that adapts output based on workspace count.
This component eliminates dual-path complexity by providing unified
reporting that works identically for count=1 and count=N scenarios.

Architecture:
- Single report generation method for all scenarios
- Adaptive output formatting based on workspace count
- Consistent report structure regardless of scenario
- Extensible reporting framework

Features:
- Unified report generation for single and multi-workspace
- Adaptive formatting based on workspace count
- Consistent report structure and content
- Multiple output formats (JSON, YAML, text)
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import yaml
import logging

from .unified_result_structures import UnifiedValidationResult, WorkspaceValidationResult

logger = logging.getLogger(__name__)


class ReportConfig:
    """Configuration for report generation."""
    
    def __init__(
        self,
        output_format: str = "json",
        include_details: bool = True,
        include_recommendations: bool = True,
        include_workspace_info: bool = True,
        include_timing: bool = True,
        include_summary_only: bool = False,
        compact_format: bool = False,
        sort_workspaces: bool = True
    ):
        """
        Initialize report configuration.
        
        Args:
            output_format: Output format ('json', 'yaml', 'text')
            include_details: Whether to include detailed validation results
            include_recommendations: Whether to include recommendations
            include_workspace_info: Whether to include workspace metadata
            include_timing: Whether to include timing information
            include_summary_only: Whether to generate summary-only report
            compact_format: Whether to use compact formatting
            sort_workspaces: Whether to sort workspaces in output
        """
        self.output_format = output_format.lower()
        self.include_details = include_details
        self.include_recommendations = include_recommendations
        self.include_workspace_info = include_workspace_info
        self.include_timing = include_timing
        self.include_summary_only = include_summary_only
        self.compact_format = compact_format
        self.sort_workspaces = sort_workspaces


class UnifiedReportGenerator:
    """
    Single report generator that adapts output based on workspace count.
    
    This class provides unified report generation that works identically
    whether generating reports for single workspace (count=1) or multiple
    workspaces (count=N), eliminating the need for separate reporting logic.
    
    Features:
    - Unified report generation for all scenarios
    - Adaptive formatting based on workspace count
    - Multiple output formats supported
    - Consistent report structure
    """
    
    def __init__(self, report_config: Optional[ReportConfig] = None):
        """
        Initialize unified report generator.
        
        Args:
            report_config: Configuration for report generation
        """
        self.report_config = report_config or ReportConfig()
    
    def generate_report(
        self,
        result: UnifiedValidationResult,
        report_config: Optional[ReportConfig] = None
    ) -> Dict[str, Any]:
        """
        Generates appropriate report format based on workspace count.
        
        This method adapts the report format based on whether the result
        represents a single workspace (count=1) or multiple workspaces (count=N),
        while maintaining a consistent overall structure.
        
        Args:
            result: Unified validation result to generate report for
            report_config: Override report configuration
            
        Returns:
            Dictionary containing the generated report
        """
        config = report_config or self.report_config
        
        logger.info(f"Generating {config.output_format} report for "
                   f"{result.workspace_count} workspace(s)")
        
        try:
            # Generate base report structure
            if result.is_single_workspace:
                report = self._generate_single_workspace_report(result, config)
            else:
                report = self._generate_multi_workspace_report(result, config)
            
            # Apply formatting based on output format
            if config.output_format == "text":
                return self._format_as_text(report, config)
            elif config.output_format == "yaml":
                return self._format_as_yaml(report, config)
            else:  # Default to JSON
                return self._format_as_json(report, config)
                
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                "error": f"Report generation failed: {str(e)}",
                "result_summary": {
                    "workspace_count": result.workspace_count,
                    "workspace_type": result.workspace_type,
                    "overall_success": result.overall_success
                }
            }
    
    def _generate_single_workspace_report(
        self,
        result: UnifiedValidationResult,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate report optimized for single workspace scenario."""
        
        # Get the single workspace result
        workspace_id = list(result.workspaces.keys())[0]
        workspace_result = result.workspaces[workspace_id]
        
        report = {
            "report_type": "single_workspace_validation",
            "generated_at": datetime.now().isoformat(),
            "workspace_root": result.workspace_root,
            "workspace_id": workspace_id,
            "workspace_type": workspace_result.workspace_type,
            "workspace_path": workspace_result.workspace_path
        }
        
        # Add validation summary
        report["validation_summary"] = {
            "overall_success": workspace_result.success,
            "validation_types_run": workspace_result.validation_types_run,
            "has_errors": workspace_result.error is not None,
            "warning_count": len(workspace_result.warnings)
        }
        
        # Add timing information if requested
        if config.include_timing:
            report["timing"] = {
                "validation_start_time": workspace_result.validation_start_time,
                "validation_end_time": workspace_result.validation_end_time,
                "validation_duration_seconds": workspace_result.get_validation_duration(),
                "total_validation_duration_seconds": result.get_validation_duration()
            }
        
        # Add detailed results if requested and not summary-only
        if config.include_details and not config.include_summary_only:
            report["validation_results"] = {}
            
            if workspace_result.has_alignment_results:
                report["validation_results"]["alignment"] = workspace_result.alignment_results
            
            if workspace_result.has_builder_results:
                report["validation_results"]["builder"] = workspace_result.builder_results
            
            # Add error and warning details
            if workspace_result.error:
                report["validation_results"]["error"] = workspace_result.error
            
            if workspace_result.warnings:
                report["validation_results"]["warnings"] = workspace_result.warnings
        
        # Add workspace information if requested
        if config.include_workspace_info and workspace_result.developer_info:
            report["workspace_info"] = workspace_result.developer_info
        
        # Add recommendations if requested
        if config.include_recommendations and result.recommendations:
            report["recommendations"] = result.recommendations
        
        return report
    
    def _generate_multi_workspace_report(
        self,
        result: UnifiedValidationResult,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate report optimized for multi-workspace scenario."""
        
        report = {
            "report_type": "multi_workspace_validation",
            "generated_at": datetime.now().isoformat(),
            "workspace_root": result.workspace_root,
            "workspace_type": result.workspace_type
        }
        
        # Add overall summary
        report["validation_summary"] = {
            "total_workspaces": result.summary.total_workspaces,
            "successful_workspaces": result.summary.successful_workspaces,
            "failed_workspaces": result.summary.failed_workspaces,
            "success_rate": result.summary.success_rate,
            "success_percentage": result.summary.success_percentage,
            "overall_success": result.overall_success,
            "has_global_error": result.global_error is not None
        }
        
        # Add timing information if requested
        if config.include_timing:
            report["timing"] = {
                "validation_start_time": result.validation_start_time,
                "validation_end_time": result.validation_end_time,
                "total_validation_duration_seconds": result.get_validation_duration()
            }
        
        # Add workspace results
        workspace_ids = list(result.workspaces.keys())
        if config.sort_workspaces:
            workspace_ids.sort()
        
        if config.include_summary_only:
            # Summary-only format
            report["workspace_summaries"] = {}
            for workspace_id in workspace_ids:
                workspace_result = result.workspaces[workspace_id]
                report["workspace_summaries"][workspace_id] = {
                    "success": workspace_result.success,
                    "workspace_type": workspace_result.workspace_type,
                    "validation_types_run": workspace_result.validation_types_run,
                    "has_errors": workspace_result.error is not None,
                    "warning_count": len(workspace_result.warnings)
                }
        else:
            # Detailed format
            report["workspace_results"] = {}
            for workspace_id in workspace_ids:
                workspace_result = result.workspaces[workspace_id]
                
                workspace_report = {
                    "workspace_id": workspace_id,
                    "workspace_type": workspace_result.workspace_type,
                    "workspace_path": workspace_result.workspace_path,
                    "success": workspace_result.success
                }
                
                # Add timing for individual workspace
                if config.include_timing:
                    workspace_report["timing"] = {
                        "validation_start_time": workspace_result.validation_start_time,
                        "validation_end_time": workspace_result.validation_end_time,
                        "validation_duration_seconds": workspace_result.get_validation_duration()
                    }
                
                # Add detailed results if requested
                if config.include_details:
                    workspace_report["validation_results"] = {}
                    
                    if workspace_result.has_alignment_results:
                        workspace_report["validation_results"]["alignment"] = workspace_result.alignment_results
                    
                    if workspace_result.has_builder_results:
                        workspace_report["validation_results"]["builder"] = workspace_result.builder_results
                    
                    # Add error and warning details
                    if workspace_result.error:
                        workspace_report["validation_results"]["error"] = workspace_result.error
                    
                    if workspace_result.warnings:
                        workspace_report["validation_results"]["warnings"] = workspace_result.warnings
                
                # Add workspace information if requested
                if config.include_workspace_info and workspace_result.developer_info:
                    workspace_report["workspace_info"] = workspace_result.developer_info
                
                report["workspace_results"][workspace_id] = workspace_report
        
        # Add failed workspace summary
        failed_workspaces = result.get_failed_workspaces()
        if failed_workspaces:
            report["failed_workspaces"] = failed_workspaces
        
        # Add global error if present
        if result.global_error:
            report["global_error"] = result.global_error
        
        # Add recommendations if requested
        if config.include_recommendations and result.recommendations:
            report["recommendations"] = result.recommendations
        
        return report
    
    def _format_as_json(
        self,
        report: Dict[str, Any],
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Format report as JSON structure."""
        # JSON format returns the report dictionary directly
        # The actual JSON serialization would be done by the caller
        return report
    
    def _format_as_yaml(
        self,
        report: Dict[str, Any],
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Format report as YAML structure."""
        # For YAML format, we return a structure optimized for YAML output
        # The actual YAML serialization would be done by the caller
        return {
            "yaml_content": report,
            "format": "yaml"
        }
    
    def _format_as_text(
        self,
        report: Dict[str, Any],
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Format report as human-readable text."""
        
        text_lines = []
        
        # Header
        text_lines.append("=" * 60)
        text_lines.append(f"WORKSPACE VALIDATION REPORT")
        text_lines.append("=" * 60)
        text_lines.append(f"Generated: {report.get('generated_at', 'Unknown')}")
        text_lines.append(f"Workspace Root: {report.get('workspace_root', 'Unknown')}")
        text_lines.append("")
        
        # Summary section
        if report["report_type"] == "single_workspace_validation":
            text_lines.extend(self._format_single_workspace_text(report, config))
        else:
            text_lines.extend(self._format_multi_workspace_text(report, config))
        
        # Recommendations section
        if config.include_recommendations and "recommendations" in report:
            text_lines.append("")
            text_lines.append("RECOMMENDATIONS:")
            text_lines.append("-" * 20)
            for i, rec in enumerate(report["recommendations"], 1):
                text_lines.append(f"{i}. {rec}")
        
        text_lines.append("")
        text_lines.append("=" * 60)
        
        return {
            "text_content": "\n".join(text_lines),
            "format": "text"
        }
    
    def _format_single_workspace_text(
        self,
        report: Dict[str, Any],
        config: ReportConfig
    ) -> List[str]:
        """Format single workspace report as text."""
        lines = []
        
        # Workspace info
        lines.append(f"Workspace ID: {report.get('workspace_id', 'Unknown')}")
        lines.append(f"Workspace Type: {report.get('workspace_type', 'Unknown')}")
        lines.append(f"Workspace Path: {report.get('workspace_path', 'Unknown')}")
        lines.append("")
        
        # Validation summary
        summary = report.get("validation_summary", {})
        lines.append("VALIDATION SUMMARY:")
        lines.append("-" * 20)
        lines.append(f"Overall Success: {'✓' if summary.get('overall_success') else '✗'}")
        lines.append(f"Validation Types: {', '.join(summary.get('validation_types_run', []))}")
        lines.append(f"Has Errors: {'Yes' if summary.get('has_errors') else 'No'}")
        lines.append(f"Warning Count: {summary.get('warning_count', 0)}")
        
        # Timing info
        if config.include_timing and "timing" in report:
            timing = report["timing"]
            lines.append("")
            lines.append("TIMING INFORMATION:")
            lines.append("-" * 20)
            if timing.get("validation_duration_seconds"):
                lines.append(f"Validation Duration: {timing['validation_duration_seconds']:.2f} seconds")
        
        return lines
    
    def _format_multi_workspace_text(
        self,
        report: Dict[str, Any],
        config: ReportConfig
    ) -> List[str]:
        """Format multi-workspace report as text."""
        lines = []
        
        # Overall summary
        summary = report.get("validation_summary", {})
        lines.append("OVERALL SUMMARY:")
        lines.append("-" * 20)
        lines.append(f"Total Workspaces: {summary.get('total_workspaces', 0)}")
        lines.append(f"Successful: {summary.get('successful_workspaces', 0)}")
        lines.append(f"Failed: {summary.get('failed_workspaces', 0)}")
        lines.append(f"Success Rate: {summary.get('success_percentage', 0):.1f}%")
        lines.append(f"Overall Success: {'✓' if summary.get('overall_success') else '✗'}")
        
        # Timing info
        if config.include_timing and "timing" in report:
            timing = report["timing"]
            lines.append("")
            lines.append("TIMING INFORMATION:")
            lines.append("-" * 20)
            if timing.get("total_validation_duration_seconds"):
                lines.append(f"Total Duration: {timing['total_validation_duration_seconds']:.2f} seconds")
        
        # Workspace results
        if "workspace_results" in report:
            lines.append("")
            lines.append("WORKSPACE RESULTS:")
            lines.append("-" * 20)
            
            for workspace_id, workspace_data in report["workspace_results"].items():
                status = "✓" if workspace_data.get("success") else "✗"
                lines.append(f"{status} {workspace_id} ({workspace_data.get('workspace_type', 'unknown')})")
        
        elif "workspace_summaries" in report:
            lines.append("")
            lines.append("WORKSPACE SUMMARIES:")
            lines.append("-" * 20)
            
            for workspace_id, workspace_summary in report["workspace_summaries"].items():
                status = "✓" if workspace_summary.get("success") else "✗"
                lines.append(f"{status} {workspace_id} ({workspace_summary.get('workspace_type', 'unknown')})")
        
        # Failed workspaces
        if "failed_workspaces" in report and report["failed_workspaces"]:
            lines.append("")
            lines.append("FAILED WORKSPACES:")
            lines.append("-" * 20)
            for workspace_id in report["failed_workspaces"]:
                lines.append(f"✗ {workspace_id}")
        
        return lines
    
    def generate_summary_report(
        self,
        result: UnifiedValidationResult
    ) -> Dict[str, Any]:
        """Generate a concise summary report."""
        summary_config = ReportConfig(
            include_details=False,
            include_summary_only=True,
            compact_format=True
        )
        
        return self.generate_report(result, summary_config)
    
    def generate_detailed_report(
        self,
        result: UnifiedValidationResult
    ) -> Dict[str, Any]:
        """Generate a detailed report with all information."""
        detailed_config = ReportConfig(
            include_details=True,
            include_recommendations=True,
            include_workspace_info=True,
            include_timing=True,
            include_summary_only=False
        )
        
        return self.generate_report(result, detailed_config)
    
    def export_report(
        self,
        result: UnifiedValidationResult,
        output_file: str,
        report_config: Optional[ReportConfig] = None
    ) -> bool:
        """
        Export report to file.
        
        Args:
            result: Validation result to export
            output_file: Path to output file
            report_config: Report configuration
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            config = report_config or self.report_config
            report = self.generate_report(result, config)
            
            with open(output_file, 'w') as f:
                if config.output_format == "yaml":
                    yaml.safe_dump(
                        report.get("yaml_content", report),
                        f,
                        default_flow_style=False,
                        indent=2
                    )
                elif config.output_format == "text":
                    f.write(report.get("text_content", str(report)))
                else:  # JSON
                    json.dump(
                        report,
                        f,
                        indent=None if config.compact_format else 2,
                        separators=(',', ':') if config.compact_format else (',', ': ')
                    )
            
            logger.info(f"Report exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export report to {output_file}: {e}")
            return False


def generate_unified_report(
    result: UnifiedValidationResult,
    output_format: str = "json",
    include_details: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to generate a unified report.
    
    Args:
        result: Validation result to generate report for
        output_format: Output format ('json', 'yaml', 'text')
        include_details: Whether to include detailed results
        
    Returns:
        Generated report dictionary
    """
    config = ReportConfig(
        output_format=output_format,
        include_details=include_details
    )
    
    generator = UnifiedReportGenerator(config)
    return generator.generate_report(result)


def export_unified_report(
    result: UnifiedValidationResult,
    output_file: str,
    output_format: str = "json",
    include_details: bool = True
) -> bool:
    """
    Convenience function to export a unified report to file.
    
    Args:
        result: Validation result to export
        output_file: Path to output file
        output_format: Output format ('json', 'yaml', 'text')
        include_details: Whether to include detailed results
        
    Returns:
        True if export successful, False otherwise
    """
    config = ReportConfig(
        output_format=output_format,
        include_details=include_details
    )
    
    generator = UnifiedReportGenerator(config)
    return generator.export_report(result, output_file, config)
