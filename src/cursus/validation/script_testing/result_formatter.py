"""
Result Formatter

Comprehensive result formatting utilities for script testing results.
Provides various output formats and visualization options for script execution results.

This is a well-designed component with 15% redundancy (Good Efficiency) that we preserve
as-is from the original implementation.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import csv
from datetime import datetime
from io import StringIO
import logging

# Import our simplified ScriptTestResult from api.py
from .api import ScriptTestResult

logger = logging.getLogger(__name__)


class ResultFormatter:
    """
    Comprehensive result formatting utilities for script testing results.
    
    Provides multiple output formats and visualization options for script
    execution results, including console output, JSON, CSV, and HTML reports.
    
    Key features:
    1. Multiple output formats (console, JSON, CSV, HTML)
    2. Customizable formatting options
    3. Summary and detailed reporting
    4. Error highlighting and analysis
    5. Performance metrics visualization
    
    Attributes:
        format_options: Dictionary of formatting configuration options
    """
    
    def __init__(self, format_options: Optional[Dict[str, Any]] = None):
        """
        Initialize the Result Formatter.
        
        Args:
            format_options: Optional formatting configuration
        """
        self.format_options = format_options or {
            "show_timestamps": True,
            "show_execution_times": True,
            "show_file_paths": True,
            "highlight_errors": True,
            "include_metadata": True,
            "max_error_length": 500,
            "date_format": "%Y-%m-%d %H:%M:%S",
        }
        
        logger.info("Initialized ResultFormatter with custom formatting options")
    
    def format_execution_results(
        self, 
        results: Dict[str, Any], 
        format_type: str = "console"
    ) -> str:
        """
        Format complete execution results in specified format.
        
        Args:
            results: Dictionary with execution results from script testing
            format_type: Output format ("console", "json", "csv", "html")
            
        Returns:
            Formatted results string
        """
        if format_type == "console":
            return self._format_console_results(results)
        elif format_type == "json":
            return self._format_json_results(results)
        elif format_type == "csv":
            return self._format_csv_results(results)
        elif format_type == "html":
            return self._format_html_results(results)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_script_result(
        self, 
        script_result: ScriptTestResult, 
        format_type: str = "console"
    ) -> str:
        """
        Format individual script result.
        
        Args:
            script_result: ScriptTestResult to format
            format_type: Output format ("console", "json", "summary")
            
        Returns:
            Formatted script result string
        """
        if format_type == "console":
            return self._format_console_script_result(script_result)
        elif format_type == "json":
            return json.dumps(self._script_result_to_dict(script_result), indent=2, default=str)
        elif format_type == "summary":
            return self._format_script_summary(script_result)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Create a comprehensive summary report.
        
        Args:
            results: Dictionary with execution results
            
        Returns:
            Formatted summary report
        """
        summary = StringIO()
        
        # Header
        summary.write("=" * 80 + "\n")
        summary.write("SCRIPT EXECUTION SUMMARY REPORT\n")
        summary.write("=" * 80 + "\n\n")
        
        # Execution overview
        summary.write("📊 EXECUTION OVERVIEW\n")
        summary.write("-" * 40 + "\n")
        summary.write(f"Total Scripts: {results.get('total_scripts', 0)}\n")
        summary.write(f"Successful: {results.get('successful_scripts', 0)}\n")
        summary.write(f"Failed: {results.get('total_scripts', 0) - results.get('successful_scripts', 0)}\n")
        
        if results.get('total_scripts', 0) > 0:
            success_rate = results.get('successful_scripts', 0) / results.get('total_scripts', 1)
            summary.write(f"Success Rate: {success_rate:.1%}\n")
        
        summary.write(f"Pipeline Success: {'✅ YES' if results.get('pipeline_success', False) else '❌ NO'}\n\n")
        
        # Script results summary
        script_results = results.get("script_results", {})
        if script_results:
            summary.write("📝 SCRIPT RESULTS\n")
            summary.write("-" * 40 + "\n")
            
            for node_name, script_result in script_results.items():
                status = "✅" if script_result.success else "❌"
                time_str = f" ({script_result.execution_time:.2f}s)" if script_result.execution_time else ""
                summary.write(f"{status} {node_name}{time_str}\n")
                
                if not script_result.success and script_result.error_message:
                    error_msg = script_result.error_message
                    if len(error_msg) > self.format_options["max_error_length"]:
                        error_msg = error_msg[:self.format_options["max_error_length"]] + "..."
                    summary.write(f"    Error: {error_msg}\n")
            
            summary.write("\n")
        
        # Execution order
        execution_order = results.get("execution_order", [])
        if execution_order:
            summary.write("🔄 EXECUTION ORDER\n")
            summary.write("-" * 40 + "\n")
            for i, node_name in enumerate(execution_order, 1):
                summary.write(f"{i:2d}. {node_name}\n")
            summary.write("\n")
        
        summary.write("=" * 80 + "\n")
        
        return summary.getvalue()
    
    def save_results_to_file(
        self, 
        results: Dict[str, Any], 
        output_path: str, 
        format_type: str = "json"
    ) -> Path:
        """
        Save results to file in specified format.
        
        Args:
            results: Dictionary with execution results
            output_path: Path to save results
            format_type: Output format ("json", "csv", "html", "txt")
            
        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            formatted_results = self._format_json_results(results)
        elif format_type == "csv":
            formatted_results = self._format_csv_results(results)
        elif format_type == "html":
            formatted_results = self._format_html_results(results)
        elif format_type == "txt":
            formatted_results = self.create_summary_report(results)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
        
        logger.info(f"Results saved to {output_file} in {format_type} format")
        return output_file
    
    def _format_console_results(self, results: Dict[str, Any]) -> str:
        """Format results for console output."""
        output = StringIO()
        
        # Header with timestamp
        output.write("\n" + "=" * 60 + "\n")
        output.write("🔧 SCRIPT EXECUTION RESULTS\n")
        if self.format_options["show_timestamps"]:
            output.write(f"Generated: {datetime.now().strftime(self.format_options['date_format'])}\n")
        output.write("=" * 60 + "\n\n")
        
        # Quick summary
        pipeline_success = results.get("pipeline_success", False)
        status_icon = "✅" if pipeline_success else "❌"
        output.write(f"{status_icon} Pipeline Status: {'SUCCESS' if pipeline_success else 'FAILED'}\n")
        
        total_scripts = results.get('total_scripts', 0)
        successful_scripts = results.get('successful_scripts', 0)
        if total_scripts > 0:
            output.write(f"📊 Scripts: {successful_scripts}/{total_scripts} successful\n")
        
        output.write("\n")
        
        # Execution order and results
        execution_order = results.get("execution_order", [])
        script_results = results.get("script_results", {})
        
        if execution_order and script_results:
            output.write("📝 SCRIPT EXECUTION DETAILS\n")
            output.write("-" * 40 + "\n")
            
            for i, node_name in enumerate(execution_order, 1):
                if node_name in script_results:
                    script_result = script_results[node_name]
                    status = "✅" if script_result.success else "❌"
                    
                    output.write(f"{i:2d}. {status} {node_name}")
                    
                    if self.format_options["show_execution_times"] and script_result.execution_time:
                        output.write(f" ({script_result.execution_time:.2f}s)")
                    
                    output.write("\n")
                    
                    if not script_result.success and script_result.error_message:
                        if self.format_options["highlight_errors"]:
                            output.write(f"     ❌ Error: {script_result.error_message}\n")
                    
                    if script_result.output_files and self.format_options["show_file_paths"]:
                        output.write(f"     📁 Outputs: {len(script_result.output_files)} files\n")
            
            output.write("\n")
        
        output.write("=" * 60 + "\n")
        
        return output.getvalue()
    
    def _format_json_results(self, results: Dict[str, Any]) -> str:
        """Format results as JSON."""
        # Create a serializable version of results
        serializable_results = self._make_serializable(results)
        
        # Add metadata
        serializable_results["_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "formatter_version": "1.0.0",
            "format_type": "json"
        }
        
        return json.dumps(serializable_results, indent=2, default=str)
    
    def _format_csv_results(self, results: Dict[str, Any]) -> str:
        """Format results as CSV."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        headers = [
            "node_name", "success", "execution_time", 
            "output_files_count", "error_message"
        ]
        writer.writerow(headers)
        
        # Script results
        script_results = results.get("script_results", {})
        execution_order = results.get("execution_order", [])
        
        for node_name in execution_order:
            if node_name in script_results:
                script_result = script_results[node_name]
                row = [
                    node_name,
                    script_result.success,
                    script_result.execution_time or 0,
                    len(script_result.output_files) if script_result.output_files else 0,
                    script_result.error_message or ""
                ]
                writer.writerow(row)
        
        return output.getvalue()
    
    def _format_html_results(self, results: Dict[str, Any]) -> str:
        """Format results as HTML."""
        html = StringIO()
        
        # HTML header
        html.write("""<!DOCTYPE html>
<html>
<head>
    <title>Script Execution Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .success { color: green; }
        .failure { color: red; }
        .summary { background-color: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .script-result { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .error { background-color: #ffe6e6; padding: 5px; margin: 5px 0; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
""")
        
        # Header
        html.write('<div class="header">')
        html.write('<h1>🔧 Script Execution Results</h1>')
        if self.format_options["show_timestamps"]:
            html.write(f'<p>Generated: {datetime.now().strftime(self.format_options["date_format"])}</p>')
        html.write('</div>')
        
        # Summary
        pipeline_success = results.get("pipeline_success", False)
        status_class = "success" if pipeline_success else "failure"
        status_text = "SUCCESS" if pipeline_success else "FAILED"
        
        html.write('<div class="summary">')
        html.write(f'<h2 class="{status_class}">Pipeline Status: {status_text}</h2>')
        
        total_scripts = results.get('total_scripts', 0)
        successful_scripts = results.get('successful_scripts', 0)
        if total_scripts > 0:
            html.write(f'<p>Scripts: {successful_scripts}/{total_scripts} successful</p>')
        
        html.write('</div>')
        
        # Script results table
        script_results = results.get("script_results", {})
        execution_order = results.get("execution_order", [])
        
        if script_results:
            html.write('<h2>Script Results</h2>')
            html.write('<table>')
            html.write('<tr><th>Script</th><th>Status</th><th>Time (s)</th><th>Outputs</th><th>Error</th></tr>')
            
            for node_name in execution_order:
                if node_name in script_results:
                    script_result = script_results[node_name]
                    status_class = "success" if script_result.success else "failure"
                    status_icon = "✅" if script_result.success else "❌"
                    
                    html.write('<tr>')
                    html.write(f'<td>{node_name}</td>')
                    html.write(f'<td class="{status_class}">{status_icon}</td>')
                    html.write(f'<td>{script_result.execution_time:.2f if script_result.execution_time else "N/A"}</td>')
                    html.write(f'<td>{len(script_result.output_files) if script_result.output_files else 0}</td>')
                    html.write(f'<td>{script_result.error_message or ""}</td>')
                    html.write('</tr>')
            
            html.write('</table>')
        
        html.write('</body></html>')
        
        return html.getvalue()
    
    def _format_console_script_result(self, script_result: ScriptTestResult) -> str:
        """Format individual script result for console."""
        output = StringIO()
        
        status = "✅ SUCCESS" if script_result.success else "❌ FAILED"
        output.write(f"{status}\n")
        
        if self.format_options["show_execution_times"] and script_result.execution_time:
            output.write(f"  Time: {script_result.execution_time:.2f}s\n")
        
        if script_result.output_files:
            output.write(f"  Outputs: {len(script_result.output_files)} files\n")
            if self.format_options["show_file_paths"]:
                for output_name, output_path in script_result.output_files.items():
                    output.write(f"    - {output_name}: {output_path}\n")
        
        if not script_result.success and script_result.error_message:
            output.write(f"  Error: {script_result.error_message}\n")
        
        return output.getvalue()
    
    def _format_script_summary(self, script_result: ScriptTestResult) -> str:
        """Format script result as summary."""
        status = "SUCCESS" if script_result.success else "FAILED"
        time_str = f" ({script_result.execution_time:.2f}s)" if script_result.execution_time else ""
        return f"{status}{time_str}"
    
    def _script_result_to_dict(self, script_result: ScriptTestResult) -> Dict[str, Any]:
        """Convert ScriptTestResult to dictionary."""
        return {
            'success': script_result.success,
            'output_files': script_result.output_files,
            'error_message': script_result.error_message,
            'execution_time': script_result.execution_time
        }
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, ScriptTestResult):
            return self._script_result_to_dict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        else:
            return obj
    
    def get_formatter_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the formatter configuration and capabilities.
        
        Returns:
            Dictionary with formatter summary information
        """
        return {
            "supported_formats": ["console", "json", "csv", "html", "txt"],
            "format_options": self.format_options,
            "features": [
                "Multiple output formats",
                "Customizable formatting",
                "Summary and detailed reporting",
                "Error highlighting",
                "Performance metrics",
                "File export capabilities"
            ],
            "formatter_type": "ResultFormatter",
        }
    
    def __str__(self) -> str:
        """String representation of the formatter."""
        return f"ResultFormatter(formats=['console', 'json', 'csv', 'html'])"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ResultFormatter("
            f"options={len(self.format_options)} configured)"
        )
