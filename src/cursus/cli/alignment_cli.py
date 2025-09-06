#!/usr/bin/env python3
"""
Command-line interface for the Unified Alignment Tester.

This CLI provides comprehensive alignment validation across all four levels:
1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment  
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment

The CLI supports both individual script validation and batch validation of all scripts.
"""

import sys
import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from ..validation.alignment.alignment_scorer import AlignmentScorer


def print_validation_summary(results: Dict[str, Any], verbose: bool = False, show_scoring: bool = False) -> None:
    """Print validation results in a formatted way."""
    script_name = results.get('script_name', 'Unknown')
    status = results.get('overall_status', 'UNKNOWN')
    
    # Status emoji and color
    if status == 'PASSING':
        status_emoji = '✅'
        status_color = 'green'
    elif status == 'FAILING':
        status_emoji = '❌'
        status_color = 'red'
    else:
        status_emoji = '⚠️'
        status_color = 'yellow'
    
    click.echo(f"\n{status_emoji} {script_name}: ", nl=False)
    click.secho(status, fg=status_color, bold=True)
    
    # Show scoring information if requested
    if show_scoring:
        try:
            scorer = AlignmentScorer(results)
            overall_score = scorer.calculate_overall_score()
            quality_rating = scorer.get_quality_rating()
            level_scores = scorer.get_level_scores()
            
            # Color-code the quality rating
            rating_colors = {
                'Excellent': 'green',
                'Good': 'green', 
                'Satisfactory': 'yellow',
                'Needs Work': 'yellow',
                'Poor': 'red'
            }
            rating_color = rating_colors.get(quality_rating, 'white')
            
            click.echo(f"📊 Overall Score: ", nl=False)
            click.secho(f"{overall_score:.1f}/100", fg=rating_color, bold=True, nl=False)
            click.echo(f" (", nl=False)
            click.secho(quality_rating, fg=rating_color, bold=True, nl=False)
            click.echo(")")
            
            if verbose:
                click.echo("📈 Level Scores:")
                level_names = {
                    'level1_script_contract': 'Level 1 (Script ↔ Contract)',
                    'level2_contract_spec': 'Level 2 (Contract ↔ Specification)',
                    'level3_spec_dependencies': 'Level 3 (Specification ↔ Dependencies)',
                    'level4_builder_config': 'Level 4 (Builder ↔ Configuration)'
                }
                
                for level_key, score in level_scores.items():
                    level_name = level_names.get(level_key, level_key)
                    click.echo(f"  • {level_name}: {score:.1f}/100")
                    
        except Exception as e:
            if verbose:
                click.echo(f"⚠️  Could not calculate scoring: {e}")
    
    # Print level-by-level results
    level_names = [
        "Script ↔ Contract",
        "Contract ↔ Specification", 
        "Specification ↔ Dependencies",
        "Builder ↔ Configuration"
    ]
    
    for level_num, level_name in enumerate(level_names, 1):
        level_key = f"level{level_num}"
        level_result = results.get(level_key, {})
        level_passed = level_result.get('passed', False)
        level_issues = level_result.get('issues', [])
        
        level_emoji = '✅' if level_passed else '❌'
        issues_text = f" ({len(level_issues)} issues)" if level_issues else ""
        
        click.echo(f"  {level_emoji} Level {level_num} ({level_name}): ", nl=False)
        level_status = 'PASS' if level_passed else 'FAIL'
        level_color = 'green' if level_passed else 'red'
        click.secho(f"{level_status}{issues_text}", fg=level_color)
        
        # Show issues if verbose or if there are critical/error issues
        if verbose or any(issue.get('severity') in ['CRITICAL', 'ERROR'] for issue in level_issues):
            for issue in level_issues:
                severity = issue.get('severity', 'INFO')
                message = issue.get('message', 'No message')
                
                severity_colors = {
                    'CRITICAL': 'red',
                    'ERROR': 'red', 
                    'WARNING': 'yellow',
                    'INFO': 'blue'
                }
                
                severity_color = severity_colors.get(severity, 'white')
                click.echo(f"    • ", nl=False)
                click.secho(f"[{severity}]", fg=severity_color, nl=False)
                click.echo(f" {message}")
                
                # Show recommendation if available
                recommendation = issue.get('recommendation')
                if recommendation and verbose:
                    click.echo(f"      💡 {recommendation}")


def _make_json_serializable(obj: Any) -> Any:
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
        return [_make_json_serializable(item) for item in obj]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Ensure keys are strings
            str_key = str(key)
            result[str_key] = _make_json_serializable(value)
        return result
    
    # Handle sets - convert to sorted lists
    if isinstance(obj, set):
        return sorted([_make_json_serializable(item) for item in obj])
    
    # Handle Path objects
    if hasattr(obj, '__fspath__'):  # Path-like objects
        return str(obj)
    
    # Handle datetime objects
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    # Handle Enum objects
    if hasattr(obj, 'value'):
        return obj.value
    
    # Handle type objects
    if isinstance(obj, type):
        return str(obj.__name__)
    
    # For everything else, try string conversion
    try:
        str_value = str(obj)
        # Avoid storing string representations of complex objects
        if '<' in str_value and '>' in str_value and 'object at' in str_value:
            return f"<{type(obj).__name__}>"
        return str_value
    except Exception:
        return f"<{type(obj).__name__}>"


def save_report(script_name: str, results: Dict[str, Any], output_dir: Path, format: str) -> None:
    """Save validation results to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        output_file = output_dir / f"{script_name}_alignment_report.json"
        try:
            # Clean the results to ensure JSON serializability
            cleaned_results = _make_json_serializable(results)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            click.echo(f"📄 JSON report saved: {output_file}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not save JSON report for {script_name}: {e}")
            # Try to save a simplified version
            try:
                simplified_results = {
                    'script_name': script_name,
                    'overall_status': results.get('overall_status', 'ERROR'),
                    'error': f'JSON serialization failed: {str(e)}',
                    'metadata': results.get('metadata', {})
                }
                cleaned_simplified = _make_json_serializable(simplified_results)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_simplified, f, indent=2, default=str)
                click.echo(f"📄 Simplified JSON report saved: {output_file}")
            except Exception as e2:
                click.echo(f"❌ Failed to save even simplified JSON report for {script_name}: {e2}")
    
    elif format == 'html':
        output_file = output_dir / f"{script_name}_alignment_report.html"
        html_content = generate_html_report(script_name, results)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        click.echo(f"🌐 HTML report saved: {output_file}")


def generate_html_report(script_name: str, results: Dict[str, Any]) -> str:
    """Generate HTML report for validation results."""
    status = results.get('overall_status', 'UNKNOWN')
    status_class = 'passing' if status == 'PASSING' else 'failing'
    timestamp = results.get('metadata', {}).get('validation_timestamp', 'Unknown')
    
    # Count issues by severity
    total_issues = 0
    critical_issues = 0
    error_issues = 0
    warning_issues = 0
    
    for level_num in range(1, 5):
        level_key = f"level{level_num}"
        level_result = results.get(level_key, {})
        issues = level_result.get('issues', [])
        total_issues += len(issues)
        
        for issue in issues:
            severity = issue.get('severity', 'ERROR')
            if severity == 'CRITICAL':
                critical_issues += 1
            elif severity == 'ERROR':
                error_issues += 1
            elif severity == 'WARNING':
                warning_issues += 1
    
    # Generate level sections
    level_sections = ""
    for level_num, level_name in enumerate([
        "Level 1: Script ↔ Contract",
        "Level 2: Contract ↔ Specification",
        "Level 3: Specification ↔ Dependencies", 
        "Level 4: Builder ↔ Configuration"
    ], 1):
        level_key = f"level{level_num}"
        level_result = results.get(level_key, {})
        level_passed = level_result.get('passed', False)
        level_issues = level_result.get('issues', [])
        
        result_class = "test-passed" if level_passed else "test-failed"
        status_text = "PASSED" if level_passed else "FAILED"
        
        issues_html = ""
        for issue in level_issues:
            severity = issue.get('severity', 'ERROR').lower()
            message = issue.get('message', 'No message')
            recommendation = issue.get('recommendation', '')
            
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


@click.group()
@click.pass_context
def alignment(ctx):
    """
    Unified Alignment Tester for Cursus Scripts.
    
    Validates alignment across all four levels:
    1. Script ↔ Contract Alignment
    2. Contract ↔ Specification Alignment  
    3. Specification ↔ Dependencies Alignment
    4. Builder ↔ Configuration Alignment
    """
    ctx.ensure_object(dict)


@alignment.command()
@click.argument('script_name')
@click.option('--scripts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing scripts (default: src/cursus/steps/scripts)')
@click.option('--contracts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing contracts (default: src/cursus/steps/contracts)')
@click.option('--specs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing specifications (default: src/cursus/steps/specs)')
@click.option('--builders-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing builders (default: src/cursus/steps/builders)')
@click.option('--configs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing configs (default: src/cursus/steps/configs)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for reports')
@click.option('--format', type=click.Choice(['json', 'html', 'both']), default='json',
              help='Output format for reports')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--show-scoring', is_flag=True, help='Show alignment scoring information')
@click.pass_context
def validate(ctx, script_name, scripts_dir, contracts_dir, specs_dir, builders_dir, 
             configs_dir, output_dir, format, verbose, show_scoring):
    """
    Validate alignment for a specific script.
    
    SCRIPT_NAME: Name of the script to validate (without .py extension)
    
    Example:
        cursus alignment validate currency_conversion --verbose
        cursus alignment validate dummy_training --output-dir ./reports --format html
    """
    # Set default directories if not provided
    project_root = Path.cwd()
    # Check if we're already in the src directory
    if project_root.name == "src":
        base_path = project_root / "cursus" / "steps"
    else:
        base_path = project_root / "src" / "cursus" / "steps"
    
    if not scripts_dir:
        scripts_dir = base_path / "scripts"
    if not contracts_dir:
        contracts_dir = base_path / "contracts"
    if not specs_dir:
        specs_dir = base_path / "specs"
    if not builders_dir:
        builders_dir = base_path / "builders"
    if not configs_dir:
        configs_dir = base_path / "configs"
    
    if verbose:
        click.echo(f"🔍 Validating script: {script_name}")
        click.echo(f"📁 Scripts directory: {scripts_dir}")
        click.echo(f"📁 Contracts directory: {contracts_dir}")
        click.echo(f"📁 Specifications directory: {specs_dir}")
        click.echo(f"📁 Builders directory: {builders_dir}")
        click.echo(f"📁 Configs directory: {configs_dir}")
    
    try:
        # Initialize the unified alignment tester
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir),
            configs_dir=str(configs_dir)
        )
        
        # Run validation
        results = tester.validate_specific_script(script_name)
        
        # Add metadata
        results['metadata'] = {
            'script_path': str(scripts_dir / f"{script_name}.py"),
            'validation_timestamp': datetime.now().isoformat(),
            'validator_version': '1.0.0'
        }
        
        # Print results
        print_validation_summary(results, verbose, show_scoring)
        
        # Save reports if output directory specified
        if output_dir:
            if format in ['json', 'both']:
                save_report(script_name, results, output_dir, 'json')
            if format in ['html', 'both']:
                save_report(script_name, results, output_dir, 'html')
        
        # Return appropriate exit code
        status = results.get('overall_status', 'UNKNOWN')
        if status == 'PASSING':
            click.echo(f"\n✅ {script_name} passed all alignment validation checks!")
            # Exit successfully without raising an exception
            sys.exit(0)
        else:
            click.echo(f"\n❌ {script_name} failed alignment validation. Please review the issues above.")
            ctx.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error validating {script_name}: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.option('--scripts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing scripts (default: src/cursus/steps/scripts)')
@click.option('--contracts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing contracts (default: src/cursus/steps/contracts)')
@click.option('--specs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing specifications (default: src/cursus/steps/specs)')
@click.option('--builders-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing builders (default: src/cursus/steps/builders)')
@click.option('--configs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing configs (default: src/cursus/steps/configs)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for reports')
@click.option('--format', type=click.Choice(['json', 'html', 'both']), default='json',
              help='Output format for reports')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--continue-on-error', is_flag=True, 
              help='Continue validation even if individual scripts fail')
@click.pass_context
def validate_all(ctx, scripts_dir, contracts_dir, specs_dir, builders_dir, configs_dir,
                 output_dir, format, verbose, continue_on_error):
    """
    Validate alignment for all scripts in the scripts directory.
    
    Discovers all Python scripts and runs comprehensive alignment validation
    for each one, generating detailed reports.
    
    Example:
        cursus alignment validate-all --output-dir ./reports --format both --verbose
    """
    try:
        # Set default directories if not provided
        project_root = Path.cwd()
        # Check if we're already in the src directory
        if project_root.name == "src":
            base_path = project_root / "cursus" / "steps"
        else:
            base_path = project_root / "src" / "cursus" / "steps"
        
        if not scripts_dir:
            scripts_dir = base_path / "scripts"
        if not contracts_dir:
            contracts_dir = base_path / "contracts"
        if not specs_dir:
            specs_dir = base_path / "specs"
        if not builders_dir:
            builders_dir = base_path / "builders"
        if not configs_dir:
            configs_dir = base_path / "configs"
        
        click.echo("🚀 Starting Comprehensive Script Alignment Validation")
        if verbose:
            click.echo(f"📁 Scripts directory: {scripts_dir}")
            click.echo(f"📁 Contracts directory: {contracts_dir}")
            click.echo(f"📁 Specifications directory: {specs_dir}")
            click.echo(f"📁 Builders directory: {builders_dir}")
            click.echo(f"📁 Configs directory: {configs_dir}")
            if output_dir:
                click.echo(f"📁 Output directory: {output_dir}")
        
        # Initialize the unified alignment tester
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir),
            configs_dir=str(configs_dir)
        )
        
        # Discover all scripts
        scripts = []
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                if not script_file.name.startswith('__'):
                    scripts.append(script_file.stem)
        
        scripts = sorted(scripts)
        click.echo(f"\n📋 Discovered {len(scripts)} scripts: {', '.join(scripts)}")
        
        # Validation results summary
        validation_summary = {
            'total_scripts': len(scripts),
            'passed_scripts': 0,
            'failed_scripts': 0,
            'error_scripts': 0,
            'validation_timestamp': datetime.now().isoformat(),
            'script_results': {}
        }
        
        # Validate each script
        for script_name in scripts:
            click.echo(f"\n{'='*60}")
            click.echo(f"🔍 VALIDATING SCRIPT: {script_name}")
            click.echo(f"{'='*60}")
            
            try:
                results = tester.validate_specific_script(script_name)
                
                # Add metadata
                results['metadata'] = {
                    'script_path': str(scripts_dir / f"{script_name}.py"),
                    'validation_timestamp': datetime.now().isoformat(),
                    'validator_version': '1.0.0'
                }
                
                # Print results
                print_validation_summary(results, verbose)
                
                # Save reports if output directory specified
                if output_dir:
                    if format in ['json', 'both']:
                        save_report(script_name, results, output_dir, 'json')
                    if format in ['html', 'both']:
                        save_report(script_name, results, output_dir, 'html')
                
                # Update summary
                status = results.get('overall_status', 'UNKNOWN')
                validation_summary['script_results'][script_name] = {
                    'status': status,
                    'timestamp': results.get('metadata', {}).get('validation_timestamp')
                }
                
                if status == 'PASSING':
                    validation_summary['passed_scripts'] += 1
                elif status == 'FAILING':
                    validation_summary['failed_scripts'] += 1
                else:
                    validation_summary['error_scripts'] += 1
                    
            except Exception as e:
                click.echo(f"❌ Failed to validate {script_name}: {e}")
                validation_summary['error_scripts'] += 1
                validation_summary['script_results'][script_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                if not continue_on_error:
                    click.echo("Stopping validation due to error. Use --continue-on-error to continue.")
                    return 1
        
        # Save overall summary
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_file = output_dir / "validation_summary.json"
            try:
                cleaned_summary = _make_json_serializable(validation_summary)
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_summary, f, indent=2, default=str)
                click.echo(f"\n📊 Validation summary saved: {summary_file}")
            except Exception as e:
                click.echo(f"⚠️  Warning: Could not save validation summary: {e}")
        
        # Print final summary
        click.echo(f"\n{'='*80}")
        click.echo("🎯 FINAL VALIDATION SUMMARY")
        click.echo(f"{'='*80}")
        
        total = validation_summary['total_scripts']
        passed = validation_summary['passed_scripts']
        failed = validation_summary['failed_scripts']
        errors = validation_summary['error_scripts']
        
        click.echo(f"📊 Total Scripts: {total}")
        click.secho(f"✅ Passed: {passed} ({passed/total*100:.1f}%)", fg='green')
        click.secho(f"❌ Failed: {failed} ({failed/total*100:.1f}%)", fg='red')
        click.secho(f"⚠️  Errors: {errors} ({errors/total*100:.1f}%)", fg='yellow')
        
        if output_dir:
            click.echo(f"\n📁 Reports saved in: {output_dir}")
        
        # List scripts by status
        if passed > 0:
            passing_scripts = [name for name, result in validation_summary['script_results'].items() 
                             if result['status'] == 'PASSING']
            click.echo(f"\n✅ PASSING SCRIPTS ({len(passing_scripts)}):")
            for script in passing_scripts:
                click.echo(f"   • {script}")
        
        if failed > 0:
            failing_scripts = [name for name, result in validation_summary['script_results'].items() 
                             if result['status'] == 'FAILING']
            click.echo(f"\n❌ FAILING SCRIPTS ({len(failing_scripts)}):")
            for script in failing_scripts:
                click.echo(f"   • {script}")
        
        if errors > 0:
            error_scripts = [name for name, result in validation_summary['script_results'].items() 
                           if result['status'] == 'ERROR']
            click.echo(f"\n⚠️  ERROR SCRIPTS ({len(error_scripts)}):")
            for script in error_scripts:
                click.echo(f"   • {script}")
        
        click.echo(f"\n{'='*80}")
        
        # Return appropriate exit code
        if failed > 0 or errors > 0:
            click.echo(f"\n⚠️  {failed + errors} script(s) failed validation. Please review the issues above.")
            ctx.exit(1)
        else:
            click.echo(f"\n🎉 All {passed} scripts passed alignment validation!")
            ctx.exit(0)
            
    except click.exceptions.Exit:
        # Re-raise Click's Exit exception to preserve proper exit handling
        raise
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        click.echo(f"❌ Fatal error during validation: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.argument('script_name')
@click.argument('level', type=click.IntRange(1, 4))
@click.option('--scripts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing scripts (default: src/cursus/steps/scripts)')
@click.option('--contracts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing contracts (default: src/cursus/steps/contracts)')
@click.option('--specs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing specifications (default: src/cursus/steps/specs)')
@click.option('--builders-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing builders (default: src/cursus/steps/builders)')
@click.option('--configs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing configs (default: src/cursus/steps/configs)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.pass_context
def validate_level(ctx, script_name, level, scripts_dir, contracts_dir, specs_dir, 
                   builders_dir, configs_dir, verbose):
    """
    Validate alignment for a specific script at a specific level.
    
    SCRIPT_NAME: Name of the script to validate (without .py extension)
    LEVEL: Validation level (1=Script↔Contract, 2=Contract↔Spec, 3=Spec↔Deps, 4=Builder↔Config)
    
    Example:
        cursus alignment validate-level currency_conversion 1 --verbose
        cursus alignment validate-level dummy_training 3
    """
    level_names = {
        1: "Script ↔ Contract",
        2: "Contract ↔ Specification",
        3: "Specification ↔ Dependencies",
        4: "Builder ↔ Configuration"
    }
    
    try:
        # Set default directories if not provided
        project_root = Path.cwd()
        # Check if we're already in the src directory
        if project_root.name == "src":
            base_path = project_root / "cursus" / "steps"
        else:
            base_path = project_root / "src" / "cursus" / "steps"
        
        if not scripts_dir:
            scripts_dir = base_path / "scripts"
        if not contracts_dir:
            contracts_dir = base_path / "contracts"
        if not specs_dir:
            specs_dir = base_path / "specs"
        if not builders_dir:
            builders_dir = base_path / "builders"
        if not configs_dir:
            configs_dir = base_path / "configs"
        
        click.echo(f"🔍 Validating {script_name} at Level {level} ({level_names[level]})")
        
        if verbose:
            click.echo(f"📁 Scripts directory: {scripts_dir}")
            click.echo(f"📁 Contracts directory: {contracts_dir}")
            click.echo(f"📁 Specifications directory: {specs_dir}")
            click.echo(f"📁 Builders directory: {builders_dir}")
            click.echo(f"📁 Configs directory: {configs_dir}")
        
        # Initialize the unified alignment tester
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir),
            configs_dir=str(configs_dir)
        )
        
        # Run validation for specific level
        results = tester.validate_specific_script(script_name)
        
        # Extract level-specific results
        level_key = f"level{level}"
        level_result = results.get(level_key, {})
        level_passed = level_result.get('passed', False)
        level_issues = level_result.get('issues', [])
        
        # Print level-specific results
        click.echo(f"\n{'='*60}")
        click.echo(f"Level {level} ({level_names[level]}) Results")
        click.echo(f"{'='*60}")
        
        status_emoji = '✅' if level_passed else '❌'
        status_text = 'PASSED' if level_passed else 'FAILED'
        status_color = 'green' if level_passed else 'red'
        
        click.echo(f"{status_emoji} Status: ", nl=False)
        click.secho(status_text, fg=status_color, bold=True)
        
        if level_issues:
            click.echo(f"\n📋 Issues ({len(level_issues)}):")
            for issue in level_issues:
                severity = issue.get('severity', 'INFO')
                message = issue.get('message', 'No message')
                
                severity_colors = {
                    'CRITICAL': 'red',
                    'ERROR': 'red', 
                    'WARNING': 'yellow',
                    'INFO': 'blue'
                }
                
                severity_color = severity_colors.get(severity, 'white')
                click.echo(f"  • ", nl=False)
                click.secho(f"[{severity}]", fg=severity_color, nl=False)
                click.echo(f" {message}")
                
                # Show recommendation if available
                recommendation = issue.get('recommendation')
                if recommendation and verbose:
                    click.echo(f"    💡 {recommendation}")
        else:
            click.echo("\n✅ No issues found!")
        
        # Return appropriate exit code
        if level_passed:
            click.echo(f"\n✅ {script_name} passed Level {level} validation!")
            return 0
        else:
            click.echo(f"\n❌ {script_name} failed Level {level} validation. Please review the issues above.")
            return 1
            
    except Exception as e:
        click.echo(f"❌ Error validating {script_name} at Level {level}: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@alignment.command()
@click.argument('script_name')
@click.option('--scripts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing scripts (default: src/cursus/steps/scripts)')
@click.option('--contracts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing contracts (default: src/cursus/steps/contracts)')
@click.option('--specs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing specifications (default: src/cursus/steps/specs)')
@click.option('--builders-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing builders (default: src/cursus/steps/builders)')
@click.option('--configs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing configs (default: src/cursus/steps/configs)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), required=True,
              help='Output directory for visualization files (required)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.pass_context
def visualize(ctx, script_name, scripts_dir, contracts_dir, specs_dir, builders_dir, 
              configs_dir, output_dir, verbose):
    """
    Generate visualization charts and scoring reports for a specific script.
    
    This command runs full alignment validation and generates:
    - High-resolution PNG chart with scoring breakdown
    - JSON scoring report with detailed metrics
    - Enhanced HTML report with scoring integration
    
    SCRIPT_NAME: Name of the script to validate (without .py extension)
    
    Example:
        cursus alignment visualize currency_conversion --output-dir ./visualizations --verbose
        cursus alignment visualize xgboost_model_evaluation --output-dir ./charts
    """
    # Set default directories if not provided
    project_root = Path.cwd()
    # Check if we're already in the src directory
    if project_root.name == "src":
        base_path = project_root / "cursus" / "steps"
    else:
        base_path = project_root / "src" / "cursus" / "steps"
    
    if not scripts_dir:
        scripts_dir = base_path / "scripts"
    if not contracts_dir:
        contracts_dir = base_path / "contracts"
    if not specs_dir:
        specs_dir = base_path / "specs"
    if not builders_dir:
        builders_dir = base_path / "builders"
    if not configs_dir:
        configs_dir = base_path / "configs"
    
    if verbose:
        click.echo(f"🎨 Generating visualization for script: {script_name}")
        click.echo(f"📁 Scripts directory: {scripts_dir}")
        click.echo(f"📁 Output directory: {output_dir}")
    
    try:
        # Initialize the unified alignment tester
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir),
            configs_dir=str(configs_dir)
        )
        
        # Run validation
        click.echo("🔍 Running alignment validation...")
        results = tester.validate_specific_script(script_name)
        
        # Initialize scorer
        scorer = AlignmentScorer(results)
        overall_score = scorer.calculate_overall_score()
        quality_rating = scorer.get_quality_rating()
        level_scores = scorer.get_level_scores()
        
        # Print scoring summary
        rating_colors = {
            'Excellent': 'green',
            'Good': 'green', 
            'Satisfactory': 'yellow',
            'Needs Work': 'yellow',
            'Poor': 'red'
        }
        rating_color = rating_colors.get(quality_rating, 'white')
        
        click.echo(f"\n📊 Alignment Scoring Results:")
        click.echo(f"Overall Score: ", nl=False)
        click.secho(f"{overall_score:.1f}/100", fg=rating_color, bold=True, nl=False)
        click.echo(f" (", nl=False)
        click.secho(quality_rating, fg=rating_color, bold=True, nl=False)
        click.echo(")")
        
        if verbose:
            click.echo("\n📈 Level-by-Level Scores:")
            level_names = {
                'level1_script_contract': 'Level 1 (Script ↔ Contract)',
                'level2_contract_spec': 'Level 2 (Contract ↔ Specification)',
                'level3_spec_dependencies': 'Level 3 (Specification ↔ Dependencies)',
                'level4_builder_config': 'Level 4 (Builder ↔ Configuration)'
            }
            
            for level_key, score in level_scores.items():
                level_name = level_names.get(level_key, level_key)
                click.echo(f"  • {level_name}: {score:.1f}/100")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualization chart
        click.echo("\n🎨 Generating visualization chart...")
        chart_path = scorer.generate_chart(
            output_dir=str(output_dir),
            script_name=script_name
        )
        click.echo(f"📊 Chart saved: {chart_path}")
        
        # Generate scoring report
        click.echo("📄 Generating scoring report...")
        scoring_report = scorer.generate_scoring_report()
        scoring_file = output_dir / f"{script_name}_alignment_scoring_report.json"
        
        with open(scoring_file, 'w', encoding='utf-8') as f:
            json.dump(scoring_report, f, indent=2)
        click.echo(f"📄 Scoring report saved: {scoring_file}")
        
        # Generate enhanced HTML report
        click.echo("🌐 Generating enhanced HTML report...")
        results['metadata'] = {
            'script_path': str(scripts_dir / f"{script_name}.py"),
            'validation_timestamp': datetime.now().isoformat(),
            'validator_version': '1.0.0',
            'chart_path': str(chart_path),
            'scoring_report_path': str(scoring_file)
        }
        
        html_file = output_dir / f"{script_name}_alignment_report.html"
        html_content = generate_html_report(script_name, results)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        click.echo(f"🌐 HTML report saved: {html_file}")
        
        # Summary
        click.echo(f"\n✅ Visualization generation complete for {script_name}!")
        click.echo(f"📁 Files generated in: {output_dir}")
        click.echo(f"  • Chart: {chart_path.name}")
        click.echo(f"  • Scoring Report: {scoring_file.name}")
        click.echo(f"  • HTML Report: {html_file.name}")
        
        return 0
        
    except Exception as e:
        click.echo(f"❌ Error generating visualization for {script_name}: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.option('--scripts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing scripts (default: src/cursus/steps/scripts)')
@click.option('--contracts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing contracts (default: src/cursus/steps/contracts)')
@click.option('--specs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing specifications (default: src/cursus/steps/specs)')
@click.option('--builders-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing builders (default: src/cursus/steps/builders)')
@click.option('--configs-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing configs (default: src/cursus/steps/configs)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), required=True,
              help='Output directory for visualization files (required)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--continue-on-error', is_flag=True, 
              help='Continue visualization even if individual scripts fail')
@click.pass_context
def visualize_all(ctx, scripts_dir, contracts_dir, specs_dir, builders_dir, configs_dir,
                  output_dir, verbose, continue_on_error):
    """
    Generate visualization charts and scoring reports for all scripts.
    
    Discovers all Python scripts and generates comprehensive visualizations
    for each one, including charts, scoring reports, and enhanced HTML reports.
    
    Example:
        cursus alignment visualize-all --output-dir ./visualizations --verbose
    """
    try:
        # Set default directories if not provided
        project_root = Path.cwd()
        if not scripts_dir:
            scripts_dir = project_root / "src" / "cursus" / "steps" / "scripts"
        if not contracts_dir:
            contracts_dir = project_root / "src" / "cursus" / "steps" / "contracts"
        if not specs_dir:
            specs_dir = project_root / "src" / "cursus" / "steps" / "specs"
        if not builders_dir:
            builders_dir = project_root / "src" / "cursus" / "steps" / "builders"
        if not configs_dir:
            configs_dir = project_root / "src" / "cursus" / "steps" / "configs"
        
        click.echo("🎨 Starting Comprehensive Alignment Visualization Generation")
        if verbose:
            click.echo(f"📁 Scripts directory: {scripts_dir}")
            click.echo(f"📁 Output directory: {output_dir}")
        
        # Initialize the unified alignment tester
        tester = UnifiedAlignmentTester(
            scripts_dir=str(scripts_dir),
            contracts_dir=str(contracts_dir),
            specs_dir=str(specs_dir),
            builders_dir=str(builders_dir),
            configs_dir=str(configs_dir)
        )
        
        # Discover all scripts
        scripts = []
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                if not script_file.name.startswith('__'):
                    scripts.append(script_file.stem)
        
        scripts = sorted(scripts)
        click.echo(f"\n📋 Discovered {len(scripts)} scripts: {', '.join(scripts)}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization results summary
        visualization_summary = {
            'total_scripts': len(scripts),
            'successful_visualizations': 0,
            'failed_visualizations': 0,
            'generation_timestamp': datetime.now().isoformat(),
            'script_results': {},
            'overall_statistics': {
                'total_charts_generated': 0,
                'total_scoring_reports_generated': 0,
                'total_html_reports_generated': 0
            }
        }
        
        # Generate visualizations for each script
        for script_name in scripts:
            click.echo(f"\n{'='*60}")
            click.echo(f"🎨 GENERATING VISUALIZATION: {script_name}")
            click.echo(f"{'='*60}")
            
            try:
                # Run validation
                results = tester.validate_specific_script(script_name)
                
                # Initialize scorer
                scorer = AlignmentScorer(results)
                overall_score = scorer.calculate_overall_score()
                quality_rating = scorer.get_quality_rating()
                
                # Print scoring summary
                rating_color = {
                    'Excellent': 'green', 'Good': 'green', 'Satisfactory': 'yellow',
                    'Needs Work': 'yellow', 'Poor': 'red'
                }.get(quality_rating, 'white')
                
                click.echo(f"📊 Score: ", nl=False)
                click.secho(f"{overall_score:.1f}/100", fg=rating_color, bold=True, nl=False)
                click.echo(f" (", nl=False)
                click.secho(quality_rating, fg=rating_color, bold=True, nl=False)
                click.echo(")")
                
                # Generate visualization files
                chart_path = scorer.generate_chart(output_dir=str(output_dir), script_name=script_name)
                scoring_report = scorer.generate_scoring_report()
                
                # Save scoring report
                scoring_file = output_dir / f"{script_name}_alignment_scoring_report.json"
                with open(scoring_file, 'w', encoding='utf-8') as f:
                    json.dump(scoring_report, f, indent=2)
                
                # Generate enhanced HTML report
                results['metadata'] = {
                    'script_path': str(scripts_dir / f"{script_name}.py"),
                    'validation_timestamp': datetime.now().isoformat(),
                    'validator_version': '1.0.0',
                    'chart_path': str(chart_path),
                    'scoring_report_path': str(scoring_file)
                }
                
                html_file = output_dir / f"{script_name}_alignment_report.html"
                html_content = generate_html_report(script_name, results)
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                click.echo(f"✅ Generated: Chart, Scoring Report, HTML Report")
                
                # Update summary
                visualization_summary['successful_visualizations'] += 1
                visualization_summary['overall_statistics']['total_charts_generated'] += 1
                visualization_summary['overall_statistics']['total_scoring_reports_generated'] += 1
                visualization_summary['overall_statistics']['total_html_reports_generated'] += 1
                
                visualization_summary['script_results'][script_name] = {
                    'status': 'SUCCESS',
                    'overall_score': overall_score,
                    'quality_rating': quality_rating,
                    'chart_path': str(chart_path),
                    'scoring_report_path': str(scoring_file),
                    'html_report_path': str(html_file),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                click.echo(f"❌ Failed to generate visualization for {script_name}: {e}")
                visualization_summary['failed_visualizations'] += 1
                visualization_summary['script_results'][script_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                if not continue_on_error:
                    click.echo("Stopping visualization due to error. Use --continue-on-error to continue.")
                    ctx.exit(1)
        
        # Save visualization summary
        summary_file = output_dir / "visualization_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(visualization_summary, f, indent=2)
        click.echo(f"\n📊 Visualization summary saved: {summary_file}")
        
        # Print final summary
        click.echo(f"\n{'='*80}")
        click.echo("🎯 VISUALIZATION GENERATION SUMMARY")
        click.echo(f"{'='*80}")
        
        total = visualization_summary['total_scripts']
        successful = visualization_summary['successful_visualizations']
        failed = visualization_summary['failed_visualizations']
        
        click.echo(f"📊 Total Scripts: {total}")
        click.secho(f"✅ Successful: {successful} ({successful/total*100:.1f}%)", fg='green')
        click.secho(f"❌ Failed: {failed} ({failed/total*100:.1f}%)", fg='red')
        
        stats = visualization_summary['overall_statistics']
        click.echo(f"\n📈 Files Generated:")
        click.echo(f"  • Charts: {stats['total_charts_generated']}")
        click.echo(f"  • Scoring Reports: {stats['total_scoring_reports_generated']}")
        click.echo(f"  • HTML Reports: {stats['total_html_reports_generated']}")
        
        click.echo(f"\n📁 All files saved in: {output_dir}")
        
        # Return appropriate exit code
        if failed > 0:
            click.echo(f"\n⚠️  {failed} script(s) failed visualization generation.")
            ctx.exit(1)
        else:
            click.echo(f"\n🎉 All {successful} visualizations generated successfully!")
            ctx.exit(0)
            
    except click.exceptions.Exit:
        # Re-raise Click's Exit exception to preserve proper exit handling
        raise
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        click.echo(f"❌ Fatal error during visualization generation: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.option('--scripts-dir', type=click.Path(exists=True, path_type=Path),
              help='Directory containing scripts (default: src/cursus/steps/scripts)')
@click.pass_context
def list_scripts(ctx, scripts_dir):
    """
    List all available scripts that can be validated.
    
    Discovers all Python scripts in the scripts directory.
    
    Example:
        cursus alignment list-scripts
    """
    try:
        # Set default directory if not provided
        project_root = Path.cwd()
        # Check if we're already in the src directory
        if project_root.name == "src":
            base_path = project_root / "cursus" / "steps"
        else:
            base_path = project_root / "src" / "cursus" / "steps"
        
        if not scripts_dir:
            scripts_dir = base_path / "scripts"
        
        click.echo("📋 Available Scripts for Alignment Validation:")
        click.echo("=" * 50)
        
        # Discover all scripts
        scripts = []
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                if not script_file.name.startswith('__'):
                    scripts.append(script_file.stem)
        
        if scripts:
            scripts = sorted(scripts)
            for script in scripts:
                click.echo(f"  • {script}")
            
            click.echo(f"\nTotal: {len(scripts)} scripts found")
            click.echo(f"\nUsage examples:")
            click.echo(f"  cursus alignment validate {scripts[0]} --verbose --show-scoring")
            click.echo(f"  cursus alignment validate-all --output-dir ./reports")
            click.echo(f"  cursus alignment validate-level {scripts[0]} 1")
            click.echo(f"  cursus alignment visualize {scripts[0]} --output-dir ./charts --verbose")
            click.echo(f"  cursus alignment visualize-all --output-dir ./visualizations")
        else:
            click.echo("  No scripts found in the scripts directory.")
            click.echo(f"  Searched in: {scripts_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error listing scripts: {e}", err=True)
        return 1


def main():
    """Main entry point for alignment CLI."""
    alignment()


if __name__ == '__main__':
    main()
