#!/usr/bin/env python3
"""
Script Generator for Individual Alignment Validation Programs

This program automatically generates individual validation scripts for each script
found in src/cursus/steps/scripts.
"""

import sys
from pathlib import Path
from typing import List

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


def discover_scripts() -> List[str]:
    """Discover all Python scripts in the scripts directory."""
    scripts_path = Path(scripts_dir)

    if not scripts_path.exists():
        print(f"⚠️  Scripts directory not found: {scripts_path}")
        return []

    scripts = []
    for script_file in scripts_path.glob("*.py"):
        if script_file.name != "__init__.py":
            script_name = script_file.stem
            scripts.append(script_name)

    return sorted(scripts)


def generate_validation_script(script_name: str) -> str:
    """Generate a validation script for the given script name."""
    script_title = script_name.replace("_", " ").title()

    template = f'''#!/usr/bin/env python3
"""
Individual Alignment Validation Script for {script_name}

This script validates the alignment between script, contract, specification,
and builder configuration for the {script_name} script.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Define workspace directory structure
# workspace_dir points to src/cursus (the main workspace)
current_file = Path(__file__).resolve()
workspace_dir = current_file.parent.parent.parent.parent.parent / "src" / "cursus" / "steps" 

# Define component directories within the workspace
scripts_dir = str(workspace_dir / "scripts")
contracts_dir = str(workspace_dir / "contracts")
specs_dir = str(workspace_dir / "specs")
builders_dir = str(workspace_dir / "builders")
configs_dir = str(workspace_dir / "configs")

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

def main():
    """Run alignment validation for {script_name} script."""
    print("🔍 {script_title} Script Alignment Validation")
    print("=" * 60)
    
    # Initialize the tester
    tester = UnifiedAlignmentTester(
        scripts_dir=scripts_dir,
        contracts_dir=contracts_dir,
        specs_dir=specs_dir,
        builders_dir=builders_dir,
        configs_dir=configs_dir
    )
    
    # Run validation for {script_name} script
    script_name = "{script_name}"
    
    try:
        results = tester.validate_specific_script(script_name)
        
        # Print detailed results
        print(f"\\n📊 VALIDATION RESULTS FOR: {{script_name}}")
        print("=" * 60)
        
        status = results.get('overall_status', 'UNKNOWN')
        status_emoji = '✅' if status == 'PASSING' else '❌' if status == 'FAILING' else '⚠️'
        print(f"{{status_emoji}} Overall Status: {{status}}")
        
        for level_num, level_name in enumerate([
            "Script ↔ Contract",
            "Contract ↔ Specification", 
            "Specification ↔ Dependencies",
            "Builder ↔ Configuration"
        ], 1):
            level_key = f"level{{level_num}}"
            level_result = results.get(level_key, {{}})
            level_passed = level_result.get('passed', False)
            level_issues = level_result.get('issues', [])
            
            status_emoji = '✅' if level_passed else '❌'
            print(f"\\n{{status_emoji}} Level {{level_num}}: {{level_name}}")
            print(f"   Status: {{'PASS' if level_passed else 'FAIL'}}")
            print(f"   Issues: {{len(level_issues)}}")
            
            # Print issues with details
            for issue in level_issues:
                severity = issue.get('severity', 'ERROR')
                message = issue.get('message', 'No message')
                category = issue.get('category', 'unknown')
                recommendation = issue.get('recommendation', '')
                
                print(f"   • {{severity}} [{{category}}]: {{message}}")
                if recommendation:
                    print(f"     💡 Recommendation: {{recommendation}}")
        
        # Save reports
        output_dir = Path(__file__).parent / "reports" / "individual"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        import json
        from datetime import datetime
        
        # Add metadata
        results['metadata'] = {{
            'script_name': script_name,
            'validation_timestamp': datetime.now().isoformat(),
            'validator_version': '1.0.0',
            'script_path': str(workspace_dir / 'scripts'/ f"{{script_name}}.py")
        }}
        
        json_file = output_dir / f"{{script_name}}_validation_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📄 JSON Report saved: {{json_file}}")
        
        # Generate and save HTML report
        try:
            html_content = generate_html_report(script_name, results)
            html_file = output_dir / f"{{script_name}}_validation_report.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"🌐 HTML Report saved: {{html_file}}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate HTML report: {{e}}")
        
        print("=" * 60)
        
        # Return status for potential automation
        return 0 if status == 'PASSING' else 1
        
    except Exception as e:
        print(f"❌ ERROR during validation: {{e}}")
        import traceback
        traceback.print_exc()
        return 2

def generate_html_report(script_name: str, results: dict) -> str:
    """Generate HTML report for the validation results."""
    status = results.get('overall_status', 'UNKNOWN')
    status_class = 'passing' if status == 'PASSING' else 'failing'
    timestamp = results.get('metadata', {{}}).get('validation_timestamp', 'Unknown')
    
    # Count issues by severity
    total_issues = 0
    critical_issues = 0
    error_issues = 0
    warning_issues = 0
    
    for level_num in range(1, 5):
        level_key = f"level{{level_num}}"
        level_result = results.get(level_key, {{}})
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
        level_key = f"level{{level_num}}"
        level_result = results.get(level_key, {{}})
        level_passed = level_result.get('passed', False)
        level_issues = level_result.get('issues', [])
        
        result_class = "test-passed" if level_passed else "test-failed"
        status_text = "PASSED" if level_passed else "FAILED"
        
        issues_html = ""
        for issue in level_issues:
            severity = issue.get('severity', 'ERROR').lower()
            message = issue.get('message', 'No message')
            category = issue.get('category', 'unknown')
            recommendation = issue.get('recommendation', '')
            
            issues_html += f"""
            <div class="issue {{severity}}">
                <strong>{{issue.get('severity', 'ERROR')}} [{{category}}]:</strong> {{message}}
                {{f'<br><em>💡 Recommendation: {{recommendation}}</em>' if recommendation else ''}}
            </div>
            """
        
        level_sections += f"""
        <div class="test-result {{result_class}}">
            <h4>{{level_name}}</h4>
            <p>Status: {{status_text}}</p>
            <p>Issues Found: {{len(level_issues)}}</p>
            {{issues_html}}
        </div>
        """
    
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Alignment Validation Report - {script_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
        .metric h3 {{ margin: 0; font-size: 2em; }}
        .metric p {{ margin: 5px 0; color: #666; }}
        .passing {{ color: #28a745; }}
        .failing {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .level-section {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .level-header {{ background-color: #e9ecef; padding: 15px; font-weight: bold; font-size: 1.2em; }}
        .test-result {{ padding: 15px; border-bottom: 1px solid #eee; }}
        .test-passed {{ border-left: 4px solid #28a745; }}
        .test-failed {{ border-left: 4px solid #dc3545; }}
        .issue {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
        .critical {{ border-left: 4px solid #dc3545; background-color: #f8d7da; }}
        .error {{ border-left: 4px solid #fd7e14; background-color: #fff3cd; }}
        .warning {{ border-left: 4px solid #ffc107; background-color: #fff3cd; }}
        .info {{ border-left: 4px solid #17a2b8; background-color: #d1ecf1; }}
        .metadata {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
        .footer {{ margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px; text-align: center; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Alignment Validation Report</h1>
        <h2>Script: {script_name}</h2>
        <p><strong>Generated:</strong> {{timestamp}}</p>
        <p><strong>Overall Status:</strong> <span class="{{status_class}}">{{status}}</span></p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{{total_issues}}</h3>
            <p>Total Issues</p>
        </div>
        <div class="metric">
            <h3>{{critical_issues}}</h3>
            <p>Critical Issues</p>
        </div>
        <div class="metric">
            <h3>{{error_issues}}</h3>
            <p>Error Issues</p>
        </div>
        <div class="metric">
            <h3>{{warning_issues}}</h3>
            <p>Warning Issues</p>
        </div>
    </div>
    
    <div class="level-section">
        <div class="level-header">📋 Alignment Validation Results</div>
        {{level_sections}}
    </div>
    
    <div class="metadata">
        <h3>📄 Metadata</h3>
        <p><strong>Script Path:</strong> {{results.get('metadata', {{}}).get('script_path', 'Unknown')}}</p>
        <p><strong>Validation Timestamp:</strong> {{timestamp}}</p>
        <p><strong>Validator Version:</strong> {{results.get('metadata', {{}}).get('validator_version', 'Unknown')}}</p>
    </div>
    
    <div class="footer">
        <p>Generated by Cursus Alignment Validation System</p>
    </div>
</body>
</html>"""
    
    return html_template

if __name__ == "__main__":
    sys.exit(main())
'''

    return template


def main():
    """Generate validation scripts for all discovered scripts."""
    print("🚀 Generating Individual Validation Scripts")
    print("=" * 60)

    # Discover all scripts
    scripts = discover_scripts()
    print(f"📋 Discovered {len(scripts)} scripts: {', '.join(scripts)}")

    # Create output directory
    output_dir = Path(__file__).parent

    generated_count = 0

    # Generate validation script for each discovered script
    for script_name in scripts:
        try:
            script_content = generate_validation_script(script_name)
            script_file = output_dir / f"validate_{script_name}.py"

            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script_content)

            # Make the script executable
            script_file.chmod(0o755)

            print(f"✅ Generated: {script_file}")
            generated_count += 1

        except Exception as e:
            print(f"❌ Failed to generate script for {script_name}: {e}")

    print(
        f"\n🎯 Successfully generated {generated_count}/{len(scripts)} validation scripts"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
