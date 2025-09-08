"""
Simple report generator for step builder types.

This script generates basic reports and creates subfolder structures with 
canonical step builder names, following the tabular_preprocessing pattern.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

)

from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.registry.step_names import get_steps_by_sagemaker_type, STEP_NAMES

def get_canonical_step_name(step_name: str) -> str:
    """
    Get canonical step name from registry.
    
    Args:
        step_name: Registry step name (e.g., "PyTorchTraining")
        
    Returns:
        Canonical builder class name (e.g., "PyTorchTrainingStepBuilder")
    """
    if step_name in STEP_NAMES:
        return STEP_NAMES[step_name].get("builder_step_name", f"{step_name}StepBuilder")
    return f"{step_name}StepBuilder"

def load_builder_class(step_name: str):
    """
    Load a builder class by step name.
    
    Args:
        step_name: Registry step name
        
    Returns:
        Builder class if found, None otherwise
    """
    try:
        if step_name not in STEP_NAMES:
            return None
        
        builder_class_name = STEP_NAMES[step_name]["builder_step_name"]
        
        # Convert step name to module name (CamelCase to snake_case)
        import re
        module_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', step_name)
        module_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', module_name).lower()
        module_name = f"builder_{module_name}_step"
        
        # Import the module
        module_path = f"cursus.steps.builders.{module_name}"
        module = __import__(module_path, fromlist=[builder_class_name])
        
        # Get the class from the module
        builder_class = getattr(module, builder_class_name)
        
        return builder_class
        
    except Exception as e:
        print(f"Failed to load {step_name} builder: {e}")
        return None

def create_step_subfolder(base_path: Path, step_type: str, step_name: str) -> Path:
    """
    Create subfolder structure for a specific step.
    
    Args:
        base_path: Base directory path
        step_type: Type of step (training, transform, etc.)
        step_name: Registry step name
        
    Returns:
        Path to the scoring_reports directory
    """
    canonical_name = get_canonical_step_name(step_name)
    
    # Create directory structure: {step_type}/{canonical_name}/scoring_reports/
    step_dir = base_path / step_type.lower() / canonical_name
    scoring_dir = step_dir / "scoring_reports"
    
    # Create directories
    step_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(exist_ok=True)
    
    # Create README for the step directory
    readme_path = step_dir / "README.md"
    if not readme_path.exists():
        create_step_readme(readme_path, step_name, canonical_name, step_type)
    
    return scoring_dir

def create_step_readme(readme_path: Path, step_name: str, canonical_name: str, step_type: str):
    """Create README file for a step directory."""
    content = f"""# {canonical_name} Test Reports

This directory contains test reports and scoring charts for the {step_name} step builder.

## Directory Structure

- `scoring_reports/` - Contains detailed test scoring reports and charts
  - `{canonical_name}_score_report.json` - Detailed test results in JSON format

## Step Information

- **Registry Name**: {step_name}
- **Builder Class**: {canonical_name}
- **Step Type**: {step_type.title()}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

The scoring reports are generated automatically by running:

```bash
python test/steps/builders/generate_simple_reports.py
```

Or for specific step types:

```bash
python test/steps/builders/generate_simple_reports.py --step-type {step_type}
```
"""
    readme_path.write_text(content)

def run_step_tests(step_name: str, step_type: str) -> Dict[str, Any]:
    """
    Run comprehensive tests for a step builder.
    
    Args:
        step_name: Registry step name
        step_type: Type of step
        
    Returns:
        Dictionary containing test results and metadata
    """
    try:
        builder_class = load_builder_class(step_name)
        if not builder_class:
            return {
                "error": f"Could not load builder class for {step_name}",
                "step_name": step_name,
                "step_type": step_type,
                "timestamp": datetime.now().isoformat()
            }
        
        # Run universal tests with scoring enabled
        tester = UniversalStepBuilderTest(
            builder_class,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        results = tester.run_all_tests()
        
        # Add metadata
        results.update({
            "step_name": step_name,
            "step_type": step_type,
            "builder_class_name": builder_class.__name__,
            "timestamp": datetime.now().isoformat(),
            "generator_version": "1.0.0"
        })
        
        return results
        
    except Exception as e:
        return {
            "error": f"Error testing {step_name}: {str(e)}",
            "step_name": step_name,
            "step_type": step_type,
            "timestamp": datetime.now().isoformat()
        }

def generate_reports_for_step_type(base_path: Path, step_type: str) -> Dict[str, Any]:
    """
    Generate reports for all steps of a given type.
    
    Args:
        base_path: Base directory path
        step_type: Type of step (training, transform, createmodel, processing)
        
    Returns:
        Summary of generated reports
    """
    # Map step types to SageMaker types
    sagemaker_type_map = {
        "training": "Training",
        "transform": "Transform", 
        "createmodel": "CreateModel",
        "processing": "Processing"
    }
    
    if step_type not in sagemaker_type_map:
        raise ValueError(f"Unknown step type: {step_type}")
    
    sagemaker_type = sagemaker_type_map[step_type]
    
    print(f"\n{'='*60}")
    print(f"Generating {step_type.title()} Step Builder Reports")
    print(f"{'='*60}")
    
    # Get all steps for this type
    steps = get_steps_by_sagemaker_type(sagemaker_type)
    if not steps:
        print(f"No {step_type} steps found in registry")
        return {"step_type": step_type, "steps_processed": 0, "errors": []}
    
    print(f"Found {len(steps)} {step_type} steps: {', '.join(steps)}")
    
    summary = {
        "step_type": step_type,
        "steps_processed": 0,
        "successful": [],
        "errors": [],
        "timestamp": datetime.now().isoformat()
    }
    
    for step_name in steps:
        print(f"\nProcessing {step_name}...")
        
        try:
            # Create subfolder structure
            scoring_dir = create_step_subfolder(base_path, step_type, step_name)
            canonical_name = get_canonical_step_name(step_name)
            
            # Run tests
            results = run_step_tests(step_name, step_type)
            
            if "error" in results:
                print(f"❌ Error testing {step_name}: {results['error']}")
                summary["errors"].append({
                    "step_name": step_name,
                    "error": results["error"]
                })
                continue
            
            # Generate JSON report
            json_path = scoring_dir / f"{canonical_name}_score_report.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Generated JSON report: {json_path}")
            
            summary["successful"].append({
                "step_name": step_name,
                "canonical_name": canonical_name,
                "json_report": str(json_path)
            })
            summary["steps_processed"] += 1
            
        except Exception as e:
            error_msg = f"Error processing {step_name}: {str(e)}"
            print(f"❌ {error_msg}")
            summary["errors"].append({
                "step_name": step_name,
                "error": error_msg
            })
    
    return summary

def generate_all_reports(base_path: Path) -> Dict[str, Any]:
    """
    Generate reports for all step types.
    
    Args:
        base_path: Base directory path
        
    Returns:
        Overall summary of all generated reports
    """
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE STEP BUILDER REPORTS")
    print(f"{'='*80}")
    print(f"Base directory: {base_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    step_types = ["training", "transform", "createmodel", "processing"]
    
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "base_path": str(base_path),
        "step_types": {},
        "total_steps_processed": 0,
        "total_successful": 0,
        "total_errors": 0
    }
    
    for step_type in step_types:
        try:
            summary = generate_reports_for_step_type(base_path, step_type)
            overall_summary["step_types"][step_type] = summary
            overall_summary["total_steps_processed"] += summary["steps_processed"]
            overall_summary["total_successful"] += len(summary["successful"])
            overall_summary["total_errors"] += len(summary["errors"])
            
        except Exception as e:
            error_msg = f"Error processing {step_type} step type: {str(e)}"
            print(f"❌ {error_msg}")
            overall_summary["step_types"][step_type] = {
                "error": error_msg,
                "steps_processed": 0
            }
            overall_summary["total_errors"] += 1
    
    # Generate overall summary report
    reports_base = base_path / "reports"
    reports_base.mkdir(exist_ok=True)
    summary_path = reports_base / "overall_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("REPORT GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total steps processed: {overall_summary['total_steps_processed']}")
    print(f"Successful: {overall_summary['total_successful']}")
    print(f"Errors: {overall_summary['total_errors']}")
    print(f"Overall summary saved: {summary_path}")
    print(f"{'='*80}")
    
    return overall_summary

def main():
    """Main function to run report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate step builder test reports")
    parser.add_argument(
        "--step-type", 
        choices=["training", "transform", "createmodel", "processing", "all"],
        default="all",
        help="Step type to generate reports for (default: all)"
    )
    parser.add_argument(
        "--base-path",
        help="Base path for report generation (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Set base path
    if args.base_path:
        base_path = Path(args.base_path)
    else:
        base_path = Path(__file__).parent
    
    try:
        if args.step_type == "all":
            generate_all_reports(base_path)
        else:
            generate_reports_for_step_type(base_path, args.step_type)
            
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
