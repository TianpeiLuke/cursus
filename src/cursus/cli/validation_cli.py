#!/usr/bin/env python3
"""
Command-line interface for the naming standard validation tools.

This CLI provides easy access to validate naming conventions across the codebase
according to the standardization rules defined in the developer guide.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..validation.naming.naming_standard_validator import NamingStandardValidator, NamingViolation


def print_violations(violations: List[NamingViolation], verbose: bool = False) -> None:
    """Print violations in a formatted way."""
    if not violations:
        print("✅ No naming violations found!")
        return
    
    print(f"❌ Found {len(violations)} naming violations:")
    print()
    
    # Group violations by component
    violations_by_component = {}
    for violation in violations:
        component = violation.component
        if component not in violations_by_component:
            violations_by_component[component] = []
        violations_by_component[component].append(violation)
    
    for component, component_violations in violations_by_component.items():
        print(f"📁 {component}:")
        for violation in component_violations:
            print(f"  • {violation}")
            if verbose and violation.suggestions:
                print(f"    💡 Suggestions: {', '.join(violation.suggestions)}")
        print()


def validate_registry(verbose: bool = False) -> int:
    """Validate all registry entries."""
    print("🔍 Validating registry entries...")
    validator = NamingStandardValidator()
    violations = validator.validate_all_registry_entries()
    
    print_violations(violations, verbose)
    return len(violations)


def validate_file_name(filename: str, file_type: str, verbose: bool = False) -> int:
    """Validate a specific file name."""
    print(f"🔍 Validating file name: {filename} (type: {file_type})")
    validator = NamingStandardValidator()
    violations = validator.validate_file_naming(filename, file_type)
    
    print_violations(violations, verbose)
    return len(violations)


def validate_step_name(step_name: str, verbose: bool = False) -> int:
    """Validate a canonical step name."""
    print(f"🔍 Validating step name: {step_name}")
    validator = NamingStandardValidator()
    violations = validator._validate_canonical_step_name(step_name, "CLI")
    
    print_violations(violations, verbose)
    return len(violations)


def validate_logical_name(logical_name: str, verbose: bool = False) -> int:
    """Validate a logical name."""
    print(f"🔍 Validating logical name: {logical_name}")
    validator = NamingStandardValidator()
    violations = validator._validate_logical_name(logical_name, "CLI")
    
    print_violations(violations, verbose)
    return len(violations)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate naming conventions according to standardization rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all registry entries
  python -m cursus.cli.validation_cli registry
  
  # Validate a file name
  python -m cursus.cli.validation_cli file builder_xgboost_training_step.py builder
  
  # Validate a step name
  python -m cursus.cli.validation_cli step XGBoostTraining
  
  # Validate a logical name
  python -m cursus.cli.validation_cli logical input_data
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including suggestions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Registry validation
    registry_parser = subparsers.add_parser(
        "registry",
        help="Validate all registry entries"
    )
    
    # File name validation
    file_parser = subparsers.add_parser(
        "file",
        help="Validate a file name"
    )
    file_parser.add_argument("filename", help="File name to validate")
    file_parser.add_argument(
        "file_type",
        choices=["builder", "config", "spec", "contract"],
        help="Type of file"
    )
    
    # Step name validation
    step_parser = subparsers.add_parser(
        "step",
        help="Validate a canonical step name"
    )
    step_parser.add_argument("step_name", help="Step name to validate")
    
    # Logical name validation
    logical_parser = subparsers.add_parser(
        "logical",
        help="Validate a logical name"
    )
    logical_parser.add_argument("logical_name", help="Logical name to validate")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "registry":
            violation_count = validate_registry(args.verbose)
        elif args.command == "file":
            violation_count = validate_file_name(args.filename, args.file_type, args.verbose)
        elif args.command == "step":
            violation_count = validate_step_name(args.step_name, args.verbose)
        elif args.command == "logical":
            violation_count = validate_logical_name(args.logical_name, args.verbose)
        else:
            parser.print_help()
            return 1
        
        if violation_count > 0:
            print(f"\n⚠️  Found {violation_count} violation(s). Please fix them to comply with naming standards.")
            return 1
        else:
            print("\n✅ All naming conventions are compliant!")
            return 0
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
