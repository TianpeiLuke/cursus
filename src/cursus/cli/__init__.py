"""Command-line interfaces for the Cursus package."""

import sys
import argparse

# Import all CLI modules
from .alignment_cli import main as alignment_main
from .builder_test_cli import main as builder_test_main
from .catalog_cli import main as catalog_main
from .registry_cli import main as registry_main
from .runtime_testing_cli import main as runtime_testing_main
from .validation_cli import main as validation_main
from .workspace_cli import main as workspace_main

__all__ = [
    "main"
]

def main():
    """Main CLI entry point - dispatcher for all Cursus CLI tools."""
    parser = argparse.ArgumentParser(
        prog='cursus.cli',
        description="Cursus CLI - Pipeline development and validation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  alignment       - Alignment validation tools
  builder-test    - Step builder testing tools  
  catalog         - Pipeline catalog management
  registry        - Registry management tools
  runtime-testing - Runtime testing for pipeline scripts
  validation      - Naming and interface validation
  workspace       - Workspace management tools

Examples:
  python -m cursus.cli runtime-testing test_script my_script.py
  python -m cursus.cli validation registry
  python -m cursus.cli catalog find --tags training,xgboost
  python -m cursus.cli workspace setup --project my_project

For help with a specific command:
  python -m cursus.cli <command> --help
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'alignment', 'builder-test', 'catalog', 
            'registry', 'runtime-testing', 'validation', 'workspace'
        ],
        help='CLI command to run'
    )
    
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the selected command'
    )
    
    # Parse only the first argument to get the command
    if len(sys.argv) < 2:
        parser.print_help()
        return 1
    
    args = parser.parse_args()
    
    # Modify sys.argv to pass remaining arguments to the selected CLI
    original_argv = sys.argv[:]
    sys.argv = [f'cursus.cli.{args.command}'] + args.args
    
    try:
        # Route to appropriate CLI module
        if args.command == 'alignment':
            return alignment_main()
        elif args.command == 'builder-test':
            return builder_test_main()
        elif args.command == 'catalog':
            return catalog_main()
        elif args.command == 'registry':
            return registry_main()
        elif args.command == 'runtime-testing':
            return runtime_testing_main()
        elif args.command == 'validation':
            return validation_main()
        elif args.command == 'workspace':
            return workspace_main()
        else:
            parser.print_help()
            return 1
    except SystemExit as e:
        # Preserve exit codes from sub-commands
        return e.code
    except Exception as e:
        print(f"Error running {args.command}: {e}")
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
