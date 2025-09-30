"""Command-line interfaces for the Cursus package."""

import sys
import argparse

# Import all CLI modules
from .alignment_cli import main as alignment_main
from .builder_test_cli import main as builder_test_main
from .catalog_cli import main as catalog_main
from .pipeline_cli import main as pipeline_main
from .registry_cli import main as registry_main
from .runtime_testing_cli import main as runtime_testing_main
from .workspace_cli import main as workspace_main

__all__ = ["main"]


def main():
    """Main CLI entry point - dispatcher for all Cursus CLI tools."""
    parser = argparse.ArgumentParser(
        prog="cursus.cli",
        description="Cursus CLI - Pipeline development and validation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  alignment       - Alignment validation tools
  builder-test    - Step builder testing tools  
  catalog         - Step catalog management
  pipeline        - Pipeline catalog management
  registry        - Registry management tools
  runtime-testing - Runtime testing for pipeline scripts
  workspace       - Workspace management tools

Examples:

  # Runtime Testing - Test pipeline scripts
  python -m cursus.cli runtime-testing test_script my_script.py
  python -m cursus.cli runtime-testing validate_contracts --step XGBoostTraining
  python -m cursus.cli runtime-testing run_integration_tests --workspace my_workspace

  # Step Catalog - Discover and manage steps
  python -m cursus.cli catalog list --framework xgboost --limit 10
  python -m cursus.cli catalog search "training" --job-type validation
  python -m cursus.cli catalog show XGBoostTraining --show-components
  python -m cursus.cli catalog frameworks --format json
  python -m cursus.cli catalog discover --workspace-dir /path/to/workspace

  # Pipeline Catalog - Manage pipeline templates
  python -m cursus.cli pipeline discover --framework pytorch --use-case "model training"
  python -m cursus.cli pipeline list --complexity advanced --format json
  python -m cursus.cli pipeline recommend --use-case "end-to-end ML pipeline"
  python -m cursus.cli pipeline show xgb_e2e_comprehensive --show-connections
  python -m cursus.cli pipeline validate --format json

  # Registry Management - Workspace and step validation
  python -m cursus.cli registry init-workspace my_developer --template advanced
  python -m cursus.cli registry list-steps --workspace my_developer --conflicts-only
  python -m cursus.cli registry validate-registry --check-conflicts
  python -m cursus.cli registry resolve-step XGBoostTraining --workspace my_developer
  python -m cursus.cli registry validate-step-definition --name MyStep --auto-correct

  # Workspace Management - Project setup and configuration
  python -m cursus.cli workspace setup --project my_project --template standard
  python -m cursus.cli workspace validate --project my_project --check-dependencies

  # Builder Testing - Test step builders and configurations
  python -m cursus.cli builder-test validate --step XGBoostTraining --config-file config.json
  python -m cursus.cli builder-test run-tests --workspace my_workspace --framework pytorch

  # Alignment Validation - Ensure component consistency
  python -m cursus.cli alignment validate --step XGBoostTraining --check-all-components
  python -m cursus.cli alignment report --workspace my_workspace --format json

For help with a specific command:
  python -m cursus.cli <command> --help

For detailed command options:
  python -m cursus.cli catalog --help
  python -m cursus.cli pipeline --help
  python -m cursus.cli registry --help
        """,
    )

    parser.add_argument(
        "command",
        choices=[
            "alignment",
            "builder-test",
            "catalog",
            "pipeline",
            "registry",
            "runtime-testing",
            "workspace",
        ],
        help="CLI command to run",
    )

    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the selected command",
    )

    # Parse only the first argument to get the command
    if len(sys.argv) < 2:
        parser.print_help()
        return 1

    args = parser.parse_args()

    # Modify sys.argv to pass remaining arguments to the selected CLI
    original_argv = sys.argv[:]
    sys.argv = [f"cursus.cli.{args.command}"] + args.args

    try:
        # Route to appropriate CLI module
        if args.command == "alignment":
            return alignment_main()
        elif args.command == "builder-test":
            return builder_test_main()
        elif args.command == "catalog":
            return catalog_main()
        elif args.command == "pipeline":
            return pipeline_main()
        elif args.command == "registry":
            return registry_main()
        elif args.command == "runtime-testing":
            return runtime_testing_main()
        elif args.command == "workspace":
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
