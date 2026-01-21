"""Command-line interfaces for the Cursus package."""

import sys
import argparse

# Import all CLI modules
from .alignment_cli import main as alignment_main
from .catalog_cli import main as catalog_main
from .compile_cli import main as compile_main
from .exec_doc_cli import main as exec_doc_main
from .registry_cli import main as registry_main

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
  catalog         - Step catalog management
  compile         - Compile DAG and config to SageMaker pipeline
  exec-doc        - Generate execution documents from DAG and config
  registry        - Registry management tools

Examples:

  # Pipeline Compilation - Compile DAG + config to SageMaker pipeline
  python -m cursus.cli compile -d dag.json -c config.json
  python -m cursus.cli compile -d dag.json -c config.json --upsert
  python -m cursus.cli compile -d dag.json -c config.json --upsert --start
  python -m cursus.cli compile -d dag.json -c config.json -o pipeline_def.json
  python -m cursus.cli compile -d dag.json -c config.json --validate-only

  # Execution Document Generation - Generate execution docs from DAG and config
  python -m cursus.cli exec-doc generate -d dag.json -c config.json
  python -m cursus.cli exec-doc generate -d dag.json -c config.json -o my_exec_doc.json
  python -m cursus.cli exec-doc generate -d dag.json -c config.json --template base_template.json
  python -m cursus.cli exec-doc generate -d dag.json -c config.json --format yaml
  python -m cursus.cli exec-doc generate -d dag.json -c config.json --role arn:aws:iam::123:role/MyRole

  # Step Catalog - Discover and manage steps
  python -m cursus.cli catalog list --framework xgboost --limit 10
  python -m cursus.cli catalog search "training" --job-type validation
  python -m cursus.cli catalog show XGBoostTraining --show-components
  python -m cursus.cli catalog frameworks --format json
  python -m cursus.cli catalog discover --workspace-dir /path/to/workspace

  # Registry Management - Workspace and step validation
  python -m cursus.cli registry init-workspace my_developer --template advanced
  python -m cursus.cli registry list-steps --workspace my_developer --conflicts-only
  python -m cursus.cli registry validate-registry --check-conflicts
  python -m cursus.cli registry resolve-step XGBoostTraining --workspace my_developer
  python -m cursus.cli registry validate-step-definition --name MyStep --auto-correct

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
            "catalog",
            "compile",
            "exec-doc",
            "registry",
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
        elif args.command == "catalog":
            return catalog_main()
        elif args.command == "compile":
            return compile_main()
        elif args.command == "exec-doc":
            return exec_doc_main()
        elif args.command == "registry":
            return registry_main()
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
