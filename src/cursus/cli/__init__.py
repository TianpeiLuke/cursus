"""Command-line interfaces for the Cursus package."""

from .runtime_testing_cli import cli as runtime_testing_cli

__all__ = [
    "runtime_testing_cli",
    "main"
]

def main():
    """Main CLI entry point - simplified runtime testing."""
    runtime_testing_cli()
