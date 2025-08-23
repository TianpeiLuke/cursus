"""Command-line interfaces for the Cursus package."""

from .runtime_cli import runtime, main as runtime_main

__all__ = [
    "runtime",
    "runtime_main"
]
