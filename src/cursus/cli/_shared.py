"""
Shared helpers for the cursus CLI command modules.

Centralizes patterns that were copy-pasted across the CLI files: constructing a
``StepCatalog`` (was instantiated ~16 times in catalog_cli), emitting JSON output, and
the standard error-handling wrapper. Keeping them here is the single source of truth so
every command behaves consistently and new commands are consistent from day one.

Engine imports are kept lazy (inside functions) so importing the CLI surface stays cheap
and does not fail when an optional engine dependency is missing.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache, wraps
from typing import Any, Callable

import click

logger = logging.getLogger("cursus.cli")

# Shared display constants (previously hardcoded/duplicated per command).
SEP_WIDTH = 50


@lru_cache(maxsize=1)
def get_catalog():
    """
    Return a process-wide, package-scoped ``StepCatalog`` singleton.

    catalog_cli previously built a fresh ``StepCatalog()`` per command (~16 sites), each
    rebuilding the discovery index. This memoizes the no-arg (package-scope) catalog. Note
    the cache is keyed on no arguments — commands that need a *workspace-scoped* catalog
    (``StepCatalog(workspace_dirs=...)``) must construct their own instance, since those
    are not interchangeable with the package-only singleton.
    """
    from ..step_catalog import StepCatalog

    return StepCatalog()


def echo_json(data: Any) -> None:
    """Emit ``data`` as pretty JSON (default=str so non-serializable values degrade)."""
    click.echo(json.dumps(data, indent=2, default=str))


def echo_header(title: str, emoji: str = "") -> None:
    """Print a consistent section header followed by a separator rule."""
    prefix = f"{emoji} " if emoji else ""
    click.echo(f"\n{prefix}{title}")
    click.echo("=" * SEP_WIDTH)


def safe_cli_command(error_label: str) -> Callable:
    """
    Decorator wrapping a command body in the standard CLI error handler.

    On an unhandled exception it echoes a uniform ``❌`` message to stderr, logs the full
    traceback via ``logger.error(..., exc_info=True)``, and exits nonzero (``SystemExit``)
    so the failure propagates as a real exit code. ``SystemExit`` / ``click`` control-flow
    exceptions are re-raised untouched so intentional exits and ``--help`` still work.

    Usage::

        @cli.command()
        @safe_cli_command("list projects")
        def list_projects(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (SystemExit, click.exceptions.Abort, click.exceptions.Exit):
                raise
            except Exception as exc:  # noqa: BLE001 - top-level CLI guard
                click.echo(f"❌ Failed to {error_label}: {exc}", err=True)
                logger.error("Error during '%s'", error_label, exc_info=True)
                raise SystemExit(1)

        return wrapper

    return decorator
