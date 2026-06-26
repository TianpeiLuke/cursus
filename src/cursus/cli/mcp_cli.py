"""
CLI commands for the cursus MCP (Model Context Protocol) tool server.

`cursus mcp serve` launches the stdio MCP server exposing the cursus tool surface;
`cursus mcp list-tools` prints the registered tools (no SDK required for listing).
Engine/SDK imports are lazy so `cursus --help` works even without the optional `mcp` SDK.

Examples:
    cursus mcp list-tools
    cursus mcp list-tools --format json
    cursus mcp serve
"""

import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group(name="mcp")
def mcp_cli():
    """Run and inspect the cursus MCP tool server."""


@mcp_cli.command(name="list-tools")
@click.option("--namespace", help="Filter to one tool namespace (e.g. catalog, dag).")
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def list_tools_cmd(namespace, format):
    """List the tools registered in the cursus MCP registry."""
    fmt = format.lower()
    try:
        from ..mcp import list_tools

        tools = list_tools(namespace=namespace)
    except Exception as e:
        click.echo(f"❌ Failed to list MCP tools: {e}", err=True)
        logger.error("MCP list-tools error", exc_info=True)
        raise SystemExit(1)

    if fmt == "json":
        click.echo(
            json.dumps(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "destructive": t.destructive,
                    }
                    for t in tools
                ],
                indent=2,
            )
        )
        return

    if not tools:
        click.echo(
            "No MCP tools registered"
            + (f" in namespace '{namespace}'." if namespace else ".")
        )
        return
    click.echo(f"{len(tools)} MCP tool(s):\n")
    for t in tools:
        flag = " [destructive]" if t.destructive else ""
        click.echo(f"  {t.name}{flag}")
        click.echo(f"      {t.description}")


@mcp_cli.command(name="serve")
def serve():
    """Run the cursus MCP server over stdio (requires the optional 'mcp' SDK)."""
    try:
        from ..mcp.server import main as server_main
    except Exception as e:
        click.echo(f"❌ Could not load the MCP server: {e}", err=True)
        logger.error("MCP serve import error", exc_info=True)
        raise SystemExit(1)

    try:
        raise SystemExit(server_main())
    except RuntimeError as e:
        # Raised when the optional mcp SDK is not installed.
        click.echo(f"❌ {e}", err=True)
        raise SystemExit(1)


def main():
    """Main entry point for the mcp CLI."""
    return mcp_cli()


if __name__ == "__main__":
    main()
