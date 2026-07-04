"""
CLI commands for the cursus MCP (Model Context Protocol) tool server.

`cursus mcp serve` launches the stdio MCP server exposing the cursus tool surface;
`cursus mcp list-tools` prints the registered tools (no SDK required for listing);
`cursus mcp help` prints the guided overview (namespaces + phases + tools) that the
`tools.help` agent tool returns. Engine/SDK imports are lazy so `cursus --help` works
even without the optional `mcp` SDK.

Examples:
    cursus mcp help
    cursus mcp help --namespace compile
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


@mcp_cli.command(name="help")
@click.option(
    "--namespace", help="Restrict to one tool namespace (e.g. catalog, compile)."
)
@click.option(
    "--phase",
    type=click.Choice(["planner", "validator", "programmer"], case_sensitive=False),
    help="Restrict to one lifecycle phase.",
)
@click.option(
    "--schema",
    "include_schema",
    is_flag=True,
    help="Include each tool's JSON input schema.",
)
@click.option(
    "--examples",
    "show_examples",
    is_flag=True,
    help="Show each tool's usage examples (auto-on when --namespace is given).",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format.",
)
def help_cmd(namespace, phase, include_schema, show_examples, format):
    """Print the guided overview of the cursus MCP toolset (the `tools.help` result)."""
    fmt = format.lower()
    try:
        from ..mcp import call_tool

        args = {"include_schema": include_schema}
        if namespace:
            args["namespace"] = namespace
        if phase:
            args["phase"] = phase.lower()
        result = call_tool("tools.help", args)
    except Exception as e:
        click.echo(f"❌ Failed to build MCP help: {e}", err=True)
        logger.error("MCP help error", exc_info=True)
        raise SystemExit(1)

    if not result.ok:
        click.echo(f"❌ {result.error}", err=True)
        raise SystemExit(1)

    data = result.data
    if fmt == "json":
        click.echo(json.dumps(data, indent=2))
        return

    # Show examples when explicitly asked, or automatically when zoomed into one
    # namespace (the global 57-tool listing stays scannable by default).
    render_examples = show_examples or bool(namespace)

    click.echo(data["overview"])
    click.echo("")
    click.echo(f"{data['shown']} of {data['total_tools']} tool(s) shown.")
    click.echo("")
    click.echo("Phases:")
    for name, info in data["phases"].items():
        click.echo(f"  {name:<10} ({info['count']:>2})  {info['description']}")
    click.echo("")
    for ns in data["namespaces"]:
        click.echo(f"{ns['namespace']} ({ns['count']}) — {ns['description']}")
        for t in ns["tools"]:
            flag = " [destructive]" if t["destructive"] else ""
            tags = f"  [{', '.join(t['tags'])}]" if t["tags"] else ""
            click.echo(f"  {t['name']}{flag}{tags}")
            click.echo(f"      {t['description']}")
            if t.get("when"):
                click.echo(f"      When: {t['when']}")
            if render_examples and t.get("examples"):
                click.echo("      Examples:")
                for ex in t["examples"]:
                    click.echo(f"        {ex}")
        click.echo("")


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
