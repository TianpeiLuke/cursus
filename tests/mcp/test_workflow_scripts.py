"""
Structural + parse gate for the shipped dynamic-workflow scripts in ``src/cursus/mcp/workflows``.

These ``.js`` files are reference/runnable orchestration artifacts (not importable Python), so the
Python test suite cannot exercise their runtime. But we CAN guard the two failure modes that would
ship a broken workflow:

  1. Syntax: the script must parse (``node --check``) — a stray apostrophe in a single-quoted string
     or an unbalanced paren has bitten these before.
  2. Structure: every script must begin with the required ``export const meta`` literal (name +
     description + phases), and every phase the body references (``phase('X')`` or ``phase: 'X'``)
     must be declared in ``meta.phases`` — the workflow runtime groups agents by that title, so a
     typo'd/undeclared phase silently drops an agent into its own orphan group.

Skips cleanly when Node.js is not on PATH (the parse gate needs the JS engine); the structural
checks that do not need Node still run via a light regex extraction.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKFLOW_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "cursus" / "mcp" / "workflows"
)
WORKFLOW_FILES = sorted(WORKFLOW_DIR.glob("*.js"))
NODE = shutil.which("node")


def _ids(paths):
    return [p.name for p in paths]


def test_workflow_dir_has_scripts():
    # Guard against a rename/move silently emptying the shipped workflow set.
    names = {p.name for p in WORKFLOW_FILES}
    assert "cursus-author-step.js" in names
    assert "cursus-configure-pipeline.js" in names


@pytest.mark.skipif(NODE is None, reason="node not on PATH — cannot run the JS parse gate")
@pytest.mark.parametrize("wf", WORKFLOW_FILES, ids=_ids(WORKFLOW_FILES))
def test_workflow_parses(wf):
    """`node --check` must succeed — the script is valid JavaScript."""
    result = subprocess.run(
        [NODE, "--check", str(wf)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{wf.name} failed node --check:\n{result.stderr}"


def _extract_meta(wf: Path) -> dict:
    """Extract the `export const meta = {...}` literal.

    Prefers a real JS eval via Node (exact); falls back to a JSON-ish parse when Node is absent.
    The body of the script runs `await agent(...)` at top level so it cannot simply be imported;
    only the leading `meta` literal is read.
    """
    src = wf.read_text()
    m = re.search(r"export const meta = (\{.*?\n\})", src, re.DOTALL)
    assert m, f"{wf.name} has no `export const meta = {{...}}` literal"
    literal = m.group(1)

    if NODE is not None:
        proc = subprocess.run(
            [NODE, "-e", f"process.stdout.write(JSON.stringify({literal}))"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"{wf.name} meta literal did not eval:\n{proc.stderr}"
        return json.loads(proc.stdout)

    # Node-absent fallback: not a full JS parser, but enough to read the string fields we assert on.
    meta = {}
    name = re.search(r"name:\s*'([^']+)'", literal)
    desc = re.search(r"description:\s*'((?:[^'\\]|\\.)*)'", literal)
    if name:
        meta["name"] = name.group(1)
    if desc:
        meta["description"] = desc.group(1)
    meta["phases"] = [
        {"title": t} for t in re.findall(r"title:\s*'([^']+)'", literal)
    ]
    return meta


@pytest.mark.parametrize("wf", WORKFLOW_FILES, ids=_ids(WORKFLOW_FILES))
def test_meta_has_required_fields(wf):
    meta = _extract_meta(wf)
    assert isinstance(meta.get("name"), str) and meta["name"], f"{wf.name}: meta.name missing"
    assert isinstance(meta.get("description"), str) and meta["description"], (
        f"{wf.name}: meta.description missing"
    )
    phases = meta.get("phases")
    assert isinstance(phases, list) and phases, f"{wf.name}: meta.phases missing/empty"
    for p in phases:
        assert isinstance(p.get("title"), str) and p["title"], (
            f"{wf.name}: a phase entry has no title"
        )


@pytest.mark.parametrize("wf", WORKFLOW_FILES, ids=_ids(WORKFLOW_FILES))
def test_meta_name_matches_filename(wf):
    """The runtime resolves a saved workflow by meta.name; keep it equal to the file stem."""
    meta = _extract_meta(wf)
    assert meta["name"] == wf.stem, (
        f"{wf.name}: meta.name '{meta['name']}' != file stem '{wf.stem}'"
    )


@pytest.mark.parametrize("wf", WORKFLOW_FILES, ids=_ids(WORKFLOW_FILES))
def test_every_referenced_phase_is_declared(wf):
    """Each phase the body references must be a declared meta.phases title.

    The runtime groups agents under the meta phase title; a phase() call (or a `phase:` agent opt)
    naming an undeclared phase drops that agent into an orphan progress group — a silent drift bug.
    Both reference forms are used across the two scripts: `phase('X')` and `phase: 'X'`.
    """
    meta = _extract_meta(wf)
    declared = {p["title"] for p in meta["phases"]}

    src = wf.read_text()
    referenced = set(re.findall(r"phase\('([^']+)'\)", src))
    referenced |= set(re.findall(r"phase:\s*'([^']+)'", src))
    assert referenced, f"{wf.name}: no phase references found (expected phase()/phase: usage)"

    undeclared = referenced - declared
    assert not undeclared, (
        f"{wf.name}: phases referenced but NOT declared in meta.phases: {sorted(undeclared)}"
    )
