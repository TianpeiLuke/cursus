"""
Discover and summarize Cursus pipeline *projects* under a root directory.

A "pipeline project" here is a deployable Cursus pipeline directory — e.g. the
``projects/*`` folders under the AmazonCursus root (``atoz_xgboost``,
``rnr_pytorch_bedrock``, ...) or the per-pipeline folders under a consumer repo such
as BuyerAbuseModsTemplate. Such a project is recognized by a configuration directory
(``pipeline_config`` or ``pipeline_configs``) holding one or more config JSON files,
typically alongside a ``dockers``/``scripts`` directory and a pipeline-definition
``.py`` module.

This is intentionally a *read-only inspection* utility built on the two existing
primitives — :mod:`cursus.core.utils` path discovery (to locate a named project) and
the config JSON each project already ships — so it answers "what pipeline projects
exist under here, and what does each contain?" without the multi-developer-workspace
authoring machinery. It does not import or require :mod:`cursus.workspace`.

Public API:
    discover_pipeline_projects(root=None, names=None) -> List[ProjectInfo]
    summarize_project(project_dir) -> Optional[ProjectInfo]

Each project's config JSON is read for its ``metadata.config_types`` (node -> config
class) and ``configuration.specific`` (per-node settings), which is how a Cursus config
file records the pipeline's nodes. No SageMaker/engine objects are constructed, so this
is cheap and safe to run anywhere.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .generic_path_discovery import find_project_folder_generic

logger = logging.getLogger(__name__)

# Directory names a Cursus project uses to hold its config JSON files.
_CONFIG_DIR_NAMES = ("pipeline_config", "pipeline_configs")
# Sibling markers that increase confidence a directory is a real pipeline project.
_PROJECT_MARKERS = ("dockers", "scripts")


@dataclass
class ConfigSummary:
    """Summary of one config JSON file inside a project."""

    file: str
    node_count: int = 0
    config_types: Dict[str, str] = field(default_factory=dict)
    nodes: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        out: Dict = {"file": self.file}
        if self.error:
            out["error"] = self.error
            return out
        out.update(
            {
                "node_count": self.node_count,
                "nodes": self.nodes,
                "config_types": self.config_types,
            }
        )
        if self.created_at:
            out["created_at"] = self.created_at
        return out


@dataclass
class ProjectInfo:
    """Summary of one discovered pipeline project."""

    name: str
    path: str
    config_dir: Optional[str] = None
    config_files: List[ConfigSummary] = field(default_factory=list)
    has_dockers: bool = False
    has_scripts: bool = False
    pipeline_modules: List[str] = field(default_factory=list)

    @property
    def config_file_count(self) -> int:
        return len(self.config_files)

    @property
    def distinct_config_types(self) -> List[str]:
        """Union of config class names referenced across all of the project's configs."""
        types = set()
        for c in self.config_files:
            types.update(c.config_types.values())
        return sorted(types)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "path": self.path,
            "config_dir": self.config_dir,
            "config_file_count": self.config_file_count,
            "config_files": [c.to_dict() for c in self.config_files],
            "distinct_config_types": self.distinct_config_types,
            "has_dockers": self.has_dockers,
            "has_scripts": self.has_scripts,
            "pipeline_modules": self.pipeline_modules,
        }


def _find_config_dir(project_dir: Path) -> Optional[Path]:
    for name in _CONFIG_DIR_NAMES:
        candidate = project_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _summarize_config_file(path: Path) -> ConfigSummary:
    """Read a Cursus config JSON and extract its node/config-type summary."""
    summary = ConfigSummary(file=path.name)
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        summary.error = f"could not parse: {exc}"
        return summary

    # Cursus config layout: {"configuration": {"shared": {...}, "specific": {node: {...}}},
    #                        "metadata": {"config_types": {node: ClassName}, "created_at": ...}}
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    config_types = metadata.get("config_types", {}) or {}
    if isinstance(config_types, dict):
        summary.config_types = {str(k): str(v) for k, v in config_types.items()}

    configuration = data.get("configuration", {}) if isinstance(data, dict) else {}
    specific = (
        configuration.get("specific", {}) if isinstance(configuration, dict) else {}
    )
    # Node names come from 'specific'; fall back to config_types keys for older layouts.
    nodes = list(specific.keys()) if isinstance(specific, dict) else []
    if not nodes:
        nodes = list(summary.config_types.keys())
    summary.nodes = sorted(nodes)
    summary.node_count = len(summary.nodes)
    summary.created_at = metadata.get("created_at")
    return summary


def _is_project_dir(project_dir: Path) -> bool:
    """A directory is a pipeline project if it has a recognizable config dir."""
    return _find_config_dir(project_dir) is not None


def summarize_project(project_dir: Union[str, Path]) -> Optional[ProjectInfo]:
    """
    Summarize a single pipeline-project directory.

    Returns a :class:`ProjectInfo`, or ``None`` if ``project_dir`` is not a recognizable
    Cursus pipeline project (no ``pipeline_config``/``pipeline_configs`` directory).
    """
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        return None
    config_dir = _find_config_dir(project_dir)
    if config_dir is None:
        return None

    info = ProjectInfo(
        name=project_dir.name,
        path=str(project_dir),
        config_dir=str(config_dir),
        has_dockers=(project_dir / "dockers").is_dir(),
        has_scripts=(project_dir / "scripts").is_dir(),
    )

    # Config JSON files (skip generated execution documents).
    for cfg_path in sorted(config_dir.glob("*.json")):
        if cfg_path.name.startswith("execute_doc") or cfg_path.name.startswith(
            "execution_doc"
        ):
            continue
        info.config_files.append(_summarize_config_file(cfg_path))

    # Pipeline-definition modules (top-level .py with a DAG/pipeline definition).
    info.pipeline_modules = sorted(
        p.name for p in project_dir.glob("*.py") if p.name != "__init__.py"
    )
    return info


def discover_pipeline_projects(
    root: Optional[Union[str, Path]] = None,
    names: Optional[List[str]] = None,
) -> List[ProjectInfo]:
    """
    Discover Cursus pipeline projects under ``root`` (or locate specific ones by name).

    Args:
        root: Directory to scan for project subdirectories. When ``None`` and ``names``
            is given, each name is located via
            :func:`cursus.core.utils.find_project_folder_generic` (cross-deployment
            search). When both are ``None``, returns an empty list (nothing to scan).
        names: Optional explicit list of project folder names. If given, only these are
            returned (located under ``root`` if provided, else via generic discovery).

    Returns:
        A list of :class:`ProjectInfo`, one per recognized pipeline project, sorted by
        name. Directories that are not pipeline projects (no config dir) are skipped.
    """
    projects: List[ProjectInfo] = []

    if names:
        for name in names:
            project_dir: Optional[Path] = None
            if root is not None:
                candidate = Path(root).expanduser().resolve() / name
                if candidate.is_dir():
                    project_dir = candidate
            if project_dir is None:
                # Cross-deployment search for a uniquely named project folder.
                project_dir = find_project_folder_generic(name)
            if project_dir is None:
                logger.warning("Project '%s' not found", name)
                continue
            info = summarize_project(project_dir)
            if info is None:
                logger.warning(
                    "Directory for '%s' (%s) is not a recognizable pipeline project",
                    name,
                    project_dir,
                )
                continue
            projects.append(info)
        return sorted(projects, key=lambda p: p.name)

    if root is None:
        logger.warning(
            "discover_pipeline_projects called with no root and no names; nothing to scan"
        )
        return []

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        logger.warning("Root '%s' is not a directory", root_path)
        return []

    for child in sorted(root_path.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "__")):
            continue
        if not _is_project_dir(child):
            continue
        info = summarize_project(child)
        if info is not None:
            projects.append(info)

    return sorted(projects, key=lambda p: p.name)
