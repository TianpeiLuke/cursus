"""
DAG Auto-Discovery Module

AST-based discovery system for pipeline DAG definitions following step_catalog patterns.
Supports workspace-aware priority system where local DAGs override package DAGs.

Features:
- AST parsing (no imports, no circular dependencies)
- Workspace prioritization (local overrides package)
- Registry integration (validation + enrichment)
- Function caching (performance)
- Convention enforcement (create_*_dag + get_dag_metadata)
"""

import ast
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from ...api.dag.base_dag import PipelineDAG
from ..shared_dags import DAGMetadata
from .catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)


@dataclass
class DAGInfo:
    """
    Rich metadata about a discovered DAG.

    Attributes:
        dag_id: Unique identifier for the DAG (derived from function name)
        dag_name: Full function name (e.g., create_xgboost_complete_e2e_dag)
        dag_path: File path where DAG is defined
        workspace_id: Identifier of workspace ("package" for package DAGs)
        framework: Framework used (xgboost, pytorch, etc.)
        complexity: Complexity level (simple, standard, comprehensive)
        features: List of features (training, evaluation, etc.)
        node_count: Number of nodes in the DAG
        edge_count: Number of edges in the DAG
        create_function: Lazy-loaded DAG creation function
        metadata_function: Lazy-loaded metadata function
        metadata: Complete DAG metadata from get_dag_metadata()
    """

    dag_id: str
    dag_name: str
    dag_path: Path
    workspace_id: str
    framework: str = "unknown"
    complexity: str = "standard"
    features: List[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    create_function: Optional[Callable[[], PipelineDAG]] = None
    metadata_function: Optional[Callable[[], DAGMetadata]] = None
    metadata: Optional[DAGMetadata] = None

    def load_functions(self) -> bool:
        """
        Dynamically load create and metadata functions.

        Returns:
            bool: True if functions loaded successfully
        """
        if self.create_function is not None and self.metadata_function is not None:
            return True  # Already loaded

        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(
                f"dag_{self.dag_id}", self.dag_path
            )
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create module spec for {self.dag_path}")
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the functions
            self.create_function = getattr(module, self.dag_name, None)
            self.metadata_function = getattr(module, "get_dag_metadata", None)

            if self.create_function is None:
                logger.error(f"Function {self.dag_name} not found in {self.dag_path}")
                return False

            if self.metadata_function is None:
                logger.warning(f"get_dag_metadata not found in {self.dag_path}")

            # Load metadata if function exists
            if self.metadata_function:
                self.metadata = self.metadata_function()
                # Update fields from metadata
                self.framework = self.metadata.framework
                self.complexity = self.metadata.complexity
                self.features = self.metadata.features
                self.node_count = self.metadata.node_count
                self.edge_count = self.metadata.edge_count

            return True

        except Exception as e:
            logger.error(f"Failed to load functions from {self.dag_path}: {e}")
            return False

    def create_dag(self) -> Optional[PipelineDAG]:
        """
        Create the pipeline DAG.

        Returns:
            PipelineDAG or None if creation fails
        """
        if self.create_function is None:
            if not self.load_functions():
                return None

        try:
            return self.create_function()
        except Exception as e:
            logger.error(f"Failed to create DAG {self.dag_id}: {e}")
            return None


class DAGAutoDiscovery:
    """
    AST-based DAG discovery with workspace support.

    Discovery Strategy:
    1. Scan workspace_dirs/dags/ (highest priority)
    2. Scan package_root/pipeline_catalog/shared_dags/
    3. Parse files using AST (no imports)
    4. Extract create_*_dag() functions
    5. Extract get_dag_metadata() functions
    6. Cross-reference with CatalogRegistry
    7. Cache results for performance

    Naming Convention:
    - DAG files must end with *_dag.py
    - Must contain create_*_dag() function
    - Should contain get_dag_metadata() function
    - DAG ID extracted from function name
    """

    def __init__(
        self,
        package_root: Optional[Path] = None,
        workspace_dirs: Optional[List[Path]] = None,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize DAG discovery.

        Args:
            package_root: Root of package (defaults to pipeline_catalog parent)
            workspace_dirs: List of workspace directories to scan
            registry_path: Path to catalog registry JSON
        """
        # Set package root
        if package_root is None:
            # Default to pipeline_catalog parent directory
            package_root = Path(__file__).parent.parent.parent.parent
        self.package_root = Path(package_root)

        # Set workspace directories
        self.workspace_dirs = [Path(d) for d in (workspace_dirs or [])]

        # Initialize registry
        try:
            self.registry = CatalogRegistry(registry_path=registry_path)
            logger.info("Initialized catalog registry for DAG discovery")
        except Exception as e:
            logger.warning(f"Failed to initialize registry: {e}")
            self.registry = None

        # Discovery cache
        self._dag_cache: Dict[str, DAGInfo] = {}
        self._discovery_complete = False

    def discover_all_dags(self, force_refresh: bool = False) -> Dict[str, DAGInfo]:
        """
        Discover all DAGs from package and workspaces.

        Args:
            force_refresh: Force rediscovery even if cache exists

        Returns:
            Dict mapping dag_id to DAGInfo
        """
        if self._discovery_complete and not force_refresh:
            return self._dag_cache

        # Clear cache if forcing refresh
        if force_refresh:
            self._dag_cache.clear()

        # 1. Scan workspace directories (highest priority)
        for workspace_dir in self.workspace_dirs:
            logger.info(f"Scanning workspace directory: {workspace_dir}")
            workspace_dags = self._scan_workspace_directory(workspace_dir)
            self._dag_cache.update(workspace_dags)
            logger.info(
                f"Found {len(workspace_dags)} DAGs in workspace {workspace_dir}"
            )

        # 2. Scan package shared_dags (lower priority, don't override workspace)
        logger.info("Scanning package shared_dags directory")
        package_dags = self._scan_package_directory()
        for dag_id, dag_info in package_dags.items():
            if dag_id not in self._dag_cache:
                self._dag_cache[dag_id] = dag_info
        logger.info(f"Found {len(package_dags)} DAGs in package")

        self._discovery_complete = True
        logger.info(f"Discovery complete: {len(self._dag_cache)} total DAGs")
        return self._dag_cache

    def _scan_package_directory(self) -> Dict[str, DAGInfo]:
        """Scan package shared_dags directory."""
        shared_dags_dir = (
            self.package_root / "cursus" / "pipeline_catalog" / "shared_dags"
        )
        if not shared_dags_dir.exists():
            logger.warning(
                f"Package shared_dags directory not found: {shared_dags_dir}"
            )
            return {}
        return self._scan_dag_directory(shared_dags_dir, "package")

    def _scan_workspace_directory(self, workspace_dir: Path) -> Dict[str, DAGInfo]:
        """Scan workspace dags directory."""
        dags_dir = workspace_dir / "dags"
        if not dags_dir.exists():
            logger.debug(f"Workspace dags directory not found: {dags_dir}")
            return {}
        return self._scan_dag_directory(dags_dir, str(workspace_dir))

    def _scan_dag_directory(
        self, dir_path: Path, workspace_id: str
    ) -> Dict[str, DAGInfo]:
        """
        Scan directory recursively for *_dag.py files.

        Args:
            dir_path: Directory to scan
            workspace_id: Identifier for this workspace

        Returns:
            Dict mapping dag_id to DAGInfo
        """
        dags = {}

        # Find all *_dag.py files recursively
        dag_files = list(dir_path.rglob("*_dag.py"))
        logger.debug(f"Found {len(dag_files)} *_dag.py files in {dir_path}")

        for file_path in dag_files:
            # Skip __pycache__ and hidden files
            if "__pycache__" in str(file_path) or file_path.name.startswith("."):
                continue

            dag_info = self._extract_dag_from_ast(file_path, workspace_id)
            if dag_info:
                dags[dag_info.dag_id] = dag_info
                logger.debug(f"Discovered DAG: {dag_info.dag_id} from {file_path}")

        return dags

    def _extract_dag_from_ast(
        self, file_path: Path, workspace_id: str
    ) -> Optional[DAGInfo]:
        """
        Extract DAG information using AST parsing (no imports).

        Args:
            file_path: Path to DAG file
            workspace_id: Identifier for workspace

        Returns:
            DAGInfo or None if extraction fails
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            # Find create_*_dag function
            create_functions = self._find_create_dag_functions(tree)
            if not create_functions:
                logger.debug(f"No create_*_dag functions found in {file_path}")
                return None

            # Use first create_*_dag function found
            create_func = create_functions[0]
            dag_id = self._extract_dag_id(create_func.name)

            # Extract basic metadata from AST (without importing)
            has_metadata_func = self._has_metadata_function(tree)

            # Create DAGInfo with lazy loading
            dag_info = DAGInfo(
                dag_id=dag_id,
                dag_name=create_func.name,
                dag_path=file_path,
                workspace_id=workspace_id,
            )

            # Try to enrich with registry metadata if available
            if self.registry:
                try:
                    registry_node = self.registry.get_pipeline_node(dag_id)
                    if registry_node:
                        zettel_meta = registry_node.get("zettelkasten_metadata", {})
                        dag_info.framework = zettel_meta.get("framework", "unknown")
                        dag_info.complexity = zettel_meta.get("complexity", "standard")
                        dag_info.features = zettel_meta.get("features", [])
                except Exception as e:
                    logger.debug(f"Could not enrich from registry: {e}")

            return dag_info

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None

    def _find_create_dag_functions(self, tree: ast.AST) -> List[ast.FunctionDef]:
        """
        Find create_*_dag() functions in AST.

        Args:
            tree: AST tree

        Returns:
            List of FunctionDef nodes
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("create_") and node.name.endswith("_dag"):
                    functions.append(node)
        return functions

    def _has_metadata_function(self, tree: ast.AST) -> bool:
        """
        Check if get_dag_metadata() function exists.

        Args:
            tree: AST tree

        Returns:
            bool: True if function exists
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_dag_metadata":
                return True
        return False

    def _extract_dag_id(self, function_name: str) -> str:
        """
        Extract DAG ID from function name.

        Args:
            function_name: Function name like create_xgboost_complete_e2e_dag

        Returns:
            dag_id like xgboost_complete_e2e
        """
        # Remove create_ prefix and _dag suffix
        dag_id = function_name.replace("create_", "").replace("_dag", "")
        return dag_id

    def load_dag_info(self, dag_id: str) -> Optional[DAGInfo]:
        """
        Load specific DAG with workspace-aware priority.

        Args:
            dag_id: DAG identifier

        Returns:
            DAGInfo or None if not found
        """
        if not self._discovery_complete:
            self.discover_all_dags()
        return self._dag_cache.get(dag_id)

    def list_available_dags(self) -> List[str]:
        """
        List all available DAG IDs.

        Returns:
            List of DAG IDs
        """
        if not self._discovery_complete:
            self.discover_all_dags()
        return list(self._dag_cache.keys())

    def get_dags_by_framework(self, framework: str) -> Dict[str, DAGInfo]:
        """
        Get all DAGs for a specific framework.

        Args:
            framework: Framework name (xgboost, pytorch, etc.)

        Returns:
            Dict mapping dag_id to DAGInfo
        """
        if not self._discovery_complete:
            self.discover_all_dags()

        return {
            dag_id: dag_info
            for dag_id, dag_info in self._dag_cache.items()
            if dag_info.framework == framework
        }

    def get_dags_by_complexity(self, complexity: str) -> Dict[str, DAGInfo]:
        """
        Get all DAGs for a specific complexity level.

        Args:
            complexity: Complexity level (simple, standard, comprehensive)

        Returns:
            Dict mapping dag_id to DAGInfo
        """
        if not self._discovery_complete:
            self.discover_all_dags()

        return {
            dag_id: dag_info
            for dag_id, dag_info in self._dag_cache.items()
            if dag_info.complexity == complexity
        }

    def search_dags(
        self,
        framework: Optional[str] = None,
        complexity: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> Dict[str, DAGInfo]:
        """
        Search DAGs by multiple criteria.

        Args:
            framework: Framework filter (optional)
            complexity: Complexity filter (optional)
            features: Features filter - DAG must have all (optional)

        Returns:
            Dict mapping dag_id to DAGInfo
        """
        if not self._discovery_complete:
            self.discover_all_dags()

        results = {}
        for dag_id, dag_info in self._dag_cache.items():
            # Apply filters
            if framework and dag_info.framework != framework:
                continue
            if complexity and dag_info.complexity != complexity:
                continue
            if features:
                # Check if DAG has all required features
                if not all(f in dag_info.features for f in features):
                    continue

            results[dag_id] = dag_info

        return results

    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about discovered DAGs.

        Returns:
            Dict with discovery statistics
        """
        if not self._discovery_complete:
            self.discover_all_dags()

        stats = {
            "total_dags": len(self._dag_cache),
            "workspace_dags": len(
                [d for d in self._dag_cache.values() if d.workspace_id != "package"]
            ),
            "package_dags": len(
                [d for d in self._dag_cache.values() if d.workspace_id == "package"]
            ),
            "by_framework": {},
            "by_complexity": {},
        }

        # Count by framework
        for dag_info in self._dag_cache.values():
            framework = dag_info.framework
            stats["by_framework"][framework] = (
                stats["by_framework"].get(framework, 0) + 1
            )

            complexity = dag_info.complexity
            stats["by_complexity"][complexity] = (
                stats["by_complexity"].get(complexity, 0) + 1
            )

        return stats
