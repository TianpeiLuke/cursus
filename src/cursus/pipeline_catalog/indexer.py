"""
Pipeline Catalog Indexer

This module provides indexing functionality for complete pipeline implementations,
distinct from the step catalog which indexes individual components.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
import inspect

logger = logging.getLogger(__name__)


class CatalogIndexer:
    """
    Indexer for complete pipeline implementations.
    
    This class scans pipeline files and builds indexes of complete end-to-end
    pipeline templates, which is different from the step catalog that indexes
    individual step components.
    """
    
    def __init__(self, catalog_root: Path):
        """
        Initialize the catalog indexer.
        
        Args:
            catalog_root: Root directory of the pipeline catalog
        """
        self.catalog_root = Path(catalog_root)
        self.index_path = self.catalog_root / "index.json"
        
        # Try to use the step catalog for enhanced functionality
        self._step_catalog = None
        try:
            from ..step_catalog import StepCatalog
            # PORTABLE: Use package-only discovery for pipeline indexing
            self._step_catalog = StepCatalog(workspace_dirs=None)
            logger.info("Using StepCatalog for enhanced pipeline indexing")
        except ImportError:
            logger.warning("StepCatalog not available, using legacy indexing")
        except Exception as e:
            logger.warning(f"Failed to initialize StepCatalog: {e}, using legacy indexing")
    
    def generate_index(self) -> Dict[str, Any]:
        """
        Generate a complete index of all pipelines.
        
        Returns:
            Dictionary containing the pipeline index
        """
        pipelines = []
        
        # Find all Python files in the catalog
        python_files = self._find_python_files(self.catalog_root)
        
        for file_path in python_files:
            try:
                entry = self._process_pipeline_file(file_path)
                if entry:
                    pipelines.append(entry)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
        
        return {"pipelines": pipelines}
    
    def _find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in a directory recursively."""
        python_files = []
        if directory.exists():
            for file_path in directory.rglob("*.py"):
                if file_path.name != "__init__.py":
                    python_files.append(file_path)
        return python_files
    
    def _process_pipeline_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single pipeline file and extract metadata."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("pipeline_module", file_path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract relative path
            rel_path = file_path.relative_to(self.catalog_root)
            
            # Extract basic information
            pipeline_id = self._extract_id(rel_path)
            name = self._extract_name(module)
            framework = self._detect_framework_from_path(rel_path)
            complexity = self._determine_complexity(rel_path, module.__doc__ or "")
            features = self._extract_features(module.__doc__ or "")
            description = self._extract_description(module.__doc__ or "")
            tags = self._extract_tags(module.__doc__ or "", rel_path)
            
            return {
                "id": pipeline_id,
                "name": name,
                "path": str(rel_path),
                "framework": framework,
                "complexity": complexity,
                "features": features,
                "description": description,
                "tags": tags,
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_id(self, rel_path: Path) -> str:
        """Extract pipeline ID from relative path."""
        # Remove .py extension and convert to kebab-case
        name = rel_path.stem
        # Handle nested paths
        if len(rel_path.parts) > 1:
            framework = rel_path.parts[0] if rel_path.parts[0] != "pipelines" else rel_path.parts[1] if len(rel_path.parts) > 1 else "unknown"
            name = f"{framework}-{name}"
        return name.lower().replace('_', '-')
    
    def _extract_name(self, module) -> str:
        """Extract pipeline name from module."""
        if hasattr(module, '__doc__') and module.__doc__:
            first_line = module.__doc__.split('\n')[0].strip()
            if first_line:
                return first_line
        
        # Fallback to filename
        if hasattr(module, '__file__'):
            return Path(module.__file__).stem.replace('_', ' ').title() + " Pipeline"
        
        return "Unknown Pipeline"
    
    def _detect_framework_from_path(self, rel_path: Path) -> str:
        """Detect framework from file path."""
        path_str = str(rel_path).lower()
        if 'xgboost' in path_str or 'xgb' in path_str:
            return 'xgboost'
        elif 'pytorch' in path_str:
            return 'pytorch'
        elif 'tensorflow' in path_str:
            return 'tensorflow'
        elif 'dummy' in path_str:
            return 'dummy'
        else:
            return 'unknown'
    
    def _determine_complexity(self, rel_path: Path, docstring: str) -> str:
        """Determine pipeline complexity."""
        path_str = str(rel_path).lower()
        content_lower = docstring.lower()
        
        if 'simple' in path_str or 'basic' in path_str or 'simple' in content_lower:
            return 'simple'
        elif ('advanced' in path_str or 'complex' in path_str or 'e2e' in path_str or 
              'end_to_end' in path_str or 'comprehensive' in path_str or 'advanced' in content_lower):
            return 'advanced'
        else:
            return 'intermediate'
    
    def _extract_features(self, docstring: str) -> List[str]:
        """Extract features from docstring."""
        features = []
        content_lower = docstring.lower()
        
        if 'training' in content_lower:
            features.append('training')
        if 'evaluation' in content_lower or 'eval' in content_lower:
            features.append('evaluation')
        if 'calibration' in content_lower:
            features.append('calibration')
        if 'registration' in content_lower or 'register' in content_lower:
            features.append('registration')
        if 'end-to-end' in content_lower or 'e2e' in content_lower:
            features.append('end_to_end')
        
        return features
    
    def _extract_description(self, docstring: str) -> str:
        """Extract description from docstring."""
        if not docstring:
            return "No description available"
        
        # Try to extract from docstring
        lines = docstring.split('\n')
        
        # Skip the first line (title) and find the first substantial paragraph
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if line and not line.startswith('Args:') and not line.startswith('Returns:'):
                return line
        
        # Fallback to first line if no description paragraph found
        first_line = lines[0].strip()
        return first_line if first_line else "No description available"
    
    def _extract_tags(self, docstring: str, rel_path: Path) -> List[str]:
        """Extract tags from docstring and path."""
        tags = []
        
        # Add framework tag
        framework = self._detect_framework_from_path(rel_path)
        if framework != 'unknown':
            tags.append(framework)
        
        # Add path-based tags
        path_str = str(rel_path).lower()
        if 'training' in path_str:
            tags.append('training')
        if 'evaluation' in path_str or 'eval' in path_str:
            tags.append('evaluation')
        if 'calibration' in path_str:
            tags.append('calibration')
        if 'e2e' in path_str or 'end_to_end' in path_str:
            tags.append('end_to_end')
        
        # Add content-based tags
        content_lower = docstring.lower()
        if 'training' in content_lower:
            tags.append('training')
        if 'evaluation' in content_lower or 'eval' in content_lower:
            tags.append('evaluation')
        if 'calibration' in content_lower:
            tags.append('calibration')
        if 'machine learning' in content_lower or 'ml' in content_lower:
            tags.append('machine_learning')
        
        return list(set(tags))  # Remove duplicates
    
    def update_index(self) -> None:
        """Update the existing index with new entries."""
        # Load existing index
        existing_index = {"pipelines": []}
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    existing_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
        
        # Generate new index
        new_index = self.generate_index()
        
        # Merge indices
        merged_index = self._merge_indices(existing_index, new_index)
        
        # Validate merged index
        is_valid, issues = self.validate_index(merged_index)
        if not is_valid:
            logger.warning(f"Index validation failed: {issues}")
        
        # Save merged index
        self.save_index(merged_index)
    
    def _merge_indices(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two indices, with new entries overriding existing ones."""
        # Create a mapping of existing pipelines by ID
        existing_by_id = {p["id"]: p for p in existing.get("pipelines", [])}
        
        # Update with new pipelines
        for pipeline in new.get("pipelines", []):
            existing_by_id[pipeline["id"]] = pipeline
        
        return {"pipelines": list(existing_by_id.values())}
    
    def validate_index(self, index: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an index structure."""
        issues = []
        
        if "pipelines" not in index:
            issues.append("Index missing 'pipelines' key")
            return False, issues
        
        pipeline_ids = set()
        required_fields = ["id", "name", "path", "framework", "complexity", "features", "description", "tags"]
        
        for i, pipeline in enumerate(index["pipelines"]):
            # Check required fields
            for field in required_fields:
                if field not in pipeline:
                    issues.append(f"Pipeline {i} missing '{field}' field")
            
            # Check for duplicate IDs
            if "id" in pipeline:
                if pipeline["id"] in pipeline_ids:
                    issues.append(f"Duplicate pipeline ID: {pipeline['id']}")
                pipeline_ids.add(pipeline["id"])
            
            # Check if file exists (if path is provided)
            if "path" in pipeline:
                file_path = self.catalog_root / pipeline["path"]
                if not file_path.exists():
                    issues.append(f"Pipeline file does not exist: {pipeline['path']}")
        
        return len(issues) == 0, issues
    
    def save_index(self, index: Dict[str, Any]) -> None:
        """Save index to file."""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index
            with open(self.index_path, 'w') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
