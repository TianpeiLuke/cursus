"""
Pipeline Catalog Indexing Tool

This module provides functionality for automatically generating and updating the
pipeline catalog index.json file by scanning the catalog directory structure and 
extracting metadata from pipeline files.

Example usage:
    ```python
    from cursus.pipeline_catalog.indexer import CatalogIndexer
    
    # Generate a new index
    indexer = CatalogIndexer()
    index = indexer.generate_index()
    
    # Save the updated index
    indexer.save_index(index)
    ```
"""

import os
import re
import json
import inspect
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Type aliases
PipelineEntry = Dict[str, Any]
PipelineIndex = Dict[str, List[PipelineEntry]]


class CatalogIndexer:
    """
    Tool for automatically generating and updating the pipeline catalog index.
    """
    
    # Module import path prefix
    MODULE_PREFIX = "src.cursus.pipeline_catalog"
    
    # Directories to scan for pipeline files
    SCAN_DIRECTORIES = [
        "frameworks",
        "components"
    ]
    
    # Keywords to identify key features in docstrings
    FEATURE_KEYWORDS = {
        "training": ["train", "learning", "fit"],
        "evaluation": ["eval", "assess", "test", "valid"],
        "calibration": ["calib", "probability"],
        "registration": ["regist", "deploy", "mims"],
        "data_loading": ["data load", "dataset", "cradle"],
        "preprocessing": ["preprocess", "transform", "feature"]
    }
    
    def __init__(self, catalog_root: Optional[Path] = None):
        """
        Initialize the catalog indexer.
        
        Args:
            catalog_root: Path to the root of the pipeline catalog directory.
                          If None, will use the directory containing this file.
        """
        if catalog_root is None:
            self.catalog_root = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.catalog_root = catalog_root
            
        self.index_path = self.catalog_root / "index.json"
    
    def generate_index(self) -> PipelineIndex:
        """
        Generate a complete pipeline catalog index by scanning the directory structure.
        
        Returns:
            Dict: The generated index containing pipeline entries
        """
        logger.info(f"Scanning pipeline catalog at: {self.catalog_root}")
        
        # Initialize the index structure
        index = {"pipelines": []}
        
        # Find and process all Python files in the catalog
        for dir_name in self.SCAN_DIRECTORIES:
            dir_path = self.catalog_root / dir_name
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                continue
                
            for file_path in self._find_python_files(dir_path):
                try:
                    pipeline_entry = self._process_pipeline_file(file_path)
                    if pipeline_entry:
                        index["pipelines"].append(pipeline_entry)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Sort pipelines by ID for consistency
        index["pipelines"].sort(key=lambda p: p.get("id", ""))
        
        logger.info(f"Generated index with {len(index['pipelines'])} pipeline entries")
        return index
    
    def save_index(self, index: PipelineIndex) -> None:
        """
        Save the generated index to the index.json file.
        
        Args:
            index: The pipeline index to save
        """
        try:
            with open(self.index_path, "w") as f:
                json.dump(index, f, indent=2)
            logger.info(f"Saved index to: {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def validate_index(self, index: PipelineIndex) -> Tuple[bool, List[str]]:
        """
        Validate the structure and content of a pipeline index.
        
        Args:
            index: The pipeline index to validate
            
        Returns:
            Tuple[bool, List[str]]: Validation result (True if valid) and list of issues
        """
        issues = []
        
        # Check overall structure
        if "pipelines" not in index:
            issues.append("Missing 'pipelines' key in index")
            return False, issues
            
        if not isinstance(index["pipelines"], list):
            issues.append("'pipelines' should be a list")
            return False, issues
        
        # Check for duplicate IDs
        pipeline_ids = []
        for i, pipeline in enumerate(index["pipelines"]):
            # Check required fields
            if "id" not in pipeline:
                issues.append(f"Pipeline at index {i} missing 'id' field")
                continue
                
            # Check for duplicate IDs
            if pipeline["id"] in pipeline_ids:
                issues.append(f"Duplicate pipeline ID: {pipeline['id']}")
            pipeline_ids.append(pipeline["id"])
            
            # Check for required fields
            required_fields = ["name", "path", "framework", "complexity", "features"]
            for field in required_fields:
                if field not in pipeline:
                    issues.append(f"Pipeline '{pipeline['id']}' missing '{field}' field")
            
            # Validate path exists
            if "path" in pipeline:
                if not (self.catalog_root / pipeline["path"]).exists():
                    issues.append(f"Pipeline '{pipeline['id']}' has invalid path: {pipeline['path']}")
        
        return len(issues) == 0, issues
    
    def update_index(self) -> None:
        """
        Update the pipeline catalog index by generating a new index and
        merging it with any existing index.
        """
        # Generate new index
        new_index = self.generate_index()
        
        # Load existing index if available
        existing_index = {"pipelines": []}
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    existing_index = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Could not load existing index, creating new one")
        
        # Merge indices, preferring entries from new_index but keeping
        # existing entries that aren't in the new index
        merged_index = self._merge_indices(existing_index, new_index)
        
        # Validate and save the merged index
        is_valid, issues = self.validate_index(merged_index)
        if not is_valid:
            for issue in issues:
                logger.warning(f"Validation issue: {issue}")
            logger.warning("Saving index despite validation issues")
            
        self.save_index(merged_index)
    
    def _find_python_files(self, directory: Path) -> List[Path]:
        """
        Find all Python files in the given directory and its subdirectories.
        
        Args:
            directory: The directory to search
            
        Returns:
            List[Path]: List of paths to Python files
        """
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    python_files.append(Path(root) / file)
        return python_files
    
    def _process_pipeline_file(self, file_path: Path) -> Optional[PipelineEntry]:
        """
        Process a potential pipeline file to extract metadata.
        
        Args:
            file_path: Path to the Python file to process
            
        Returns:
            Optional[Dict]: Pipeline entry or None if not a valid pipeline file
        """
        # Get relative path from catalog root
        rel_path = file_path.relative_to(self.catalog_root)
        
        # Skip test files and utility files
        if "test" in str(rel_path).lower() or "utils" in str(rel_path).lower():
            return None
            
        # Load the module to inspect its contents
        module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        full_module_name = f"{self.MODULE_PREFIX}.{module_name}"
        
        # Try to import the module to inspect it
        spec = importlib.util.spec_from_file_location(full_module_name, file_path)
        if spec is None or spec.loader is None:
            logger.warning(f"Could not load module: {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"Error loading module {file_path}: {str(e)}")
            return None
            
        # Check if this is a pipeline module
        if not hasattr(module, "create_dag") or not hasattr(module, "create_pipeline"):
            return None
            
        # Extract pipeline information
        pipeline_id = self._extract_id(rel_path)
        pipeline_name = self._extract_name(module)
        docstring = inspect.getdoc(module) or ""
        
        # Extract features from docstring
        features = self._extract_features(docstring)
        
        # Determine framework from path
        framework = "generic"
        if "xgboost" in str(rel_path):
            framework = "xgboost"
        elif "pytorch" in str(rel_path):
            framework = "pytorch"
            
        # Determine complexity from file structure and content
        complexity = self._determine_complexity(rel_path, docstring)
        
        # Extract tags
        tags = self._extract_tags(docstring, rel_path)
        
        # Extract description
        description = self._extract_description(docstring)
        
        # Create the pipeline entry
        entry = {
            "id": pipeline_id,
            "name": pipeline_name,
            "path": str(rel_path),
            "framework": framework,
            "complexity": complexity,
            "features": list(features),
            "description": description,
            "tags": tags
        }
        
        return entry
    
    def _extract_id(self, rel_path: Path) -> str:
        """
        Extract a pipeline ID from the file path.
        
        Args:
            rel_path: Relative path to the pipeline file
            
        Returns:
            str: The pipeline ID
        """
        # Start with the filename without extension
        pipeline_id = rel_path.stem
        
        # Replace underscores with hyphens
        pipeline_id = pipeline_id.replace("_", "-")
        
        # If in a framework directory, prefix with framework name
        path_parts = rel_path.parts
        if len(path_parts) >= 2 and path_parts[0] == "frameworks":
            framework = path_parts[1]
            if not pipeline_id.startswith(framework):
                pipeline_id = f"{framework}-{pipeline_id}"
                
        return pipeline_id
    
    def _extract_name(self, module) -> str:
        """
        Extract a human-readable name for the pipeline.
        
        Args:
            module: The imported module
            
        Returns:
            str: The pipeline name
        """
        # Try to get name from module docstring first line
        if module.__doc__:
            first_line = module.__doc__.strip().split("\n")[0]
            return first_line
            
        # Fall back to ID-based name
        if hasattr(module, "__file__"):
            path = Path(module.__file__)
            name_parts = path.stem.split("_")
            return " ".join(part.capitalize() for part in name_parts) + " Pipeline"
            
        return "Unnamed Pipeline"
    
    def _extract_features(self, docstring: str) -> Set[str]:
        """
        Extract pipeline features from the docstring.
        
        Args:
            docstring: The module docstring
            
        Returns:
            Set[str]: Set of feature names
        """
        features = set()
        
        # Look for features based on keywords
        for feature, keywords in self.FEATURE_KEYWORDS.items():
            if any(keyword.lower() in docstring.lower() for keyword in keywords):
                features.add(feature)
                
        return features
    
    def _determine_complexity(self, rel_path: Path, docstring: str) -> str:
        """
        Determine the pipeline complexity.
        
        Args:
            rel_path: Relative path to the pipeline file
            docstring: The module docstring
            
        Returns:
            str: Complexity level ("simple", "standard", or "advanced")
        """
        path_str = str(rel_path)
        
        # End-to-end pipelines are usually advanced
        if "end_to_end" in path_str or "e2e" in path_str:
            return "advanced"
            
        # Simple pipelines
        if "simple" in path_str or "basic" in path_str or "dataload" in path_str:
            return "simple"
            
        # Check docstring for complexity hints
        if docstring:
            if "advanced" in docstring.lower() or "complex" in docstring.lower():
                return "advanced"
            if "simple" in docstring.lower() or "basic" in docstring.lower():
                return "simple"
                
        # Default to standard
        return "standard"
    
    def _extract_tags(self, docstring: str, rel_path: Path) -> List[str]:
        """
        Extract tags for the pipeline.
        
        Args:
            docstring: The module docstring
            rel_path: Relative path to the pipeline file
            
        Returns:
            List[str]: List of tags
        """
        tags = []
        
        # Add framework tag
        if "xgboost" in str(rel_path):
            tags.append("xgboost")
        elif "pytorch" in str(rel_path):
            tags.append("pytorch")
            
        # Add feature tags
        features = self._extract_features(docstring)
        for feature in features:
            tags.append(feature)
            
        # Add complexity tags
        complexity = self._determine_complexity(rel_path, docstring)
        if complexity == "simple":
            tags.append("beginner")
        elif complexity == "advanced":
            tags.append("advanced")
            
        # Add special tags
        if "end_to_end" in str(rel_path) or "e2e" in str(rel_path):
            tags.append("end-to-end")
            
        return tags
    
    def _extract_description(self, docstring: str) -> str:
        """
        Extract a description from the docstring.
        
        Args:
            docstring: The module docstring
            
        Returns:
            str: The description
        """
        if not docstring:
            return "No description available"
            
        # Use the first paragraph after the title
        paragraphs = [p.strip() for p in docstring.split("\n\n")]
        if len(paragraphs) > 1:
            return paragraphs[1]
            
        # If only one paragraph, use that
        return paragraphs[0]
    
    def _merge_indices(self, existing_index: PipelineIndex, new_index: PipelineIndex) -> PipelineIndex:
        """
        Merge two pipeline indices, preserving entries that may only exist in one.
        
        Args:
            existing_index: The existing index
            new_index: The newly generated index
            
        Returns:
            Dict: The merged index
        """
        # Create a mapping of IDs to entries for both indices
        existing_entries = {entry.get("id"): entry for entry in existing_index.get("pipelines", [])}
        new_entries = {entry.get("id"): entry for entry in new_index.get("pipelines", [])}
        
        # Merge entries, preferring new entries but preserving entries only in existing
        merged_entries = {}
        
        # Add all new entries
        for pipeline_id, entry in new_entries.items():
            merged_entries[pipeline_id] = entry
            
        # Add entries that only exist in the existing index
        for pipeline_id, entry in existing_entries.items():
            if pipeline_id not in merged_entries:
                merged_entries[pipeline_id] = entry
                
        # Convert back to a list and sort by ID
        merged_pipelines = list(merged_entries.values())
        merged_pipelines.sort(key=lambda p: p.get("id", ""))
        
        return {"pipelines": merged_pipelines}


def main():
    """
    Main function to run the indexer as a script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Catalog Indexer")
    parser.add_argument(
        "--validate-only", 
        action="store_true", 
        help="Only validate the existing index without updating"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force update without merging with existing index"
    )
    args = parser.parse_args()
    
    indexer = CatalogIndexer()
    
    if args.validate_only:
        # Only validate the existing index
        try:
            with open(indexer.index_path, "r") as f:
                index = json.load(f)
            
            is_valid, issues = indexer.validate_index(index)
            if is_valid:
                logger.info("Index validation successful")
            else:
                logger.warning("Index validation failed:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        except Exception as e:
            logger.error(f"Failed to validate index: {str(e)}")
    else:
        # Generate and update index
        if args.force:
            # Generate new index from scratch
            new_index = indexer.generate_index()
            indexer.save_index(new_index)
        else:
            # Update existing index
            indexer.update_index()


if __name__ == "__main__":
    main()
