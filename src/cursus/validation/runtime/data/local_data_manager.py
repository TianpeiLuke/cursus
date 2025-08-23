"""Local Data Manager for managing local real data files for pipeline testing."""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class LocalDataManager:
    """Manages local real data files for pipeline testing"""
    
    def __init__(self, workspace_dir: str):
        """Initialize with workspace directory"""
        self.workspace_dir = Path(workspace_dir)
        self.local_data_dir = self.workspace_dir / "local_data"
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifest file if it doesn't exist
        self.manifest_path = self.local_data_dir / "data_manifest.yaml"
        if not self.manifest_path.exists():
            self._create_default_manifest()
        
        logger.info(f"LocalDataManager initialized with data directory: {self.local_data_dir}")
    
    def get_data_for_script(self, script_name: str) -> Optional[Dict[str, str]]:
        """Get local data file paths for a specific script"""
        manifest = self._load_manifest()
        
        if script_name in manifest.get("scripts", {}):
            script_data = manifest["scripts"][script_name]
            data_paths = {}
            
            for data_key, file_info in script_data.items():
                file_path = self.local_data_dir / file_info["path"]
                if file_path.exists():
                    data_paths[data_key] = str(file_path)
                else:
                    logger.warning(f"Local data file not found: {file_path}")
            
            return data_paths if data_paths else None
        
        logger.info(f"No local data configured for script: {script_name}")
        return None
    
    def prepare_data_for_execution(self, script_name: str, target_dir: str):
        """Copy local data files to execution directory"""
        data_paths = self.get_data_for_script(script_name)
        if not data_paths:
            logger.info(f"No local data to prepare for script: {script_name}")
            return
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for data_key, source_path in data_paths.items():
            target_file = target_path / Path(source_path).name
            shutil.copy2(source_path, target_file)
            logger.info(f"Copied local data file: {source_path} -> {target_file}")
    
    def add_data_for_script(self, script_name: str, data_key: str, 
                           file_path: str, description: str = "") -> bool:
        """Add a local data file for a script"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {file_path}")
                return False
            
            # Create script directory in local data
            script_dir = self.local_data_dir / script_name
            script_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to local data directory
            target_file = script_dir / source_path.name
            shutil.copy2(source_path, target_file)
            
            # Update manifest
            manifest = self._load_manifest()
            if "scripts" not in manifest:
                manifest["scripts"] = {}
            if script_name not in manifest["scripts"]:
                manifest["scripts"][script_name] = {}
            
            # Determine file format from extension
            file_format = source_path.suffix.lower().lstrip('.')
            if file_format == 'pkl':
                file_format = 'pickle'
            
            manifest["scripts"][script_name][data_key] = {
                "path": f"{script_name}/{source_path.name}",
                "format": file_format,
                "description": description or f"Local data file for {script_name}"
            }
            
            self._save_manifest(manifest)
            logger.info(f"Added local data file for {script_name}: {data_key} -> {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add local data file: {str(e)}")
            return False
    
    def list_data_for_script(self, script_name: str) -> Dict[str, Dict[str, Any]]:
        """List all local data files for a script"""
        manifest = self._load_manifest()
        return manifest.get("scripts", {}).get(script_name, {})
    
    def list_all_scripts(self) -> List[str]:
        """List all scripts that have local data configured"""
        manifest = self._load_manifest()
        return list(manifest.get("scripts", {}).keys())
    
    def remove_data_for_script(self, script_name: str, data_key: str = None) -> bool:
        """Remove local data for a script (specific key or all data)"""
        try:
            manifest = self._load_manifest()
            
            if script_name not in manifest.get("scripts", {}):
                logger.warning(f"No local data found for script: {script_name}")
                return False
            
            if data_key:
                # Remove specific data key
                if data_key in manifest["scripts"][script_name]:
                    file_info = manifest["scripts"][script_name][data_key]
                    file_path = self.local_data_dir / file_info["path"]
                    if file_path.exists():
                        file_path.unlink()
                    del manifest["scripts"][script_name][data_key]
                    logger.info(f"Removed local data: {script_name}.{data_key}")
                else:
                    logger.warning(f"Data key not found: {script_name}.{data_key}")
                    return False
            else:
                # Remove all data for script
                script_dir = self.local_data_dir / script_name
                if script_dir.exists():
                    shutil.rmtree(script_dir)
                del manifest["scripts"][script_name]
                logger.info(f"Removed all local data for script: {script_name}")
            
            self._save_manifest(manifest)
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove local data: {str(e)}")
            return False
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load data manifest configuration"""
        try:
            with open(self.manifest_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load manifest, using empty: {str(e)}")
            return {}
    
    def _save_manifest(self, manifest: Dict[str, Any]):
        """Save data manifest configuration"""
        try:
            with open(self.manifest_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=True)
        except Exception as e:
            logger.error(f"Failed to save manifest: {str(e)}")
    
    def _create_default_manifest(self):
        """Create default manifest file"""
        default_manifest = {
            "version": "1.0",
            "description": "Local data manifest for pipeline testing",
            "scripts": {
                "example_script": {
                    "input_data": {
                        "path": "example_script/input.csv",
                        "format": "csv",
                        "description": "Example input data file"
                    }
                }
            }
        }
        
        self._save_manifest(default_manifest)
        logger.info("Created default data manifest")
