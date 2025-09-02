"""Local Data Manager for managing local real data files for pipeline testing."""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class LocalDataManager:
    """Manages local real data files for pipeline testing with workspace awareness"""
    
    def __init__(self, workspace_dir: str, workspace_root: str = None):
        """Initialize with workspace directory and optional workspace root for workspace-aware data management"""
        self.workspace_dir = Path(workspace_dir)
        self.local_data_dir = self.workspace_dir / "local_data"
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 5: Workspace-aware data management
        self.workspace_root = workspace_root
        self.workspace_data_contexts = {}
        
        # Create manifest file if it doesn't exist
        self.manifest_path = self.local_data_dir / "data_manifest.yaml"
        if not self.manifest_path.exists():
            self._create_default_manifest()
        
        logger.info(f"LocalDataManager initialized with data directory: {self.local_data_dir}")
        if workspace_root:
            logger.info(f"Workspace-aware data management enabled for: {workspace_root}")
    
    def get_data_for_script(self, script_name: str, developer_id: str = None) -> Optional[Dict[str, str]]:
        """Get local data file paths for a specific script with workspace context"""
        manifest = self._load_manifest()
        
        # Phase 5: Try workspace-specific data first
        if developer_id and self.workspace_root:
            workspace_key = f"{developer_id}:{script_name}"
            if workspace_key in manifest.get("workspace_scripts", {}):
                script_data = manifest["workspace_scripts"][workspace_key]
                data_paths = self._resolve_data_paths(script_data, workspace_key)
                if data_paths:
                    logger.info(f"Found workspace-specific data for {workspace_key}")
                    return data_paths
        
        # Fallback to general script data
        if script_name in manifest.get("scripts", {}):
            script_data = manifest["scripts"][script_name]
            data_paths = self._resolve_data_paths(script_data, script_name)
            if data_paths:
                logger.info(f"Found general data for script: {script_name}")
                return data_paths
        
        logger.info(f"No local data configured for script: {script_name} (developer: {developer_id or 'any'})")
        return None
    
    def _resolve_data_paths(self, script_data: Dict[str, Any], context_key: str) -> Optional[Dict[str, str]]:
        """Resolve data file paths from script data configuration"""
        data_paths = {}
        
        for data_key, file_info in script_data.items():
            file_path = self.local_data_dir / file_info["path"]
            if file_path.exists():
                data_paths[data_key] = str(file_path)
            else:
                logger.warning(f"Local data file not found for {context_key}: {file_path}")
        
        return data_paths if data_paths else None
    
    def prepare_data_for_execution(self, script_name: str, target_dir: str, developer_id: str = None):
        """Copy local data files to execution directory with workspace context"""
        data_paths = self.get_data_for_script(script_name, developer_id)
        if not data_paths:
            logger.info(f"No local data to prepare for script: {script_name} (developer: {developer_id or 'any'})")
            return
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Phase 5: Track workspace data usage
        if developer_id and self.workspace_root:
            if developer_id not in self.workspace_data_contexts:
                self.workspace_data_contexts[developer_id] = {
                    'scripts_with_data': [],
                    'total_files_prepared': 0,
                    'data_sources': set()
                }
            
            context = self.workspace_data_contexts[developer_id]
            context['scripts_with_data'].append(script_name)
            context['total_files_prepared'] += len(data_paths)
        
        for data_key, source_path in data_paths.items():
            target_file = target_path / Path(source_path).name
            shutil.copy2(source_path, target_file)
            logger.info(f"Copied local data file: {source_path} -> {target_file}")
            
            # Track data source
            if developer_id and self.workspace_root:
                self.workspace_data_contexts[developer_id]['data_sources'].add(str(Path(source_path).parent))
    
    def add_data_for_script(self, script_name: str, data_key: str, 
                           file_path: str, description: str = "", developer_id: str = None) -> bool:
        """Add a local data file for a script with optional workspace context"""
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
            
            # Update manifest with workspace awareness
            manifest = self._load_manifest()
            
            # Determine file format from extension
            file_format = source_path.suffix.lower().lstrip('.')
            if file_format == 'pkl':
                file_format = 'pickle'
            
            file_entry = {
                "path": f"{script_name}/{source_path.name}",
                "format": file_format,
                "description": description or f"Local data file for {script_name}"
            }
            
            # Phase 5: Add to workspace-specific section if developer_id provided
            if developer_id and self.workspace_root:
                if "workspace_scripts" not in manifest:
                    manifest["workspace_scripts"] = {}
                
                workspace_key = f"{developer_id}:{script_name}"
                if workspace_key not in manifest["workspace_scripts"]:
                    manifest["workspace_scripts"][workspace_key] = {}
                
                manifest["workspace_scripts"][workspace_key][data_key] = file_entry
                logger.info(f"Added workspace-specific local data file for {workspace_key}: {data_key} -> {target_file}")
            else:
                # Add to general scripts section
                if "scripts" not in manifest:
                    manifest["scripts"] = {}
                if script_name not in manifest["scripts"]:
                    manifest["scripts"][script_name] = {}
                
                manifest["scripts"][script_name][data_key] = file_entry
                logger.info(f"Added local data file for {script_name}: {data_key} -> {target_file}")
            
            self._save_manifest(manifest)
            return True
            
        except Exception as e:
            logger.error(f"Failed to add local data file: {str(e)}")
            return False
    
    def list_data_for_script(self, script_name: str, developer_id: str = None) -> Dict[str, Dict[str, Any]]:
        """List all local data files for a script with workspace context"""
        manifest = self._load_manifest()
        
        # Phase 5: Check workspace-specific data first
        if developer_id and self.workspace_root:
            workspace_key = f"{developer_id}:{script_name}"
            workspace_data = manifest.get("workspace_scripts", {}).get(workspace_key, {})
            if workspace_data:
                return workspace_data
        
        # Fallback to general script data
        return manifest.get("scripts", {}).get(script_name, {})
    
    def list_all_scripts(self, developer_id: str = None) -> List[str]:
        """List all scripts that have local data configured with optional workspace filtering"""
        manifest = self._load_manifest()
        scripts = set()
        
        # Add general scripts
        scripts.update(manifest.get("scripts", {}).keys())
        
        # Phase 5: Add workspace-specific scripts
        if developer_id and self.workspace_root:
            workspace_scripts = manifest.get("workspace_scripts", {})
            for workspace_key in workspace_scripts.keys():
                if workspace_key.startswith(f"{developer_id}:"):
                    script_name = workspace_key.split(":", 1)[1]
                    scripts.add(script_name)
        elif self.workspace_root:
            # If no specific developer but workspace is enabled, show all workspace scripts
            workspace_scripts = manifest.get("workspace_scripts", {})
            for workspace_key in workspace_scripts.keys():
                script_name = workspace_key.split(":", 1)[1]
                scripts.add(script_name)
        
        return list(scripts)
    
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
    
    def get_workspace_data_summary(self) -> Dict[str, Any]:
        """Get summary of workspace data usage and contexts.
        
        Returns:
            Dictionary containing workspace data summary
        """
        if not self.workspace_root:
            return {"error": "No workspace context available"}
        
        manifest = self._load_manifest()
        
        summary = {
            'workspace_root': self.workspace_root,
            'general_scripts': len(manifest.get("scripts", {})),
            'workspace_scripts': len(manifest.get("workspace_scripts", {})),
            'developer_contexts': {},
            'data_usage_stats': dict(self.workspace_data_contexts)
        }
        
        # Analyze workspace-specific scripts by developer
        workspace_scripts = manifest.get("workspace_scripts", {})
        developer_script_counts = {}
        
        for workspace_key in workspace_scripts.keys():
            developer_id = workspace_key.split(":", 1)[0]
            if developer_id not in developer_script_counts:
                developer_script_counts[developer_id] = 0
            developer_script_counts[developer_id] += 1
        
        summary['developer_contexts'] = developer_script_counts
        
        # Convert sets to lists for JSON serialization
        for dev_id, context in summary['data_usage_stats'].items():
            if 'data_sources' in context:
                context['data_sources'] = list(context['data_sources'])
        
        return summary
    
    def validate_workspace_data_availability(self, script_name: str, developer_id: str = None) -> Dict[str, Any]:
        """Validate data availability for workspace-aware execution.
        
        Args:
            script_name: Name of the script
            developer_id: Optional developer ID
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'script_name': script_name,
            'developer_id': developer_id,
            'data_available': False,
            'data_sources': [],
            'missing_files': [],
            'recommendations': []
        }
        
        try:
            data_paths = self.get_data_for_script(script_name, developer_id)
            
            if data_paths:
                validation_result['data_available'] = True
                validation_result['data_sources'] = list(data_paths.keys())
                
                # Check if all files actually exist
                for data_key, file_path in data_paths.items():
                    if not Path(file_path).exists():
                        validation_result['missing_files'].append({
                            'data_key': data_key,
                            'file_path': file_path
                        })
                
                if validation_result['missing_files']:
                    validation_result['recommendations'].append(
                        "Some data files are missing - check file paths and availability"
                    )
                else:
                    validation_result['recommendations'].append(
                        "All configured data sources are available"
                    )
            else:
                validation_result['recommendations'].extend([
                    f"No local data configured for script '{script_name}'",
                    "Consider adding local data files using add_data_for_script()",
                    "Or use synthetic data source for testing"
                ])
                
                if developer_id:
                    validation_result['recommendations'].append(
                        f"Try checking data availability without developer filter"
                    )
        
        except Exception as e:
            validation_result['error'] = str(e)
            validation_result['recommendations'].append(f"Error validating data: {e}")
        
        return validation_result
