"""Workspace manager for efficient data caching and organization."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
import json
import hashlib
from datetime import datetime, timedelta
import logging

class WorkspaceConfig(BaseModel):
    """Configuration for test workspace management."""
    base_dir: Path
    max_cache_size_gb: float = 10.0
    cache_retention_days: int = 7
    auto_cleanup: bool = True

class CacheEntry(BaseModel):
    """Entry in the data cache."""
    key: str
    local_path: Path
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int

class WorkspaceManager:
    """Manages test workspace and data caching."""
    
    def __init__(self, config: WorkspaceConfig):
        """Initialize with configuration."""
        self.config = config
        self.cache_index_path = config.base_dir / ".cache_index.json"
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.logger = logging.getLogger(__name__)
        self._load_cache_index()
    
    def setup_workspace(self, workspace_name: str) -> Path:
        """Set up a new test workspace.
        
        Args:
            workspace_name: Name of workspace to create
            
        Returns:
            Path to created workspace directory
        """
        workspace_dir = self.config.base_dir / workspace_name
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        (workspace_dir / "inputs").mkdir(exist_ok=True)
        (workspace_dir / "outputs").mkdir(exist_ok=True)
        (workspace_dir / "logs").mkdir(exist_ok=True)
        (workspace_dir / "cache").mkdir(exist_ok=True)
        (workspace_dir / "s3_data").mkdir(exist_ok=True)
        
        return workspace_dir
    
    def cleanup_workspace(self, workspace_name: str):
        """Clean up a test workspace.
        
        Args:
            workspace_name: Name of workspace to clean
        """
        workspace_dir = self.config.base_dir / workspace_name
        if workspace_dir.exists():
            # First remove cache entries that point to this workspace
            keys_to_remove = []
            for key, entry in self.cache_entries.items():
                if str(workspace_dir) in str(entry.local_path):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache_entries[key]
            
            # Now remove the directory
            shutil.rmtree(workspace_dir)
            self.logger.info(f"Cleaned up workspace: {workspace_name}")
            
            # Save updated cache index
            self._save_cache_index()
    
    def cache_data(self, data_key: str, source_path: Path, 
                  workspace_dir: Path) -> Path:
        """Cache data in the workspace.
        
        Args:
            data_key: Unique identifier for the data
            source_path: Path to the source data file
            workspace_dir: Path to workspace directory
            
        Returns:
            Path to cached file
        """
        # Generate cache key
        cache_key = self._generate_cache_key(data_key, source_path)
        
        # Check if already cached
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]
            if entry.local_path.exists():
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._save_cache_index()
                return entry.local_path
        
        # Cache the data
        cache_dir = workspace_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / f"{cache_key}_{source_path.name}"
        
        shutil.copy2(source_path, cached_path)
        
        # Update cache index
        self.cache_entries[cache_key] = CacheEntry(
            key=cache_key,
            local_path=cached_path,
            size_bytes=cached_path.stat().st_size,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1
        )
        
        self._save_cache_index()
        
        # Perform cleanup if needed
        if self.config.auto_cleanup:
            self._cleanup_cache()
        
        return cached_path
    
    def get_workspace_info(self, workspace_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about workspaces.
        
        Args:
            workspace_name: Optional specific workspace name to get info for
            
        Returns:
            Dictionary with workspace information
        """
        if workspace_name:
            workspace_dir = self.config.base_dir / workspace_name
            if not workspace_dir.exists():
                return {"error": f"Workspace not found: {workspace_name}"}
            
            return self._get_single_workspace_info(workspace_name, workspace_dir)
        else:
            # Get info for all workspaces
            workspaces = {}
            for workspace_dir in self.config.base_dir.iterdir():
                if workspace_dir.is_dir() and not workspace_dir.name.startswith('.'):
                    workspaces[workspace_dir.name] = self._get_single_workspace_info(
                        workspace_dir.name, workspace_dir
                    )
            
            return {
                "workspaces": workspaces,
                "cache_size_gb": self._get_total_cache_size() / (1024**3),
                "max_cache_size_gb": self.config.max_cache_size_gb,
                "cache_entries": len(self.cache_entries)
            }
    
    def _get_single_workspace_info(self, name: str, workspace_dir: Path) -> Dict[str, Any]:
        """Get information about a single workspace.
        
        Args:
            name: Name of workspace
            workspace_dir: Path to workspace directory
            
        Returns:
            Dictionary with workspace information
        """
        # Count files in standard directories
        input_count = sum(1 for _ in (workspace_dir / "inputs").glob("**/*") if _.is_file())
        output_count = sum(1 for _ in (workspace_dir / "outputs").glob("**/*") if _.is_file())
        log_count = sum(1 for _ in (workspace_dir / "logs").glob("**/*") if _.is_file())
        cache_count = sum(1 for _ in (workspace_dir / "cache").glob("**/*") if _.is_file())
        s3_data_count = sum(1 for _ in (workspace_dir / "s3_data").glob("**/*") if _.is_file())
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in workspace_dir.glob("**/*") if f.is_file())
        
        return {
            "name": name,
            "path": str(workspace_dir),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024**2),
            "files": {
                "inputs": input_count,
                "outputs": output_count,
                "logs": log_count,
                "cache": cache_count,
                "s3_data": s3_data_count
            },
            "last_modified": datetime.fromtimestamp(
                max((f.stat().st_mtime for f in workspace_dir.glob("**/*") if f.is_file()), 
                    default=workspace_dir.stat().st_mtime)
            ).isoformat()
        }
    
    def _generate_cache_key(self, data_key: str, source_path: Path) -> str:
        """Generate a unique cache key.
        
        Args:
            data_key: Unique identifier for the data
            source_path: Path to the source data file
            
        Returns:
            Unique cache key
        """
        content = f"{data_key}_{source_path.stat().st_mtime}_{source_path.stat().st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up old cache entries.
        
        This removes entries that are:
        1. Older than the retention period
        2. Entries that make the cache exceed the size limit (LRU policy)
        """
        current_time = datetime.now()
        retention_threshold = current_time - timedelta(days=self.config.cache_retention_days)
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.cache_entries.items():
            if entry.last_accessed < retention_threshold:
                if entry.local_path.exists():
                    try:
                        entry.local_path.unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file {entry.local_path}: {e}")
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache_entries[key]
        
        # Check cache size and remove least recently used if needed
        total_size_gb = self._get_total_cache_size() / (1024**3)
        
        if total_size_gb > self.config.max_cache_size_gb:
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.cache_entries.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove entries until under size limit
            for key, entry in sorted_entries:
                if entry.local_path.exists():
                    try:
                        entry.local_path.unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file {entry.local_path}: {e}")
                del self.cache_entries[key]
                
                total_size_gb = self._get_total_cache_size() / (1024**3)
                if total_size_gb <= self.config.max_cache_size_gb:
                    break
        
        self._save_cache_index()
    
    def _get_total_cache_size(self) -> int:
        """Get total size of cache in bytes.
        
        Returns:
            Total cache size in bytes
        """
        return sum(entry.size_bytes for entry in self.cache_entries.values())
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        if self.cache_index_path.exists():
            try:
                with open(self.cache_index_path) as f:
                    data = json.load(f)
                
                for key, entry_data in data.items():
                    self.cache_entries[key] = CacheEntry(
                        key=entry_data['key'],
                        local_path=Path(entry_data['local_path']),
                        size_bytes=entry_data['size_bytes'],
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                        access_count=entry_data['access_count']
                    )
            except Exception as e:
                # If index is corrupted, start fresh
                self.logger.error(f"Error loading cache index: {e}")
                self.cache_entries = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        data = {}
        for key, entry in self.cache_entries.items():
            data[key] = {
                'key': entry.key,
                'local_path': str(entry.local_path),
                'size_bytes': entry.size_bytes,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count
            }
        
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_index_path, 'w') as f:
            json.dump(data, f, indent=2)
