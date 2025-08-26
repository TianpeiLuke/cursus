"""Unit tests for WorkspaceManager class."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.cursus.validation.runtime.integration.workspace_manager import (
    WorkspaceManager,
    WorkspaceConfig,
    CacheEntry
)


class TestWorkspaceConfig:
    """Test WorkspaceConfig model."""
    
    def test_workspace_config_creation(self):
        """Test creating WorkspaceConfig with valid data."""
        config = WorkspaceConfig(
            base_dir=Path("/tmp/test"),
            max_cache_size_gb=5.0,
            cache_retention_days=3,
            auto_cleanup=False
        )
        
        assert config.base_dir == Path("/tmp/test")
        assert config.max_cache_size_gb == 5.0
        assert config.cache_retention_days == 3
        assert config.auto_cleanup is False
    
    def test_workspace_config_defaults(self):
        """Test WorkspaceConfig with default values."""
        config = WorkspaceConfig(base_dir=Path("/tmp/test"))
        
        assert config.base_dir == Path("/tmp/test")
        assert config.max_cache_size_gb == 10.0
        assert config.cache_retention_days == 7
        assert config.auto_cleanup is True


class TestCacheEntry:
    """Test CacheEntry model."""
    
    def test_cache_entry_creation(self):
        """Test creating CacheEntry with valid data."""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            local_path=Path("/tmp/test.txt"),
            size_bytes=1024,
            created_at=now,
            last_accessed=now,
            access_count=5
        )
        
        assert entry.key == "test_key"
        assert entry.local_path == Path("/tmp/test.txt")
        assert entry.size_bytes == 1024
        assert entry.created_at == now
        assert entry.last_accessed == now
        assert entry.access_count == 5


class TestWorkspaceManager:
    """Test WorkspaceManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def workspace_config(self, temp_dir):
        """Create test workspace configuration."""
        return WorkspaceConfig(
            base_dir=temp_dir,
            max_cache_size_gb=1.0,
            cache_retention_days=1,
            auto_cleanup=True
        )
    
    @pytest.fixture
    def workspace_manager(self, workspace_config):
        """Create WorkspaceManager instance for testing."""
        return WorkspaceManager(workspace_config)
    
    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create a sample file for testing."""
        sample_file = temp_dir / "sample.txt"
        sample_file.write_text("This is a test file content.")
        return sample_file
    
    def test_workspace_manager_initialization(self, workspace_manager, temp_dir):
        """Test WorkspaceManager initialization."""
        assert workspace_manager.config.base_dir == temp_dir
        assert workspace_manager.cache_index_path == temp_dir / ".cache_index.json"
        assert isinstance(workspace_manager.cache_entries, dict)
        assert len(workspace_manager.cache_entries) == 0
    
    def test_setup_workspace(self, workspace_manager):
        """Test workspace setup functionality."""
        workspace_name = "test_workspace"
        workspace_dir = workspace_manager.setup_workspace(workspace_name)
        
        # Check workspace directory exists
        assert workspace_dir.exists()
        assert workspace_dir.is_dir()
        assert workspace_dir.name == workspace_name
        
        # Check standard subdirectories exist
        assert (workspace_dir / "inputs").exists()
        assert (workspace_dir / "outputs").exists()
        assert (workspace_dir / "logs").exists()
        assert (workspace_dir / "cache").exists()
        assert (workspace_dir / "s3_data").exists()
    
    def test_setup_workspace_existing(self, workspace_manager):
        """Test setting up workspace that already exists."""
        workspace_name = "existing_workspace"
        
        # Create workspace twice
        workspace_dir1 = workspace_manager.setup_workspace(workspace_name)
        workspace_dir2 = workspace_manager.setup_workspace(workspace_name)
        
        # Should return same directory
        assert workspace_dir1 == workspace_dir2
        assert workspace_dir1.exists()
    
    def test_cleanup_workspace(self, workspace_manager):
        """Test workspace cleanup functionality."""
        workspace_name = "cleanup_test"
        workspace_dir = workspace_manager.setup_workspace(workspace_name)
        
        # Add some files to workspace
        test_file = workspace_dir / "inputs" / "test.txt"
        test_file.write_text("test content")
        
        # Add cache entry pointing to this workspace
        cache_key = "test_cache_key"
        cache_file = workspace_dir / "cache" / "cached_file.txt"
        cache_file.write_text("cached content")
        
        workspace_manager.cache_entries[cache_key] = CacheEntry(
            key=cache_key,
            local_path=cache_file,
            size_bytes=100,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1
        )
        
        # Cleanup workspace
        workspace_manager.cleanup_workspace(workspace_name)
        
        # Check workspace is removed
        assert not workspace_dir.exists()
        
        # Check cache entry is removed
        assert cache_key not in workspace_manager.cache_entries
    
    def test_cleanup_nonexistent_workspace(self, workspace_manager):
        """Test cleaning up workspace that doesn't exist."""
        # Should not raise exception
        workspace_manager.cleanup_workspace("nonexistent_workspace")
    
    def test_cache_data_new_file(self, workspace_manager, sample_file):
        """Test caching new data file."""
        workspace_name = "cache_test"
        workspace_dir = workspace_manager.setup_workspace(workspace_name)
        
        # Cache the sample file
        cached_path = workspace_manager.cache_data(
            data_key="test_data",
            source_path=sample_file,
            workspace_dir=workspace_dir
        )
        
        # Check cached file exists
        assert cached_path.exists()
        assert cached_path.parent == workspace_dir / "cache"
        assert cached_path.read_text() == sample_file.read_text()
        
        # Check cache entry was created
        assert len(workspace_manager.cache_entries) == 1
        cache_entry = list(workspace_manager.cache_entries.values())[0]
        assert cache_entry.local_path == cached_path
        assert cache_entry.access_count == 1
    
    def test_cache_data_existing_file(self, workspace_manager, sample_file):
        """Test caching file that's already cached."""
        workspace_name = "cache_existing_test"
        workspace_dir = workspace_manager.setup_workspace(workspace_name)
        
        # Cache the file twice
        cached_path1 = workspace_manager.cache_data(
            data_key="test_data",
            source_path=sample_file,
            workspace_dir=workspace_dir
        )
        cached_path2 = workspace_manager.cache_data(
            data_key="test_data",
            source_path=sample_file,
            workspace_dir=workspace_dir
        )
        
        # Should return same path
        assert cached_path1 == cached_path2
        
        # Check access count increased
        cache_entry = list(workspace_manager.cache_entries.values())[0]
        assert cache_entry.access_count == 2
    
    def test_get_workspace_info_single(self, workspace_manager):
        """Test getting info for single workspace."""
        workspace_name = "info_test"
        workspace_dir = workspace_manager.setup_workspace(workspace_name)
        
        # Add some test files
        (workspace_dir / "inputs" / "input1.txt").write_text("input")
        (workspace_dir / "outputs" / "output1.txt").write_text("output")
        (workspace_dir / "logs" / "log1.txt").write_text("log")
        
        info = workspace_manager.get_workspace_info(workspace_name)
        
        assert info["name"] == workspace_name
        assert info["path"] == str(workspace_dir)
        assert info["files"]["inputs"] == 1
        assert info["files"]["outputs"] == 1
        assert info["files"]["logs"] == 1
        assert info["files"]["cache"] == 0
        assert info["files"]["s3_data"] == 0
        assert info["total_size_bytes"] > 0
        assert "last_modified" in info
    
    def test_get_workspace_info_nonexistent(self, workspace_manager):
        """Test getting info for nonexistent workspace."""
        info = workspace_manager.get_workspace_info("nonexistent")
        assert "error" in info
        assert "not found" in info["error"].lower()
    
    def test_get_workspace_info_all(self, workspace_manager):
        """Test getting info for all workspaces."""
        # Create multiple workspaces
        workspace_manager.setup_workspace("workspace1")
        workspace_manager.setup_workspace("workspace2")
        
        info = workspace_manager.get_workspace_info()
        
        assert "workspaces" in info
        assert "cache_size_gb" in info
        assert "max_cache_size_gb" in info
        assert "cache_entries" in info
        
        assert len(info["workspaces"]) == 2
        assert "workspace1" in info["workspaces"]
        assert "workspace2" in info["workspaces"]
    
    def test_generate_cache_key(self, workspace_manager, sample_file):
        """Test cache key generation."""
        key1 = workspace_manager._generate_cache_key("data1", sample_file)
        key2 = workspace_manager._generate_cache_key("data1", sample_file)
        key3 = workspace_manager._generate_cache_key("data2", sample_file)
        
        # Same data should generate same key
        assert key1 == key2
        
        # Different data should generate different key
        assert key1 != key3
        
        # Keys should be MD5 hashes (32 characters)
        assert len(key1) == 32
        assert all(c in "0123456789abcdef" for c in key1)
    
    def test_cleanup_cache_expired_entries(self, workspace_manager, sample_file):
        """Test cleanup of expired cache entries."""
        workspace_dir = workspace_manager.setup_workspace("cleanup_test")
        
        # Create cache entry with old timestamp
        old_time = datetime.now() - timedelta(days=2)
        cache_key = "old_entry"
        cache_file = workspace_dir / "cache" / "old_file.txt"
        cache_file.write_text("old content")
        
        workspace_manager.cache_entries[cache_key] = CacheEntry(
            key=cache_key,
            local_path=cache_file,
            size_bytes=100,
            created_at=old_time,
            last_accessed=old_time,
            access_count=1
        )
        
        # Run cleanup
        workspace_manager._cleanup_cache()
        
        # Check expired entry was removed
        assert cache_key not in workspace_manager.cache_entries
        assert not cache_file.exists()
    
    def test_cleanup_cache_size_limit(self, workspace_manager, temp_dir):
        """Test cleanup when cache exceeds size limit."""
        workspace_dir = workspace_manager.setup_workspace("size_test")
        
        # Create multiple large cache entries
        now = datetime.now()
        for i in range(3):
            cache_key = f"large_entry_{i}"
            cache_file = workspace_dir / "cache" / f"large_file_{i}.txt"
            # Create file larger than 1GB limit (simulated with size_bytes)
            cache_file.write_text("content")
            
            workspace_manager.cache_entries[cache_key] = CacheEntry(
                key=cache_key,
                local_path=cache_file,
                size_bytes=500 * 1024 * 1024,  # 500MB each
                created_at=now - timedelta(minutes=i),  # Different access times
                last_accessed=now - timedelta(minutes=i),
                access_count=1
            )
        
        # Run cleanup
        workspace_manager._cleanup_cache()
        
        # Should remove oldest entries to get under 1GB limit
        assert len(workspace_manager.cache_entries) < 3
    
    def test_get_total_cache_size(self, workspace_manager):
        """Test total cache size calculation."""
        # Add some cache entries
        workspace_manager.cache_entries["entry1"] = CacheEntry(
            key="entry1",
            local_path=Path("/tmp/file1"),
            size_bytes=1000,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1
        )
        workspace_manager.cache_entries["entry2"] = CacheEntry(
            key="entry2",
            local_path=Path("/tmp/file2"),
            size_bytes=2000,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1
        )
        
        total_size = workspace_manager._get_total_cache_size()
        assert total_size == 3000
    
    def test_load_cache_index_existing(self, workspace_manager, temp_dir):
        """Test loading existing cache index."""
        # Create cache index file
        cache_data = {
            "test_key": {
                "key": "test_key",
                "local_path": "/tmp/test.txt",
                "size_bytes": 100,
                "created_at": "2023-01-01T12:00:00",
                "last_accessed": "2023-01-01T12:00:00",
                "access_count": 1
            }
        }
        
        cache_index_path = temp_dir / ".cache_index.json"
        with open(cache_index_path, 'w') as f:
            json.dump(cache_data, f)
        
        # Create new manager to load index
        config = WorkspaceConfig(base_dir=temp_dir)
        manager = WorkspaceManager(config)
        
        # Check cache entry was loaded
        assert "test_key" in manager.cache_entries
        entry = manager.cache_entries["test_key"]
        assert entry.key == "test_key"
        assert entry.size_bytes == 100
        assert entry.access_count == 1
    
    def test_load_cache_index_corrupted(self, workspace_manager, temp_dir):
        """Test loading corrupted cache index."""
        # Create corrupted cache index file
        cache_index_path = temp_dir / ".cache_index.json"
        cache_index_path.write_text("invalid json content")
        
        # Create new manager to load index
        config = WorkspaceConfig(base_dir=temp_dir)
        manager = WorkspaceManager(config)
        
        # Should start with empty cache
        assert len(manager.cache_entries) == 0
    
    def test_save_cache_index(self, workspace_manager, temp_dir):
        """Test saving cache index."""
        # Add cache entry
        now = datetime.now()
        workspace_manager.cache_entries["test_key"] = CacheEntry(
            key="test_key",
            local_path=Path("/tmp/test.txt"),
            size_bytes=100,
            created_at=now,
            last_accessed=now,
            access_count=1
        )
        
        # Save index
        workspace_manager._save_cache_index()
        
        # Check file was created
        cache_index_path = temp_dir / ".cache_index.json"
        assert cache_index_path.exists()
        
        # Check content
        with open(cache_index_path) as f:
            data = json.load(f)
        
        assert "test_key" in data
        assert data["test_key"]["key"] == "test_key"
        assert data["test_key"]["size_bytes"] == 100
        assert data["test_key"]["access_count"] == 1
    
    @patch('shutil.copy2')
    def test_cache_data_copy_failure(self, mock_copy, workspace_manager, sample_file):
        """Test handling of file copy failure during caching."""
        mock_copy.side_effect = OSError("Permission denied")
        
        workspace_dir = workspace_manager.setup_workspace("copy_fail_test")
        
        with pytest.raises(OSError):
            workspace_manager.cache_data(
                data_key="test_data",
                source_path=sample_file,
                workspace_dir=workspace_dir
            )
    
    @patch('pathlib.Path.unlink')
    def test_cleanup_cache_delete_failure(self, mock_unlink, workspace_manager):
        """Test handling of file deletion failure during cleanup."""
        mock_unlink.side_effect = OSError("Permission denied")
        
        workspace_dir = workspace_manager.setup_workspace("delete_fail_test")
        
        # Create expired cache entry
        old_time = datetime.now() - timedelta(days=2)
        cache_key = "old_entry"
        cache_file = workspace_dir / "cache" / "old_file.txt"
        cache_file.write_text("old content")
        
        workspace_manager.cache_entries[cache_key] = CacheEntry(
            key=cache_key,
            local_path=cache_file,
            size_bytes=100,
            created_at=old_time,
            last_accessed=old_time,
            access_count=1
        )
        
        # Should not raise exception, just log warning
        workspace_manager._cleanup_cache()
        
        # Entry should still be removed from index even if file deletion failed
        assert cache_key not in workspace_manager.cache_entries
    
    def test_workspace_manager_with_auto_cleanup_disabled(self, temp_dir):
        """Test WorkspaceManager with auto cleanup disabled."""
        config = WorkspaceConfig(
            base_dir=temp_dir,
            auto_cleanup=False
        )
        manager = WorkspaceManager(config)
        
        workspace_dir = manager.setup_workspace("no_cleanup_test")
        sample_file = temp_dir / "sample.txt"
        sample_file.write_text("test content")
        
        with patch.object(manager, '_cleanup_cache') as mock_cleanup:
            manager.cache_data(
                data_key="test_data",
                source_path=sample_file,
                workspace_dir=workspace_dir
            )
            
            # Cleanup should not be called
            mock_cleanup.assert_not_called()
