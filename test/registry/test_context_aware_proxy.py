"""
Test suite for Context-Aware Registry Proxy (Section 2.2).

Tests thread-local workspace context management, context managers,
and global instance coordination.
"""

import pytest
import threading
import time
from unittest.mock import patch, MagicMock

from src.cursus.registry.hybrid.proxy import (
    ContextAwareRegistryProxy,
    set_workspace_context,
    get_workspace_context,
    clear_workspace_context,
    workspace_context,
    get_global_registry_manager,
    get_enhanced_compatibility,
    get_context_aware_proxy,
    reset_global_instances,
    get_workspace_from_environment,
    auto_set_workspace_from_environment,
    validate_workspace_context,
    debug_workspace_context,
    ensure_workspace_context,
    get_effective_workspace_context,
    sync_all_contexts,
    get_context_status
)


class TestContextAwareRegistryProxy:
    """Test the ContextAwareRegistryProxy class."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_instances()
        clear_workspace_context()
    
    def teardown_method(self):
        """Clean up test environment."""
        reset_global_instances()
        clear_workspace_context()
    
    def test_proxy_initialization(self):
        """Test proxy can be initialized with registry manager."""
        # Mock the registry manager to avoid file system dependencies
        mock_registry_manager = MagicMock()
        proxy = ContextAwareRegistryProxy(mock_registry_manager)
        
        assert proxy.registry_manager == mock_registry_manager
        assert proxy.compatibility_layer is not None
    
    @patch('src.cursus.registry.hybrid.proxy.get_workspace_context')
    def test_proxy_automatic_context_usage(self, mock_get_context):
        """Test proxy automatically uses workspace context."""
        mock_get_context.return_value = "test_workspace"
        
        mock_registry_manager = MagicMock()
        mock_compatibility_layer = MagicMock()
        
        proxy = ContextAwareRegistryProxy(mock_registry_manager)
        proxy.compatibility_layer = mock_compatibility_layer
        
        # Test get_step_names uses automatic context
        proxy.get_step_names()
        mock_compatibility_layer.get_step_names.assert_called_once_with("test_workspace")
        
        # Test get_step_definition uses automatic context
        proxy.get_step_definition("TestStep")
        mock_registry_manager.get_step_definition.assert_called_once_with("TestStep", "test_workspace")


class TestWorkspaceContextManagement:
    """Test thread-local workspace context management."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
    
    def test_set_and_get_workspace_context(self):
        """Test basic workspace context operations."""
        # Initially no context
        assert get_workspace_context() is None
        
        # Set context
        set_workspace_context("test_workspace")
        assert get_workspace_context() == "test_workspace"
        
        # Clear context
        clear_workspace_context()
        assert get_workspace_context() is None
    
    def test_workspace_context_manager(self):
        """Test workspace context manager."""
        # Initially no context
        assert get_workspace_context() is None
        
        # Use context manager
        with workspace_context("temp_workspace"):
            assert get_workspace_context() == "temp_workspace"
        
        # Context should be cleared after manager
        assert get_workspace_context() is None
    
    def test_workspace_context_manager_with_existing_context(self):
        """Test context manager preserves existing context."""
        # Set initial context
        set_workspace_context("original_workspace")
        
        # Use context manager with different context
        with workspace_context("temp_workspace"):
            assert get_workspace_context() == "temp_workspace"
        
        # Original context should be restored
        assert get_workspace_context() == "original_workspace"
    
    def test_thread_isolation(self):
        """Test workspace context is isolated between threads."""
        results = {}
        
        def thread_function(thread_id, workspace_id):
            set_workspace_context(workspace_id)
            time.sleep(0.1)  # Allow other threads to run
            results[thread_id] = get_workspace_context()
        
        # Start multiple threads with different contexts
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=thread_function,
                args=(f"thread_{i}", f"workspace_{i}")
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Each thread should have its own context
        assert results["thread_0"] == "workspace_0"
        assert results["thread_1"] == "workspace_1"
        assert results["thread_2"] == "workspace_2"


class TestEnvironmentIntegration:
    """Test environment variable integration."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
    
    @patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_workspace'})
    def test_get_workspace_from_environment(self):
        """Test getting workspace from environment variable."""
        workspace_id = get_workspace_from_environment()
        assert workspace_id == "env_workspace"
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_workspace_from_environment_none(self):
        """Test getting workspace from environment when not set."""
        workspace_id = get_workspace_from_environment()
        assert workspace_id is None
    
    @patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_workspace'})
    def test_auto_set_workspace_from_environment(self):
        """Test automatically setting workspace from environment."""
        # Initially no context
        assert get_workspace_context() is None
        
        # Auto-set from environment
        result = auto_set_workspace_from_environment()
        assert result is True
        assert get_workspace_context() == "env_workspace"
    
    def test_auto_set_workspace_from_environment_no_override(self):
        """Test auto-set doesn't override existing context."""
        # Set explicit context
        set_workspace_context("explicit_workspace")
        
        with patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_workspace'}):
            result = auto_set_workspace_from_environment()
            assert result is False
            assert get_workspace_context() == "explicit_workspace"
    
    @patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_workspace'})
    def test_get_effective_workspace_context(self):
        """Test effective workspace context resolution."""
        # No thread-local context, should use environment
        assert get_effective_workspace_context() == "env_workspace"
        
        # Set thread-local context, should use that instead
        set_workspace_context("thread_workspace")
        assert get_effective_workspace_context() == "thread_workspace"


class TestContextValidation:
    """Test workspace context validation and debugging."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
        reset_global_instances()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
        reset_global_instances()
    
    def test_validate_workspace_context_no_context(self):
        """Test validation when no context is set."""
        validation = validate_workspace_context()
        
        assert validation['current_context'] is None
        assert validation['is_valid'] is False
        assert validation['context_source'] == 'none'
        assert len(validation['recommendations']) > 0
    
    @patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_workspace'})
    def test_validate_workspace_context_environment_available(self):
        """Test validation when environment context is available."""
        validation = validate_workspace_context()
        
        assert validation['current_context'] is None
        assert validation['environment_context'] == 'env_workspace'
        assert validation['context_source'] == 'environment_available'
        assert any('auto_set_workspace_from_environment' in rec for rec in validation['recommendations'])
    
    def test_debug_workspace_context(self):
        """Test debug information generation."""
        debug_info = debug_workspace_context()
        
        assert "=== Workspace Context Debug Information ===" in debug_info
        assert "Current Context: None" in debug_info
        assert "Environment Context: None" in debug_info
        assert "Context Source: none" in debug_info
        assert "Is Valid: False" in debug_info
    
    def test_ensure_workspace_context_with_parameter(self):
        """Test ensure_workspace_context with provided workspace ID."""
        workspace_id = ensure_workspace_context("test_workspace")
        assert workspace_id == "test_workspace"
        assert get_workspace_context() == "test_workspace"
    
    def test_ensure_workspace_context_with_existing_context(self):
        """Test ensure_workspace_context with existing context."""
        set_workspace_context("existing_workspace")
        workspace_id = ensure_workspace_context("different_workspace")
        assert workspace_id == "existing_workspace"  # Should use existing
    
    @patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_workspace'})
    def test_ensure_workspace_context_from_environment(self):
        """Test ensure_workspace_context uses environment when no context set."""
        workspace_id = ensure_workspace_context()
        assert workspace_id == "env_workspace"
        assert get_workspace_context() == "env_workspace"
    
    def test_ensure_workspace_context_no_context_available(self):
        """Test ensure_workspace_context raises error when no context available."""
        with pytest.raises(ValueError) as exc_info:
            ensure_workspace_context()
        
        assert "No workspace context available" in str(exc_info.value)
        assert "set_workspace_context" in str(exc_info.value)
        assert "CURSUS_WORKSPACE_ID" in str(exc_info.value)


class TestGlobalInstanceCoordination:
    """Test global instance coordination."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_instances()
    
    def teardown_method(self):
        """Clean up test environment."""
        reset_global_instances()
    
    def test_global_registry_manager_singleton(self):
        """Test global registry manager is singleton."""
        manager1 = get_global_registry_manager()
        manager2 = get_global_registry_manager()
        
        assert manager1 is manager2
    
    def test_global_compatibility_layer_singleton(self):
        """Test global compatibility layer is singleton."""
        layer1 = get_enhanced_compatibility()
        layer2 = get_enhanced_compatibility()
        
        assert layer1 is layer2
    
    def test_context_aware_proxy_creation(self):
        """Test context-aware proxy creation."""
        proxy = get_context_aware_proxy()
        assert isinstance(proxy, ContextAwareRegistryProxy)
        assert proxy.registry_manager is not None
        assert proxy.compatibility_layer is not None
    
    def test_reset_global_instances(self):
        """Test resetting global instances."""
        # Initialize instances
        manager1 = get_global_registry_manager()
        layer1 = get_enhanced_compatibility()
        
        # Reset instances
        reset_global_instances()
        
        # New instances should be different
        manager2 = get_global_registry_manager()
        layer2 = get_enhanced_compatibility()
        
        assert manager1 is not manager2
        assert layer1 is not layer2


class TestContextSynchronization:
    """Test context synchronization utilities."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
        reset_global_instances()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
        reset_global_instances()
    
    def test_sync_all_contexts(self):
        """Test synchronizing all context-aware components."""
        # Set workspace context
        set_workspace_context("sync_test_workspace")
        
        # Sync all contexts
        sync_all_contexts()
        
        # Verify context is synchronized
        assert get_workspace_context() == "sync_test_workspace"
    
    def test_sync_all_contexts_with_parameter(self):
        """Test syncing all contexts with specific workspace."""
        # Sync to specific workspace
        sync_all_contexts("param_workspace")
        
        # Verify context was set
        assert get_workspace_context() == "param_workspace"
    
    def test_get_context_status(self):
        """Test getting comprehensive context status."""
        set_workspace_context("status_test_workspace")
        
        status = get_context_status()
        
        assert status['thread_local_context'] == "status_test_workspace"
        assert status['effective_context'] == "status_test_workspace"
        assert 'global_instances_initialized' in status
        assert 'contexts_synchronized' in status


class TestDecorators:
    """Test workspace context decorators."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
    
    def test_with_workspace_context_decorator(self):
        """Test with_workspace_context decorator."""
        from src.cursus.registry.hybrid.proxy import with_workspace_context
        
        @with_workspace_context("decorator_workspace")
        def test_function():
            return get_workspace_context()
        
        # Function should run with specified context
        result = test_function()
        assert result == "decorator_workspace"
        
        # Context should be cleared after function
        assert get_workspace_context() is None
    
    @patch.dict('os.environ', {'CURSUS_WORKSPACE_ID': 'env_decorator_workspace'})
    def test_auto_workspace_context_decorator(self):
        """Test auto_workspace_context decorator."""
        from src.cursus.registry.hybrid.proxy import auto_workspace_context
        
        @auto_workspace_context
        def test_function():
            return get_workspace_context()
        
        # Function should automatically use environment context
        result = test_function()
        assert result == "env_decorator_workspace"


class TestThreadSafety:
    """Test thread safety of context management."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
        reset_global_instances()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
        reset_global_instances()
    
    def test_concurrent_global_instance_creation(self):
        """Test thread-safe global instance creation."""
        managers = []
        
        def create_manager():
            manager = get_global_registry_manager()
            managers.append(manager)
        
        # Start multiple threads simultaneously
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_manager)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All managers should be the same instance
        assert len(set(id(manager) for manager in managers)) == 1
    
    def test_concurrent_context_operations(self):
        """Test concurrent context operations don't interfere."""
        results = {}
        
        def context_operations(thread_id):
            # Each thread sets its own context
            set_workspace_context(f"workspace_{thread_id}")
            time.sleep(0.05)  # Allow context switching
            
            # Use context manager
            with workspace_context(f"temp_{thread_id}"):
                temp_context = get_workspace_context()
                time.sleep(0.05)
                results[f"{thread_id}_temp"] = temp_context
            
            # Original context should be restored
            final_context = get_workspace_context()
            results[f"{thread_id}_final"] = final_context
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=context_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify each thread had correct contexts
        for i in range(3):
            assert results[f"{i}_temp"] == f"temp_{i}"
            assert results[f"{i}_final"] == f"workspace_{i}"


class TestErrorHandling:
    """Test error handling in context management."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_workspace_context()
    
    def teardown_method(self):
        """Clean up test environment."""
        clear_workspace_context()
    
    def test_ensure_workspace_context_error(self):
        """Test ensure_workspace_context raises helpful error."""
        with pytest.raises(ValueError) as exc_info:
            ensure_workspace_context()
        
        error_message = str(exc_info.value)
        assert "No workspace context available" in error_message
        assert "set_workspace_context" in error_message
        assert "CURSUS_WORKSPACE_ID" in error_message
    
    def test_context_manager_exception_handling(self):
        """Test context manager handles exceptions properly."""
        set_workspace_context("original_context")
        
        try:
            with workspace_context("temp_context"):
                assert get_workspace_context() == "temp_context"
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Original context should be restored even after exception
        assert get_workspace_context() == "original_context"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
