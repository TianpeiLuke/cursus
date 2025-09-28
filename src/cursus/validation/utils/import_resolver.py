"""
Centralized Import Resolution Utility for Validation Scripts

This module provides robust import resolution that works across different deployment scenarios:
- Development mode (running from source)
- Installed package mode (pip installed)
- Submodule mode (cursus as a submodule)
- Different execution contexts (various working directories)
"""

import sys
import os
import ast
from pathlib import Path
from typing import Optional, List
import logging
import importlib

logger = logging.getLogger(__name__)


class ImportResolver:
    """
    Centralized import resolution for validation scripts.
    
    Handles multiple deployment scenarios and execution contexts to ensure
    consistent module resolution across all validation scripts.
    """
    
    _setup_complete = False
    _project_root: Optional[Path] = None
    _src_dir: Optional[Path] = None
    
    @classmethod
    def ensure_cursus_imports(cls) -> bool:
        """
        Ensure cursus package can be imported regardless of execution context.
        
        This is the main entry point that validation scripts should call.
        It's safe to call multiple times - setup only happens once.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if cls._setup_complete:
            return True
            
        try:
            # Strategy 1: Try importing as installed package first
            if cls._try_installed_import():
                cls._setup_complete = True
                logger.debug("Using installed cursus package")
                return True
                
            # Strategy 2: Try relative import pattern (deployment-agnostic)
            if cls._try_relative_import_pattern():
                cls._setup_complete = True
                logger.debug("Using relative import pattern")
                return True
                
            # Strategy 3: Try StepCatalog integration (leverages existing system)
            if cls._try_step_catalog_discovery():
                cls._setup_complete = True
                logger.debug("Using StepCatalog discovery")
                return True
                
            # Strategy 4: Setup development imports (fallback)
            if cls._setup_development_imports():
                cls._setup_complete = True
                logger.debug(f"Setup development imports from {cls._src_dir}")
                return True
                
            # Strategy 5: Last resort - try common patterns
            if cls._try_fallback_patterns():
                cls._setup_complete = True
                logger.debug("Using fallback import patterns")
                return True
                
            logger.error("Failed to setup cursus imports with all strategies")
            return False
            
        except Exception as e:
            logger.error(f"Error during import setup: {e}")
            return False
    
    @classmethod
    def _try_installed_import(cls) -> bool:
        """Try importing cursus as an installed package."""
        try:
            import cursus
            # Verify it's a real cursus package by checking for key modules
            from cursus.validation.alignment import unified_alignment_tester
            return True
        except ImportError:
            return False
    
    @classmethod
    def _try_relative_import_pattern(cls) -> bool:
        """
        Try deployment-agnostic relative import pattern.
        
        Uses relative imports with package parameter following the
        cursus package portability architecture design.
        """
        try:
            # Try relative import pattern (deployment-agnostic)
            # This works across PyPI, source, container, and serverless deployments
            module = importlib.import_module('..validation.alignment.unified_alignment_tester', 
                                           package=__package__)
            logger.debug("Relative import pattern successful")
            return True
        except ImportError as e:
            logger.debug(f"Relative import pattern failed: {e}")
            return False
    
    @classmethod
    def _try_step_catalog_discovery(cls) -> bool:
        """
        Try using existing StepCatalog for import resolution.
        
        Leverages the unified discovery system from the cursus package
        portability architecture for consistent component access.
        """
        try:
            # Try to use existing StepCatalog system
            from cursus.step_catalog import StepCatalog
            catalog = StepCatalog()
            
            # Verify catalog can discover validation components
            available_steps = catalog.list_available_steps()
            if len(available_steps) > 0:
                # Test that we can access validation components through catalog
                from cursus.validation.alignment import unified_alignment_tester
                logger.debug(f"StepCatalog discovery successful with {len(available_steps)} steps")
                return True
            else:
                logger.debug("StepCatalog discovery found no steps")
                return False
                
        except ImportError as e:
            logger.debug(f"StepCatalog discovery failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"StepCatalog discovery error: {e}")
            return False
    
    @classmethod
    def _validate_cursus_structure_with_ast(cls, package_path: Path) -> bool:
        """
        Validate cursus package structure using AST parsing.
        
        Follows the AST-based discovery pattern from the portability
        architecture to safely validate components before importing.
        """
        try:
            # Check key validation modules exist and are syntactically valid
            validation_tester = package_path / 'validation' / 'alignment' / 'unified_alignment_tester.py'
            if validation_tester.exists():
                with open(validation_tester, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())  # Validate syntax without importing
                logger.debug(f"AST validation successful for {validation_tester}")
                return True
            else:
                logger.debug(f"Validation tester not found: {validation_tester}")
                return False
        except (SyntaxError, OSError) as e:
            logger.debug(f"AST validation failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"AST validation error: {e}")
            return False
    
    @classmethod
    def _setup_development_imports(cls) -> bool:
        """Setup imports for development mode (running from source)."""
        try:
            project_root = cls._find_project_root()
            if not project_root:
                return False
                
            src_dir = project_root / "src"
            if not src_dir.exists():
                logger.debug(f"Source directory not found: {src_dir}")
                return False
                
            # Add src directory to Python path
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
                logger.debug(f"Added to sys.path: {src_str}")
            
            # Verify the import works
            try:
                import cursus
                from cursus.validation.alignment import unified_alignment_tester
                cls._project_root = project_root
                cls._src_dir = src_dir
                return True
            except ImportError as e:
                logger.debug(f"Import verification failed: {e}")
                # Remove from sys.path if verification failed
                if src_str in sys.path:
                    sys.path.remove(src_str)
                return False
                
        except Exception as e:
            logger.debug(f"Development import setup failed: {e}")
            return False
    
    @classmethod
    def _find_project_root(cls) -> Optional[Path]:
        """
        Find project root by looking for characteristic files.
        
        Searches upward from current file location for:
        1. pyproject.toml (primary indicator)
        2. setup.py (fallback)
        3. src/cursus directory structure
        """
        # Start from this file's location
        current = Path(__file__).resolve()
        
        # Search upward through parent directories
        for parent in [current] + list(current.parents):
            # Primary indicator: pyproject.toml
            if (parent / "pyproject.toml").exists():
                # Verify it's the cursus project by checking for src/cursus
                if (parent / "src" / "cursus").exists():
                    logger.debug(f"Found project root via pyproject.toml: {parent}")
                    return parent
            
            # Secondary indicator: setup.py with src/cursus
            if (parent / "setup.py").exists() and (parent / "src" / "cursus").exists():
                logger.debug(f"Found project root via setup.py: {parent}")
                return parent
                
            # Tertiary indicator: direct src/cursus structure
            if (parent / "src" / "cursus" / "__init__.py").exists():
                logger.debug(f"Found project root via src/cursus structure: {parent}")
                return parent
        
        logger.debug("Could not find project root")
        return None
    
    @classmethod
    def _try_fallback_patterns(cls) -> bool:
        """Try common fallback patterns for import resolution."""
        fallback_patterns = [
            # Common development patterns
            Path.cwd() / "src",
            Path.cwd().parent / "src", 
            Path.cwd().parent.parent / "src",
            
            # Common execution patterns
            Path(__file__).parent.parent.parent.parent.parent / "src",
            Path(__file__).parent.parent.parent.parent / "src",
        ]
        
        for src_candidate in fallback_patterns:
            if src_candidate.exists() and (src_candidate / "cursus").exists():
                try:
                    src_str = str(src_candidate)
                    if src_str not in sys.path:
                        sys.path.insert(0, src_str)
                    
                    # Test import
                    import cursus
                    from cursus.validation.alignment import unified_alignment_tester
                    
                    cls._src_dir = src_candidate
                    logger.debug(f"Fallback pattern worked: {src_candidate}")
                    return True
                    
                except ImportError:
                    # Remove from sys.path if it didn't work
                    if src_str in sys.path:
                        sys.path.remove(src_str)
                    continue
        
        return False
    
    @classmethod
    def get_project_info(cls) -> dict:
        """
        Get information about the current project setup.
        
        Returns:
            dict: Project information including paths and setup status
        """
        return {
            "setup_complete": cls._setup_complete,
            "project_root": str(cls._project_root) if cls._project_root else None,
            "src_dir": str(cls._src_dir) if cls._src_dir else None,
            "cursus_in_path": any("cursus" in path for path in sys.path),
            "sys_path_entries": [path for path in sys.path if "cursus" in path or "src" in path],
        }
    
    @classmethod
    def reset(cls):
        """Reset the resolver state (mainly for testing)."""
        cls._setup_complete = False
        cls._project_root = None
        cls._src_dir = None


# Convenience function for easy import
def ensure_cursus_imports() -> bool:
    """
    Convenience function to ensure cursus imports work.
    
    This is the main function that validation scripts should call.
    
    Returns:
        bool: True if setup successful, False otherwise
        
    Example:
        from cursus.validation.utils.import_resolver import ensure_cursus_imports
        
        if not ensure_cursus_imports():
            print("Failed to setup cursus imports")
            sys.exit(1)
    """
    return ImportResolver.ensure_cursus_imports()


def get_project_info() -> dict:
    """
    Get information about the current project setup.
    
    Returns:
        dict: Project information including paths and setup status
    """
    return ImportResolver.get_project_info()


# Auto-setup when module is imported (optional convenience)
def auto_setup_imports():
    """
    Automatically setup imports when this module is imported.
    
    This provides a convenience import pattern:
        from cursus.validation.utils.import_resolver import auto_setup_imports
        auto_setup_imports()  # or just importing triggers it
    """
    success = ensure_cursus_imports()
    if not success:
        logger.warning("Auto-setup of cursus imports failed")
    return success


if __name__ == "__main__":
    # Test the import resolver
    print("Testing Import Resolver...")
    print("=" * 50)
    
    success = ensure_cursus_imports()
    print(f"Setup successful: {success}")
    
    if success:
        print("\nProject Info:")
        info = get_project_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\nTesting imports...")
        try:
            from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
            print("✅ UnifiedAlignmentTester import successful")
        except ImportError as e:
            print(f"❌ UnifiedAlignmentTester import failed: {e}")
    else:
        print("❌ Setup failed - cannot test imports")
