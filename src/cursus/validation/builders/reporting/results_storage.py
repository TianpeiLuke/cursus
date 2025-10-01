"""
Test Results Storage Module

This module provides organized storage functionality for dynamic universal builder
test results, supporting automatic timestamped file saving and directory management.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class BuilderTestResultsStorage:
    """
    Helper class for saving test results to organized directory structure.
    
    This class provides automatic results storage for the dynamic universal builder
    testing system, organizing results by command type and providing timestamped
    filenames to prevent overwrites.
    """
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], command_type: str, identifier: str = None) -> str:
        """
        Save test results to organized directory structure.
        
        Args:
            results: Test results to save (dictionary format)
            command_type: Type of command ('all_builders' or 'single_builder')
            identifier: Optional identifier for single builder tests (e.g., step type or canonical name)
            
        Returns:
            Path where results were saved
            
        Raises:
            Exception: If results cannot be saved
        """
        try:
            # Create results directory structure
            results_dir = BuilderTestResultsStorage.get_results_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if command_type == "all_builders":
                subdir = results_dir / "all_builders"
                if identifier:
                    filename = f"all_builders_{identifier}_{timestamp}.json"
                else:
                    filename = f"all_builders_{timestamp}.json"
            elif command_type == "single_builder":
                subdir = results_dir / "individual"
                if identifier:
                    filename = f"{identifier}_{timestamp}.json"
                else:
                    filename = f"single_builder_{timestamp}.json"
            else:
                subdir = results_dir
                filename = f"results_{timestamp}.json"
            
            # Create directory and save file
            output_path = subdir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            raise
    
    @staticmethod
    def get_results_directory() -> Path:
        """
        Get the base results directory path.
        
        Returns:
            Path to the test results directory
        """
        return Path("test/steps/builders/results")
    
    @staticmethod
    def ensure_results_directory() -> None:
        """
        Ensure results directory structure exists.
        
        Creates the necessary directory structure and .gitignore file
        if they don't already exist.
        """
        try:
            results_dir = BuilderTestResultsStorage.get_results_directory()
            
            # Create subdirectories
            (results_dir / "all_builders").mkdir(parents=True, exist_ok=True)
            (results_dir / "individual").mkdir(parents=True, exist_ok=True)
            
            # Create .gitignore if it doesn't exist
            gitignore_path = results_dir / ".gitignore"
            if not gitignore_path.exists():
                with open(gitignore_path, 'w') as f:
                    f.write("*.json\n!.gitignore\n")
                logger.debug(f"Created .gitignore at: {gitignore_path}")
            
            logger.debug(f"Results directory structure ensured at: {results_dir}")
            
        except Exception as e:
            logger.error(f"Error ensuring results directory: {e}")
            raise
    
    @staticmethod
    def list_saved_results(command_type: str = None) -> Dict[str, list]:
        """
        List previously saved test results.
        
        Args:
            command_type: Optional filter by command type ('all_builders' or 'single_builder')
            
        Returns:
            Dictionary with lists of result files by type
        """
        try:
            results_dir = BuilderTestResultsStorage.get_results_directory()
            saved_results = {
                'all_builders': [],
                'single_builder': []
            }
            
            if not results_dir.exists():
                return saved_results
            
            # List all_builders results
            if command_type is None or command_type == 'all_builders':
                all_builders_dir = results_dir / "all_builders"
                if all_builders_dir.exists():
                    for json_file in all_builders_dir.glob("*.json"):
                        saved_results['all_builders'].append({
                            'path': str(json_file),
                            'name': json_file.name,
                            'modified': datetime.fromtimestamp(json_file.stat().st_mtime)
                        })
            
            # List single_builder results
            if command_type is None or command_type == 'single_builder':
                individual_dir = results_dir / "individual"
                if individual_dir.exists():
                    for json_file in individual_dir.glob("*.json"):
                        saved_results['single_builder'].append({
                            'path': str(json_file),
                            'name': json_file.name,
                            'modified': datetime.fromtimestamp(json_file.stat().st_mtime)
                        })
            
            # Sort by modification time (newest first)
            for result_type in saved_results:
                saved_results[result_type].sort(key=lambda x: x['modified'], reverse=True)
            
            return saved_results
            
        except Exception as e:
            logger.error(f"Error listing saved results: {e}")
            return {'all_builders': [], 'single_builder': []}
    
    @staticmethod
    def load_saved_results(file_path: str) -> Dict[str, Any]:
        """
        Load previously saved test results.
        
        Args:
            file_path: Path to the saved results file
            
        Returns:
            Dictionary containing the loaded test results
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.debug(f"Loaded test results from: {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading saved results from {file_path}: {e}")
            raise
    
    @staticmethod
    def cleanup_old_results(days_to_keep: int = 30) -> int:
        """
        Clean up old test result files.
        
        Args:
            days_to_keep: Number of days of results to keep (default: 30)
            
        Returns:
            Number of files deleted
        """
        try:
            results_dir = BuilderTestResultsStorage.get_results_directory()
            if not results_dir.exists():
                return 0
            
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            deleted_count = 0
            
            # Clean up all_builders directory
            all_builders_dir = results_dir / "all_builders"
            if all_builders_dir.exists():
                for json_file in all_builders_dir.glob("*.json"):
                    if json_file.stat().st_mtime < cutoff_time:
                        json_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old result file: {json_file}")
            
            # Clean up individual directory
            individual_dir = results_dir / "individual"
            if individual_dir.exists():
                for json_file in individual_dir.glob("*.json"):
                    if json_file.stat().st_mtime < cutoff_time:
                        json_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old result file: {json_file}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old result files (older than {days_to_keep} days)")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old results: {e}")
            return 0
