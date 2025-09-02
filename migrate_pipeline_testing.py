#!/usr/bin/env python3
"""
Migration script to transition from pipeline_testing to developer_workspaces structure.

This script helps migrate existing pipeline testing data and configurations
from the old pipeline_testing directory to the new workspace-aware structure.
"""

import os
import shutil
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_pipeline_testing_data(source_dir: str = "./pipeline_testing", 
                                 target_base: str = "./developer_workspaces/developers",
                                 developer_id: str = "developer_1",
                                 dry_run: bool = False):
    """
    Migrate pipeline testing data to developer workspace structure.
    
    Args:
        source_dir: Source pipeline_testing directory
        target_base: Base directory for developer workspaces
        developer_id: Target developer ID
        dry_run: If True, only show what would be migrated without actually doing it
    """
    source_path = Path(source_dir)
    target_path = Path(target_base) / developer_id
    
    if not source_path.exists():
        logger.warning(f"Source directory {source_path} does not exist. Nothing to migrate.")
        return
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Migrating from {source_path} to {target_path}")
    
    # Create target directory structure
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "test").mkdir(exist_ok=True)
        (target_path / "validation_reports").mkdir(exist_ok=True)
    
    # Migration mappings
    migrations = [
        # Data directories
        (source_path / "inputs", target_path / "inputs"),
        (source_path / "outputs", target_path / "outputs"),
        (source_path / "local_data", target_path / "local_data"),
        (source_path / "s3_data", target_path / "s3_data"),
        (source_path / "synthetic_data", target_path / "synthetic_data"),
        
        # Logs and metadata
        (source_path / "logs", target_path / "logs"),
        (source_path / "metadata", target_path / "metadata"),
    ]
    
    migrated_count = 0
    for src, dst in migrations:
        if src.exists():
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Migrating {src} -> {dst}")
            if not dry_run:
                if dst.exists():
                    logger.warning(f"Target {dst} already exists. Merging contents...")
                    # Merge directories
                    for item in src.rglob('*'):
                        if item.is_file():
                            relative_path = item.relative_to(src)
                            target_file = dst / relative_path
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, target_file)
                            logger.debug(f"Copied {item} -> {target_file}")
                else:
                    shutil.copytree(src, dst)
            migrated_count += 1
        else:
            logger.debug(f"Source {src} does not exist, skipping")
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Migration completed. {migrated_count} directories processed.")
    
    if not dry_run:
        # Create a README in the target directory
        readme_content = f"""# Developer Workspace: {developer_id}

This workspace was migrated from the legacy pipeline_testing structure.

## Directory Structure

- `inputs/` - Input data for pipeline steps
- `outputs/` - Output data from pipeline steps  
- `local_data/` - Local test data files
- `s3_data/` - Downloaded S3 data for testing
- `synthetic_data/` - Generated synthetic test data
- `logs/` - Execution logs
- `metadata/` - Pipeline execution metadata
- `test/` - Developer-specific tests
- `validation_reports/` - Validation reports
- `src/cursus_dev/steps/` - Developer step implementations (to be created)

## Migration Information

- Migrated from: {source_path.absolute()}
- Migration date: {Path(__file__).stat().st_mtime if Path(__file__).exists() else 'Unknown'}
- Target developer: {developer_id}

## Next Steps

1. Review migrated data for completeness
2. Update any hardcoded paths in scripts to use the new structure
3. Create step implementations in `src/cursus_dev/steps/` as needed
4. Test workspace functionality with the new structure
"""
        
        readme_path = target_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created README at {readme_path}")

def create_workspace_structure(base_dir: str = "./developer_workspaces", 
                              developers: list = None,
                              dry_run: bool = False):
    """
    Create the complete developer workspace structure.
    
    Args:
        base_dir: Base directory for workspaces
        developers: List of developer IDs to create workspaces for
        dry_run: If True, only show what would be created
    """
    if developers is None:
        developers = ["developer_1", "developer_2", "developer_3"]
    
    base_path = Path(base_dir)
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Creating workspace structure at {base_path}")
    
    # Main structure
    main_dirs = [
        "workspace_manager",
        "shared_resources", 
        "validation_pipeline",
        "integration_staging/staging_areas",
        "integration_staging/validation_results",
        "integration_staging/integration_reports"
    ]
    
    for dir_path in main_dirs:
        full_path = base_path / dir_path
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Creating {full_path}")
        if not dry_run:
            full_path.mkdir(parents=True, exist_ok=True)
    
    # Developer workspaces
    for dev_id in developers:
        dev_path = base_path / "developers" / dev_id
        
        # Developer directory structure
        dev_dirs = [
            "src/cursus_dev/steps/builders",
            "src/cursus_dev/steps/configs", 
            "src/cursus_dev/steps/contracts",
            "src/cursus_dev/steps/specs",
            "src/cursus_dev/steps/scripts",
            "test",
            "validation_reports"
        ]
        
        for dir_path in dev_dirs:
            full_path = dev_path / dir_path
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Creating {full_path}")
            if not dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
        
        # Create developer README
        if not dry_run:
            readme_content = f"""# Developer Workspace: {dev_id}

## Directory Structure

### Step Implementation
- `src/cursus_dev/steps/builders/` - Step builder implementations
- `src/cursus_dev/steps/configs/` - Step configuration files
- `src/cursus_dev/steps/contracts/` - Step contract definitions
- `src/cursus_dev/steps/specs/` - Step specification files
- `src/cursus_dev/steps/scripts/` - Step script implementations

### Testing and Validation
- `test/` - Developer-specific tests
- `validation_reports/` - Validation reports and results

### Data Directories (created as needed)
- `inputs/` - Input data for pipeline steps
- `outputs/` - Output data from pipeline steps
- `local_data/` - Local test data files
- `s3_data/` - Downloaded S3 data for testing
- `synthetic_data/` - Generated synthetic test data
- `logs/` - Execution logs
- `metadata/` - Pipeline execution metadata

## Workspace Isolation Principle

Everything that happens within this developer workspace stays in this workspace.
Only code within `src/cursus/` (the shared core) is shared across all workspaces.

## Usage

1. Implement your step components in the appropriate `src/cursus_dev/steps/` subdirectories
2. Create tests in the `test/` directory
3. Use the workspace-aware runtime validation tools for testing
4. Generate validation reports in `validation_reports/`

## Integration

When ready to integrate your work:
1. Use the integration staging area: `../../integration_staging/`
2. Follow the workspace-to-production pathway
3. Coordinate with other developers through the shared integration process
"""
            
            readme_path = dev_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            logger.info(f"Created developer README at {readme_path}")

def cleanup_old_structure(directories_to_remove: list = None, dry_run: bool = False):
    """
    Clean up old directory structure after successful migration.
    
    Args:
        directories_to_remove: List of directories to remove
        dry_run: If True, only show what would be removed
    """
    if directories_to_remove is None:
        directories_to_remove = [
            "./deployment_validation",
            "./health_check_workspace", 
            "./test_workspace",
            "./workspace"
        ]
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaning up old directory structure")
    
    for dir_path in directories_to_remove:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Removing {path}")
            if not dry_run:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        else:
            logger.debug(f"Directory {path} does not exist, skipping")

def main():
    """Main migration function with CLI interface."""
    parser = argparse.ArgumentParser(description="Migrate pipeline_testing to developer_workspaces structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--source", default="./pipeline_testing", help="Source pipeline_testing directory")
    parser.add_argument("--target-base", default="./developer_workspaces/developers", help="Target base directory")
    parser.add_argument("--developer", default="developer_1", help="Target developer ID")
    parser.add_argument("--create-structure", action="store_true", help="Create complete workspace structure")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old directory structure")
    parser.add_argument("--all", action="store_true", help="Run complete migration (create structure + migrate data + cleanup)")
    
    args = parser.parse_args()
    
    try:
        if args.all:
            logger.info("Running complete migration process...")
            
            # Step 1: Create workspace structure
            logger.info("Step 1: Creating workspace structure")
            create_workspace_structure(dry_run=args.dry_run)
            
            # Step 2: Migrate pipeline_testing data
            logger.info("Step 2: Migrating pipeline_testing data")
            migrate_pipeline_testing_data(
                source_dir=args.source,
                target_base=args.target_base,
                developer_id=args.developer,
                dry_run=args.dry_run
            )
            
            # Step 3: Cleanup old structure (only if not dry run)
            if not args.dry_run:
                logger.info("Step 3: Cleaning up old structure")
                cleanup_old_structure(dry_run=args.dry_run)
            else:
                logger.info("Step 3: Would clean up old structure (skipped in dry run)")
                
        else:
            # Individual operations
            if args.create_structure:
                create_workspace_structure(dry_run=args.dry_run)
            
            if Path(args.source).exists():
                migrate_pipeline_testing_data(
                    source_dir=args.source,
                    target_base=args.target_base,
                    developer_id=args.developer,
                    dry_run=args.dry_run
                )
            
            if args.cleanup:
                cleanup_old_structure(dry_run=args.dry_run)
        
        logger.info("Migration process completed successfully!")
        
        if args.dry_run:
            logger.info("This was a dry run. To actually perform the migration, run without --dry-run")
        else:
            logger.info("Next steps:")
            logger.info("1. Review migrated data for completeness")
            logger.info("2. Update any hardcoded paths in your code")
            logger.info("3. Test the new workspace structure")
            logger.info("4. Remove the old pipeline_testing directory when ready")
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
