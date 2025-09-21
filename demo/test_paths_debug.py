#!/usr/bin/env python3
"""
Debug script to test the absolute paths fix for portable source directory issue.
This script runs the same tests as the notebook to identify the issues.
"""

import os
import sys
from pathlib import Path
import traceback

def main():
    print("=== DEBUGGING PORTABLE PATH RESOLUTION ISSUE ===\n")
    
    # Get current working directory (should be demo/)
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    print(f"Current directory name: {current_dir.name}")
    
    # Add project root to Python path
    project_root = current_dir.parent
    src_path = project_root / 'src'
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"Added to Python path: {src_path}")
    
    print(f"Project root: {project_root}")
    print(f"Project root (absolute): {project_root.resolve()}")
    print()
    
    # Test Current (Problematic) Approach
    print("=== TEST CURRENT (PROBLEMATIC) APPROACH ===")
    current_source_dir = current_dir.parent / 'dockers' / 'xgboost_atoz'
    print(f"Current approach source_dir: {current_source_dir}")
    print(f"Is absolute: {current_source_dir.is_absolute()}")
    print(f"Exists: {current_source_dir.exists()}")
    
    # Check scripts directory
    current_scripts_dir = current_source_dir / 'scripts'
    print(f"Scripts directory: {current_scripts_dir}")
    print(f"Scripts directory exists: {current_scripts_dir.exists()}")
    
    # Check for the specific script
    target_script = current_scripts_dir / 'tabular_preprocessing.py'
    print(f"Target script: {target_script}")
    print(f"Target script exists: {target_script.exists()}")
    print()
    
    # Test Fixed Approach (Absolute Paths)
    print("=== TEST FIXED APPROACH (ABSOLUTE PATHS) ===")
    project_root_abs = current_dir.parent.resolve()
    fixed_source_dir = project_root_abs / 'dockers' / 'xgboost_atoz'
    
    print(f"Fixed approach source_dir: {fixed_source_dir}")
    print(f"Is absolute: {fixed_source_dir.is_absolute()}")
    print(f"Exists: {fixed_source_dir.exists()}")
    
    # Check scripts directory
    fixed_scripts_dir = fixed_source_dir / 'scripts'
    print(f"Scripts directory: {fixed_scripts_dir}")
    print(f"Scripts directory exists: {fixed_scripts_dir.exists()}")
    
    # Check for the specific script
    fixed_target_script = fixed_scripts_dir / 'tabular_preprocessing.py'
    print(f"Target script: {fixed_target_script}")
    print(f"Target script exists: {fixed_target_script.exists()}")
    print()
    
    # Test Config Creation with Absolute Paths
    print("=== TEST CONFIG CREATION WITH ABSOLUTE PATHS ===")
    try:
        # Import the config classes
        from cursus.core.base.config_base import BasePipelineConfig
        from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
        from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
        
        # Create base config with absolute source directory
        base_config = BasePipelineConfig(
            bucket="test-bucket",
            current_date="2025-09-21",
            region="NA",
            aws_region="us-east-1",
            author="test-user",
            role="arn:aws:iam::123456789012:role/TestRole",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            source_dir=str(fixed_source_dir)  # Use absolute path
        )
        
        print(f"✅ Base config created successfully")
        print(f"Base config source_dir: {base_config.source_dir}")
        print(f"Base config portable_source_dir: {base_config.portable_source_dir}")
        
        # Create processing config with absolute paths
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir=str(fixed_scripts_dir),  # Use absolute path
            processing_instance_type_large='ml.m5.12xlarge',
            processing_instance_type_small='ml.m5.4xlarge'
        )
        
        print(f"✅ Processing config created successfully")
        print(f"Processing config source_dir: {processing_config.processing_source_dir}")
        print(f"Processing config portable_processing_source_dir: {processing_config.portable_processing_source_dir}")
        
        # Create tabular preprocessing config
        tabular_config = TabularPreprocessingConfig.from_base_config(
            processing_config,
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        print(f"✅ Tabular config created successfully")
        print(f"Tabular config script_path: {tabular_config.script_path}")
        print(f"Tabular config full_script_path: {tabular_config.full_script_path}")
        
    except Exception as e:
        print(f"❌ Error creating configs: {e}")
        traceback.print_exc()
        return
    
    print()
    
    # Verify Path Resolution
    print("=== VERIFY PATH RESOLUTION ===")
    script_path = Path(tabular_config.script_path)
    print(f"Script path: {script_path}")
    print(f"Script path is absolute: {script_path.is_absolute()}")
    print(f"Script path exists: {script_path.exists()}")
    
    # If it's a relative path, try to resolve it from current directory
    if not script_path.is_absolute():
        resolved_from_current = current_dir / script_path
        print(f"Resolved from current dir: {resolved_from_current}")
        print(f"Resolved from current dir exists: {resolved_from_current.exists()}")
        
        resolved_from_project_root = project_root_abs / script_path
        print(f"Resolved from project root: {resolved_from_project_root}")
        print(f"Resolved from project root exists: {resolved_from_project_root.exists()}")
    
    print()
    
    # Test Step Builder Creation
    print("=== TEST STEP BUILDER CREATION ===")
    try:
        # Import step builder
        from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        
        # Create step builder with the config
        step_builder = TabularPreprocessingStepBuilder(config=tabular_config)
        print("✅ Step builder created successfully!")
        
        # Test script path resolution
        builder_script_path = step_builder.config.get_portable_script_path()
        print(f"Builder script path: {builder_script_path}")
        
        # Test if the path exists
        if builder_script_path:
            path_obj = Path(builder_script_path)
            if path_obj.is_absolute():
                print(f"Absolute path exists: {path_obj.exists()}")
            else:
                # Try resolving from different base directories
                from_current = current_dir / builder_script_path
                from_project = project_root_abs / builder_script_path
                print(f"Relative path from current dir exists: {from_current.exists()}")
                print(f"Relative path from project root exists: {from_project.exists()}")
                
    except Exception as e:
        print(f"❌ Error creating step builder: {e}")
        traceback.print_exc()
    
    print()
    
    # Test Different Relative Path Calculations
    print("=== TESTING DIFFERENT RELATIVE PATH CALCULATIONS ===")
    
    # Test what the correct relative path should be from different locations
    test_locations = {
        "demo/": current_dir,
        "project_root/": project_root_abs,
        "src/": project_root_abs / "src",
        "src/cursus/": project_root_abs / "src" / "cursus",
        "src/cursus/steps/": project_root_abs / "src" / "cursus" / "steps",
        "src/cursus/steps/configs/": project_root_abs / "src" / "cursus" / "steps" / "configs",
        "src/cursus/steps/builders/": project_root_abs / "src" / "cursus" / "steps" / "builders",
    }
    
    target_absolute = fixed_target_script
    print(f"Target script (absolute): {target_absolute}")
    print(f"Target exists: {target_absolute.exists()}")
    print()
    
    for location_name, location_path in test_locations.items():
        print(f"--- From {location_name} ---")
        print(f"Location: {location_path}")
        
        try:
            # Calculate relative path from this location to target
            relative_path = target_absolute.relative_to(location_path)
            print(f"Direct relative_to: {relative_path}")
            
            # Test if this relative path works
            resolved_path = location_path / relative_path
            print(f"Resolves to: {resolved_path}")
            print(f"Resolved path exists: {resolved_path.exists()}")
            
        except ValueError:
            # If direct relative_to fails, calculate using common parent
            try:
                # Find common parent
                common_parts = []
                loc_parts = location_path.parts
                target_parts = target_absolute.parts
                
                for p1, p2 in zip(loc_parts, target_parts):
                    if p1 == p2:
                        common_parts.append(p1)
                    else:
                        break
                
                if common_parts:
                    common_parent = Path(*common_parts)
                    
                    # Calculate relative path via common parent
                    loc_to_common = location_path.relative_to(common_parent)
                    target_to_common = target_absolute.relative_to(common_parent)
                    
                    up_levels = len(loc_to_common.parts)
                    relative_parts = ['..'] * up_levels + list(target_to_common.parts)
                    calculated_relative = str(Path(*relative_parts))
                    
                    print(f"Common parent: {common_parent}")
                    print(f"Calculated relative: {calculated_relative}")
                    
                    # Test if this relative path works
                    resolved_path = location_path / calculated_relative
                    print(f"Resolves to: {resolved_path}")
                    print(f"Resolved path exists: {resolved_path.exists()}")
                else:
                    print("No common parent found")
                    
            except Exception as e:
                print(f"Error calculating relative path: {e}")
        
        print()
    
    # Test the current generated path from different locations
    print("=== TESTING CURRENT GENERATED PATH ===")
    current_generated_path = "../../../../dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
    print(f"Current generated path: {current_generated_path}")
    print()
    
    for location_name, location_path in test_locations.items():
        resolved = location_path / current_generated_path
        print(f"From {location_name}: {resolved}")
        print(f"  Exists: {resolved.exists()}")
        print(f"  Resolved: {resolved.resolve()}")
        print()
    
    # Summary and Recommendations
    print("=== SUMMARY AND RECOMMENDATIONS ===")
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root_abs}")
    print(f"Target script exists: {fixed_target_script.exists()}")
    print(f"Current generated relative path: {current_generated_path}")
    
    # Find which location the current path works from
    working_locations = []
    for location_name, location_path in test_locations.items():
        resolved = location_path / current_generated_path
        if resolved.exists():
            working_locations.append(location_name)
    
    if working_locations:
        print(f"\n✅ Current path works from: {', '.join(working_locations)}")
    else:
        print(f"\n❌ Current path doesn't work from any tested location")
        
    print(f"\nThe path calculation should be relative to the execution context (likely demo/ or project_root/)")

if __name__ == "__main__":
    main()
