"""
Example usage of the Cradle Config Widget in a Jupyter notebook

This script demonstrates how to use the widget with the new JSON file workflow
to replace manual configuration blocks.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

from cursus.core.base.config_base import BasePipelineConfig
from cursus.api.cradle_ui.jupyter_widget import create_cradle_config_widget
from cursus.api.cradle_ui.utils.config_loader import load_cradle_config_from_json


def main():
    """Example usage of the Cradle Config Widget with JSON file workflow."""
    
    # Create base configuration
    base_config = BasePipelineConfig(
        bucket="my-pipeline-bucket",
        current_date="2025-10-06",
        region="NA",
        aws_region="us-east-1",
        author="demo-user",
        role="arn:aws:iam::123456789012:role/demo-role",
        service_name="DemoService",
        pipeline_version="1.0.0",
        project_root_folder="demo-project",
        framework_version="2.1.0",
        py_version="py310",
        source_dir="/path/to/source"
    )
    
    # Initialize config list
    config_list = [base_config]
    
    print("🎯 Cradle Config Widget Example - JSON File Workflow")
    print("=" * 60)
    
    # Create training configuration widget
    print("\n1. Creating Training Configuration Widget...")
    training_widget = create_cradle_config_widget(
        base_config=base_config,
        job_type="training",
        height="700px"
    )
    
    print("✅ Training widget created successfully!")
    print("   In a Jupyter notebook, you would call: training_widget.display()")
    
    # Create calibration configuration widget
    print("\n2. Creating Calibration Configuration Widget...")
    calibration_widget = create_cradle_config_widget(
        base_config=base_config,
        job_type="calibration",
        height="700px"
    )
    
    print("✅ Calibration widget created successfully!")
    print("   In a Jupyter notebook, you would call: calibration_widget.display()")
    
    # Show the new JSON file workflow
    print("\n📝 New JSON File Workflow:")
    print("=" * 60)
    print("1. Display the widget in your notebook:")
    print("   training_widget.display()")
    print()
    print("2. Complete the 4-step configuration in the UI")
    print("3. Click 'Finish' in the UI to generate the configuration")
    print("4. Click 'Get Configuration' button to save to JSON file")
    print("5. Load the configuration from the saved JSON file:")
    print("   config = load_cradle_config_from_json('cradle_config_training.json')")
    print("   config_list.append(config)")
    
    # Show example of loading from JSON
    print("\n🔧 Example JSON Loading Code:")
    print("=" * 60)
    print("""
# After saving the configuration to JSON file:
from cursus.api.cradle_ui.utils.config_loader import load_cradle_config_from_json

# Load the configuration (handles all nested objects properly)
training_config = load_cradle_config_from_json('cradle_config_training.json')
config_list.append(training_config)

# The config object is identical to manual configuration!
print(f"Job Type: {training_config.job_type}")
print(f"Data Sources: {len(training_config.data_sources_spec.data_sources)}")
""")
    
    print("\n🎉 Benefits of JSON File Workflow:")
    print("=" * 60)
    print("✅ Reliable: No complex iframe communication issues")
    print("✅ Transparent: Users can see and modify the saved configuration")
    print("✅ Reusable: Save configurations for later use or sharing")
    print("✅ Debuggable: Easy to inspect and troubleshoot configurations")
    print("✅ Compatible: Works in all Jupyter environments")
    print("✅ Same Result: Identical CradleDataLoadingConfig objects as manual approach")


# Example notebook cells for the new workflow
NOTEBOOK_CELL_WIDGET = '''
### **REQUIRED: [STEP 1.0] Cradle Data Loading Config** - Interactive UI Version

# Import the widget
from cursus.api.cradle_ui.jupyter_widget import create_cradle_config_widget

# Create and display the training configuration widget
print("🎯 Training Data Configuration")
training_cradle_widget = create_cradle_config_widget(
    base_config=base_config,
    job_type="training",
    height="700px"
)
training_cradle_widget.display()

print("📝 Instructions:")
print("1. Complete the 4-step configuration in the UI above")
print("2. Click 'Finish' to generate the configuration")
print("3. Click 'Get Configuration' to save to JSON file")
print("4. Run the next cell to load the configuration")
'''

NOTEBOOK_CELL_LOAD_CONFIG = '''
# Load the training configuration from JSON file
from cursus.api.cradle_ui.utils.config_loader import load_cradle_config_from_json

# Update this path to match where you saved your configuration
config_file_path = 'cradle_config_training.json'  # Update this path!

try:
    # Load the configuration (properly handles all nested objects)
    training_cradle_data_load_config = load_cradle_config_from_json(config_file_path)
    
    print("✅ Training configuration loaded successfully!")
    print(f"Job Type: {training_cradle_data_load_config.job_type}")
    print(f"Data Sources: {len(training_cradle_data_load_config.data_sources_spec.data_sources)}")
    
    # Add to config list (same as manual configuration)
    config_list.append(training_cradle_data_load_config)
    print(f"✅ Added to config_list. Total configs: {len(config_list)}")
    
except FileNotFoundError:
    print("⚠️ Configuration file not found.")
    print("Please complete the UI configuration and save the JSON file first.")
except Exception as e:
    print(f"❌ Error loading configuration: {str(e)}")
'''

NOTEBOOK_CELL_CALIBRATION = '''
# Create and display the calibration configuration widget
print("🎯 Calibration Data Configuration")
calibration_cradle_widget = create_cradle_config_widget(
    base_config=base_config,
    job_type="calibration",
    height="700px"
)
calibration_cradle_widget.display()
'''

NOTEBOOK_CELL_LOAD_CALIBRATION = '''
# Load the calibration configuration from JSON file
try:
    # Update this path to match where you saved your calibration configuration
    calibration_config_file_path = 'cradle_config_calibration.json'  # Update this path!
    
    calibration_cradle_data_load_config = load_cradle_config_from_json(calibration_config_file_path)
    
    print("✅ Calibration configuration loaded successfully!")
    config_list.append(calibration_cradle_data_load_config)
    print(f"✅ Added to config_list. Total configs: {len(config_list)}")
    
except FileNotFoundError:
    print("⚠️ Calibration configuration file not found.")
    print("Please complete the UI configuration and save the JSON file first.")
except Exception as e:
    print(f"❌ Error loading calibration configuration: {str(e)}")
'''

# Comparison with manual approach
COMPARISON_EXAMPLE = """
## 🔄 Comparison: Manual vs Widget Approach

### BEFORE (Manual - Complex):
```python
training_cradle_data_load_config = create_cradle_data_load_config(
    base_config=base_config,
    job_type='training',
    mds_field_list=mds_field_list,
    start_date=training_start_datetime,
    end_date=training_end_datetime,
    service_name=service_name,
    tag_edx_provider=tag_edx_provider,
    tag_edx_subject=tag_edx_subject,
    tag_edx_dataset=tag_edx_dataset,
    etl_job_id=etl_job_id,
    cradle_account=cradle_account,
    org_id=org_id,
    edx_manifest_comment=edx_manifest_comment,
    cluster_type=cluster_type,
    output_format=output_format,
    output_save_mode="ERRORIFEXISTS",
    use_dedup_sql=True,
    tag_schema=tag_schema,
    mds_join_key='objectId',
    edx_join_key='order_id',
    join_type='JOIN'
    # ... 20+ parameters total!
)
```

### AFTER (Widget - Simple):
```python
# Step 1: Create and display widget
widget = create_cradle_config_widget(base_config=base_config, job_type="training")
widget.display()

# Step 2: Complete UI, click "Get Configuration", save JSON file

# Step 3: Load configuration
config = load_cradle_config_from_json('cradle_config_training.json')
config_list.append(config)
```

### Result: Identical CradleDataLoadingConfig objects, but much easier to create!
"""

# Usage instructions
USAGE_INSTRUCTIONS = """
## 📋 How to Replace Manual Configuration with Widget

### Step 1: Start the UI server (if not already running)
```bash
cd src/cursus/api/cradle_ui
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### Step 2: Replace manual configuration blocks
In your demo_config.ipynb, find sections like:
```
### **REQUIRED: [STEP 1.0] Cradle Data Loading Config**
```

Replace the entire manual configuration block with the widget code.

### Step 3: Use the new JSON file workflow
1. Run the cell with the widget creation code
2. Complete the 4-step configuration in the embedded UI
3. Click "Finish" in the UI to generate the configuration
4. Click "Get Configuration" button to save to JSON file
5. Run the cell to load the configuration from JSON file
6. The config object is automatically added to your config_list

### Benefits:
✅ **User-friendly**: Visual interface instead of 20+ manual parameters
✅ **Error prevention**: Built-in validation and error checking  
✅ **Reliable**: File-based approach works in all environments
✅ **Reusable**: Save and share configurations easily
✅ **Identical result**: Same CradleDataLoadingConfig objects as manual approach
✅ **Debuggable**: Easy to inspect and modify saved configurations
"""


if __name__ == "__main__":
    main()
    print("\n" + COMPARISON_EXAMPLE)
    print(USAGE_INSTRUCTIONS)
