# Pipeline Configuration UI Demo - DAG-Driven Approach

This notebook demonstrates the **DAG-driven configuration UI workflow** that replicates the exact pattern from `demo_config.ipynb`:

1. **Load Pipeline DAG** - Import the complete XGBoost E2E pipeline DAG
2. **Analyze DAG Structure** - Discover required configuration classes
3. **Interactive Configuration** - Use native widgets to collect user input for each required config
4. **Build Config List** - Assemble all configurations into a list
5. **Merge and Save** - Call `merge_and_save_configs()` to create unified JSON

## Key Features

- **🎯 DAG-Driven Discovery**: Only shows configurations needed for the specific pipeline
- **🔄 Progressive Configuration**: Base → Processing → Step-specific configs
- **📋 Enhanced Clipboard Support**: Direct Ctrl+V pasting into individual fields
- **💾 Unified Export**: Same `merge_and_save_configs()` workflow as demo_config.ipynb
- **🌍 Universal Compatibility**: Works in SageMaker, local Jupyter, JupyterLab, etc.

---

## Step 1: Environment Setup and DAG Loading

Import required modules and load the pipeline DAG:

```python
# Setup imports for universal environment - uses existing robust infrastructure
import os
from pathlib import Path
from datetime import datetime

# Environment detection
def detect_environment():
    """Detect if running in SageMaker environment."""
    sagemaker_indicators = [
        '/opt/ml' in str(Path.cwd()),
        'SageMaker' in os.environ.get('AWS_EXECUTION_ENV', ''),
        os.path.exists('/opt/ml/code'),
        'sagemaker' in str(Path.home()).lower()
    ]
    return any(sagemaker_indicators)

IS_SAGEMAKER = detect_environment()

print(f"🔍 Environment Detection:")
print(f"   • SageMaker Environment: {'✅ Yes' if IS_SAGEMAKER else '❌ No'}")
print(f"   • Current Directory: {Path.cwd()}")
print(f"   • Home Directory: {Path.home()}")

# Use existing robust infrastructure for imports and discovery
print("\n🎯 Initializing cursus configuration system...")
try:
    # Import the existing robust core system
    from cursus.api.config_ui.core.core import UniversalConfigCore
    from cursus.api.config_ui.widgets.native import create_native_config_widget
    
    # Initialize with robust discovery and error handling
    config_core = UniversalConfigCore()
    
    # Test the discovery system
    config_classes = config_core.discover_config_classes()
    
    print("✅ SUCCESS: Cursus configuration system initialized")
    print(f"   • Discovered {len(config_classes)} configuration classes")
    print(f"   • Using robust StepCatalog-based discovery")
    print("   💡 This approach handles all deployment scenarios automatically")
    
except ImportError as e:
    print(f"❌ FAILED: Could not import cursus configuration system")
    print(f"   Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Ensure cursus package is installed: pip install .")
    print("   2. Check that you're in the correct environment")
    print("   3. Verify cursus package installation")
    raise ImportError("Could not initialize cursus configuration system")

print(f"\n🎉 Setup Complete!")
print(f"   • Environment: {'SageMaker' if IS_SAGEMAKER else 'Local/Other'}")
print(f"   • Configuration classes available: {len(config_classes)}")
print(f"   • Ready for DAG-driven Config UI!")
```

```python
# Import all required modules
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.core.config_fields import merge_and_save_configs

# Import the complete E2E DAG
from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag

# Import native widgets
from cursus.api.config_ui.widgets.native import create_native_config_widget

# Import step catalog for configuration discovery
from cursus.step_catalog.step_catalog import StepCatalog
from cursus.step_catalog.adapters.config_resolver import StepConfigResolverAdapter

print("✅ All modules imported successfully!")
print("🎯 Ready for DAG-driven configuration workflow!")
```

---

## Step 2: Load and Analyze Pipeline DAG

Load the complete XGBoost E2E pipeline DAG and analyze its structure:

```python
# Load the complete XGBoost E2E pipeline DAG
pipeline_dag = create_xgboost_complete_e2e_dag()

print("📊 Pipeline DAG Analysis:")
print(f"   • Total Nodes: {len(pipeline_dag.nodes)}")
print(f"   • Total Edges: {len(pipeline_dag.edges)}")
print("\n🔍 Discovered Pipeline Steps:")
for i, node in enumerate(pipeline_dag.nodes, 1):
    print(f"   {i}. {node}")

print("\n🔗 Pipeline Dependencies:")
for edge in pipeline_dag.edges:
    print(f"   • {edge[0]} → {edge[1]}")
```

```python
# Initialize step catalog and resolver for configuration discovery
step_catalog = StepCatalog()
config_resolver = StepConfigResolverAdapter()

# Discover required configuration classes based on DAG nodes
print("🔍 Discovering Required Configuration Classes...")
print("="*50)

required_configs = []
node_to_config_map = {
    "CradleDataLoading_training": "CradleDataLoadConfig",
    "CradleDataLoading_calibration": "CradleDataLoadConfig", 
    "TabularPreprocessing_training": "TabularPreprocessingConfig",
    "TabularPreprocessing_calibration": "TabularPreprocessingConfig",
    "XGBoostTraining": "XGBoostTrainingConfig",
    "XGBoostModelEval_calibration": "XGBoostModelEvalConfig",
    "ModelCalibration_calibration": "ModelCalibrationConfig",
    "Package": "PackageConfig",
    "Payload": "PayloadConfig",
    "Registration": "RegistrationConfig"
}

# Build list of unique required configurations
unique_configs = set()
for node in pipeline_dag.nodes:
    if node in node_to_config_map:
        config_class_name = node_to_config_map[node]
        if config_class_name not in unique_configs:
            unique_configs.add(config_class_name)
            required_configs.append({
                "config_class_name": config_class_name,
                "example_node": node,
                "is_specialized": config_class_name == "CradleDataLoadConfig"
            })

print(f"⚙️ Required Configurations ({len(required_configs)} unique types):")
for i, config in enumerate(required_configs, 1):
    specialized = " (Specialized UI)" if config["is_specialized"] else ""
    print(f"   {i}. ✅ {config['config_class_name']}{specialized}")

# Calculate hidden configs (total available - required)
all_config_classes = step_catalog.discover_config_classes()
hidden_count = len(all_config_classes) - len(required_configs)
print(f"\n❌ Hidden: {hidden_count} other config types not needed for this pipeline")

print(f"\n📋 Configuration Workflow:")
print(f"   Base Config → Processing Config → {len(required_configs)} Specific Configs")
```

---

## Step 3: Initialize Configuration List

Initialize the configuration list that will be populated through the UI workflow:

```python
# Initialize the config list - this will be populated as we go through each configuration step
config_list = []

print("📋 Configuration List Initialized")
print(f"   • Current count: {len(config_list)} configurations")
print(f"   • Target count: {len(required_configs) + 2} configurations (Base + Processing + {len(required_configs)} specific)")
print("\n💡 This list will be populated as you complete each configuration step below.")
```

---

## Step 4: Base Pipeline Configuration

Configure the base pipeline settings that are shared across all steps:

```python
# Create Base Pipeline Configuration Widget
print("🏗️ Step 1: Base Pipeline Configuration")
print("="*40)
print("\n📋 This configuration is shared across ALL pipeline steps.")
print("🔥 Essential fields (marked with *) must be filled by user.")
print("⚙️ System fields have defaults but can be customized.")
print("\n💡 Enhanced Clipboard Support:")
print("   • Copy any text with Ctrl+C")
print("   • Click in any field below")
print("   • Paste with Ctrl+V - text appears instantly!")

base_config_widget = create_native_config_widget("BasePipelineConfig")
base_config_widget.display()
```

```python
# Save Base Configuration to Config List
base_config_data = base_config_widget.get_config()

if base_config_data:
    # Create BasePipelineConfig instance
    base_config = BasePipelineConfig(**base_config_data)
    config_list.append(base_config)
    
    print("✅ Base Configuration Saved!")
    print(f"   • Configuration fields: {len(base_config_data)}")
    print(f"   • Config list count: {len(config_list)}")
    print(f"   • Author: {base_config.author}")
    print(f"   • Service: {base_config.service_name}")
    print(f"   • Region: {base_config.region}")
else:
    print("⚠️ Base configuration not saved yet.")
    print("💡 Please fill out the base configuration form above and click 'Save Configuration'.")
```

---

## Step 5: Processing Step Configuration

Configure processing-specific settings that are shared across processing steps:

```python
# Create Processing Step Configuration Widget (inherits from Base Config)
print("⚙️ Step 2: Processing Step Configuration")
print("="*40)
print("\n📋 This configuration is shared across all PROCESSING steps.")
print("💾 Inherits all values from Base Configuration automatically.")
print("🎯 Only processing-specific fields need to be configured.")

if len(config_list) > 0 and isinstance(config_list[0], BasePipelineConfig):
    # Use base config for inheritance
    processing_config_widget = create_native_config_widget(
        "ProcessingStepConfigBase", 
        base_config=config_list[0]
    )
    processing_config_widget.display()
else:
    print("❌ Base configuration required first!")
    print("💡 Please complete the Base Configuration step above before proceeding.")
```

```python
# Save Processing Configuration to Config List
if len(config_list) > 0:
    processing_config_data = processing_config_widget.get_config()
    
    if processing_config_data:
        # Create ProcessingStepConfigBase instance using inheritance
        processing_config = ProcessingStepConfigBase.from_base_config(
            config_list[0],  # base_config
            **processing_config_data
        )
        config_list.append(processing_config)
        
        print("✅ Processing Configuration Saved!")
        print(f"   • Configuration fields: {len(processing_config_data)}")
        print(f"   • Config list count: {len(config_list)}")
        print(f"   • Instance type: {processing_config.processing_instance_type_large}")
        print(f"   • Source dir: {processing_config.processing_source_dir}")
    else:
        print("⚠️ Processing configuration not saved yet.")
        print("💡 Please fill out the processing configuration form above and click 'Save Configuration'.")
else:
    print("❌ Base configuration required first!")
```

---

## Step 6: Step-Specific Configurations

Configure each step-specific configuration discovered from the DAG:

```python
# Display configuration progress
print("🎯 Step-Specific Configuration Progress")
print("="*40)
print(f"\n📊 Current Status:")
print(f"   • Base Config: {'✅ Complete' if len(config_list) >= 1 else '❌ Pending'}")
print(f"   • Processing Config: {'✅ Complete' if len(config_list) >= 2 else '❌ Pending'}")
print(f"   • Step-Specific Configs: {max(0, len(config_list) - 2)}/{len(required_configs)} complete")

if len(config_list) >= 2:
    print("\n🚀 Ready for step-specific configurations!")
    print("💡 Each configuration below inherits from Base + Processing configs automatically.")
else:
    print("\n⚠️ Complete Base and Processing configurations first.")
    print("💡 Step-specific configurations will appear once prerequisites are met.")
```

### 6.1 TabularPreprocessingConfig

```python
# TabularPreprocessingConfig
print("📊 TabularPreprocessingConfig")
print("="*30)

if len(config_list) >= 2:
    tabular_preprocessing_widget = create_native_config_widget(
        "TabularPreprocessingConfig",
        base_config=config_list[1]  # ProcessingStepConfigBase
    )
    
    print("\n💾 Inherits from Base + Processing configurations automatically.")
    print("🎯 Configure tabular preprocessing specific settings:")
    
    tabular_preprocessing_widget.display()
else:
    print("❌ Base and Processing configurations required first!")
```

```python
# Save TabularPreprocessingConfig
if len(config_list) >= 2:
    tabular_config_data = tabular_preprocessing_widget.get_config()
    
    if tabular_config_data:
        from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
        
        tabular_config = TabularPreprocessingConfig.from_base_config(
            config_list[1],  # processing_step_config
            **tabular_config_data
        )
        config_list.append(tabular_config)
        
        print("✅ TabularPreprocessingConfig saved!")
        print(f"   • Job type: {tabular_config.job_type}")
        print(f"   • Label name: {tabular_config.label_name}")
        print(f"   • Config list count: {len(config_list)}")
    else:
        print("⚠️ TabularPreprocessingConfig not saved yet.")
else:
    print("❌ Prerequisites not met.")
```

### 6.2 XGBoostTrainingConfig

```python
# XGBoostTrainingConfig
print("🚀 XGBoostTrainingConfig")
print("="*25)

if len(config_list) >= 1:
    xgboost_training_widget = create_native_config_widget(
        "XGBoostTrainingConfig",
        base_config=config_list[0]  # BasePipelineConfig
    )
    
    print("\n💾 Inherits from Base configuration automatically.")
    print("🎯 Configure XGBoost training specific settings:")
    
    xgboost_training_widget.display()
else:
    print("❌ Base configuration required first!")
```

```python
# Save XGBoostTrainingConfig
if len(config_list) >= 1:
    xgboost_training_data = xgboost_training_widget.get_config()
    
    if xgboost_training_data:
        from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
        
        xgboost_training_config = XGBoostTrainingConfig.from_base_config(
            config_list[0],  # base_config
            **xgboost_training_data
        )
        config_list.append(xgboost_training_config)
        
        print("✅ XGBoostTrainingConfig saved!")
        print(f"   • Training instance: {xgboost_training_config.training_instance_type}")
        print(f"   • Entry point: {xgboost_training_config.training_entry_point}")
        print(f"   • Config list count: {len(config_list)}")
    else:
        print("⚠️ XGBoostTrainingConfig not saved yet.")
else:
    print("❌ Base configuration required first!")
```

---

## Step 7: Final Configuration Summary

Review all completed configurations before merging:

```python
# Display final configuration summary
print("📋 Final Configuration Summary")
print("="*40)
print(f"\n✅ Configuration Complete - {len(config_list)} configurations ready!")
print("\n📊 Configuration List:")
for i, config in enumerate(config_list, 1):
    config_name = config.__class__.__name__
    print(f"   {i}. {config_name}")

print(f"\n🎯 Ready for merge_and_save_configs() - Same workflow as demo_config.ipynb!")
print(f"💡 This will create the unified hierarchical JSON structure.")
```

---

## Step 8: Merge and Save Configurations

Use the same `merge_and_save_configs()` function as in demo_config.ipynb:

```python
# Merge and Save Config List - Exact same workflow as demo_config.ipynb
print("💾 Merge and Save Configurations")
print("="*35)

if len(config_list) >= 3:  # At least Base + Processing + 1 specific config
    # Create output directory
    current_dir = Path.cwd()
    config_dir = current_dir / 'pipeline_config' / 'config_demo_xgboost_pipeline_v2'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output filename
    config_file_name = 'config_demo_xgboost_pipeline.json'
    config_file_path = str(config_dir / config_file_name)
    
    print(f"📁 Output directory: {config_dir}")
    print(f"📄 Output filename: {config_file_name}")
    print(f"\n🔄 Calling merge_and_save_configs()...")
    
    # Call the same merge_and_save_configs function as demo_config.ipynb
    merged_config = merge_and_save_configs(config_list, config_file_path)
    
    print(f"\n✅ Configuration merge completed successfully!")
    print(f"📄 Saved to: {config_file_path}")
    print(f"📊 Merged {len(config_list)} configurations into unified JSON")
    
    # Display structure preview
    if merged_config:
        print(f"\n📋 Generated Structure:")
        print(f"   • Shared fields: {len(merged_config.get('shared', {}))} fields")
        print(f"   • Processing shared: {len(merged_config.get('processing_shared', {}))} fields")
        print(f"   • Specific configs: {len(merged_config.get('specific', {}))} config types")
        print(f"   • Step list: {len(merged_config.get('step_list', []))} steps")
        print(f"   • Inverted index: {len(merged_config.get('inverted_index', {}))} field mappings")
        
else:
    print("⚠️ Insufficient configurations for merge.")
    print(f"💡 Current count: {len(config_list)}, minimum required: 3")
    print("🔄 Please complete more configuration steps above.")
```

---

## Step 9: Verification and Next Steps

Verify the generated configuration and provide next steps:

```python
# Verification and Next Steps
print("🎉 DAG-Driven Configuration UI Demo Complete!")
print("="*50)

if len(config_list) >= 3:
    print("\n✅ Successfully demonstrated the complete workflow:")
    print("   1. ✅ Loaded Pipeline DAG (XGBoost Complete E2E)")
    print("   2. ✅ Analyzed DAG structure and discovered required configs")
    print("   3. ✅ Used native widgets for interactive configuration")
    print("   4. ✅ Built configuration list with inheritance")
    print("   5. ✅ Called merge_and_save_configs() - same as demo_config.ipynb")
    
    print(f"\n📊 Results:")
    print(f"   • Total configurations: {len(config_list)}")
    print(f"   • DAG nodes analyzed: {len(pipeline_dag.nodes)}")
    print(f"   • Required config types: {len(required_configs)}")
    print(f"   • Hidden config types: {hidden_count}")
    
    print(f"\n🎯 Key Benefits Demonstrated:")
    print(f"   • 🎯 DAG-Driven Discovery: Only relevant configs shown")
    print(f"   • 🔄 Progressive Configuration: Base → Processing → Specific")
    print(f"   • 📋 Enhanced Clipboard Support: Direct Ctrl+V pasting")
    print(f"   • 💾 Unified Export: Same merge_and_save_configs() workflow")
    print(f"   • 🌍 Universal Compatibility: Works in any Jupyter environment")
    
    print(f"\n📁 Generated Files:")
    config_files = list(Path.cwd().glob("pipeline_config/**/config_*.json"))
    for config_file in config_files[-3:]:  # Show last 3 files
        print(f"   📄 {config_file}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Copy this notebook to your SageMaker environment")
    print(f"   2. Run the setup cells to detect environment and import modules")
    print(f"   3. Execute the DAG analysis to see your pipeline's required configs")
    print(f"   4. Use the native widgets to configure each step interactively")
    print(f"   5. Call merge_and_save_configs() to generate unified JSON")
    print(f"   6. Use the generated config file for pipeline execution")
    
else:
    print("\n⚠️ Demo partially complete - some configurations still pending.")
    print("💡 Run the configuration cells above to complete the full workflow.")

print(f"\n🎉 DAG-driven configuration UI successfully replicated demo_config.ipynb workflow!")
```

---

## Summary

This markdown file provides all the code blocks and text sections needed to create the DAG-driven configuration UI demo notebook. Simply copy each code block into code cells and each text section into markdown cells in your Jupyter notebook.

The workflow demonstrates:
1. **DAG-driven configuration discovery** - Only shows relevant configs
2. **Progressive configuration** - Base → Processing → Step-specific
3. **Native widget integration** - Enhanced clipboard support
4. **Same merge_and_save_configs() workflow** - Exact same function call as demo_config.ipynb
5. **Universal compatibility** - Works in SageMaker and local environments
