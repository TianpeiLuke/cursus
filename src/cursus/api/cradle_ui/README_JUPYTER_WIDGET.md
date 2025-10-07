# Cradle Data Load Config - Jupyter Widget

This directory contains a Jupyter notebook widget that provides an interactive UI for configuring Cradle data loading directly within Jupyter notebooks.

## 🎯 Purpose

Replace complex manual configuration blocks in notebooks (like `demo_config.ipynb`) with an intuitive, visual interface that generates the same `CradleDataLoadConfig` objects.

## 📁 Files

- `jupyter_widget.py` - Main widget implementation
- `example_notebook_usage.py` - Usage examples and instructions
- `README_JUPYTER_WIDGET.md` - This documentation

## 🚀 Quick Start

### 1. Start the UI Server

```bash
cd src/cursus/api/cradle_ui
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Use in Jupyter Notebook

Replace your manual configuration block with:

```python
# Import the widget
from cursus.api.cradle_ui.jupyter_widget import create_cradle_config_widget

# Create and display the widget
cradle_widget = create_cradle_config_widget(
    base_config=base_config,
    job_type="training"
)
cradle_widget.display()
```

### 3. Get the Configuration

After completing the UI configuration:

```python
# Get the generated config object
training_cradle_data_load_config = cradle_widget.get_config()

# Add to your config list (same as before)
config_list.append(training_cradle_data_load_config)
```

## 🔄 Replacing Manual Configuration

### Before (Manual Configuration)

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
)
```

### After (Widget Configuration)

```python
# Much simpler!
cradle_widget = create_cradle_config_widget(base_config=base_config, job_type="training")
cradle_widget.display()

# After UI completion:
training_cradle_data_load_config = cradle_widget.get_config()
config_list.append(training_cradle_data_load_config)
```

## 🎨 Features

### ✅ **User-Friendly Interface**
- 4-step guided wizard
- Visual form fields with validation
- Pre-filled defaults based on best practices

### ✅ **Error Prevention**
- Built-in validation and error checking
- Real-time feedback on configuration issues
- Prevents invalid configurations

### ✅ **Seamless Integration**
- Works with existing notebook workflows
- Same output as manual configuration
- Compatible with `config_list.append()` pattern

### ✅ **Flexible Configuration**
- Support for all job types (training, validation, testing, calibration)
- Dynamic data source configuration (MDS, EDX, ANDES)
- Customizable output formats and settings

## 📋 Widget API

### `create_cradle_config_widget()`

```python
def create_cradle_config_widget(
    base_config=None,           # Base pipeline configuration
    job_type: str = "training", # Job type: training, validation, testing, calibration
    width: str = "100%",        # Widget width
    height: str = "800px",      # Widget height
    server_port: int = 8001     # UI server port
) -> CradleConfigWidget
```

### Widget Methods

- `display()` - Display the widget in the notebook
- `get_config()` - Get the generated CradleDataLoadConfig object
- `_handle_config_result()` - Internal method for processing UI results

## 🔧 Advanced Usage

### Multiple Configurations

```python
# Training configuration
training_widget = create_cradle_config_widget(base_config=base_config, job_type="training")
training_widget.display()

# Calibration configuration  
calibration_widget = create_cradle_config_widget(base_config=base_config, job_type="calibration")
calibration_widget.display()

# Get both configs
training_config = training_widget.get_config()
calibration_config = calibration_widget.get_config()

config_list.extend([training_config, calibration_config])
```

### Custom Widget Size

```python
# Larger widget for better visibility
widget = create_cradle_config_widget(
    base_config=base_config,
    job_type="training",
    height="1000px"
)
widget.display()
```

## 🛠️ Requirements

- `ipywidgets` - For Jupyter widget functionality
- `requests` - For API communication
- Running Cradle UI server on specified port

## 🐛 Troubleshooting

### Server Not Running
If you see "Server Not Available", start the server:
```bash
uvicorn cursus.api.cradle_ui.app:app --host 0.0.0.0 --port 8001 --reload
```

### Widget Not Displaying
Ensure you have `ipywidgets` installed and enabled:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Configuration Not Generated
1. Complete all 4 steps in the UI
2. Click "Finish" button
3. Wait for success message
4. Then call `widget.get_config()`

## 📝 Example Notebook Integration

See `example_notebook_usage.py` for complete examples of how to integrate the widget into your existing notebooks.

## 🎯 Benefits

1. **Reduced Complexity** - No need to remember parameter names and formats
2. **Error Prevention** - Built-in validation prevents configuration mistakes  
3. **User Experience** - Visual interface is more intuitive than code
4. **Consistency** - Generates the same objects as manual configuration
5. **Flexibility** - Easy to modify configurations through the UI
6. **Integration** - Works seamlessly with existing notebook workflows

The widget provides the exact same `CradleDataLoadConfig` objects that manual configuration produces, but through a much more user-friendly interface.
