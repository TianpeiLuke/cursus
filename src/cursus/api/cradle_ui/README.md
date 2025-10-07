# Cradle Data Load Config UI

A web-based user interface for creating and managing `CradleDataLoadConfig` objects through a guided wizard interface.

## Overview

This UI provides a step-by-step wizard that guides users through the complex process of creating a `CradleDataLoadConfig`. The interface is organized into 4 main pages corresponding to the top-level configuration sections:

1. **Data Sources Configuration** - Configure time range and data sources (MDS, EDX, ANDES)
2. **Transform Configuration** - Configure SQL transformation and job splitting options
3. **Output Configuration** - Configure output schema and format options
4. **Cradle Job Configuration** - Configure cluster and job execution settings

## Architecture

The implementation consists of:

### Backend (Python/FastAPI)
- **FastAPI Application** (`app.py`) - Main application with CORS and error handling
- **API Routes** (`api/routes.py`) - REST endpoints for validation, config building, and export
- **Validation Service** (`services/validation_service.py`) - Server-side validation logic
- **Config Builder Service** (`services/config_builder.py`) - Configuration building and export
- **Field Extractors** (`utils/field_extractors.py`) - Dynamic form schema generation
- **Request/Response Schemas** (`schemas/`) - Pydantic models for API validation

### Frontend (HTML/CSS/JavaScript)
- **Static HTML Interface** (`static/index.html`) - Complete wizard interface
- **Responsive Design** - Mobile-friendly responsive layout
- **Dynamic Forms** - Context-aware forms that update based on data source type
- **Real-time Validation** - Client-side validation with error messages

## Features

### User Experience
- **Guided Workflow** - Step-by-step process reduces complexity and errors
- **Dynamic Forms** - Forms update based on user selections (e.g., data source type)
- **Real-time Validation** - Immediate feedback prevents invalid configurations
- **Progress Tracking** - Visual progress indicator shows completion status
- **Configuration Summary** - Review all settings before completion

### Technical Features
- **Type Safety** - Leverages existing Pydantic validation from `CradleDataLoadConfig`
- **Extensible Design** - Easy to add new data source types or configuration sections
- **Export Functionality** - Export configurations as JSON or Python code
- **Error Handling** - Comprehensive error handling with user-friendly messages
- **API Documentation** - Auto-generated OpenAPI/Swagger documentation

## Installation & Setup

### Prerequisites
- Python 3.8+
- FastAPI
- Pydantic
- Uvicorn (for running the server)

### Install Dependencies
```bash
pip install fastapi uvicorn pydantic
```

### Running the Application

1. **Start the FastAPI server:**
```bash
cd src/cursus/api/cradle_ui
python app.py
```

2. **Access the application:**
- Web UI: http://localhost:8000/static/index.html
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### Configuration Management
- `GET /api/cradle-ui/config-defaults` - Get default values for all fields
- `POST /api/cradle-ui/validate-step` - Validate a specific step's configuration
- `POST /api/cradle-ui/validate-data-source` - Validate a single data source
- `POST /api/cradle-ui/build-config` - Build final CradleDataLoadConfig
- `POST /api/cradle-ui/export-config` - Export configuration as JSON/Python

### Schema Information
- `GET /api/cradle-ui/field-schema/{config_type}` - Get field schema for dynamic forms

### Health & Status
- `GET /api/cradle-ui/health` - Health check endpoint

## Usage Guide

### Step 1: Data Sources Configuration
1. Set the start and end dates in `YYYY-MM-DDTHH:MM:SS` format
2. Add data source blocks by clicking "Add Data Source"
3. For each data source:
   - Enter a logical name (e.g., "RAW_MDS_NA", "TAGS")
   - Select the data source type (MDS, EDX, or ANDES)
   - Fill in the type-specific configuration fields
   - Add output schema fields as needed

### Step 2: Transform Configuration
1. Enter the SQL transformation query
2. Optionally enable job splitting:
   - Set days per split
   - Provide merge SQL for combining split results

### Step 3: Output Configuration
1. Define the output schema (list of field names)
2. Select output format (PARQUET, CSV, JSON, etc.)
3. Choose save mode (ERRORIFEXISTS, OVERWRITE, etc.)
4. Configure advanced options as needed

### Step 4: Cradle Job Configuration
1. Enter the Cradle account name
2. Select cluster type (STANDARD, SMALL, MEDIUM, LARGE)
3. Set job retry count
4. Add extra Spark arguments if needed

### Step 5: Completion
1. Select the job type (training, validation, testing, calibration)
2. Review the configuration summary
3. Click "Finish" to generate the final configuration

## Configuration Examples

### Basic MDS + EDX Configuration
```json
{
  "job_type": "training",
  "data_sources_spec": {
    "start_date": "2025-01-01T00:00:00",
    "end_date": "2025-04-17T00:00:00",
    "data_sources": [
      {
        "data_source_name": "RAW_MDS_NA",
        "data_source_type": "MDS",
        "mds_data_source_properties": {
          "service_name": "AtoZ",
          "region": "NA",
          "output_schema": [
            {"field_name": "objectId", "field_type": "STRING"},
            {"field_name": "transactionDate", "field_type": "STRING"}
          ]
        }
      }
    ]
  },
  "transform_spec": {
    "transform_sql": "SELECT mds.objectId, mds.transactionDate FROM mds_source mds"
  },
  "output_spec": {
    "output_schema": ["objectId", "transactionDate"],
    "output_format": "PARQUET"
  },
  "cradle_job_spec": {
    "cradle_account": "Buyer-Abuse-RnD-Dev",
    "cluster_type": "STANDARD"
  }
}
```

## Development

### Adding New Data Source Types
1. Add the new config class to `config_cradle_data_loading_step.py`
2. Update `field_extractors.py` to include the new type
3. Add validation logic in `validation_service.py`
4. Update the frontend to handle the new type

### Extending Validation
1. Add new validation rules to `validation_service.py`
2. Update client-side validation in the HTML file
3. Add corresponding error messages

### Customizing the UI
1. Modify the CSS in `static/index.html` for styling changes
2. Update the JavaScript for new functionality
3. Add new form fields as needed

## Testing

### Manual Testing
1. Start the server and navigate to the web interface
2. Go through each step of the wizard
3. Test different data source types and configurations
4. Verify validation messages appear for invalid inputs
5. Check that the final configuration is generated correctly

### API Testing
Use the interactive API documentation at `/docs` to test individual endpoints.

## Troubleshooting

### Common Issues
1. **Import Errors** - Ensure all dependencies are installed and the Python path is correct
2. **Validation Failures** - Check that all required fields are filled and in the correct format
3. **Server Errors** - Check the console logs for detailed error messages

### Debug Mode
Set logging level to DEBUG in `app.py` for more detailed error information.

## Future Enhancements

### Planned Features
1. **Configuration Templates** - Pre-built templates for common use cases
2. **Save/Load Drafts** - Save work in progress and resume later
3. **Bulk Import** - Import configurations from CSV or Excel files
4. **Version Control** - Track configuration changes and history
5. **Collaboration** - Multi-user editing and approval workflows

### Integration Opportunities
1. **Pipeline Integration** - Direct integration with pipeline execution
2. **Monitoring Dashboard** - Real-time status of configurations in use
3. **Analytics** - Usage analytics and optimization suggestions

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include validation for new fields
4. Update documentation for any changes
5. Test thoroughly before submitting changes

## License

This project is part of the Cursus framework and follows the same licensing terms.
