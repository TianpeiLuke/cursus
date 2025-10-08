"""
Cradle Data Load Config UI Package

A web-based user interface for creating and managing CradleDataLoadingConfig objects
through a guided wizard interface.

This package provides:
- FastAPI-based REST API for configuration management
- HTML/CSS/JavaScript frontend with step-by-step wizard
- Server-side validation using existing Pydantic models
- Export functionality for JSON and Python code
- Dynamic form generation based on configuration schemas

Usage:
    To run the application:
    
    ```python
    from cursus.api.cradle_ui.app import app
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    ```
    
    Or run directly:
    
    ```bash
    cd src/cursus/api/cradle_ui
    python app.py
    ```

Components:
- app: FastAPI application with routes and middleware
- api: REST API endpoints for configuration management
- services: Business logic for validation and config building
- schemas: Pydantic models for request/response validation
- utils: Utility functions for field extraction and schema generation
- static: Frontend HTML/CSS/JavaScript files
"""

from .app import app
from .services import ValidationService, ConfigBuilderService
from .utils import (
    extract_field_schema,
    get_data_source_variant_schemas,
    get_all_config_schemas,
    get_field_defaults,
    get_field_validation_rules
)

__version__ = "1.0.0"
__author__ = "Cursus Framework"
__description__ = "Web UI for Cradle Data Load Configuration"

__all__ = [
    "app",
    "ValidationService", 
    "ConfigBuilderService",
    "extract_field_schema",
    "get_data_source_variant_schemas",
    "get_all_config_schemas", 
    "get_field_defaults",
    "get_field_validation_rules"
]
