---
tags:
  - design
  - ui
  - configuration
  - user-interface
keywords:
  - cradle
  - data
  - load
  - config
  - ui
  - user interface
  - form
  - wizard
  - multi-page
  - hierarchical config
topics:
  - user interface design
  - configuration management
  - form design
  - wizard interface
language: python, javascript, html, css
date of note: 2025-01-06
---

# Cradle Data Load Config UI Design

## Overview

This design document outlines the development of a multi-page wizard UI to help users fill in the complex hierarchical fields of `CradleDataLoadConfig`. The UI guides users through a structured 4-page workflow, with each page corresponding to one of the top-level configuration sections, and provides dynamic form updates based on user selections.

## Problem Statement

The current process for creating a `CradleDataLoadConfig` is complex and error-prone:

1. **Deeply nested hierarchy**: The configuration has 4 top-level configs, each with their own nested structures
2. **Dynamic field requirements**: DataSourceConfig has 3 variants (MDS, EDX, Andes) with different field sets
3. **Complex interdependencies**: Fields across different sections depend on each other
4. **User experience challenges**: No guided workflow for non-expert users
5. **Error-prone manual creation**: Easy to miss required fields or create invalid configurations

## Design Goals

1. **Guided Workflow**: Provide a step-by-step wizard interface that guides users through configuration creation
2. **Dynamic Forms**: Update form fields dynamically based on user selections (e.g., data source type)
3. **Validation**: Provide real-time validation and clear error messages
4. **Hierarchical Organization**: Organize the UI to match the natural hierarchy of the configuration
5. **User-Friendly**: Make the complex configuration accessible to non-expert users
6. **Extensible**: Design the UI to be easily extensible for future configuration types

## Configuration Hierarchy Analysis

Based on the `CradleDataLoadConfig` structure:

```
CradleDataLoadConfig (Top Level)
├── job_type (Essential - filled at the end)
├── data_sources_spec: DataSourcesSpecificationConfig (Page 1)
│   ├── start_date (Essential)
│   ├── end_date (Essential)
│   └── data_sources: List[DataSourceConfig] (Essential)
│       ├── data_source_name (Essential)
│       ├── data_source_type (Essential - MDS/EDX/ANDES)
│       └── [variant-specific properties based on type]
│           ├── MdsDataSourceConfig (if type=MDS)
│           │   ├── service_name (Essential)
│           │   ├── region (Essential)
│           │   ├── output_schema (Essential)
│           │   ├── org_id (System default=0)
│           │   └── use_hourly_edx_data_set (System default=False)
│           ├── EdxDataSourceConfig (if type=EDX)
│           │   ├── edx_provider (Essential)
│           │   ├── edx_subject (Essential)
│           │   ├── edx_dataset (Essential)
│           │   ├── edx_manifest_key (Essential)
│           │   └── schema_overrides (Essential)
│           └── AndesDataSourceConfig (if type=ANDES)
│               ├── provider (Essential)
│               ├── table_name (Essential)
│               └── andes3_enabled (System default=True)
├── transform_spec: TransformSpecificationConfig (Page 2)
│   ├── transform_sql (Essential)
│   └── job_split_options: JobSplitOptionsConfig
│       ├── split_job (System default=False)
│       ├── days_per_split (System default=7)
│       └── merge_sql (Essential if split_job=True)
├── output_spec: OutputSpecificationConfig (Page 3)
│   ├── output_schema (Essential)
│   ├── job_type (Essential - inherited from parent)
│   ├── pipeline_s3_loc (Essential - inherited from parent)
│   ├── output_format (System default="PARQUET")
│   ├── output_save_mode (System default="ERRORIFEXISTS")
│   ├── output_file_count (System default=0)
│   ├── keep_dot_in_output_schema (System default=False)
│   └── include_header_in_s3_output (System default=True)
└── cradle_job_spec: CradleJobSpecificationConfig (Page 4)
    ├── cradle_account (Essential)
    ├── cluster_type (System default="STANDARD")
    ├── extra_spark_job_arguments (System default="")
    └── job_retry_count (System default=1)
```

## User Experience Design

### Overall Workflow

1. **Page 1: Data Sources Configuration**
   - Configure start/end dates
   - Add data source blocks dynamically
   - Each block allows selection of data source type and fills corresponding fields

2. **Page 2: Transform Configuration**
   - Configure SQL transformation
   - Configure job splitting options (optional)

3. **Page 3: Output Configuration**
   - Configure output schema and format
   - Configure output options

4. **Page 4: Cradle Job Configuration**
   - Configure cluster and job settings

5. **Final Step: Job Type Selection & Completion**
   - Select job type (training/validation/testing/calibration)
   - Generate final configuration

### Page-by-Page User Experience

#### Page 1: Data Sources Configuration
```
┌─────────────────────────────────────────────────────────────┐
│ Step 1 of 4: Data Sources Configuration                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Time Range                                                  │
│ ┌─────────────────┐  ┌─────────────────┐                   │
│ │ Start Date      │  │ End Date        │                   │
│ │ [YYYY-MM-DD...] │  │ [YYYY-MM-DD...] │                   │
│ └─────────────────┘  └─────────────────┘                   │
│                                                             │
│ Data Sources                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Data Source 1                               [Remove]    │ │
│ │ ┌─────────────────┐  ┌─────────────────────────────────┐ │ │
│ │ │ Source Name     │  │ Source Type                     │ │ │
│ │ │ [text input]    │  │ [MDS ▼] [EDX] [ANDES]          │ │ │
│ │ └─────────────────┘  └─────────────────────────────────┘ │ │
│ │                                                         │ │
│ │ MDS Configuration                                       │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │ │
│ │ │Service Name │ │Region       │ │Output Schema        │ │ │
│ │ │[text input] │ │[NA ▼]       │ │[+ Add Field]        │ │ │
│ │ └─────────────┘ └─────────────┘ └─────────────────────┘ │ │
│ │                                                         │ │
│ │ Advanced Options (Optional)                             │ │
│ │ ☐ Use Hourly EDX Data Set                              │ │
│ │ Org ID: [0]                                            │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [+ Add Data Source]                                         │
│                                                             │
│ ┌─────────────┐                           ┌─────────────┐   │
│ │   Cancel    │                           │    Next     │   │
│ └─────────────┘                           └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### Page 2: Transform Configuration
```
┌─────────────────────────────────────────────────────────────┐
│ Step 2 of 4: Transform Configuration                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ SQL Transformation *                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ SELECT                                                  │ │
│ │   mds.objectId,                                         │ │
│ │   mds.transactionDate,                                  │ │
│ │   edx.is_abuse                                          │ │
│ │ FROM mds_source mds                                     │ │
│ │ JOIN edx_source edx ON mds.objectId = edx.order_id     │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Job Splitting Options                                       │
│ ☐ Enable Job Splitting                                     │
│                                                             │
│ [When enabled, shows:]                                      │
│ Days per Split: [7]                                        │
│ Merge SQL *:                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ SELECT * FROM INPUT                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐           ┌─────────────┐ │
│ │   Back      │  │   Cancel    │           │    Next     │ │
│ └─────────────┘  └─────────────┘           └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Page 3: Output Configuration
```
┌─────────────────────────────────────────────────────────────┐
│ Step 3 of 4: Output Configuration                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Output Schema *                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ objectId          [Remove]                              │ │
│ │ transactionDate   [Remove]                              │ │
│ │ is_abuse          [Remove]                              │ │
│ └─────────────────────────────────────────────────────────┘ │
│ [+ Add Field]                                               │
│                                                             │
│ Output Format                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [PARQUET ▼] [CSV] [JSON] [ION] [UNESCAPED_TSV]         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Save Mode                                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [ERRORIFEXISTS ▼] [OVERWRITE] [APPEND] [IGNORE]        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Advanced Options                                            │
│ Output File Count: [0] (0 = auto-split)                   │
│ ☐ Keep dots in output schema                               │
│ ☑ Include header in S3 output                             │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐           ┌─────────────┐ │
│ │   Back      │  │   Cancel    │           │    Next     │ │
│ └─────────────┘  └─────────────┘           └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Page 4: Cradle Job Configuration
```
┌─────────────────────────────────────────────────────────────┐
│ Step 4 of 4: Cradle Job Configuration                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Cradle Account *                                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Buyer-Abuse-RnD-Dev                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Cluster Configuration                                       │
│ Cluster Type:                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [STANDARD ▼] [SMALL] [MEDIUM] [LARGE]                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Job Retry Count: [1]                                       │
│                                                             │
│ Extra Spark Arguments (Optional):                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐           ┌─────────────┐ │
│ │   Back      │  │   Cancel    │           │    Next     │ │
│ └─────────────┘  └─────────────┘           └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Final Step: Job Type & Completion
```
┌─────────────────────────────────────────────────────────────┐
│ Complete Configuration                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Job Type *                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ○ Training                                              │ │
│ │ ○ Validation                                            │ │
│ │ ○ Testing                                               │ │
│ │ ○ Calibration                                           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Configuration Summary                                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Data Sources: 2 configured                              │ │
│ │ Time Range: 2025-01-01 to 2025-04-17                   │ │
│ │ Transform: Custom SQL provided                          │ │
│ │ Output: PARQUET format, 3 fields                       │ │
│ │ Cluster: STANDARD                                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐           ┌─────────────┐ │
│ │   Back      │  │   Cancel    │           │   Finish    │ │
│ └─────────────┘  └─────────────┘           └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend Components

#### Core UI Components
```
src/cursus/api/cradle_ui/
├── components/
│   ├── common/
│   │   ├── WizardLayout.js          # Main wizard container
│   │   ├── NavigationButtons.js     # Back/Next/Cancel buttons
│   │   ├── ProgressIndicator.js     # Step progress indicator
│   │   ├── ValidationMessage.js     # Error/success messages
│   │   └── FormField.js            # Reusable form field wrapper
│   ├── data_sources/
│   │   ├── DataSourcesPage.js       # Page 1 container
│   │   ├── TimeRangeSection.js      # Start/end date inputs
│   │   ├── DataSourceBlock.js       # Individual data source block
│   │   ├── DataSourceTypeSelector.js # MDS/EDX/ANDES selector
│   │   ├── MdsConfigForm.js         # MDS-specific fields
│   │   ├── EdxConfigForm.js         # EDX-specific fields
│   │   └── AndesConfigForm.js       # ANDES-specific fields
│   ├── transform/
│   │   ├── TransformPage.js         # Page 2 container
│   │   ├── SqlEditor.js             # SQL text area with syntax highlighting
│   │   └── JobSplitOptions.js       # Job splitting configuration
│   ├── output/
│   │   ├── OutputPage.js            # Page 3 container
│   │   ├── OutputSchemaEditor.js    # Output schema field list
│   │   └── OutputOptionsForm.js     # Format, save mode, etc.
│   ├── cradle_job/
│   │   ├── CradleJobPage.js         # Page 4 container
│   │   └── ClusterConfigForm.js     # Cluster and job settings
│   └── completion/
│       ├── CompletionPage.js        # Final step container
│       ├── JobTypeSelector.js       # Job type radio buttons
│       └── ConfigSummary.js         # Configuration summary display
├── hooks/
│   ├── useConfigState.js            # Global configuration state management
│   ├── useValidation.js             # Form validation logic
│   └── useWizardNavigation.js       # Wizard navigation logic
├── services/
│   ├── configService.js             # Configuration CRUD operations
│   ├── validationService.js         # Backend validation calls
│   └── exportService.js             # Configuration export/download
├── utils/
│   ├── configDefaults.js            # Default values for fields
│   ├── validationRules.js           # Client-side validation rules
│   └── configTransforms.js          # Data transformation utilities
└── styles/
    ├── wizard.css                   # Wizard-specific styles
    ├── forms.css                    # Form styling
    └── components.css               # Component-specific styles
```

#### Backend API Components
```
src/cursus/api/cradle_ui/
├── api/
│   ├── __init__.py
│   ├── routes.py                    # FastAPI routes
│   ├── models.py                    # Pydantic request/response models
│   └── dependencies.py              # FastAPI dependencies
├── services/
│   ├── __init__.py
│   ├── config_builder.py            # Configuration object building
│   ├── validation_service.py        # Server-side validation
│   └── export_service.py            # Configuration export/serialization
├── schemas/
│   ├── __init__.py
│   ├── request_schemas.py           # API request schemas
│   └── response_schemas.py          # API response schemas
└── utils/
    ├── __init__.py
    ├── config_helpers.py            # Configuration utility functions
    └── field_extractors.py          # Field extraction from config classes
```

### State Management Architecture

#### Global Configuration State
```javascript
// useConfigState.js
const initialState = {
  currentStep: 1,
  isValid: {
    step1: false,
    step2: false,
    step3: false,
    step4: false
  },
  data: {
    dataSourcesSpec: {
      startDate: '',
      endDate: '',
      dataSources: []
    },
    transformSpec: {
      transformSql: '',
      jobSplitOptions: {
        splitJob: false,
        daysPerSplit: 7,
        mergeSql: ''
      }
    },
    outputSpec: {
      outputSchema: [],
      outputFormat: 'PARQUET',
      outputSaveMode: 'ERRORIFEXISTS',
      outputFileCount: 0,
      keepDotInOutputSchema: false,
      includeHeaderInS3Output: true
    },
    cradleJobSpec: {
      cradleAccount: '',
      clusterType: 'STANDARD',
      extraSparkJobArguments: '',
      jobRetryCount: 1
    },
    jobType: ''
  },
  errors: {},
  isDirty: false
};
```

#### Data Source Block State
```javascript
// Individual data source block state
const dataSourceBlockState = {
  id: 'unique-id',
  dataSourceName: '',
  dataSourceType: 'MDS', // 'MDS' | 'EDX' | 'ANDES'
  mdsProperties: {
    serviceName: '',
    region: 'NA',
    outputSchema: [],
    orgId: 0,
    useHourlyEdxDataSet: false
  },
  edxProperties: {
    edxProvider: '',
    edxSubject: '',
    edxDataset: '',
    edxManifestKey: '',
    schemaOverrides: []
  },
  andesProperties: {
    provider: '',
    tableName: '',
    andes3Enabled: true
  },
  isValid: false,
  errors: {}
};
```

### API Design

#### REST API Endpoints
```python
# routes.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from .models import *
from .services.config_builder import ConfigBuilderService
from .services.validation_service import ValidationService

router = APIRouter(prefix="/api/cradle-ui", tags=["cradle-ui"])

@router.get("/config-defaults")
async def get_config_defaults() -> ConfigDefaultsResponse:
    """Get default values for all configuration fields"""
    pass

@router.post("/validate-step")
async def validate_step(request: StepValidationRequest) -> StepValidationResponse:
    """Validate a specific step's configuration"""
    pass

@router.post("/validate-data-source")
async def validate_data_source(request: DataSourceValidationRequest) -> ValidationResponse:
    """Validate a single data source configuration"""
    pass

@router.post("/build-config")
async def build_config(request: ConfigBuildRequest) -> ConfigBuildResponse:
    """Build the final CradleDataLoadConfig from UI data"""
    pass

@router.post("/export-config")
async def export_config(request: ConfigExportRequest) -> ConfigExportResponse:
    """Export configuration as JSON or Python code"""
    pass

@router.get("/field-schema/{config_type}")
async def get_field_schema(config_type: str) -> FieldSchemaResponse:
    """Get field schema for dynamic form generation"""
    pass
```

#### Request/Response Models
```python
# models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class DataSourceBlockData(BaseModel):
    id: str
    data_source_name: str
    data_source_type: str  # 'MDS' | 'EDX' | 'ANDES'
    mds_properties: Optional[Dict[str, Any]] = None
    edx_properties: Optional[Dict[str, Any]] = None
    andes_properties: Optional[Dict[str, Any]] = None

class StepValidationRequest(BaseModel):
    step: int
    data: Dict[str, Any]

class StepValidationResponse(BaseModel):
    is_valid: bool
    errors: Dict[str, List[str]]
    warnings: Dict[str, List[str]]

class ConfigBuildRequest(BaseModel):
    data_sources_spec: Dict[str, Any]
    transform_spec: Dict[str, Any]
    output_spec: Dict[str, Any]
    cradle_job_spec: Dict[str, Any]
    job_type: str

class ConfigBuildResponse(BaseModel):
    success: bool
    config: Optional[Dict[str, Any]] = None
    errors: List[str] = []
```

### Dynamic Form Generation

#### Field Schema System
```python
# field_extractors.py
from typing import Dict, List, Any
from src.cursus.steps.configs.config_cradle_data_loading_step import *

def extract_field_schema(config_class) -> Dict[str, Any]:
    """Extract field schema from Pydantic config class for UI generation"""
    schema = {
        "fields": {},
        "categories": {}
    }
    
    # Get field categories using the three-tier system
    if hasattr(config_class, 'categorize_fields'):
        categories = config_class.categorize_fields()
        schema["categories"] = categories
    
    # Extract field information from Pydantic model
    for field_name, field_info in config_class.model_fields.items():
        schema["fields"][field_name] = {
            "type": str(field_info.annotation),
            "required": field_info.is_required(),
            "default": field_info.default if field_info.default is not None else None,
            "description": field_info.description,
            "validation": extract_validation_rules(field_info)
        }
    
    return schema

def get_data_source_variant_schemas() -> Dict[str, Dict[str, Any]]:
    """Get schemas for all data source variants"""
    return {
        "MDS": extract_field_schema(MdsDataSourceConfig),
        "EDX": extract_field_schema(EdxDataSourceConfig),
        "ANDES": extract_field_schema(AndesDataSourceConfig)
    }
```

#### Dynamic Component Rendering
```javascript
// DataSourceBlock.js
import React, { useState, useEffect } from 'react';
import MdsConfigForm from './MdsConfigForm';
import EdxConfigForm from './EdxConfigForm';
import AndesConfigForm from './AndesConfigForm';

const DataSourceBlock = ({ blockData, onUpdate, onRemove }) => {
  const [localData, setLocalData] = useState(blockData);
  
  const handleTypeChange = (newType) => {
    // Clear variant-specific properties when type changes
    const updatedData = {
      ...localData,
      dataSourceType: newType,
      mdsProperties: newType === 'MDS' ? {} : null,
      edxProperties: newType === 'EDX' ? {} : null,
      andesProperties: newType === 'ANDES' ? {} : null
    };
    setLocalData(updatedData);
    onUpdate(updatedData);
  };
  
  const renderVariantForm = () => {
    switch (localData.dataSourceType) {
      case 'MDS':
        return (
          <MdsConfigForm
            data={localData.mdsProperties}
            onChange={(data) => handleVariantChange('mdsProperties', data)}
          />
        );
      case 'EDX':
        return (
          <EdxConfigForm
            data={localData.edxProperties}
            onChange={(data) => handleVariantChange('edxProperties', data)}
          />
        );
      case 'ANDES':
        return (
          <AndesConfigForm
            data={localData.andesProperties}
            onChange={(data) => handleVariantChange('andesProperties', data)}
          />
        );
      default:
        return null;
    }
  };
  
  return (
    <div className="data-source-block">
      <div className="block-header">
        <input
          type="text"
          placeholder="Data Source Name"
          value={localData.dataSourceName}
          onChange={(e) => handleFieldChange('dataSourceName', e.target.value)}
        />
        <select
          value={localData.dataSourceType}
          onChange={(e) => handleTypeChange(e.target.value)}
        >
          <option value="MDS">MDS</option>
          <option value="EDX">EDX</option>
          <option value="ANDES">ANDES</option>
        </select>
        <button onClick={() => onRemove(localData.id)}>Remove</button>
      </div>
      
      <div className="variant-form">
        {renderVariantForm()}
      </div>
    </div>
  );
};
```

### Validation System

#### Client-Side Validation
```javascript
// validationRules.js
export const validationRules = {
  dataSourcesSpec: {
    startDate: {
      required: true,
      pattern: /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/,
      message: 'Start date must be in format YYYY-MM-DDTHH:MM:SS'
    },
    endDate: {
      required: true,
      pattern: /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/,
      message: 'End date must be in format YYYY-MM-DDTHH:MM:SS'
    },
    dataSources: {
      required: true,
      minLength: 1,
      message: 'At least one data source is required'
    }
  },
  mdsDataSource: {
    serviceName: {
      required: true,
      message: 'Service name is required'
    },
    region: {
      required: true,
      enum: ['NA', 'EU', 'FE'],
      message: 'Region must be one of NA, EU, FE'
    },
    outputSchema: {
      required: true,
      minLength: 1,
      message: 'At least one output field is required'
    }
  },
  // ... more validation rules
};

// useValidation.js
export const useValidation = () => {
  const validateField = (fieldPath, value, rules) => {
    const errors = [];
    
    if (rules.required && (!value || value.length === 0)) {
      errors.push(rules.message || 'This field is required');
    }
    
    if (rules.pattern && value && !rules.pattern.test(value)) {
      errors.push(rules.message || 'Invalid format');
    }
    
    if (rules.enum && value && !rules.enum.includes(value)) {
      errors.push(rules.message || `Must be one of: ${rules.enum.join(', ')}`);
    }
    
    return errors;
  };
  
  const validateStep = (stepNumber, data) => {
    // Step-specific validation logic
    const errors = {};
    
    switch (stepNumber) {
      case 1:
        return validateDataSourcesStep(data);
      case 2:
        return validateTransformStep(data);
      case 3:
        return validateOutputStep(data);
      case 4:
        return validateCradleJobStep(data);
      default:
        return {};
    }
  };
  
  return { validateField, validateStep };
};
```

#### Server-Side Validation
```python
# validation_service.py
from typing import Dict, List, Any, Optional
from src.cursus.steps.configs.config_cradle_data_loading_step import *
from pydantic import ValidationError

class ValidationService:
    """Server-side validation service for CradleDataLoadConfig"""
    
    def validate_step_data(self, step: int, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate data for a specific step"""
        errors = {}
        
        try:
            if step == 1:
                errors.update(self._validate_data_sources_spec(data))
            elif step == 2:
                errors.update(self._validate_transform_spec(data))
            elif step == 3:
                errors.update(self._validate_output_spec(data))
            elif step == 4:
                errors.update(self._validate_cradle_job_spec(data))
        except Exception as e:
            errors['general'] = [f"Validation error: {str(e)}"]
        
        return errors
    
    def _validate_data_sources_spec(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate DataSourcesSpecificationConfig data"""
        errors = {}
        
        try:
            # Validate time range
            start_date = data.get('startDate', '')
            end_date = data.get('endDate', '')
            
            if not start_date:
                errors['startDate'] = ['Start date is required']
            if not end_date:
                errors['endDate'] = ['End date is required']
            
            # Validate data sources
            data_sources = data.get('dataSources', [])
            if not data_sources:
                errors['dataSources'] = ['At least one data source is required']
            
            for i, ds in enumerate(data_sources):
                ds_errors = self._validate_data_source(ds)
                if ds_errors:
                    errors[f'dataSources[{i}]'] = ds_errors
                    
        except Exception as e:
            errors['general'] = [f"Data sources validation error: {str(e)}"]
        
        return errors
    
    def _validate_data_source(self, data: Dict[str, Any]) -> List[str]:
        """Validate individual DataSourceConfig"""
        errors = []
        
        data_source_type = data.get('dataSourceType')
        if not data_source_type:
            errors.append('Data source type is required')
            return errors
        
        try:
            if data_source_type == 'MDS':
                mds_config = MdsDataSourceConfig(**data.get('mdsProperties', {}))
            elif data_source_type == 'EDX':
                edx_config = EdxDataSourceConfig(**data.get('edxProperties', {}))
            elif data_source_type == 'ANDES':
                andes_config = AndesDataSourceConfig(**data.get('andesProperties', {}))
            else:
                errors.append(f'Invalid data source type: {data_source_type}')
                
        except ValidationError as e:
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                errors.append(f"{field}: {error['msg']}")
        
        return errors
    
    def build_final_config(self, ui_data: Dict[str, Any]) -> CradleDataLoadConfig:
        """Build final CradleDataLoadConfig from UI data"""
        
        # Build data sources
        data_sources = []
        for ds_data in ui_data['dataSourcesSpec']['dataSources']:
            ds_type = ds_data['dataSourceType']
            
            if ds_type == 'MDS':
                mds_props = MdsDataSourceConfig(**ds_data['mdsProperties'])
                data_source = DataSourceConfig(
                    data_source_name=ds_data['dataSourceName'],
                    data_source_type=ds_type,
                    mds_data_source_properties=mds_props
                )
            elif ds_type == 'EDX':
                edx_props = EdxDataSourceConfig(**ds_data['edxProperties'])
                data_source = DataSourceConfig(
                    data_source_name=ds_data['dataSourceName'],
                    data_source_type=ds_type,
                    edx_data_source_properties=edx_props
                )
            elif ds_type == 'ANDES':
                andes_props = AndesDataSourceConfig(**ds_data['andesProperties'])
                data_source = DataSourceConfig(
                    data_source_name=ds_data['dataSourceName'],
                    data_source_type=ds_type,
                    andes_data_source_properties=andes_props
                )
            
            data_sources.append(data_source)
        
        # Build specifications
        data_sources_spec = DataSourcesSpecificationConfig(
            start_date=ui_data['dataSourcesSpec']['startDate'],
            end_date=ui_data['dataSourcesSpec']['endDate'],
            data_sources=data_sources
        )
        
        job_split_options = JobSplitOptionsConfig(**ui_data['transformSpec']['jobSplitOptions'])
        transform_spec = TransformSpecificationConfig(
            transform_sql=ui_data['transformSpec']['transformSql'],
            job_split_options=job_split_options
        )
        
        output_spec = OutputSpecificationConfig(**ui_data['outputSpec'])
        cradle_job_spec = CradleJobSpecificationConfig(**ui_data['cradleJobSpec'])
        
        # Build final config
        config = CradleDataLoadConfig(
            job_type=ui_data['jobType'],
            data_sources_spec=data_sources_spec,
            transform_spec=transform_spec,
            output_spec=output_spec,
            cradle_job_spec=cradle_job_spec
        )
        
        return config
```

## Implementation Plan

### Phase 1: Backend API Foundation (Week 1-2)
1. **Setup FastAPI application structure**
   - Create API routes and models
   - Implement field schema extraction from Pydantic classes
   - Build validation service with server-side validation

2. **Configuration Builder Service**
   - Implement config building from UI data
   - Add export functionality (JSON, Python code)
   - Create comprehensive error handling

3. **Testing Infrastructure**
   - Unit tests for validation service
   - Integration tests for config building
   - API endpoint testing

### Phase 2: Frontend Core Components (Week 3-4)
1. **Wizard Framework**
   - Implement WizardLayout and navigation
   - Create ProgressIndicator and NavigationButtons
   - Build state management with useConfigState hook

2. **Form Infrastructure**
   - Create reusable FormField components
   - Implement validation system with useValidation hook
   - Build dynamic form generation utilities

3. **Basic Page Structure**
   - Create page containers for all 4 steps
   - Implement basic form layouts
   - Add navigation between pages

### Phase 3: Dynamic Data Source Forms (Week 5-6)
1. **Data Source Block System**
   - Implement DataSourceBlock with dynamic type switching
   - Create variant-specific forms (MDS, EDX, ANDES)
   - Add/remove data source functionality

2. **Advanced Form Features**
   - Schema field editors with add/remove functionality
   - SQL editor with syntax highlighting
   - Real-time validation and error display

3. **Integration Testing**
   - Test dynamic form switching
   - Validate data source configurations
   - End-to-end wizard flow testing

### Phase 4: Advanced Features & Polish (Week 7-8)
1. **Enhanced User Experience**
   - Configuration summary and preview
   - Save/load draft configurations
   - Export functionality integration

2. **Validation & Error Handling**
   - Comprehensive client-side validation
   - Server-side validation integration
   - User-friendly error messages

3. **Documentation & Deployment**
   - User documentation and help system
   - Deployment configuration
   - Performance optimization

## Benefits

### User Experience Improvements
1. **Guided Workflow**: Step-by-step process reduces complexity and errors
2. **Dynamic Forms**: Context-aware forms show only relevant fields
3. **Real-time Validation**: Immediate feedback prevents invalid configurations
4. **Visual Organization**: Clear hierarchy matches the configuration structure

### Development Benefits
1. **Reusable Components**: Modular design allows reuse for other config types
2. **Type Safety**: Leverages existing Pydantic validation
3. **Maintainable**: Clear separation of concerns between UI and business logic
4. **Extensible**: Easy to add new data source types or configuration sections

### Business Value
1. **Reduced Onboarding Time**: Non-experts can create configurations quickly
2. **Fewer Errors**: Guided workflow and validation prevent common mistakes
3. **Increased Adoption**: Lower barrier to entry for using CradleDataLoadConfig
4. **Better Documentation**: UI serves as interactive documentation

## Technical Considerations

### Performance
- **Lazy Loading**: Load form schemas on demand
- **Debounced Validation**: Avoid excessive API calls during typing
- **Optimistic Updates**: Update UI immediately, validate in background

### Security
- **Input Sanitization**: Sanitize all user inputs before processing
- **CSRF Protection**: Implement CSRF tokens for state-changing operations
- **Rate Limiting**: Prevent abuse of validation endpoints

### Accessibility
- **Keyboard Navigation**: Full keyboard support for wizard navigation
- **Screen Reader Support**: Proper ARIA labels and descriptions
- **High Contrast**: Support for high contrast themes
- **Focus Management**: Proper focus handling during navigation

### Browser Compatibility
- **Modern Browsers**: Target Chrome 90+, Firefox 88+, Safari 14+
- **Progressive Enhancement**: Basic functionality without JavaScript
- **Responsive Design**: Mobile-friendly responsive layout

## Future Enhancements

### Advanced Features
1. **Configuration Templates**: Pre-built templates for common use cases
2. **Bulk Import**: Import configurations from CSV or Excel files
3. **Version Control**: Track configuration changes and history
4. **Collaboration**: Multi-user editing and approval workflows

### Integration Opportunities
1. **Pipeline Integration**: Direct integration with pipeline execution
2. **Monitoring Dashboard**: Real-time status of configurations in use
3. **Analytics**: Usage analytics and optimization suggestions
4. **API Integration**: REST API for programmatic configuration management

### Extensibility
1. **Plugin System**: Allow custom data source types via plugins
2. **Custom Validators**: User-defined validation rules
3. **Theming**: Customizable UI themes and branding
4. **Localization**: Multi-language support

## Conclusion

This UI design provides a comprehensive solution for simplifying the creation of `CradleDataLoadConfig` objects. By implementing a guided wizard interface with dynamic forms and real-time validation, we can significantly reduce the complexity and error rate of configuration creation while maintaining the full power and flexibility of the underlying configuration system.

The modular architecture ensures that the solution is maintainable, extensible, and can serve as a foundation for similar configuration UIs in the future. The phased implementation plan allows for iterative development and early user feedback, ensuring the final product meets user needs effectively.
