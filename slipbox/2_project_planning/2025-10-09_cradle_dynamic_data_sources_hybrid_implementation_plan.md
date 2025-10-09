---
tags:
  - project
  - implementation
  - refactoring
  - cradle_ui
  - config_ui
  - dynamic_data_sources
  - hybrid_approach
keywords:
  - cradle dynamic data sources
  - hybrid single page ui
  - multiple data sources
  - dynamic widget system
  - data source management
topics:
  - cradle data loading config ui enhancement
  - dynamic data sources implementation
  - hybrid approach architecture
  - multiple data source support
language: python, javascript, html, css
date of note: 2025-10-09
---

# Cradle Dynamic Data Sources Hybrid Implementation Plan

## Executive Summary

This implementation plan provides a detailed roadmap for **enhancing the Cradle Data Loading Config UI** with **dynamic data sources support** using a **hybrid approach**. The solution builds upon the completed 2025-10-08 single-page refactoring by adding dynamic functionality specifically for the Data Sources section while maintaining the static approach for other sections.

### Key Discovery: Hierarchical Data Structure Analysis

After comprehensive analysis of `src/cursus/steps/configs/config_cradle_data_loading_step.py`, the key discovery is that **the CradleDataLoadingConfig has a sophisticated hierarchical structure** that naturally supports multiple data sources:

**Critical Finding**: The `DataSourcesSpecificationConfig.data_sources` field is defined as `List[DataSourceConfig]`, confirming that **multiple data sources are natively supported** in the configuration structure. Each `DataSourceConfig` can have different type-specific properties (`MdsDataSourceConfig`, `EdxDataSourceConfig`, or `AndesDataSourceConfig`).

### Strategic Approach: Hybrid Enhancement vs. Complete Redesign

Rather than redesigning the entire single-page form, this plan focuses on **targeted enhancement** of the Data Sources section with dynamic functionality while preserving the proven static approach for other sections.

## Problem Statement and Current State Analysis

### Current State Assessment

**✅ Completed from 2025-10-08 Plan:**
- ✅ Single-page form architecture implemented
- ✅ Static field definitions for all sections
- ✅ Data transformation logic for single data source
- ✅ ValidationService integration
- ✅ Enhanced field types (datetime, code_editor, tag_list, radio)
- ✅ 3-tier field categorization system

**❌ Current Limitation (Static Data Sources):**
- ❌ **Single Data Source Only**: Current implementation assumes exactly one data source
- ❌ **Static Field Definitions**: Cannot handle variable number of data sources
- ❌ **No Add/Remove Functionality**: Users cannot dynamically manage data sources
- ❌ **Type-Specific Field Limitation**: Cannot switch between MDS/EDX/ANDES dynamically

### Gap Analysis: Static vs. Dynamic Requirements

| Component | Current State (Static) | Required State (Dynamic) | Implementation Effort |
|-----------|------------------------|--------------------------|----------------------|
| **🔴 CRITICAL: Data Sources Section** | ❌ Single data source with conditional fields | **Multiple data sources with add/remove** | **60%** |
| **🔴 CRITICAL: Field Rendering** | ❌ Static field definitions | **Dynamic field rendering per data source type** | **25%** |
| **🔴 CRITICAL: Data Collection** | ❌ Single data source collection | **Multiple data sources collection** | **10%** |
| Time Range Fields | ✅ Working | No changes needed | 0% |
| Transform Configuration | ✅ Working | No changes needed | 0% |
| Output Configuration | ✅ Working | No changes needed | 0% |
| Job Configuration | ✅ Working | No changes needed | 0% |

**Total Implementation Effort: ~95% (primarily data sources section enhancement)**

### 🔴 **Critical Architecture Enhancement: From Static to Hybrid**

**Current Architecture (Static Single Data Source):**
```
Single-Page Form:
├── Section 0: Inherited Fields (static - working)
├── Section 1: Data Sources Configuration (STATIC - LIMITATION)
│   ├── Time Range Fields (static - working)
│   └── Single Data Source Fields (static - PROBLEM)
│       ├── data_source_name: text
│       ├── data_source_type: dropdown
│       ├── mds_service: text (conditional)
│       └── ... (static conditional fields)
├── Section 2: Transform Configuration (static - working)
├── Section 3: Output Configuration (static - working)
└── Section 4: Job Configuration (static - working)
```

**Target Architecture (Hybrid with Dynamic Data Sources):**
```
Hybrid Single-Page Form:
├── Section 0: Inherited Fields (static - unchanged)
├── Section 1: Data Sources Configuration (HYBRID - ENHANCED)
│   ├── Time Range Fields (static - unchanged)
│   └── Data Sources List (DYNAMIC - NEW)
│       ├── DataSource[0]: {type: MDS, fields: [mds_service, mds_region, ...]}
│       ├── DataSource[1]: {type: EDX, fields: [edx_provider, edx_subject, ...]}
│       ├── DataSource[n]: {type: ANDES, fields: [andes_provider, ...]}
│       ├── [+ Add Data Source] button
│       └── [Remove] buttons for each data source
├── Section 2: Transform Configuration (static - unchanged)
├── Section 3: Output Configuration (static - unchanged)
└── Section 4: Job Configuration (static - unchanged)
```

**The Benefits:**
- **Preserves 80% of Completed Work**: Only Data Sources section needs enhancement
- **Focused Complexity**: Dynamic functionality isolated to one section
- **User Experience**: Maintains single-page overview with needed flexibility
- **Manageable Implementation**: Clear separation between static and dynamic sections

## Solution Architecture

### Hybrid Enhancement Approach

**✅ SELECTED: Hybrid Data Sources Enhancement**
- Leverages 95%+ existing single-page form infrastructure
- 2-3 weeks implementation
- Focused dynamic functionality for Data Sources section only
- Maintains consistency with other static sections
- Better user experience with complete field overview plus dynamic data sources

**❌ REJECTED: Complete Dynamic Form System**
- Would require 6+ weeks to rebuild entire form system
- Unnecessary complexity for sections that work well as static
- Risk of introducing new issues in working sections
- Maintenance burden of complex dynamic form framework

### Technical Architecture

```python
# Hybrid Architecture (Minimal Changes to Existing System)
src/cursus/api/config_ui/
├── core/
│   ├── core.py                        # ✅ Minor enhancement (hybrid field discovery)
│   └── field_definitions.py           # ✅ Enhancement (hybrid field definitions)
├── widgets/
│   ├── widget.py                      # ✅ Enhancement (dynamic data sources widget)
│   ├── data_sources_manager.py        # 🆕 NEW: Dynamic data sources management
│   └── specialized_widgets.py         # ✅ No changes needed
└── enhanced_widget.py                 # ✅ No changes needed

# Implementation Flow (Hybrid Enhancement):
1. Create DataSourcesManager class for dynamic data source management
2. Enhance field definitions to support hybrid static/dynamic sections
3. Add dynamic data sources widget creation to UniversalConfigWidget
4. Implement data collection from multiple dynamic data sources
5. Enhance data transformation for multiple data sources
6. Test and validate with existing MultiStepWizard
```

### User Experience Workflow (Enhanced)

**Step 1: Standard Multi-Step Wizard Flow (Unchanged)**
```python
from cursus.api.config_ui.enhanced_widget import create_enhanced_pipeline_widget

# Create enhanced widget (no special handling needed)
enhanced_widget = create_enhanced_pipeline_widget(dag, base_config)
enhanced_widget.display()
```

**Step 2: Hybrid Cradle Configuration (Enhanced with Sub-Config Grouping)**
```
Multi-Step Wizard Progress: ●●●○○○○ (3/7)
┌─────────────────────────────────────────────────────────────┐
│ 🏗️ Cradle Data Load Configuration                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 💾 Inherited Fields - Smart Defaults                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Author: [test-user]     Bucket: [test-bucket]          │ │
│ │ Role: [arn:aws:iam::123456789012:role/test-role]       │ │
│ │ Region: [NA]  Service: [AtoZ]  Version: [1.0.0]        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📊 Data Sources Specification (data_sources_spec)          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Time Range                                              │ │
│ │ Start Date: [2025-01-01T00:00:00] End: [2025-04-17...] │ │
│ │                                                         │ │
│ │ Data Sources (Dynamic List)                             │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ 📊 Data Source 1                       [Remove]    │ │ │
│ │ │ Type: [MDS ▼]    Name: [RAW_MDS_NA]                │ │ │
│ │ │ Service: [AtoZ]  Region: [NA ▼]                    │ │ │
│ │ │ Schema: [objectId, transactionDate, is_abuse]      │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ 📊 Data Source 2                       [Remove]    │ │ │
│ │ │ Type: [EDX ▼]    Name: [RAW_EDX_EU]                │ │ │
│ │ │ Provider: [provider1] Subject: [subject1]          │ │ │
│ │ │ Dataset: [dataset1]   Manifest: [manifest1]        │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ │ [+ Add Data Source]                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ⚙️ Transform Specification (transform_spec)                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Transform SQL:                                          │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ SELECT mds.objectId, mds.transactionDate,           │ │ │
│ │ │        edx.is_abuse                                 │ │ │
│ │ │ FROM mds_source mds                                 │ │ │
│ │ │ JOIN edx_source edx ON mds.objectId = edx.order_id │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ │                                                         │ │
│ │ Job Split Options:                                      │ │
│ │ ☐ Split Job    Days per Split: [7]                     │ │
│ │ Merge SQL: [SELECT * FROM INPUT]                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📤 Output Specification (output_spec)                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Output Schema: [objectId, transactionDate, is_abuse]    │ │
│ │ Format: [PARQUET ▼]  Save Mode: [ERRORIFEXISTS ▼]      │ │
│ │ File Count: [0]  ☐ Keep dots  ☑ Include header         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 🎛️ Cradle Job Specification (cradle_job_spec)              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Cradle Account: [Buyer-Abuse-RnD-Dev]                  │ │
│ │ Cluster Type: [STANDARD ▼]  Retry Count: [1]           │ │
│ │ Extra Spark Arguments: [                              ] │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 🎯 Job Type Selection                                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ○ Training    ○ Validation    ○ Testing    ○ Calibration │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 🔧 Advanced Options                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ S3 Input Override: [                                 ] │ │
│ │ (Optional: Skip Cradle data pull, use S3 prefix)       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │        [💾 Complete Configuration]  [❌ Cancel]        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Step 3: Standard Workflow Continuation (Unchanged)**
```python
# User clicks "Complete Configuration"
# Form data collected including multiple data sources
# Data transformed to CradleDataLoadingConfig with List[DataSourceConfig]
# Workflow continues to next step
# All configs saved together with save_all_merged()
```

## Detailed Implementation Analysis

### ✅ **Perfect Compatibility with Existing System**

The hybrid approach integrates seamlessly with the existing config UI infrastructure:

#### **1. DataSourcesManager Class (NEW)**

**Core Dynamic Data Sources Management:**
```python
class DataSourcesManager:
    """Manages dynamic data sources with add/remove functionality."""
    
    def __init__(self, initial_data_sources=None):
        self.data_sources = initial_data_sources or [self._create_default_data_source()]
        self.container = widgets.VBox()
        self.data_source_widgets = []
        self._render_data_sources()
    
    def _create_default_data_source(self):
        """Create default MDS data source based on config analysis."""
        return {
            "data_source_name": "RAW_MDS_NA",
            "data_source_type": "MDS",
            "mds_service": "AtoZ",
            "mds_region": "NA",
            "output_schema": ["objectId", "transactionDate"]
        }
    
    def add_data_source(self, source_type="MDS"):
        """Add new data source with type-specific default values."""
        new_source = self._create_data_source_template(source_type)
        self.data_sources.append(new_source)
        self._refresh_ui()
    
    def remove_data_source(self, index):
        """Remove data source at index (minimum 1 data source required)."""
        if len(self.data_sources) > 1:
            self.data_sources.pop(index)
            self._refresh_ui()
    
    def _refresh_ui(self):
        """Refresh the entire data sources UI."""
        self._render_data_sources()
    
    def get_all_data_sources(self):
        """Collect data from all data source widgets."""
        collected_data = []
        for i, widget_group in enumerate(self.data_source_widgets):
            source_data = self._collect_data_source_data(widget_group, i)
            collected_data.append(source_data)
        return collected_data
```

#### **2. Data Source Field Templates (Based on Config Analysis)**

**Type-Specific Field Templates from Config Structure:**
```python
# Based on analysis of MdsDataSourceConfig, EdxDataSourceConfig, AndesDataSourceConfig
DATA_SOURCE_FIELD_TEMPLATES = {
    "MDS": {
        "required_fields": ["data_source_name", "service_name", "region", "output_schema"],
        "optional_fields": ["org_id", "use_hourly_edx_data_set"],
        "field_definitions": {
            "data_source_name": {"type": "text", "default": "RAW_MDS_NA", "tier": "essential"},
            "service_name": {"type": "text", "default": "AtoZ", "tier": "essential"},
            "region": {"type": "dropdown", "options": ["NA", "EU", "FE"], "default": "NA", "tier": "essential"},
            "output_schema": {"type": "tag_list", "default": ["objectId", "transactionDate"], "tier": "essential"},
            "org_id": {"type": "number", "default": 0, "tier": "system"},
            "use_hourly_edx_data_set": {"type": "checkbox", "default": False, "tier": "system"}
        }
    },
    "EDX": {
        "required_fields": ["data_source_name", "edx_provider", "edx_subject", "edx_dataset", "edx_manifest_key"],
        "optional_fields": ["schema_overrides"],
        "field_definitions": {
            "data_source_name": {"type": "text", "default": "RAW_EDX_EU", "tier": "essential"},
            "edx_provider": {"type": "text", "default": "", "tier": "essential"},
            "edx_subject": {"type": "text", "default": "", "tier": "essential"},
            "edx_dataset": {"type": "text", "default": "", "tier": "essential"},
            "edx_manifest_key": {"type": "text", "placeholder": '["xxx",...]', "tier": "essential"},
            "schema_overrides": {"type": "tag_list", "default": [], "tier": "system"}
        }
    },
    "ANDES": {
        "required_fields": ["data_source_name", "provider", "table_name"],
        "optional_fields": ["andes3_enabled"],
        "field_definitions": {
            "data_source_name": {"type": "text", "default": "RAW_ANDES_NA", "tier": "essential"},
            "provider": {"type": "text", "default": "", "tier": "essential"},
            "table_name": {"type": "text", "default": "", "tier": "essential"},
            "andes3_enabled": {"type": "checkbox", "default": True, "tier": "system"}
        }
    }
}
```

#### **3. Hybrid Field Definition Integration (Enhanced with Sub-Config Grouping)**

**Modified Field Definition Function (Grouped by Sub-Config):**
```python
def get_cradle_data_loading_fields_hybrid() -> List[Dict[str, Any]]:
    """Get hybrid field definition with sub-config grouping and dynamic data sources section."""
    return [
        # Inherited fields (from BasePipelineConfig) - UNCHANGED
        {"name": "author", "type": "text", "tier": "inherited", "required": True},
        {"name": "bucket", "type": "text", "tier": "inherited", "required": True},
        {"name": "role", "type": "text", "tier": "inherited", "required": True},
        {"name": "region", "type": "dropdown", "options": ["NA", "EU", "FE"], "tier": "inherited"},
        {"name": "service_name", "type": "text", "tier": "inherited", "required": True},
        {"name": "pipeline_version", "type": "text", "tier": "inherited", "required": True},
        {"name": "project_root_folder", "type": "text", "tier": "inherited", "required": True},
        
        # === DATA SOURCES SPECIFICATION (data_sources_spec) ===
        # Time range fields (static)
        {"name": "start_date", "type": "datetime", "section": "data_sources_spec", "required": True,
         "placeholder": "YYYY-MM-DDTHH:MM:SS", "description": "Start date for data loading"},
        {"name": "end_date", "type": "datetime", "section": "data_sources_spec", "required": True,
         "placeholder": "YYYY-MM-DDTHH:MM:SS", "description": "End date for data loading"},
        
        # Data sources - DYNAMIC (special handling)
        {"name": "data_sources", "type": "dynamic_data_sources", "section": "data_sources_spec", "required": True,
         "description": "Configure one or more data sources for your job"},
        
        # === TRANSFORM SPECIFICATION (transform_spec) ===
        {"name": "transform_sql", "type": "code_editor", "language": "sql", "section": "transform_spec", "required": True,
         "height": "200px", "default": "SELECT * FROM input_data", "description": "SQL transformation query"},
        {"name": "split_job", "type": "checkbox", "section": "transform_spec", "default": False,
         "description": "Enable job splitting for large datasets"},
        {"name": "days_per_split", "type": "number", "section": "transform_spec", "default": 7,
         "conditional": "split_job==True", "description": "Number of days per split"},
        {"name": "merge_sql", "type": "textarea", "section": "transform_spec", "default": "SELECT * FROM INPUT",
         "conditional": "split_job==True", "description": "SQL for merging split results"},
        
        # === OUTPUT SPECIFICATION (output_spec) ===
        {"name": "output_schema", "type": "tag_list", "section": "output_spec", "required": True,
         "default": ["objectId", "transactionDate", "is_abuse"], "description": "Output schema field names"},
        {"name": "output_format", "type": "dropdown", "section": "output_spec", "default": "PARQUET",
         "options": ["PARQUET", "CSV", "JSON", "ION", "UNESCAPED_TSV"], "description": "Output file format"},
        {"name": "output_save_mode", "type": "dropdown", "section": "output_spec", "default": "ERRORIFEXISTS",
         "options": ["ERRORIFEXISTS", "OVERWRITE", "APPEND", "IGNORE"], "description": "Save mode for output"},
        {"name": "output_file_count", "type": "number", "section": "output_spec", "default": 0,
         "description": "Number of output files (0 = auto-split)"},
        {"name": "keep_dot_in_output_schema", "type": "checkbox", "section": "output_spec", "default": False,
         "description": "Keep dots in output schema field names"},
        {"name": "include_header_in_s3_output", "type": "checkbox", "section": "output_spec", "default": True,
         "description": "Include header row in S3 output"},
        
        # === CRADLE JOB SPECIFICATION (cradle_job_spec) ===
        {"name": "cradle_account", "type": "text", "section": "cradle_job_spec", "required": True,
         "default": "Buyer-Abuse-RnD-Dev", "description": "Cradle account for job execution"},
        {"name": "cluster_type", "type": "dropdown", "section": "cradle_job_spec", "default": "STANDARD",
         "options": ["STANDARD", "SMALL", "MEDIUM", "LARGE"], "description": "Cluster type for job execution"},
        {"name": "job_retry_count", "type": "number", "section": "cradle_job_spec", "default": 1,
         "description": "Number of retries for failed jobs"},
        {"name": "extra_spark_job_arguments", "type": "textarea", "section": "cradle_job_spec", "default": "",
         "description": "Additional Spark job arguments"},
        
        # === ROOT LEVEL FIELDS ===
        {"name": "job_type", "type": "radio", "section": "root", "required": True,
         "options": ["training", "validation", "testing", "calibration"], "description": "Type of job to execute"},
        
        # === ADVANCED OPTIONS ===
        {"name": "s3_input_override", "type": "text", "section": "advanced", "default": None,
         "description": "If set, skip Cradle data pull and use this S3 prefix directly"}
    ]
```

#### **4. Enhanced Section Creation with Sub-Config Grouping**

**Section-Based Field Organization:**
```python
def _create_field_sections_by_subconfig(self, fields: List[Dict]) -> List[widgets.Widget]:
    """Create field sections organized by sub-config structure."""
    
    # Group fields by section
    sections = {
        "inherited": [],
        "data_sources_spec": [],
        "transform_spec": [],
        "output_spec": [],
        "cradle_job_spec": [],
        "root": [],
        "advanced": []
    }
    
    for field in fields:
        section = field.get("section", "inherited")
        sections[section].append(field)
    
    section_widgets = []
    
    # Inherited Fields Section
    if sections["inherited"]:
        inherited_section = self._create_field_section(
            "💾 Inherited Fields - Smart Defaults",
            sections["inherited"],
            "linear-gradient(135deg, #f0f8ff 0%, #e0f2fe 100%)",
            "#007bff",
            "Auto-filled from parent configurations"
        )
        section_widgets.append(inherited_section)
    
    # Data Sources Specification Section (HYBRID - static + dynamic)
    if sections["data_sources_spec"]:
        data_sources_section = self._create_data_sources_specification_section(
            sections["data_sources_spec"]
        )
        section_widgets.append(data_sources_section)
    
    # Transform Specification Section
    if sections["transform_spec"]:
        transform_section = self._create_field_section(
            "⚙️ Transform Specification (transform_spec)",
            sections["transform_spec"],
            "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "#f59e0b",
            "Configure SQL transformation and job splitting options"
        )
        section_widgets.append(transform_section)
    
    # Output Specification Section
    if sections["output_spec"]:
        output_section = self._create_field_section(
            "📤 Output Specification (output_spec)",
            sections["output_spec"],
            "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
            "#3b82f6",
            "Configure output schema and format options"
        )
        section_widgets.append(output_section)
    
    # Cradle Job Specification Section
    if sections["cradle_job_spec"]:
        job_section = self._create_field_section(
            "🎛️ Cradle Job Specification (cradle_job_spec)",
            sections["cradle_job_spec"],
            "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)",
            "#3b82f6",
            "Configure cluster and job execution settings"
        )
        section_widgets.append(job_section)
    
    # Root Level Fields Section
    if sections["root"]:
        root_section = self._create_field_section(
            "🎯 Job Type Selection",
            sections["root"],
            "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "#f59e0b",
            "Select the job type for this configuration"
        )
        section_widgets.append(root_section)
    
    # Advanced Options Section
    if sections["advanced"]:
        advanced_section = self._create_field_section(
            "🔧 Advanced Options",
            sections["advanced"],
            "linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%)",
            "#8b5cf6",
            "Optional advanced configuration settings"
        )
        section_widgets.append(advanced_section)
    
    return section_widgets

def _create_data_sources_specification_section(self, fields: List[Dict]) -> widgets.Widget:
    """Create the hybrid data sources specification section."""
    
    # Section header
    header_html = f"""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                border-left: 4px solid #f59e0b; 
                padding: 12px; border-radius: 8px 8px 0 0; margin-bottom: 0;'>
        <h4 style='margin: 0; color: #1f2937; display: flex; align-items: center;'>
            📊 Data Sources Specification (data_sources_spec)
        </h4>
        <p style='margin: 5px 0 0 0; font-size: 12px; color: #6b7280; font-style: italic;'>
            Configure time range and data sources for your job
        </p>
    </div>
    """
    header = widgets.HTML(header_html)
    
    # Create static fields (time range)
    static_fields = [f for f in fields if f["type"] != "dynamic_data_sources"]
    static_widgets = []
    
    for field in static_fields:
        field_widget_data = self._create_enhanced_field_widget(field)
        self.widgets[field["name"]] = field_widget_data["widget"]
        static_widgets.append(field_widget_data["container"])
    
    # Create dynamic data sources widget
    dynamic_field = next((f for f in fields if f["type"] == "dynamic_data_sources"), None)
    if dynamic_field:
        dynamic_widget_data = self._create_dynamic_data_sources_widget(dynamic_field)
        self.widgets[dynamic_field["name"]] = dynamic_widget_data["widget"]
        static_widgets.append(dynamic_widget_data["container"])
    
    # Combine all widgets
    content_container = widgets.VBox(
        static_widgets,
        layout=widgets.Layout(
            padding='20px',
            background='white',
            border='1px solid #e5e7eb',
            border_top='none',
            border_radius='0 0 8px 8px'
        )
    )
    
    return widgets.VBox([header, content_container], layout=widgets.Layout(margin='0 0 20px 0'))
```

### ✅ **Enhanced Data Collection with Sub-Config Structure**

**Data Collection Respecting Sub-Config Organization:**
```python
def _save_current_step(self) -> bool:
    """Enhanced save with sub-config organization and dynamic data sources support."""
    if self.current_step not in self.step_widgets:
        return True
    
    step_widget = self.step_widgets[self.current_step]
    step = self.steps[self.current_step]
    
    # Standard form data collection
    form_data = {}
    for field_name, widget in step_widget.widgets.items():
        if field_name == "data_sources":
            # Special handling for dynamic data sources
            if hasattr(widget, 'get_all_data_sources'):
                form_data[field_name] = widget.get_all_data_sources()
            else:
                form_data[field_name] = []
        else:
            value = widget.value
            
            # Handle special field types with enhanced conversion
            field_info = next((f for f in step_widget.fields if f["name"] == field_name), None)
            if field_info:
                field_type = field_info["type"]
                
                if field_type == "tag_list":
                    # Convert comma-separated string back to list
                    if isinstance(value, str):
                        value = [item.strip() for item in value.split(",") if item.strip()]
                    elif not isinstance(value, list):
                        value = []
                elif field_type == "radio":
                    # Radio button value is already correct
                    pass
                elif field_type == "datetime":
                    # Keep as string, validation happens in config creation
                    value = str(value) if value else ""
                elif field_type == "code_editor":
                    # Keep as string for SQL code
                    value = str(value) if value else ""
                elif field_type == "textarea":
                    # Keep as string
                    value = str(value) if value else ""
                elif field_type == "dropdown":
                    # Dropdown value is already correct
                    pass
                elif field_type == "number":
                    try:
                        value = float(value) if value != "" else field_info.get("default", 0.0)
                    except (ValueError, TypeError):
                        value = field_info.get("default", 0.0)
                elif field_type == "checkbox":
                    value = bool(value)
            
            form_data[field_name] = value
    
    # Enhanced config creation with ValidationService integration
    config_class = step["config_class"]
    config_class_name = step["config_class_name"]
    
    if config_class_name == "CradleDataLoadingConfig":
        # Transform flat form data to nested ui_data structure for multiple data sources
        ui_data = self._transform_cradle_form_data_hybrid(form_data)
        
        # REUSE ORIGINAL VALIDATION AND CONFIG BUILDING LOGIC
        try:
            from cursus.api.cradle_ui.services.validation_service import ValidationService
            validation_service = ValidationService()
            config_instance = validation_service.build_final_config(ui_data)
            logger.info(f"Created CradleDataLoadingConfig using ValidationService with {len(form_data.get('data_sources', []))} data sources")
        except ImportError as e:
            logger.warning(f"ValidationService not available: {e}, falling back to direct config creation")
            # Fallback: Create config directly (less robust but functional)
            config_instance = config_class(**ui_data)
        except Exception as e:
            logger.error(f"ValidationService failed: {e}, falling back to direct config creation")
            # Fallback: Create config directly
            config_instance = config_class(**ui_data)
    else:
        # Standard config creation for other classes
        config_instance = config_class(**form_data)
    
    # Store completed configuration with BOTH step title and class name for inheritance
    step_key = step["title"]
    
    self.completed_configs[step_key] = config_instance
    self.completed_configs[config_class_name] = config_instance
    
    logger.info(f"Step '{step_key}' saved successfully with sub-config organization")
    return True

def _transform_cradle_form_data_hybrid(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform hybrid form data with multiple data sources into ui_data structure.
    
    This creates the exact same nested structure that ValidationService.build_final_config() expects,
    but now supports multiple data sources from the dynamic data sources manager.
    
    Args:
        form_data: Flat form data from hybrid single-page form with dynamic data sources
        
    Returns:
        Nested ui_data structure compatible with ValidationService
    """
    logger.debug(f"Transforming hybrid cradle form data with {len(form_data.get('data_sources', []))} data sources")
    
    # Transform multiple data sources
    data_source_configs = []
    for source_data in form_data.get("data_sources", []):
        source_type = source_data.get("data_source_type")
        
        # Create type-specific properties based on actual config structure
        if source_type == "MDS":
            properties = {
                "mds_data_source_properties": {
                    "service_name": source_data.get("service_name"),
                    "region": source_data.get("region"),
                    "output_schema": source_data.get("output_schema", []),
                    "org_id": source_data.get("org_id", 0),
                    "use_hourly_edx_data_set": source_data.get("use_hourly_edx_data_set", False)
                }
            }
        elif source_type == "EDX":
            properties = {
                "edx_data_source_properties": {
                    "edx_provider": source_data.get("edx_provider"),
                    "edx_subject": source_data.get("edx_subject"),
                    "edx_dataset": source_data.get("edx_dataset"),
                    "edx_manifest_key": source_data.get("edx_manifest_key"),
                    "schema_overrides": source_data.get("schema_overrides", [])
                }
            }
        elif source_type == "ANDES":
            properties = {
                "andes_data_source_properties": {
                    "provider": source_data.get("provider"),
                    "table_name": source_data.get("table_name"),
                    "andes3_enabled": source_data.get("andes3_enabled", True)
                }
            }
        else:
            # Default to MDS if type is unknown
            properties = {
                "mds_data_source_properties": {
                    "service_name": "AtoZ",
                    "region": "NA",
                    "output_schema": ["objectId", "transactionDate"],
                    "org_id": 0,
                    "use_hourly_edx_data_set": False
                }
            }
        
        # Create data source config
        data_source_config = {
            "data_source_name": source_data.get("data_source_name"),
            "data_source_type": source_type,
            **properties
        }
        data_source_configs.append(data_source_config)
    
    # Create ui_data structure that matches ValidationService.build_final_config() expectations
    ui_data = {
        # Root level fields (BasePipelineConfig)
        "job_type": form_data.get("job_type", "training"),
        "author": form_data.get("author", "test-user"),
        "bucket": form_data.get("bucket", "test-bucket"),
        "role": form_data.get("role", "arn:aws:iam::123456789012:role/test-role"),
        "region": form_data.get("region", "NA"),
        "service_name": form_data.get("service_name", "test-service"),
        "pipeline_version": form_data.get("pipeline_version", "1.0.0"),
        "project_root_folder": form_data.get("project_root_folder", "test-project"),
        
        # LEVEL 3: Nested specification structures (exact match with ValidationService expectations)
        "data_sources_spec": {
            "start_date": form_data.get("start_date", "2025-01-01T00:00:00"),
            "end_date": form_data.get("end_date", "2025-04-17T00:00:00"),
            "data_sources": data_source_configs  # Multiple data sources
        },
        
        "transform_spec": {
            "transform_sql": form_data.get("transform_sql", "SELECT * FROM input_data"),
            "job_split_options": {
                "split_job": form_data.get("split_job", False),
                "days_per_split": form_data.get("days_per_split", 7),
                "merge_sql": form_data.get("merge_sql", "SELECT * FROM INPUT") if form_data.get("split_job") else None
            }
        },
        
        "output_spec": {
            "output_schema": form_data.get("output_schema", ["objectId", "transactionDate", "is_abuse"]),
            "pipeline_s3_loc": f"s3://{form_data.get('bucket', 'test-bucket')}/{form_data.get('project_root_folder', 'test-project')}",
            "output_format": form_data.get("output_format", "PARQUET"),
            "output_save_mode": form_data.get("output_save_mode", "ERRORIFEXISTS"),
            "output_file_count": form_data.get("output_file_count", 0),
            "keep_dot_in_output_schema": form_data.get("keep_dot_in_output_schema", False),
            "include_header_in_s3_output": form_data.get("include_header_in_s3_output", True)
        },
        
        "cradle_job_spec": {
            "cradle_account": form_data.get("cradle_account", "Buyer-Abuse-RnD-Dev"),
            "cluster_type": form_data.get("cluster_type", "STANDARD"),
            "extra_spark_job_arguments": form_data.get("extra_spark_job_arguments", ""),
            "job_retry_count": form_data.get("job_retry_count", 1)
        }
    }
    
    # Add optional fields if present
    if form_data.get("s3_input_override"):
        ui_data["s3_input_override"] = form_data["s3_input_override"]
    
    logger.debug(f"Transformed ui_data structure with {len(data_source_configs)} data sources")
    return ui_data
```

## Implementation Plan: **Hybrid Dynamic Data Sources Enhancement**

### **Phase 1: Dynamic Data Sources Infrastructure (Week 1)**

#### **Objective**: Create dynamic data sources management system

**Day 1-2: 🔴 CRITICAL - DataSourcesManager Class Creation**
- [ ] **HIGH PRIORITY** - Create `src/cursus/api/config_ui/widgets/data_sources_manager.py`
- [ ] **HIGH PRIORITY** - Implement DataSourcesManager class with add/remove functionality
- [ ] **HIGH PRIORITY** - Create data source field templates based on config analysis
- [ ] **HIGH PRIORITY** - Implement dynamic widget rendering for each data source type

**Implementation Details:**

**File 1: `src/cursus/api/config_ui/widgets/data_sources_manager.py` (NEW FILE)**
```python
"""
Dynamic Data Sources Manager for Cradle Data Loading Configuration

Provides dynamic add/remove functionality for multiple data sources with type-specific fields.
"""

import logging
from typing import Any, Dict, List, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

logger = logging.getLogger(__name__)

# Type-specific field templates based on config analysis
DATA_SOURCE_FIELD_TEMPLATES = {
    "MDS": {
        "required_fields": ["data_source_name", "service_name", "region", "output_schema"],
        "optional_fields": ["org_id", "use_hourly_edx_data_set"],
        "field_definitions": {
            "data_source_name": {"type": "text", "default": "RAW_MDS_NA"},
            "service_name": {"type": "text", "default": "AtoZ"},
            "region": {"type": "dropdown", "options": ["NA", "EU", "FE"], "default": "NA"},
            "output_schema": {"type": "tag_list", "default": ["objectId", "transactionDate"]},
            "org_id": {"type": "number", "default": 0},
            "use_hourly_edx_data_set": {"type": "checkbox", "default": False}
        }
    },
    "EDX": {
        "required_fields": ["data_source_name", "edx_provider", "edx_subject", "edx_dataset", "edx_manifest_key"],
        "optional_fields": ["schema_overrides"],
        "field_definitions": {
            "data_source_name": {"type": "text", "default": "RAW_EDX_EU"},
            "edx_provider": {"type": "text", "default": ""},
            "edx_subject": {"type": "text", "default": ""},
            "edx_dataset": {"type": "text", "default": ""},
            "edx_manifest_key": {"type": "text", "placeholder": '["xxx",...]'},
            "schema_overrides": {"type": "tag_list", "default": []}
        }
    },
    "ANDES": {
        "required_fields": ["data_source_name", "provider", "table_name"],
        "optional_fields": ["andes3_enabled"],
        "field_definitions": {
            "data_source_name": {"type": "text", "default": "RAW_ANDES_NA"},
            "provider": {"type": "text", "default": ""},
            "table_name": {"type": "text", "default": ""},
            "andes3_enabled": {"type": "checkbox", "default": True}
        }
    }
}

class DataSourcesManager:
    """Manages dynamic data sources with add/remove functionality."""
    
    def __init__(self, initial_data_sources=None):
        self.data_sources = initial_data_sources or [self._create_default_data_source()]
        self.container = widgets.VBox()
        self.data_source_widgets = []
        self._render_data_sources()
    
    def _create_default_data_source(self):
        """Create default MDS data source based on config analysis."""
        return {
            "data_source_name": "RAW_MDS_NA",
            "data_source_type": "MDS",
            "service_name": "AtoZ",
            "region": "NA",
            "output_schema": ["objectId", "transactionDate"]
        }
    
    def _create_data_source_template(self, source_type):
        """Create data source template with type-specific defaults."""
        template = DATA_SOURCE_FIELD_TEMPLATES[source_type]
        new_source = {"data_source_type": source_type}
        
        for field_name, field_def in template["field_definitions"].items():
            new_source[field_name] = field_def["default"]
        
        return new_source
    
    def add_data_source(self, source_type="MDS"):
        """Add new data source with type-specific default values."""
        new_source = self._create_data_source_template(source_type)
        self.data_sources.append(new_source)
        self._refresh_ui()
    
    def remove_data_source(self, index):
        """Remove data source at index (minimum 1 data source required)."""
        if len(self.data_sources) > 1:
            self.data_sources.pop(index)
            self._refresh_ui()
    
    def _refresh_ui(self):
        """Refresh the entire data sources UI."""
        self._render_data_sources()
    
    def _render_data_sources(self):
        """Render all data sources with add/remove functionality."""
        self.data_source_widgets = []
        
        with self.container:
            clear_output(wait=True)
            
            # Render each data source
            for i, source_data in enumerate(self.data_sources):
                widget_group = self._create_data_source_widget(source_data, i)
                self.data_source_widgets.append(widget_group)
                display(widget_group["widget"])
            
            # Add data source button
            add_button = widgets.Button(
                description="+ Add Data Source",
                button_style='info',
                layout=widgets.Layout(width='150px', margin='10px 0')
            )
            
            def on_add_click(button):
                self.add_data_source()
            
            add_button.on_click(on_add_click)
            display(add_button)
    
    def _create_data_source_widget(self, source_data, index):
        """Create widget for a single data source with type-specific fields."""
        source_type = source_data.get("data_source_type", "MDS")
        template = DATA_SOURCE_FIELD_TEMPLATES[source_type]
        
        # Header with type selector and remove button
        header_html = f"""
        <div style='background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); 
                    border: 1px solid #d1d5db; border-radius: 8px 8px 0 0; 
                    padding: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <h4 style='margin: 0; color: #374151;'>📊 Data Source {index + 1}</h4>
            <div style='font-size: 12px; color: #6b7280;'>Type: {source_type}</div>
        </div>
        """
        header_widget = widgets.HTML(header_html)
        
        # Type selector dropdown
        type_dropdown = widgets.Dropdown(
            options=["MDS", "EDX", "ANDES"],
            value=source_type,
            description="Type:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px')
        )
        
        # Remove button (disabled if only one data source)
        remove_button = widgets.Button(
            description="Remove",
            button_style='danger',
            layout=widgets.Layout(width='80px'),
            disabled=(len(self.data_sources) <= 1)
        )
        
        # Type-specific fields container
        fields_container = widgets.VBox()
        
        # Create type-specific fields
        field_widgets = {}
        for field_name, field_def in template["field_definitions"].items():
            field_widget = self._create_field_widget(field_name, field_def, source_data.get(field_name))
            field_widgets[field_name] = field_widget
            fields_container.children += (field_widget,)
        
        # Event handlers
        def on_type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # Update data source type and refresh fields
                self.data_sources[index]["data_source_type"] = change['new']
                self._refresh_single_data_source(index)
        
        def on_remove_click(button):
            self.remove_data_source(index)
        
        type_dropdown.observe(on_type_change)
        remove_button.on_click(on_remove_click)
        
        # Controls row
        controls = widgets.HBox([
            type_dropdown,
            remove_button
        ], layout=widgets.Layout(padding='10px'))
        
        # Complete data source widget
        data_source_widget = widgets.VBox([
            header_widget,
            controls,
            fields_container
        ], layout=widgets.Layout(
            border='1px solid #d1d5db',
            border_radius='8px',
            margin='10px 0'
        ))
        
        return {
            "widget": data_source_widget,
            "type_dropdown": type_dropdown,
            "remove_button": remove_button,
            "field_widgets": field_widgets,
            "fields_container": fields_container
        }
    
    def _create_field_widget(self, field_name, field_def, current_value):
        """Create individual field widget based on field definition."""
        field_type = field_def["type"]
        default_value = current_value if current_value is not None else field_def.get("default")
        
        if field_type == "text":
            return widgets.Text(
                value=str(default_value) if default_value else "",
                description=f"{field_name}:",
                placeholder=field_def.get("placeholder", ""),
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px', margin='5px 0')
            )
        elif field_type == "dropdown":
            return widgets.Dropdown(
                options=field_def["options"],
                value=default_value if default_value in field_def["options"] else field_def["options"][0],
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='200px', margin='5px 0')
            )
        elif field_type == "tag_list":
            value_str = ", ".join(default_value) if isinstance(default_value, list) else str(default_value)
            return widgets.Text(
                value=value_str,
                description=f"{field_name}:",
                placeholder="Enter comma-separated values",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='400px', margin='5px 0')
            )
        elif field_type == "number":
            return widgets.FloatText(
                value=float(default_value) if default_value else 0.0,
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='150px', margin='5px 0')
            )
        elif field_type == "checkbox":
            return widgets.Checkbox(
                value=bool(default_value),
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(margin='5px 0')
            )
        else:
            # Default to text
            return widgets.Text(
                value=str(default_value) if default_value else "",
                description=f"{field_name}:",
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px', margin='5px 0')
            )
    
    def _collect_data_source_data(self, widget_group, index):
        """Collect data from a single data source widget group."""
        source_data = {}
        
        # Get data source type
        source_data["data_source_type"] = widget_group["type_dropdown"].value
        
        # Get field values
        for field_name, field_widget in widget_group["field_widgets"].items():
            value = field_widget.value
            
            # Convert tag_list back to list
            field_def = DATA_SOURCE_FIELD_TEMPLATES[source_data["data_source_type"]]["field_definitions"][field_name]
            if field_def["type"] == "tag_list" and isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            
            source_data[field_name] = value
        
        return source_data
    
    def get_all_data_sources(self):
        """Collect data from all data source widgets."""
        collected_data = []
        for i, widget_group in enumerate(self.data_source_widgets):
            source_data = self._collect_data_source_data(widget_group, i)
            collected_data.append(source_data)
        return collected_data
    
    def _refresh_single_data_source(self, index):
        """Refresh a single data source when type changes."""
        # Update the data source data
        new_type = self.data_sources[index]["data_source_type"]
        self.data_sources[index] = self._create_data_source_template(new_type)
        self.data_sources[index]["data_source_name"] = f"RAW_{new_type}_NA"
        
        # Refresh entire UI (simpler than partial refresh)
        self._refresh_ui()
```

**Day 3-4: 🔴 CRITICAL - Integration with UniversalConfigWidget**
- [ ] **HIGH PRIORITY** - Update `src/cursus/api/config_ui/widgets/widget.py` to support dynamic data sources
- [ ] **HIGH PRIORITY** - Implement `_create_dynamic_data_sources_widget()` method
- [ ] **HIGH PRIORITY** - Update field section creation to use sub-config grouping
- [ ] **HIGH PRIORITY** - Test dynamic data sources widget creation and rendering

**Implementation Details:**

**File 2: `src/cursus/api/config_ui/widgets/widget.py` (ENHANCEMENT)**
```python
# Add import for DataSourcesManager
from .data_sources_manager import DataSourcesManager

# Enhance UniversalConfigWidget class
class UniversalConfigWidget:
    def render(self):
        """Enhanced render with sub-config grouping support."""
        if self._is_rendered:
            return
        
        with self.output:
            clear_output(wait=True)
            
            # Create modern title
            title_html = f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='margin: 0; display: flex; align-items: center;'>
                    ⚙️ {self.config_class_name}
                    <span style='margin-left: auto; font-size: 14px; opacity: 0.8;'>Configuration</span>
                </h2>
            </div>
            """
            title = widgets.HTML(title_html)
            display(title)
            
            # Check if this is CradleDataLoadingConfig for sub-config grouping
            if self.config_class_name == "CradleDataLoadingConfig":
                form_sections = self._create_field_sections_by_subconfig(self.fields)
            else:
                # Use existing 3-tier categorization for other configs
                form_sections = self._create_field_sections_by_tier(self.fields)
            
            # Create action buttons
            button_section = self._create_action_buttons()
            form_sections.append(button_section)
            
            # Display all sections
            form_box = widgets.VBox(form_sections, layout=widgets.Layout(padding='10px'))
            display(form_box)
        
        self._is_rendered = True
    
    def _create_dynamic_data_sources_widget(self, field: Dict) -> Dict:
        """Create dynamic data sources widget section."""
        
        # Initialize data sources manager
        initial_data = self.values.get("data_sources", [])
        data_sources_manager = DataSourcesManager(initial_data)
        
        # Create section container
        section_container = widgets.VBox([
            widgets.HTML("<h5 style='margin: 10px 0; color: #374151;'>Data Sources (Dynamic List)</h5>"),
            data_sources_manager.container
        ])
        
        return {
            "widget": data_sources_manager,  # Store manager for data collection
            "container": section_container
        }
```

**Day 5: Field Discovery Integration**
- [ ] **HIGH PRIORITY** - Update `src/cursus/api/config_ui/core/field_definitions.py` with hybrid field definitions
- [ ] **HIGH PRIORITY** - Update `src/cursus/api/config_ui/core/core.py` to use hybrid field definitions for CradleDataLoadingConfig
- [ ] **HIGH PRIORITY** - Test field discovery integration with sub-config grouping
- [ ] **HIGH PRIORITY** - Validate section creation and dynamic data sources integration

**Implementation Details:**

**File 3: `src/cursus/api/config_ui/core/field_definitions.py` (ENHANCEMENT)**
```python
# Add hybrid field definition function
def get_cradle_data_loading_fields_hybrid() -> List[Dict[str, Any]]:
    """Get hybrid field definition with sub-config grouping and dynamic data sources section."""
    # [Complete field definition from above]
    
**File 4: `src/cursus/api/config_ui/core/core.py` (ENHANCEMENT)**
```python
def _get_form_fields(self, config_class) -> List[Dict[str, Any]]:
    """Get form fields for configuration class with hybrid support."""
    config_class_name = config_class.__name__
    
    # Special handling for CradleDataLoadingConfig with hybrid approach
    if config_class_name == "CradleDataLoadingConfig":
        from .field_definitions import get_cradle_data_loading_fields_hybrid
        return get_cradle_data_loading_fields_hybrid()
    
    # Standard field discovery for other classes
    return self._discover_fields_from_pydantic(config_class)
```

### **Phase 2: Enhanced Data Transformation and Validation (Week 2)**

#### **Objective**: Implement multiple data sources support in data transformation

**Day 1-2: 🔴 CRITICAL - Enhanced Data Transformation**
- [ ] **HIGH PRIORITY** - Update `_transform_cradle_form_data_hybrid()` method in MultiStepWizard
- [ ] **HIGH PRIORITY** - Implement multiple data sources transformation logic
- [ ] **HIGH PRIORITY** - Test ValidationService integration with multiple data sources
- [ ] **HIGH PRIORITY** - Validate config creation with List[DataSourceConfig]

**Day 3-4: Enhanced Form Data Collection**
- [ ] **HIGH PRIORITY** - Update `_save_current_step()` method to handle dynamic data sources
- [ ] **HIGH PRIORITY** - Implement special data collection for DataSourcesManager
- [ ] **HIGH PRIORITY** - Test form data collection with multiple data sources
- [ ] **HIGH PRIORITY** - Validate data transformation and config creation end-to-end

**Day 5: Integration Testing**
- [ ] **MEDIUM PRIORITY** - Test complete workflow from form to config creation
- [ ] **MEDIUM PRIORITY** - Test backward compatibility with single data source
- [ ] **MEDIUM PRIORITY** - Validate save_all_merged functionality with multiple data sources
- [ ] **MEDIUM PRIORITY** - Performance testing with multiple data sources

### **Phase 3: Testing and Validation (Week 3)**

#### **Objective**: Comprehensive testing and validation of hybrid system

**Day 1-2: Unit Testing**
- [ ] **HIGH PRIORITY** - Create unit tests for DataSourcesManager class
- [ ] **HIGH PRIORITY** - Test data source field templates and widget creation
- [ ] **HIGH PRIORITY** - Test add/remove data source functionality
- [ ] **HIGH PRIORITY** - Test data collection from multiple data sources

**Day 3-4: Integration Testing**
- [ ] **HIGH PRIORITY** - Test hybrid form rendering with sub-config grouping
- [ ] **HIGH PRIORITY** - Test complete workflow with multiple data sources
- [ ] **HIGH PRIORITY** - Test ValidationService integration with multiple data sources
- [ ] **HIGH PRIORITY** - Test backward compatibility with existing workflows

**Day 5: User Experience Testing**
- [ ] **MEDIUM PRIORITY** - Test form usability with dynamic data sources
- [ ] **MEDIUM PRIORITY** - Validate add/remove data source UX
- [ ] **MEDIUM PRIORITY** - Test form validation and error handling
- [ ] **MEDIUM PRIORITY** - Performance testing with large numbers of data sources

### **Phase 4: Documentation and Deployment (Week 4)**

#### **Objective**: Complete documentation and production deployment

**Day 1-2: Documentation**
- [ ] **MEDIUM PRIORITY** - Update user documentation with hybrid approach
- [ ] **MEDIUM PRIORITY** - Create examples with multiple data sources
- [ ] **MEDIUM PRIORITY** - Document DataSourcesManager API
- [ ] **MEDIUM PRIORITY** - Create troubleshooting guide for dynamic data sources

**Day 3-4: Production Readiness**
- [ ] **HIGH PRIORITY** - Performance optimization for dynamic widgets
- [ ] **HIGH PRIORITY** - Security validation for dynamic form handling
- [ ] **HIGH PRIORITY** - Error recovery testing for dynamic data sources
- [ ] **HIGH PRIORITY** - Load testing with multiple concurrent users

**Day 5: Deployment and Monitoring**
- [ ] **HIGH PRIORITY** - Deploy hybrid solution to production
- [ ] **HIGH PRIORITY** - Monitor for any issues with dynamic data sources
- [ ] **HIGH PRIORITY** - Collect user feedback on hybrid approach
- [ ] **HIGH PRIORITY** - Performance monitoring and optimization

## Testing Strategy

### Unit Tests

```python
def test_data_sources_manager_creation():
    """Test DataSourcesManager initialization and default data source creation."""
    manager = DataSourcesManager()
    
    # Should have one default MDS data source
    assert len(manager.data_sources) == 1
    assert manager.data_sources[0]["data_source_type"] == "MDS"
    assert manager.data_sources[0]["data_source_name"] == "RAW_MDS_NA"

def test_add_remove_data_sources():
    """Test adding and removing data sources."""
    manager = DataSourcesManager()
    
    # Add EDX data source
    manager.add_data_source("EDX")
    assert len(manager.data_sources) == 2
    assert manager.data_sources[1]["data_source_type"] == "EDX"
    
    # Remove data source
    manager.remove_data_source(1)
    assert len(manager.data_sources) == 1
    
    # Cannot remove last data source
    manager.remove_data_source(0)
    assert len(manager.data_sources) == 1

def test_data_source_field_templates():
    """Test data source field templates for all types."""
    for source_type in ["MDS", "EDX", "ANDES"]:
        template = DATA_SOURCE_FIELD_TEMPLATES[source_type]
        
        # Should have required and optional fields
        assert "required_fields" in template
        assert "field_definitions" in template
        assert len(template["required_fields"]) > 0
        
        # All required fields should be in field definitions
        for field_name in template["required_fields"]:
            assert field_name in template["field_definitions"]

def test_hybrid_field_definition():
    """Test hybrid field definition with sub-config grouping."""
    fields = get_cradle_data_loading_fields_hybrid()
    
    # Should have fields from all sections
    sections = set(f.get("section", "inherited") for f in fields)
    expected_sections = {"inherited", "data_sources_spec", "transform_spec", "output_spec", "cradle_job_spec", "root", "advanced"}
    assert sections == expected_sections
    
    # Should have dynamic data sources field
    dynamic_fields = [f for f in fields if f["type"] == "dynamic_data_sources"]
    assert len(dynamic_fields) == 1
    assert dynamic_fields[0]["section"] == "data_sources_spec"

def test_multiple_data_sources_transformation():
    """Test transformation of multiple data sources to config structure."""
    form_data = {
        "job_type": "training",
        "start_date": "2025-01-01T00:00:00",
        "end_date": "2025-04-17T00:00:00",
        "data_sources": [
            {
                "data_source_name": "RAW_MDS_NA",
                "data_source_type": "MDS",
                "service_name": "AtoZ",
                "region": "NA",
                "output_schema": ["objectId", "transactionDate"]
            },
            {
                "data_source_name": "RAW_EDX_EU",
                "data_source_type": "EDX",
                "edx_provider": "provider1",
                "edx_subject": "subject1",
                "edx_dataset": "dataset1",
                "edx_manifest_key": '["manifest1"]'
            }
        ],
        "transform_sql": "SELECT * FROM mds_source JOIN edx_source",
        "output_schema": ["objectId", "transactionDate", "is_abuse"],
        "cradle_account": "test-account"
    }
    
    ui_data = _transform_cradle_form_data_hybrid(form_data)
    
    # Verify multiple data sources structure
    assert "data_sources_spec" in ui_data
    assert len(ui_data["data_sources_spec"]["data_sources"]) == 2
    
    # Verify first data source (MDS)
    mds_source = ui_data["data_sources_spec"]["data_sources"][0]
    assert mds_source["data_source_type"] == "MDS"
    assert "mds_data_source_properties" in mds_source
    
    # Verify second data source (EDX)
    edx_source = ui_data["data_sources_spec"]["data_sources"][1]
    assert edx_source["data_source_type"] == "EDX"
    assert "edx_data_source_properties" in edx_source
```

### Integration Tests

```python
def test_hybrid_cradle_form_rendering():
    """Test that hybrid cradle config renders with sub-config sections."""
    form_data = {
        "config_class": CradleDataLoadingConfig,
        "config_class_name": "CradleDataLoadingConfig",
        "fields": get_cradle_data_loading_fields_hybrid(),
        "values": {},
        "pre_populated_instance": None
    }
    
    widget = UniversalConfigWidget(form_data, is_final_step=True)
    
    # Should render without errors
    widget.render()
    
    # Should have dynamic data sources widget
    assert "data_sources" in widget.widgets
    assert hasattr(widget.widgets["data_sources"], "get_all_data_sources")

def test_multiple_data_sources_config_creation():
    """Test creating CradleDataLoadingConfig from multiple data sources."""
    # Simulate form submission with multiple data sources
    form_data = create_test_multiple_data_sources_form_data()
    ui_data = _transform_cradle_form_data_hybrid(form_data)
    
    # Should create valid config with multiple data sources
    config = CradleDataLoadingConfig(**ui_data)
    
    # Verify config structure
    assert config.job_type == "training"
    assert config.data_sources_spec is not None
    assert len(config.data_sources_spec.data_sources) == 2
    assert config.data_sources_spec.data_sources[0].data_source_type == "MDS"
    assert config.data_sources_spec.data_sources[1].data_source_type == "EDX"
```

## Success Metrics and Validation

### **Technical Metrics**
- **Dynamic Data Sources Support**: 100% (multiple data sources with add/remove functionality)
- **Sub-Config Organization**: 100% (fields grouped by data_sources_spec, transform_spec, output_spec, cradle_job_spec)
- **Code Reuse**: 85%+ (leverages existing single-page form infrastructure)
- **Test Coverage**: >95% for dynamic data sources functionality
- **Performance**: Same or better than static approach

### **User Experience Metrics**
- **Configuration Flexibility**: 100% (users can configure any number of data sources)
- **Type-Specific Fields**: 100% (MDS/EDX/ANDES fields show correctly based on type)
- **Add/Remove UX**: Smooth and intuitive data source management
- **Form Organization**: Clear sub-config grouping improves understanding
- **User Satisfaction**: >4.5/5 (target)

### **Integration Quality Metrics**
- **Backward Compatibility**: 100% (no breaking changes to other config steps)
- **Field Completeness**: 100% (all original fields preserved with sub-config organization)
- **Data Integrity**: 100% (correct nested config structure with multiple data sources)
- **Workflow Integration**: 100% (seamless MultiStepWizard integration)

## Risk Assessment: **LOW-MEDIUM RISK**

### **Risk Level: LOW-MEDIUM**
- **Technical Risk**: Low-Medium (new dynamic widget system, but building on proven infrastructure)
- **User Experience Risk**: Low (hybrid approach maintains familiar patterns)
- **Performance Risk**: Low (dynamic widgets have minimal overhead)
- **Compatibility Risk**: Low (backward compatible with existing workflows)

### **Mitigation Strategies**
- **Comprehensive Testing**: End-to-end testing of dynamic data sources functionality
- **Gradual Rollout**: Phased deployment with monitoring
- **Fallback Plan**: Can revert to static single data source if needed
- **User Training**: Documentation and examples for dynamic data sources

### **Contingency Plans**
- **If dynamic widgets don't work**: Fall back to static multiple data source fields
- **If performance issues**: Optimize widget rendering and data collection
- **If user feedback negative**: Iterate on UX based on specific feedback
- **If validation fails**: Enhance error handling and validation messages

## Expected Benefits and Impact

### **Immediate Benefits (Week 1-2)**
- **Multiple Data Sources Support**: Users can configure any number of data sources
- **Type-Specific Fields**: Proper field sets for MDS/EDX/ANDES data source types
- **Dynamic Add/Remove**: Intuitive data source management functionality
- **Sub-Config Organization**: Clear grouping by configuration structure

### **Short-term Benefits (Month 1-2)**
- **Better User Experience**: More flexible and powerful data source configuration
- **Reduced Configuration Errors**: Type-specific validation for each data source
- **Improved Understanding**: Sub-config grouping clarifies configuration structure
- **Enhanced Productivity**: Faster configuration with dynamic management

### **Long-term Benefits (Month 3+)**
- **Foundation for Advanced Features**: Easier to add data source templates, validation, etc.
- **Scalable Architecture**: Easy to add new data source types
- **Consistent Patterns**: Establishes patterns for other dynamic configuration sections
- **Better Developer Experience**: Clear architecture for dynamic widget development

## Future Enhancements

### Advanced Dynamic Features

1. **Data Source Templates**: Pre-built templates for common data source configurations
2. **Drag-and-Drop Reordering**: Allow users to reorder data sources
3. **Data Source Validation**: Real-time validation for each data source
4. **Import/Export**: Import data sources from JSON or export for reuse
5. **Data Source Preview**: Preview data source configuration before adding

### Enhanced UX Features

1. **Collapsible Data Sources**: Allow users to collapse configured data sources
2. **Data Source Duplication**: Duplicate existing data sources with modifications
3. **Bulk Operations**: Add multiple data sources at once
4. **Configuration Wizard**: Step-by-step wizard for complex data source setup
5. **Smart Defaults**: Intelligent defaults based on existing data sources

### Integration Improvements

1. **Better Inheritance**: Inherit data source configurations from parent configs
2. **Validation Integration**: Enhanced validation with data source-specific rules
3. **Export Options**: Export data sources separately or as part of complete config
4. **Version Control**: Track changes to data source configurations
5. **Configuration History**: Maintain history of data source changes

## Conclusion

### **Key Discovery: Hybrid Approach Optimal for Dynamic Data Sources**

The comprehensive analysis reveals that **the hybrid approach with sub-config grouping is the optimal solution** for supporting multiple dynamic data sources while maintaining the benefits of the single-page form architecture.

### **Strategic Benefits of Hybrid Dynamic Data Sources Enhancement**

1. **✅ Preserves Completed Work**: Leverages 85%+ of the 2025-10-08 single-page refactoring
2. **✅ Focused Enhancement**: Dynamic functionality isolated to Data Sources section only
3. **✅ Sub-Config Organization**: Fields grouped by actual configuration structure (data_sources_spec, transform_spec, output_spec, cradle_job_spec)
4. **✅ Multiple Data Sources Support**: Native support for any number of MDS/EDX/ANDES data sources
5. **✅ Type-Specific Fields**: Proper field sets for each data source type
6. **✅ Enhanced User Experience**: Dynamic add/remove functionality with clear organization

### **Implementation Approach: Targeted Enhancement**

**Recommended Strategy:**
1. **Phase 1 (Week 1)**: Create DataSourcesManager and integrate with UniversalConfigWidget
2. **Phase 2 (Week 2)**: Enhance data transformation for multiple data sources
3. **Phase 3 (Week 3)**: Comprehensive testing and validation
4. **Phase 4 (Week 4)**: Documentation and production deployment

**Risk Mitigation:**
- **Low-Medium Technical Risk**: Building on proven single-page form infrastructure
- **Backward Compatibility**: No breaking changes to existing workflows
- **Focused Implementation**: Only Data Sources section needs dynamic functionality
- **Easy Rollback**: Can revert to static approach if needed

### **Expected Outcomes**

**Technical Improvements:**
- **100% Multiple Data Sources Support**: Users can configure any number of data sources
- **100% Type-Specific Field Support**: MDS/EDX/ANDES fields show correctly based on type
- **85%+ Code Reuse**: Leverages existing single-page form infrastructure
- **95%+ Test Coverage**: Comprehensive testing for dynamic functionality

**User Experience Improvements:**
- **Dynamic Data Source Management**: Intuitive add/remove functionality
- **Sub-Config Organization**: Clear grouping by configuration structure improves understanding
- **Type-Specific Validation**: Proper validation for each data source type
- **Enhanced Flexibility**: Support for complex multi-data-source configurations

**Development Benefits:**
- **Focused Complexity**: Dynamic functionality isolated to one section
- **Maintainable Architecture**: Clear separation between static and dynamic sections
- **Extensible Design**: Easy to add new data source types or dynamic sections
