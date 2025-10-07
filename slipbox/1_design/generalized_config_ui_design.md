---
tags:
  - design
  - ui
  - configuration
  - user-interface
  - generalization
  - architecture
keywords:
  - config
  - ui
  - user interface
  - form
  - wizard
  - multi-step
  - hierarchical config
  - jupyter widget
  - generalized
  - from_base_config
  - pipeline_dag
topics:
  - user interface design
  - configuration management
  - form design
  - wizard interface
  - jupyter integration
  - generalization architecture
language: python, javascript, html, css
date of note: 2025-10-06
---

# Generalized Config UI Design - Universal Configuration Interface System

## Overview

This document describes the design for a **generalized configuration UI system** that extends the successful Cradle Data Load Config UI pattern to support **all configuration types** in the Cursus framework. The system provides a universal interface for creating, editing, and managing any configuration that follows the `BasePipelineConfig` pattern with `.from_base_config()` method support.

**Status: 🎯 DESIGN PHASE - Ready for Implementation**

## Requirements

### Functional Requirements

#### R1: PipelineDAG-Driven Configuration Discovery
- **Primary Input**: Users provide a `PipelineDAG` as input to the UI system
- **Smart Filtering**: UI automatically discovers and displays only the configuration classes required by the specific DAG nodes
- **Relevance Focus**: Users see only configurations relevant to their pipeline, eliminating confusion from unused config types
- **Dynamic Discovery**: Configuration list updates automatically based on DAG structure changes

#### R2: 3-Tier Configuration Architecture Support
- **Tier 1 (Essential User Inputs)**: Required fields marked with `*`, no defaults, must be filled by user
- **Tier 2 (System Inputs)**: Optional fields with defaults, pre-populated but user-modifiable
- **Tier 3 (Derived Fields)**: Private/computed fields completely hidden from UI, calculated automatically

#### R3: Universal Configuration Support
- **All Config Types**: Support any configuration class inheriting from `BasePipelineConfig`
- **Automatic UI Generation**: Generate forms automatically from configuration class definitions using introspection
- **Seamless Integration**: Work with existing `.from_base_config()` inheritance patterns
- **Backward Compatibility**: Existing manual configuration creation continues to work

#### R4: Hierarchical Configuration Workflow
- **Progressive Disclosure**: Users fill common fields once (Base Config), then specific fields for each step
- **Inheritance Pre-population**: Derived configurations automatically inherit and pre-populate fields from parent configs
- **Registry-Based Routing**: Use step registry to determine correct inheritance patterns (Base-only vs Processing-based)
- **Visual Inheritance Indicators**: Clear display of which fields are inherited vs. new

#### R5: Specialized Configuration Handling
- **Complex Config Detection**: Automatically detect configurations requiring specialized UI (e.g., CradleDataLoadConfig)
- **Specialized UI Integration**: Seamlessly integrate with existing specialized UIs (cradle_ui)
- **Sub-config Filtering**: Hide sub-configurations from main discovery to prevent user confusion
- **Unified Experience**: Maintain consistent experience across simple and complex configurations

### Non-Functional Requirements

#### Performance Requirements
- **Fast Discovery**: Configuration discovery and UI generation < 2 seconds
- **Responsive UI**: Form interactions and field updates < 100ms response time
- **Efficient Rendering**: Support 50+ configuration fields without performance degradation

#### Usability Requirements
- **Intuitive Workflow**: Users can complete configuration without documentation
- **Error Prevention**: Real-time validation prevents invalid configurations
- **Visual Clarity**: Clear distinction between required, optional, and inherited fields
- **Mobile Responsive**: Functional on tablet and desktop devices

#### Technical Requirements
- **Framework Integration**: Seamless integration with existing Cursus step catalog and registry systems
- **Validation Consistency**: All Pydantic validation rules preserved and enforced
- **Security**: Input sanitization and validation to prevent injection attacks
- **Extensibility**: Easy addition of new configuration types without UI code changes

## User Experience Design

### Primary User Journey: PipelineDAG-Driven Configuration

The core user experience centers around providing a `PipelineDAG` as input, which drives the entire configuration process:

#### Step 1: Pipeline DAG Input & Analysis
```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Universal Configuration UI - Pipeline-Driven Approach   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📋 Step 1: Provide Your Pipeline DAG                       │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📁 Upload DAG File:                                     │ │
│ │ [Choose File] my_xgboost_pipeline.py                    │ │
│ │                                                         │ │
│ │ OR                                                      │ │
│ │                                                         │ │
│ │ 🔗 Import from Catalog:                                │ │
│ │ [Select DAG ▼] XGBoost Complete E2E Pipeline           │ │
│ │                                                         │ │
│ │ OR                                                      │ │
│ │                                                         │ │
│ │ 💻 Provide DAG Object:                                 │ │
│ │ pipeline_dag = create_xgboost_complete_e2e_dag()        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [Analyze Pipeline DAG] [Preview DAG Structure]             │
└─────────────────────────────────────────────────────────────┘
```

#### Step 2: DAG Analysis & Relevant Configuration Discovery
```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Pipeline Analysis Results                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 🔍 Discovered Pipeline Steps:                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Step 1: cradle_data_loading                             │ │
│ │ Step 2: tabular_preprocessing_training                  │ │
│ │ Step 3: xgboost_training                                │ │
│ │ Step 4: xgboost_model_creation                          │ │
│ │ Step 5: model_registration                              │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ⚙️ Required Configurations (Only These Will Be Shown):     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ✅ CradleDataLoadConfig                                 │ │
│ │ ✅ TabularPreprocessingConfig                           │ │
│ │ ✅ XGBoostTrainingConfig                                │ │
│ │ ✅ XGBoostModelConfig                                   │ │
│ │ ✅ RegistrationConfig                                   │ │
│ │                                                         │ │
│ │ ❌ Hidden: 47 other config types not needed            │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📋 Configuration Workflow:                                 │
│ Base Config → Processing Config → 5 Specific Configs       │
│                                                             │
│ [Start Configuration Workflow]                             │
└─────────────────────────────────────────────────────────────┘
```

#### Step 3: Hierarchical Configuration Workflow
```
┌─────────────────────────────────────────────────────────────┐
│ 🏗️ Configuration Workflow - Step 1 of 7                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📋 Base Pipeline Configuration (Required for All Steps)    │
│                                                             │
│ ┌─ 🔥 Essential User Inputs (Tier 1) ─────────────────────┐ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────┐ │ │
│ │ │ 👤 author *                     │ │ 🪣 bucket *     │ │ │
│ │ │ [empty - user must fill]        │ │ [empty]         │ │ │
│ │ └─────────────────────────────────┘ └─────────────────┘ │ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────┐ │ │
│ │ │ 🔐 role *                       │ │ 🌍 region *     │ │ │
│ │ │ [empty - user must fill]        │ │ [NA ▼]          │ │ │
│ │ └─────────────────────────────────┘ └─────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─ ⚙️ System Inputs (Tier 2) ─────────────────────────────┐ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────┐ │ │
│ │ │ 🎯 model_class                  │ │ 📅 current_date │ │ │
│ │ │ [xgboost] (pre-filled)          │ │ [2025-10-07]    │ │ │
│ │ └─────────────────────────────────┘ └─────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Progress: ●○○○○○○ (1/7)                                     │
│ [Continue to Processing Config]                            │
└─────────────────────────────────────────────────────────────┘
```

#### Step 4: Processing Configuration (Conditional)
```
┌─────────────────────────────────────────────────────────────┐
│ ⚙️ Configuration Workflow - Step 2 of 7                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📋 Processing Configuration (For Processing-Based Steps)   │
│                                                             │
│ ┌─ 💾 Inherited from Base Config ─────────────────────────┐ │
│ │ • 👤 Author: john-doe    • 🪣 Bucket: my-bucket        │ │
│ │ • 🔐 Role: MyRole        • 🌍 Region: NA                │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─ ⚙️ Processing-Specific Fields ─────────────────────────┐ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────┐ │ │
│ │ │ 🖥️ instance_type                │ │ 📊 volume_size  │ │ │
│ │ │ [ml.m5.2xlarge] (default)       │ │ [500] GB        │ │ │
│ │ └─────────────────────────────────┘ └─────────────────┘ │ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────┐ │ │
│ │ │ 📁 processing_source_dir        │ │ 🎯 entry_point  │ │ │
│ │ │ [src/processing]                │ │ [main.py]       │ │ │
│ │ └─────────────────────────────────┘ └─────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Progress: ●●○○○○○ (2/7)                                     │
│ [Continue to Step-Specific Configs]                       │
└─────────────────────────────────────────────────────────────┘
```

#### Step 5: Step-Specific Configurations (DAG-Driven)
```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Configuration Workflow - Step 3 of 7                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📋 CradleDataLoadConfig (Step: cradle_data_loading)        │
│                                                             │
│ ┌─ 🎛️ Specialized Configuration ─────────────────────────┐ │
│ │ This step uses a specialized 4-step wizard interface:  │ │
│ │                                                         │ │
│ │ 1️⃣ Data Sources Configuration                          │ │
│ │ 2️⃣ Transform Specification                             │ │
│ │ 3️⃣ Output Configuration                                │ │
│ │ 4️⃣ Cradle Job Settings                                 │ │
│ │                                                         │ │
│ │ [Open CradleDataLoadConfig Wizard]                     │ │
│ │ (Base config will be pre-filled automatically)        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Progress: ●●●○○○○ (3/7)                                     │
│ [Continue to Next Step] [Skip This Step]                  │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Configuration Workflow - Step 4 of 7                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📋 TabularPreprocessingConfig (Step: preprocessing)        │
│                                                             │
│ ┌─ 💾 Inherited Configuration ────────────────────────────┐ │
│ │ Auto-filled from Base + Processing Config:              │ │
│ │ • 👤 Author: john-doe    • 🖥️ Instance: ml.m5.2xlarge  │ │
│ │ • 📁 Source: src/processing                             │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─ 🎯 Step-Specific Fields ───────────────────────────────┐ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────┐ │ │
│ │ │ 🏷️ job_type *                   │ │ 🎯 label_name * │ │ │
│ │ │ [training ▼]                    │ │ [is_abuse]      │ │ │
│ │ └─────────────────────────────────┘ └─────────────────┘ │ │
│ │                                                         │ │
│ │ 📊 Feature Selection:                                   │ │
│ │ ☑ PAYMETH  ☑ claim_reason  ☐ claimantInfo_status      │ │
│ │ ☑ claimAmount_value  ☑ COMP_DAYOB  ☐ shipment_weight  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Progress: ●●●●○○○ (4/7)                                     │
│ [Continue to Next Step]                                    │
└─────────────────────────────────────────────────────────────┘
```

#### Step 6: Configuration Completion & Export
```
┌─────────────────────────────────────────────────────────────┐
│ ✅ Configuration Complete - All Steps Configured           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📋 Configuration Summary:                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ✅ Base Configuration (BasePipelineConfig)             │ │
│ │ ✅ Processing Configuration (ProcessingStepConfigBase) │ │
│ │ ✅ CradleDataLoadConfig                                 │ │
│ │ ✅ TabularPreprocessingConfig                           │ │
│ │ ✅ XGBoostTrainingConfig                                │ │
│ │ ✅ XGBoostModelConfig                                   │ │
│ │ ✅ RegistrationConfig                                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 🎯 Ready for Pipeline Execution:                           │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ config_list = [                                         │ │
│ │     base_config,                                        │ │
│ │     processing_step_config,                             │ │
│ │     cradle_data_load_config,                            │ │
│ │     tabular_preprocessing_config,                       │ │
│ │     xgboost_training_config,                            │ │
│ │     xgboost_model_config,                               │ │
│ │     registration_config                                 │ │
│ │ ]                                                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [💾 Export Configuration] [🚀 Execute Pipeline]            │
│ [📋 Save as Template] [🔄 Modify Configuration]           │
└─────────────────────────────────────────────────────────────┘
```

### Key User Experience Benefits

#### 🎯 **DAG-Driven Relevance**
- **Focused Experience**: Users see only configurations needed for their specific pipeline
- **Reduced Cognitive Load**: No confusion from 50+ unused configuration types
- **Intelligent Filtering**: System automatically determines required configs from DAG structure
- **Dynamic Updates**: Configuration list updates when DAG changes

#### 🔄 **Progressive Configuration Flow**
- **Logical Sequence**: Base → Processing → Step-specific configurations
- **Inheritance Visualization**: Clear display of inherited vs. new fields
- **Smart Pre-population**: Fields automatically filled from parent configurations
- **Validation at Each Step**: Immediate feedback prevents errors early

#### 🎨 **Modern, Intuitive Interface**
- **Visual Progress Tracking**: Clear progress indicators (●●●○○○○)
- **Contextual Help**: Field descriptions and inheritance summaries
- **Responsive Design**: Works on desktop and tablet devices
- **Consistent Experience**: Unified interface across all configuration types

## UI Layout Design

### Modern Card-Based Layout Architecture

The UI employs a **modern, compact yet vivid card-based layout** that emphasizes visual hierarchy, micro-interactions, and delightful user experience.

#### Visual Design Principles

**1. 🎨 Vivid Visual Hierarchy**
- **Emoji Icons**: Each field has contextual emoji for instant recognition (👤 author, 🪣 bucket, 🔐 role)
- **Color-Coded Sections**: Different gradient backgrounds for logical grouping
- **Card-Based Layout**: Elevated cards with subtle shadows and rounded corners
- **Progressive Disclosure**: Collapsible sections for advanced options

**2. 🚀 Interactive Elements**
- **Smart Toggles**: Visual toggle switches instead of plain checkboxes
- **Progress Indicators**: Slider bars for numeric values with visual feedback
- **Multi-Select Cards**: Feature selection with card-based checkboxes
- **Hover Animations**: Subtle micro-interactions on field focus

**3. 📱 Modern Input Patterns**
- **Floating Labels**: Labels that animate above input fields when focused
- **Contextual Validation**: Real-time validation with inline success/error states
- **Smart Dropdowns**: Searchable dropdowns with icons and descriptions
- **File Picker Integration**: Drag-and-drop file selection with preview

#### Layout Structure Example

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🎯 TabularPreprocessingConfig                                    [⚙️ Settings] │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ ┌─ 🔥 Essential Configuration (Tier 1) ───────────────────────────────────────┐ │
│ │                                                                             │ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────────────────────┐     │ │
│ │ │ 👤 author *                     │ │ 🪣 bucket *                     │     │ │
│ │ │ ┌─────────────────────────────┐ │ │ ┌─────────────────────────────┐ │     │ │
│ │ │ │ john-doe                    │ │ │ │ my-pipeline-bucket          │ │     │ │
│ │ │ └─────────────────────────────┘ │ │ └─────────────────────────────┘ │     │ │
│ │ │ Pipeline author or owner        │ │ S3 bucket for pipeline assets   │     │ │
│ │ └─────────────────────────────────┘ └─────────────────────────────────┘     │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ ┌─ 🎯 Specific Configuration ──────────────────────────────────────────────────┐ │
│ │                                                                             │ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────────────────────┐     │ │
│ │ │ 🏷️ job_type *                   │ │ 🎯 label_name *                 │     │ │
│ │ │ ┌─────────────────────────────┐ │ │ ┌─────────────────────────────┐ │     │ │
│ │ │ │ training            ▼       │ │ │ │ is_abuse                    │ │     │ │
│ │ │ └─────────────────────────────┘ │ │ └─────────────────────────────┘ │     │ │
│ │ │ Processing job type             │ │ Target label column name        │     │ │
│ │ └─────────────────────────────────┘ └─────────────────────────────────┘     │ │
│ │                                                                             │ │
│ │ ┌─────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ 📊 Feature Selection                                                    │ │ │
│ │ │ ┌─────────────────────────────┐ ┌─────────────────────────────────┐   │ │ │
│ │ │ │ 🔤 Categorical Features     │ │ 🔢 Numerical Features           │   │ │ │
│ │ │ │ ☑ PAYMETH                   │ │ ☑ claimAmount_value             │   │ │ │
│ │ │ │ ☑ claim_reason              │ │ ☑ COMP_DAYOB                    │   │ │ │
│ │ │ │ ☐ claimantInfo_status       │ │ ☐ shipment_weight               │   │ │ │
│ │ │ └─────────────────────────────┘ └─────────────────────────────────┘   │ │ │
│ │ └─────────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ ┌─ 💾 Inherited Configuration ─────────────────────────────────────────────────┐ │
│ │ 📋 Auto-filled from Base + Processing Config:                              │ │
│ │ • 👤 Author: john-doe                    • 🪣 Bucket: my-pipeline-bucket   │ │
│ │ • 🔐 Role: arn:aws:iam::123:role/MyRole  • 🌍 Region: NA                   │ │
│ │ • 🖥️ Instance: ml.m5.2xlarge             • 📁 Source: src/processing       │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                        [💾 Save Configuration] [📤 Export JSON]            │ │
│ │                        [🔄 Reset to Defaults] [👁️ Preview Config]          │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### CSS Implementation (Modern Card-Based)

```css
/* Modern card-based layout */
.config-section {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
    margin-bottom: 24px;
    overflow: hidden;
    transition: all 0.3s ease;
}

.config-section:hover {
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    transform: translateY(-2px);
}

/* Section headers with gradients */
.field-group-section.essential {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
}

.field-group-section.processing {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-left: 4px solid #3b82f6;
}

.field-group-section.specific {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border-left: 4px solid #10b981;
}

.field-group-section.inherited {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    border-left: 4px solid #8b5cf6;
}

/* Modern form grid */
.form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

/* Enhanced field styling */
.field-group {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    position: relative;
}

.field-group.required {
    border-left: 4px solid #ef4444;
}

.field-group.required::before {
    content: "✱";
    position: absolute;
    top: 8px;
    right: 12px;
    color: #ef4444;
    font-weight: bold;
}

/* Modern input styling */
.form-input {
    width: 100%;
    padding: 12px 16px;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 14px;
    transition: all 0.3s ease;
    background: #ffffff;
}

.form-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Toggle switches */
.toggle-switch {
    position: relative;
    display: inline-block;
    width: 48px;
    height: 24px;
}

.toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #cbd5e1;
    transition: 0.3s;
    border-radius: 24px;
}

input:checked + .toggle-slider {
    background-color: #10b981;
}

/* Feature selection cards */
.feature-selection {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin: 16px 0;
}

.feature-card {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.feature-card.selected {
    border-color: #10b981;
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
}

/* Action buttons */
.config-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
    padding: 20px;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 0 0 16px 16px;
}

.btn {
    padding: 12px 24px;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.btn-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

.btn-secondary {
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
    color: white;
}

.btn-success {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
}

/* Responsive design */
@media (max-width: 768px) {
    .form-row {
        grid-template-columns: 1fr;
        gap: 16px;
    }
    
    .feature-selection {
        grid-template-columns: 1fr;
    }
    
    .config-actions {
        flex-direction: column;
    }
}
```

### 3-Tier Configuration Architecture & Field Display Strategy

**✅ DESIGN DECISION: Based on actual code analysis of `src/cursus/core/base/config_base.py`**

The Cursus framework implements a sophisticated **3-tier configuration architecture** that determines which fields should be displayed in the UI and how they should be presented to users.

#### **3-Tier Architecture Overview**

From examining `BasePipelineConfig` and derived classes, the configuration system follows this structure:

**Tier 1: Essential User Inputs (Required Fields)**
- Fields that users **must** explicitly provide
- No default values - require user input
- Marked with `*` in UI to indicate required status
- Detected via `field_info.is_required()` returning `True`

**Tier 2: System Inputs with Defaults (Optional Fields)**  
- Fields with reasonable defaults that users can override
- Pre-populated in UI with default values
- Users can modify if needed for customization
- Detected via `field_info.is_required()` returning `False`

**Tier 3: Derived Fields (Private/Computed)**
- Private attributes with public property accessors
- Computed automatically from Tier 1 + Tier 2 fields
- **NEVER displayed in UI** - completely hidden from users
- Detected via `PrivateAttr` or property methods

#### **Actual Code Implementation**

The configuration classes provide a built-in method for field categorization:

```python
# From BasePipelineConfig.categorize_fields()
def categorize_fields(self) -> Dict[str, List[str]]:
    """Categorize all fields into three tiers"""
    categories = {
        "essential": [],  # Tier 1: Required, public
        "system": [],     # Tier 2: Optional (has default), public  
        "derived": []     # Tier 3: Public properties (HIDDEN from UI)
    }
    
    model_fields = self.__class__.model_fields
    
    for field_name, field_info in model_fields.items():
        if field_name.startswith("_"):
            continue  # Skip private fields
            
        if field_info.is_required():
            categories["essential"].append(field_name)
        else:
            categories["system"].append(field_name)
    
    # Find derived properties (hidden from UI)
    for attr_name in dir(self):
        if (not attr_name.startswith("_") 
            and attr_name not in model_fields
            and isinstance(getattr(type(self), attr_name, None), property)):
            categories["derived"].append(attr_name)
    
    return categories
```

## Technical Architecture

### Universal Configuration Engine with 3-Tier Support

```python
class UniversalConfigCore:
    """Core engine for universal configuration management with 3-tier architecture support."""
    
    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        """Initialize with existing step catalog infrastructure."""
        from cursus.step_catalog.step_catalog import StepCatalog
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
        
        # Simple field type mapping
        self.field_types = {
            str: "text", int: "number", float: "number", bool: "checkbox",
            list: "list", dict: "keyvalue"
        }
    
    def create_pipeline_config_widget(self, pipeline_dag: PipelineDAG, **kwargs):
        """
        Create DAG-driven pipeline configuration widget with inheritance support.
        
        Uses the same infrastructure as DynamicPipelineTemplate but for discovery
        rather than resolution of existing configurations.
        """
        # Use existing StepConfigResolverAdapter (matches production pattern)
        from cursus.step_catalog.adapters.config_resolver import StepConfigResolverAdapter
        resolver = StepConfigResolverAdapter()
        
        # Extract DAG nodes (matches DynamicPipelineTemplate._create_config_map pattern)
        dag_nodes = list(pipeline_dag.nodes)
        
        # Discover required config classes (UI-specific, not resolution)
        required_config_classes = self._discover_required_config_classes(dag_nodes, resolver)
        
        # Create multi-step wizard with inheritance support
        workflow_steps = self._create_workflow_structure(required_config_classes)
        
        return MultiStepWizard(workflow_steps)
    
    def _discover_required_config_classes(self, dag_nodes: List[str], resolver: StepConfigResolverAdapter) -> List[Dict]:
        """
        Discover what configuration classes are needed for the DAG nodes.
        
        This is different from production resolve_config_map() because:
        - Production: Maps nodes to existing config instances from saved file
        - UI: Discovers what config classes users need to create from scratch
        
        Args:
            dag_nodes: List of DAG node names (extracted same as production)
            resolver: StepConfigResolverAdapter instance
            
        Returns:
            List of required configuration class information
        """
        required_configs = []
        
        for node_name in dag_nodes:
            # Use step catalog to determine required config class
            step_info = resolver.catalog.get_step_info(node_name)
            
            if step_info and step_info.config_class:
                config_class = resolver.catalog.get_config_class(step_info.config_class)
                if config_class:
                    required_configs.append({
                        "node_name": node_name,
                        "config_class_name": step_info.config_class,
                        "config_class": config_class,
                        "inheritance_pattern": self._get_inheritance_pattern(config_class),
                        "is_specialized": self._is_specialized_config(config_class)
                    })
            else:
                # Fallback: Try to infer from node name patterns
                inferred_config = self._infer_config_class_from_node_name(node_name, resolver)
                if inferred_config:
                    required_configs.append(inferred_config)
        
        return required_configs
    
    def _infer_config_class_from_node_name(self, node_name: str, resolver: StepConfigResolverAdapter) -> Optional[Dict]:
        """
        Fallback method to infer config class from node name patterns.
        
        Uses similar pattern matching logic as StepConfigResolverAdapter
        but for discovering requirements rather than resolving instances.
        """
        # Use resolver's pattern matching capabilities
        try:
            # Get all available config classes from catalog
            available_config_classes = resolver.catalog.discover_config_classes()
            
            # Use resolver's pattern matching to find best match
            for class_name, config_class in available_config_classes.items():
                # Simple heuristic: check if node name contains config type keywords
                config_base = class_name.lower().replace("config", "").replace("step", "")
                if config_base in node_name.lower():
                    return {
                        "node_name": node_name,
                        "config_class_name": class_name,
                        "config_class": config_class,
                        "inheritance_pattern": self._get_inheritance_pattern(config_class),
                        "is_specialized": self._is_specialized_config(config_class),
                        "inferred": True
                    }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not infer config class for node {node_name}: {e}")
            return None
    
    def _create_workflow_structure(self, required_configs: List[Dict]) -> List[Dict]:
        """Create logical workflow structure for configuration steps."""
        workflow_steps = []
        
        # Step 1: Always start with Base Configuration
        workflow_steps.append({
            "step_number": 1,
            "title": "Base Configuration",
            "config_class": BasePipelineConfig,
            "type": "base",
            "required": True
        })
        
        # Step 2: Add Processing Configuration if any configs need it
        processing_based_configs = [
            config for config in required_configs 
            if config["inheritance_pattern"] == "processing_based"
        ]
        
        if processing_based_configs:
            workflow_steps.append({
                "step_number": 2,
                "title": "Processing Configuration",
                "config_class": ProcessingStepConfigBase,
                "type": "processing",
                "required": True
            })
        
        # Step 3+: Add specific configurations
        step_number = len(workflow_steps) + 1
        for config in required_configs:
            workflow_steps.append({
                "step_number": step_number,
                "title": config["config_class_name"],
                "config_class": config["config_class"],
                "step_name": config["node_name"],
                "type": "specific",
                "inheritance_pattern": config["inheritance_pattern"],
                "is_specialized": config["is_specialized"],
                "required": True
            })
            step_number += 1
        
        return workflow_steps
    
    def _get_form_fields_with_tiers(self, config_class: Type[BasePipelineConfig], field_categories: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract form fields with 3-tier categorization - only Tier 1 + Tier 2."""
        fields = []
        
        # Only include essential (Tier 1) and system (Tier 2) fields
        # Derived fields (Tier 3) are completely excluded from UI
        fields_to_include = field_categories["essential"] + field_categories["system"]
        
        for field_name, field_info in config_class.model_fields.items():
            if field_name in fields_to_include:
                fields.append({
                    "name": field_name,
                    "type": self.field_types.get(field_info.annotation, "text"),
                    "required": field_info.is_required(),  # True for Tier 1, False for Tier 2
                    "tier": "essential" if field_info.is_required() else "system",
                    "description": field_info.description or "",
                    "default": field_info.default if hasattr(field_info, 'default') else None
                })
        
        return fields
    
    def _get_inheritance_pattern(self, config_class: Type[BasePipelineConfig]) -> str:
        """Determine inheritance pattern for a configuration class."""
        # Check if config inherits from ProcessingStepConfigBase
        for base_class in config_class.__mro__:
            if base_class.__name__ == "ProcessingStepConfigBase":
                return "processing_based"
        
        # Special handling for CradleDataLoadConfig
        if config_class.__name__ == "CradleDataLoadConfig":
            return "base_only_specialized"
        
        # Default: inherits from BasePipelineConfig only
        return "base_only"
    
    def _is_specialized_config(self, config_class: Type[BasePipelineConfig]) -> bool:
        """Check if configuration requires specialized UI."""
        specialized_configs = {
            "CradleDataLoadConfig": True,
            # Add other specialized configs here as needed
        }
        return specialized_configs.get(config_class.__name__, False)
```

### PipelineDAG-Driven Configuration Discovery

The system's core innovation is using `PipelineDAG` as the primary input to drive configuration discovery and UI generation:

```python
class DAGConfigurationManager:
    """Manages PipelineDAG-driven configuration discovery and UI generation."""
    
    def __init__(self, step_catalog: StepCatalog):
        self.step_catalog = step_catalog
        self.config_resolver = StepConfigResolverAdapter()
    
    def analyze_pipeline_dag(self, pipeline_dag: PipelineDAG) -> Dict[str, Any]:
        """
        Analyze PipelineDAG to discover required configuration classes.
        
        Args:
            pipeline_dag: The pipeline DAG to analyze
            
        Returns:
            Dict containing discovered steps, required configs, and workflow structure
        """
        # Extract step names from DAG nodes
        discovered_steps = []
        for node in pipeline_dag.nodes:
            discovered_steps.append({
                "step_name": node.name,
                "step_type": node.step_type,
                "dependencies": node.dependencies
            })
        
        # Resolve to configuration classes
        config_map = self.config_resolver.resolve_config_map(pipeline_dag.nodes, {})
        
        # Filter to only required configurations
        required_configs = []
        for node_name, config_instance in config_map.items():
            if config_instance:
                config_class = type(config_instance)
                required_configs.append({
                    "config_class_name": config_class.__name__,
                    "config_class": config_class,
                    "step_name": node_name,
                    "inheritance_pattern": self._get_inheritance_pattern(config_class),
                    "is_specialized": self._is_specialized_config(config_class)
                })
        
        # Determine workflow structure
        workflow_steps = self._create_workflow_structure(required_configs)
        
        return {
            "discovered_steps": discovered_steps,
            "required_configs": required_configs,
            "workflow_steps": workflow_steps,
            "total_steps": len(workflow_steps),
            "hidden_configs_count": self._count_total_configs() - len(required_configs)
        }
    
    def _create_workflow_structure(self, required_configs: List[Dict]) -> List[Dict]:
        """Create logical workflow structure for configuration steps."""
        workflow_steps = []
        
        # Step 1: Always start with Base Configuration
        workflow_steps.append({
            "step_number": 1,
            "title": "Base Configuration",
            "config_class": BasePipelineConfig,
            "type": "base",
            "required": True
        })
        
        # Step 2: Add Processing Configuration if any configs need it
        processing_based_configs = [
            config for config in required_configs 
            if config["inheritance_pattern"] == "processing_based"
        ]
        
        if processing_based_configs:
            workflow_steps.append({
                "step_number": 2,
                "title": "Processing Configuration",
                "config_class": ProcessingStepConfigBase,
                "type": "processing",
                "required": True
            })
        
        # Step 3+: Add specific configurations
        step_number = len(workflow_steps) + 1
        for config in required_configs:
            workflow_steps.append({
                "step_number": step_number,
                "title": config["config_class_name"],
                "config_class": config["config_class"],
                "step_name": config["step_name"],
                "type": "specific",
                "inheritance_pattern": config["inheritance_pattern"],
                "is_specialized": config["is_specialized"],
                "required": True
            })
            step_number += 1
        
        return workflow_steps
    
    def _is_specialized_config(self, config_class: Type[BasePipelineConfig]) -> bool:
        """Check if configuration requires specialized UI."""
        specialized_configs = {
            "CradleDataLoadConfig": True,
            # Add other specialized configs here
        }
        return specialized_configs.get(config_class.__name__, False)
    
    def _count_total_configs(self) -> int:
        """Count total available configuration classes."""
        config_classes = self.step_catalog.discover_config_classes()
        return len(config_classes)
```

### JavaScript Implementation for DAG-Driven UI

```javascript
class DAGConfigurationUI {
    constructor() {
        this.currentStep = 0;
        this.workflowSteps = [];
        this.configurationData = {};
        this.pipelineDAG = null;
    }
    
    async initializeFromDAG(pipelineDAG) {
        """Initialize configuration UI from PipelineDAG input."""
        
        this.pipelineDAG = pipelineDAG;
        
        // Step 1: Analyze DAG to discover required configurations
        const analysisResult = await this.analyzePipelineDAG(pipelineDAG);
        
        // Step 2: Display analysis results to user
        this.displayDAGAnalysis(analysisResult);
        
        // Step 3: Initialize workflow steps
        this.workflowSteps = analysisResult.workflow_steps;
        
        // Step 4: Start configuration workflow
        this.startConfigurationWorkflow();
    }
    
    async analyzePipelineDAG(pipelineDAG) {
        """Send DAG to backend for analysis."""
        
        const response = await fetch('/api/config-ui/analyze-dag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pipeline_dag: pipelineDAG
            })
        });
        
        return await response.json();
    }
    
    displayDAGAnalysis(analysisResult) {
        """Display DAG analysis results to user."""
        
        const analysisContainer = document.getElementById('dag-analysis');
        
        analysisContainer.innerHTML = `
            <div class="analysis-results">
                <h3>📊 Pipeline Analysis Results</h3>
                
                <div class="discovered-steps">
                    <h4>🔍 Discovered Pipeline Steps:</h4>
                    <ul>
                        ${analysisResult.discovered_steps.map(step => 
                            `<li>Step: ${step.step_name} (${step.step_type})</li>`
                        ).join('')}
                    </ul>
                </div>
                
                <div class="required-configs">
                    <h4>⚙️ Required Configurations (Only These Will Be Shown):</h4>
                    <ul>
                        ${analysisResult.required_configs.map(config => 
                            `<li>✅ ${config.config_class_name}</li>`
                        ).join('')}
                    </ul>
                    <p>❌ Hidden: ${analysisResult.hidden_configs_count} other config types not needed</p>
                </div>
                
                <div class="workflow-summary">
                    <h4>📋 Configuration Workflow:</h4>
                    <p>Base Config → Processing Config → ${analysisResult.required_configs.length} Specific Configs</p>
                </div>
                
                <button class="btn btn-primary" onclick="this.startConfigurationWorkflow()">
                    Start Configuration Workflow
                </button>
            </div>
        `;
    }
    
    startConfigurationWorkflow() {
        """Start the step-by-step configuration workflow."""
        
        this.currentStep = 0;
        this.renderCurrentStep();
    }
    
    async renderCurrentStep() {
        """Render the current configuration step."""
        
        if (this.currentStep >= this.workflowSteps.length) {
            this.renderCompletionSummary();
            return;
        }
        
        const step = this.workflowSteps[this.currentStep];
        const stepContainer = document.getElementById('config-step');
        
        // Update progress indicator
        this.updateProgressIndicator();
        
        // Render step header
        stepContainer.innerHTML = `
            <div class="config-step-header">
                <h2>🏗️ Configuration Workflow - Step ${step.step_number} of ${this.workflowSteps.length}</h2>
                <h3>📋 ${step.title}</h3>
            </div>
            <div id="step-content"></div>
            <div class="step-actions">
                <button class="btn btn-secondary" onclick="this.previousStep()" ${this.currentStep === 0 ? 'disabled' : ''}>
                    Previous
                </button>
                <button class="btn btn-primary" onclick="this.nextStep()">
                    Continue to Next Step
                </button>
            </div>
        `;
        
        // Render step-specific content
        if (step.type === 'base') {
            await this.renderBaseConfigStep();
        } else if (step.type === 'processing') {
            await this.renderProcessingConfigStep();
        } else if (step.type === 'specific') {
            await this.renderSpecificConfigStep(step);
        }
    }
    
    async renderSpecificConfigStep(step) {
        """Render a specific configuration step."""
        
        const stepContent = document.getElementById('step-content');
        
        if (step.is_specialized) {
            // Handle specialized configurations (e.g., CradleDataLoadConfig)
            stepContent.innerHTML = `
                <div class="specialized-config">
                    <h4>🎛️ Specialized Configuration</h4>
                    <p>This step uses a specialized wizard interface:</p>
                    <button class="btn btn-primary" onclick="this.openSpecializedWizard('${step.config_class_name}')">
                        Open ${step.config_class_name} Wizard
                    </button>
                    <p><small>(Base config will be pre-filled automatically)</small></p>
                </div>
            `;
        } else {
            // Handle standard configurations
            await this.renderStandardConfigForm(step);
        }
    }
    
    updateProgressIndicator() {
        """Update the visual progress indicator."""
        
        const progressContainer = document.getElementById('progress-indicator');
        const totalSteps = this.workflowSteps.length;
        
        let progressHTML = 'Progress: ';
        for (let i = 0; i < totalSteps; i++) {
            if (i <= this.currentStep) {
                progressHTML += '●';
            } else {
                progressHTML += '○';
            }
        }
        progressHTML += ` (${this.currentStep + 1}/${totalSteps})`;
        
        progressContainer.innerHTML = progressHTML;
    }
    
    nextStep() {
        """Move to the next configuration step."""
        
        // Collect current step data
        this.collectCurrentStepData();
        
        // Move to next step
        this.currentStep++;
        this.renderCurrentStep();
    }
    
    previousStep() {
        """Move to the previous configuration step."""
        
        if (this.currentStep > 0) {
            this.currentStep--;
            this.renderCurrentStep();
        }
    }
    
    renderCompletionSummary() {
        """Render the final completion summary."""
        
        const stepContainer = document.getElementById('config-step');
        
        stepContainer.innerHTML = `
            <div class="completion-summary">
                <h2>✅ Configuration Complete - All Steps Configured</h2>
                
                <div class="config-summary">
                    <h3>📋 Configuration Summary:</h3>
                    <ul>
                        ${this.workflowSteps.map(step => 
                            `<li>✅ ${step.title}</li>`
                        ).join('')}
                    </ul>
                </div>
                
                <div class="export-options">
                    <h3>🎯 Ready for Pipeline Execution:</h3>
                    <div class="action-buttons">
                        <button class="btn btn-primary" onclick="this.exportConfiguration()">
                            💾 Export Configuration
                        </button>
                        <button class="btn btn-success" onclick="this.executePipeline()">
                            🚀 Execute Pipeline
                        </button>
                        <button class="btn btn-secondary" onclick="this.saveAsTemplate()">
                            📋 Save as Template
                        </button>
                        <button class="btn btn-secondary" onclick="this.modifyConfiguration()">
                            🔄 Modify Configuration
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
}
```

## Implementation Benefits

### 🎯 **DAG-Driven Approach Benefits**

**✅ User Experience Benefits:**
- **Focused Workflow**: Users see only configurations relevant to their specific pipeline
- **Reduced Complexity**: No confusion from 50+ unused configuration types
- **Intelligent Guidance**: System automatically determines required steps
- **Dynamic Adaptation**: UI adapts to different pipeline structures

**✅ Technical Benefits:**
- **Automatic Discovery**: Leverages existing step catalog and resolver systems
- **Registry Integration**: Uses actual step registry for accurate configuration mapping
- **Inheritance Awareness**: Properly handles configuration inheritance patterns
- **Validation Consistency**: All Pydantic validation rules preserved

**✅ Architectural Benefits:**
- **Scalable Design**: Easy to add new configuration types without UI changes
- **Maintainable Code**: Clear separation between DAG analysis and UI generation
- **Extensible Framework**: Supports both simple and specialized configurations
- **Future-Proof**: Adapts automatically as new step types are added

This comprehensive design provides a **cohesive, logical, and user-focused** approach to universal configuration management, with the PipelineDAG-driven discovery as the core innovation that makes the system both powerful and intuitive.
