---
tags:
  - code
  - core
  - config_fields
  - cradle_config
  - factory_functions
keywords:
  - create_cradle_data_load_config
  - create_training_and_calibration_configs
  - CradleDataLoadConfig
  - factory pattern
  - configuration creation
topics:
  - configuration management
  - factory pattern
  - cradle data loading
language: python
date of note: 2025-09-07
---

# Cradle Configuration Factory

Helper functions for creating CradleDataLoadConfig objects with minimal inputs, deriving nested configurations from essential user inputs.

## Overview

The `cradle_config_factory` module provides utilities to simplify the creation of complex CradleDataLoadConfig objects by deriving nested configurations from essential user inputs. This module implements the factory pattern to reduce the complexity of creating data loading configurations for the Cradle system.

The factory functions handle the creation of nested data source configurations, transform specifications, output specifications, and job specifications from a minimal set of user inputs. This approach significantly reduces the boilerplate code required to create complete configuration objects while ensuring all necessary components are properly configured.

## Classes and Methods

### Functions
- [`create_cradle_data_load_config`](#create_cradle_data_load_config) - Create a single CradleDataLoadConfig with minimal inputs
- [`create_training_and_calibration_configs`](#create_training_and_calibration_configs) - Create both training and calibration configs with consistent settings

### Helper Functions
- [`_map_region_to_aws_region`](#_map_region_to_aws_region) - Map marketplace region to AWS region
- [`_create_field_schema`](#_create_field_schema) - Convert field names to schema dictionaries
- [`_format_edx_manifest_key`](#_format_edx_manifest_key) - Format EDX manifest key with date components
- [`_create_edx_manifest`](#_create_edx_manifest) - Create EDX manifest ARN with date components
- [`_generate_transform_sql`](#_generate_transform_sql) - Generate SQL query to join MDS and EDX data
- [`_get_all_fields`](#_get_all_fields) - Get combined list of all fields from MDS and EDX sources

## API Reference

### create_cradle_data_load_config

create_cradle_data_load_config(_base_config_, _job_type_, _mds_field_list_, _start_date_, _end_date_, _tag_edx_provider_, _tag_edx_subject_, _tag_edx_dataset_, _etl_job_id_, _edx_manifest_comment=None_, _service_name=None_, _cradle_account="Buyer-Abuse-RnD-Dev"_, _org_id=0_, _cluster_type="STANDARD"_, _output_format="PARQUET"_, _output_save_mode="ERRORIFEXISTS"_, _split_job=False_, _days_per_split=7_, _merge_sql=None_, _s3_input_override=None_, _transform_sql=None_, _tag_schema=None_, _use_dedup_sql=None_, _mds_join_key="objectId"_, _edx_join_key="order_id"_, _join_type="JOIN"_)

Create a CradleDataLoadConfig with minimal required inputs. This helper function simplifies the creation of a CradleDataLoadConfig by handling the generation of nested configurations from essential user inputs.

**Parameters:**
- **base_config** (_BasePipelineConfig_) – Base configuration for inheritance
- **job_type** (_str_) – Type of job ('training' or 'calibration')
- **mds_field_list** (_List[str]_) – List of fields to include from MDS
- **start_date** (_str_) – Start date for data pull (format: YYYY-MM-DDT00:00:00)
- **end_date** (_str_) – End date for data pull (format: YYYY-MM-DDT00:00:00)
- **tag_edx_provider** (_str_) – EDX provider for tags
- **tag_edx_subject** (_str_) – EDX subject for tags
- **tag_edx_dataset** (_str_) – EDX dataset for tags
- **etl_job_id** (_str_) – ETL job ID for the EDX manifest
- **edx_manifest_comment** (_Optional[str]_) – Optional comment for EDX manifest key
- **service_name** (_Optional[str]_) – Name of the MDS service (if not in base_config)
- **cradle_account** (_str_) – Cradle account name (default: "Buyer-Abuse-RnD-Dev")
- **org_id** (_int_) – Default organization ID for regional MDS bucket (default: 0)
- **cluster_type** (_str_) – Cradle cluster type (default: "STANDARD")
- **output_format** (_str_) – Output format (default: "PARQUET")
- **output_save_mode** (_str_) – Output save mode (default: "ERRORIFEXISTS")
- **split_job** (_bool_) – Whether to split the job (default: False)
- **days_per_split** (_int_) – Days per split if splitting (default: 7)
- **merge_sql** (_Optional[str]_) – SQL to merge split results, required if split_job=True
- **s3_input_override** (_Optional[str]_) – S3 input override
- **transform_sql** (_Optional[str]_) – Custom transform SQL, auto-generated if not provided
- **tag_schema** (_Optional[List[str]]_) – Schema for tag data, default provided if not specified
- **use_dedup_sql** (_Optional[bool]_) – Whether to use dedup SQL format (default: same as split_job)
- **mds_join_key** (_str_) – Join key field name from MDS (default: "objectId")
- **edx_join_key** (_str_) – Join key field name from EDX (default: "order_id")
- **join_type** (_str_) – SQL join type (default: "JOIN")

**Returns:**
- **CradleDataLoadConfig** – A fully configured CradleDataLoadConfig object

```python
from cursus.core.config_fields.cradle_config_factory import create_cradle_data_load_config

# Create training data loading config
training_config = create_cradle_data_load_config(
    base_config=base_pipeline_config,
    job_type="training",
    mds_field_list=["objectId", "transactionDate", "amount", "marketplace_id"],
    start_date="2025-01-01T00:00:00",
    end_date="2025-01-31T00:00:00",
    tag_edx_provider="buyer-abuse",
    tag_edx_subject="tags",
    tag_edx_dataset="abuse_labels",
    etl_job_id="training_job_2025_01"
)

# Create calibration config with custom settings
calibration_config = create_cradle_data_load_config(
    base_config=base_pipeline_config,
    job_type="calibration",
    mds_field_list=["objectId", "transactionDate", "amount"],
    start_date="2025-02-01T00:00:00",
    end_date="2025-02-28T00:00:00",
    tag_edx_provider="buyer-abuse",
    tag_edx_subject="tags",
    tag_edx_dataset="abuse_labels",
    etl_job_id="calibration_job_2025_02",
    split_job=True,
    days_per_split=14,
    merge_sql="SELECT * FROM INPUT ORDER BY transactionDate"
)
```

### create_training_and_calibration_configs

create_training_and_calibration_configs(_base_config_, _mds_field_list_, _tag_edx_provider_, _tag_edx_subject_, _tag_edx_dataset_, _etl_job_id_, _training_start_date_, _training_end_date_, _calibration_start_date_, _calibration_end_date_, _service_name=None_, _edx_manifest_comment=None_, _cradle_account="Buyer-Abuse-RnD-Dev"_, _cluster_type="STANDARD"_, _output_format="PARQUET"_, _output_save_mode="ERRORIFEXISTS"_, _split_job=False_, _days_per_split=7_, _merge_sql=None_, _transform_sql=None_, _tag_schema=None_)

Create both training and calibration CradleDataLoadConfig objects with consistent settings.

**Parameters:**
- **base_config** (_BasePipelineConfig_) – Base config for inheritance
- **mds_field_list** (_List[str]_) – List of fields to include from MDS
- **tag_edx_provider** (_str_) – EDX provider for tags
- **tag_edx_subject** (_str_) – EDX subject for tags
- **tag_edx_dataset** (_str_) – EDX dataset for tags
- **etl_job_id** (_str_) – ETL job ID for the EDX manifest
- **training_start_date** (_str_) – Training data start date
- **training_end_date** (_str_) – Training data end date
- **calibration_start_date** (_str_) – Calibration data start date
- **calibration_end_date** (_str_) – Calibration data end date
- **service_name** (_Optional[str]_) – Name of the MDS service (if not in base_config)
- **edx_manifest_comment** (_Optional[str]_) – Optional comment for EDX manifest key
- **cradle_account** (_str_) – Cradle account name (default: "Buyer-Abuse-RnD-Dev")
- **cluster_type** (_str_) – Cradle cluster type (default: "STANDARD")
- **output_format** (_str_) – Output format (default: "PARQUET")
- **output_save_mode** (_str_) – Output save mode (default: "ERRORIFEXISTS")
- **split_job** (_bool_) – Whether to split the job (default: False)
- **days_per_split** (_int_) – Days per split if splitting (default: 7)
- **merge_sql** (_Optional[str]_) – SQL to merge split results
- **transform_sql** (_Optional[str]_) – Custom transform SQL
- **tag_schema** (_Optional[List[str]]_) – Schema for tag data

**Returns:**
- **Dict[str, CradleDataLoadConfig]** – Dictionary with 'training' and 'calibration' configs

```python
from cursus.core.config_fields.cradle_config_factory import create_training_and_calibration_configs

# Create both training and calibration configs at once
configs = create_training_and_calibration_configs(
    base_config=base_pipeline_config,
    mds_field_list=["objectId", "transactionDate", "amount", "marketplace_id"],
    tag_edx_provider="buyer-abuse",
    tag_edx_subject="tags",
    tag_edx_dataset="abuse_labels",
    etl_job_id="pipeline_job_2025_01",
    training_start_date="2025-01-01T00:00:00",
    training_end_date="2025-01-31T00:00:00",
    calibration_start_date="2025-02-01T00:00:00",
    calibration_end_date="2025-02-28T00:00:00"
)

training_config = configs["training"]
calibration_config = configs["calibration"]
```

### Helper Functions

#### _map_region_to_aws_region

_map_region_to_aws_region(_region_)

Map marketplace region to AWS region.

**Parameters:**
- **region** (_str_) – Marketplace region ('NA', 'EU', 'FE')

**Returns:**
- **str** – AWS region name

#### _create_field_schema

_create_field_schema(_fields_)

Convert a list of field names to schema dictionaries.

**Parameters:**
- **fields** (_List[str]_) – List of field names

**Returns:**
- **List[Dict[str, str]]** – List of schema dictionaries

#### _format_edx_manifest_key

_format_edx_manifest_key(_etl_job_id_, _start_date_, _end_date_, _comment=None_)

Format an EDX manifest key with date components and optional comment.

**Parameters:**
- **etl_job_id** (_str_) – ETL job ID
- **start_date** (_str_) – Start date string
- **end_date** (_str_) – End date string
- **comment** (_Optional[str]_) – Optional comment or region code

**Returns:**
- **str** – Properly formatted EDX manifest key

#### _create_edx_manifest

_create_edx_manifest(_provider_, _subject_, _dataset_, _etl_job_id_, _start_date_, _end_date_, _comment=None_)

Create an EDX manifest ARN with date components.

**Parameters:**
- **provider** (_str_) – EDX provider name
- **subject** (_str_) – EDX subject
- **dataset** (_str_) – EDX dataset name
- **etl_job_id** (_str_) – ETL job ID
- **start_date** (_str_) – Start date string
- **end_date** (_str_) – End date string
- **comment** (_Optional[str]_) – Optional comment or region code

**Returns:**
- **str** – Properly formatted EDX manifest ARN

#### _generate_transform_sql

_generate_transform_sql(_mds_source_name_, _edx_source_name_, _mds_field_list_, _tag_schema_, _mds_join_key="objectId"_, _edx_join_key="order_id"_, _join_type="JOIN"_, _use_dedup_sql=False_)

Generate a SQL query to join MDS and EDX data with configurable join keys.

**Parameters:**
- **mds_source_name** (_str_) – Logical name for MDS source
- **edx_source_name** (_str_) – Logical name for EDX source
- **mds_field_list** (_List[str]_) – List of fields from MDS
- **tag_schema** (_List[str]_) – List of fields from EDX tags
- **mds_join_key** (_str_) – Join key field name from MDS (default: "objectId")
- **edx_join_key** (_str_) – Join key field name from EDX (default: "order_id")
- **join_type** (_str_) – SQL join type (default: "JOIN")
- **use_dedup_sql** (_bool_) – Whether to use the deduplication SQL format with subquery (default: False)

**Returns:**
- **str** – SQL query string

#### _get_all_fields

_get_all_fields(_mds_fields_, _tag_fields_)

Get a combined list of all fields from MDS and EDX sources. This function handles case-insensitivity to avoid duplicate columns in SQL SELECT statements.

**Parameters:**
- **mds_fields** (_List[str]_) – List of MDS fields
- **tag_fields** (_List[str]_) – List of tag fields

**Returns:**
- **List[str]** – Combined and deduplicated list of fields

## Related Documentation

- [Configuration Fields Overview](README.md) - System overview and integration
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Used for step name generation
- [Configuration Constants](constants.md) - Defines constants used in factory functions
