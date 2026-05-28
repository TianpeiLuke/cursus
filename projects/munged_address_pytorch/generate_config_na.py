"""
Generate config_NA.json for Munged Address Detection pipeline using DAGConfigFactory.

Usage:
    python projects/munged_address_pytorch/generate_config_na.py

Produces: projects/munged_address_pytorch/pipeline_configs/config_NA.json
"""

import sys
from pathlib import Path

from cursus.api.factory.dag_config_factory import DAGConfigFactory
from cursus.steps.configs.utils import merge_and_save_configs
from cursus.steps.configs.config_cradle_data_loading_step import (
    CradleJobSpecificationConfig,
    DataSourcesSpecificationConfig,
    DataSourceConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    AndesDataSourceConfig,
    TransformSpecificationConfig,
    JobSplitOptionsConfig,
    OutputSpecificationConfig,
)

from munged_address_pytorch_na import create_munged_address_training_dag

# ============================================================================
# Shared Constants — EDX datasets (must match between producer and consumer)
# ============================================================================

# Tags dataset: EdxUploading_tagging WRITES → CradleDataLoading_munged READS
# Single ARN is the source of truth for both producer and consumer
EDX_TAGS_ARN = 'arn:amazon:edx:iad::manifest/trms-abuse-analytics/munged-address/munged-address-tags/["munged_na"]'

# ============================================================================
# Step 1: Create DAG
# ============================================================================

dag = create_munged_address_training_dag()
print(f"DAG created: {len(dag.nodes)} nodes")

# ============================================================================
# Step 2: Initialize Factory
# ============================================================================

factory = DAGConfigFactory(dag)
config_map = factory.get_config_class_map()

print(f"\nDAG Node → Config Class Mapping ({len(config_map)} steps):")
for node, cls in config_map.items():
    print(f"  {node:<40} → {cls.__name__}")

# ============================================================================
# Step 3: Base Config (shared across all steps)
# ============================================================================

BUCKET = "sandboxdependency-abuse-secureaisandboxteamshare-1l77v9am252um"
ROLE = "arn:aws:iam::601857636239:role/SandboxRole-lukexie-us-east-1"

factory.set_base_config(
    bucket=BUCKET,
    role=ROLE,
    region="NA",
    aws_region="us-east-1",
    # author="bjjin",  # Production owner
    author="lukexie",  # Testing — revert to bjjin before production
    service_name="MungedAddressDetection",
    pipeline_version="1.0.0",
    model_class="pytorch",
    project_root_folder="munged_address_pytorch",
    source_dir="dockers",
    use_secure_pypi=True,
)
print("\n✅ Base config set")

# ============================================================================
# Step 4: Base Processing Config
# ============================================================================

factory.set_base_processing_config(
    processing_source_dir="dockers/scripts",
    processing_instance_type_large="ml.m5.12xlarge",
    processing_instance_type_small="ml.m5.4xlarge",
)
print("✅ Base processing config set")

# ============================================================================
# Step 6: Per-Step Configs
# ============================================================================

# --- Phase 1: Tagging (SINK path) ---

# Tagging SQL: UNION of FAP (Andes) + MO orders (EDX from ETLM)
TAGGING_SQL = """
SELECT DISTINCT fraud_object_id AS order_id
FROM FAP_ACTIONS
WHERE marketplace_id IN (1, 7, 526970, 771770)
  AND fraud_object_type_id = 30
  AND performed_by NOT LIKE '%Fortress%'
  AND fraud_action_type_id = 8
  AND (performed_action_text LIKE '%munged%')
UNION
SELECT DISTINCT order_id
FROM MO_ORDERS_EDX
WHERE marketplace_id IN (1, 7, 526970, 771770)
""".strip()

factory.set_step_config(
    "CradleDataLoading_tagging",
    job_type="tagging",
    data_sources_spec=DataSourcesSpecificationConfig(
        start_date="2024-01-01T00:00:00",
        end_date="2026-01-01T00:00:00",
        data_sources=[
            DataSourceConfig(
                data_source_name="FAP_ACTIONS",
                data_source_type="ANDES",
                andes_data_source_properties=AndesDataSourceConfig(
                    provider="booker",
                    table_name="frdg_fraud_actions_performed",
                ),
            ),
            DataSourceConfig(
                data_source_name="MO_ORDERS_EDX",
                data_source_type="EDX",
                edx_data_source_properties=EdxDataSourceConfig(
                    edx_provider="trms-abuse-analytics",
                    edx_subject="munged-address",
                    edx_dataset="mo-tagged-orders",
                    edx_manifest_key="munged_mo_na",
                ),
            ),
        ],
    ),
    transform_spec=TransformSpecificationConfig(transform_sql=TAGGING_SQL),
    output_spec=OutputSpecificationConfig(output_schema=["order_id"]),
    cradle_job_spec=CradleJobSpecificationConfig(
        # Original notebook uses 'BRP-ML-Payment-Generate-Data' (no permission)
        cradle_account="BRP-ML-Abuse",
        cluster_type="MEDIUM",
    ),
)
print("✅ CradleDataLoading_tagging configured")

factory.set_step_config(
    "EdxUploading_tagging",
    edx_arn=EDX_TAGS_ARN,
)
print("✅ EdxUploading_tagging configured")

# --- Phase 2-3: Data Loading ---

# Munged addresses: MDS JOIN EDX tag_file
MUNGED_SQL = """
SELECT mds.objectId, mds.orderId, mds.orderDate, mds.saddr, mds.marketplaceId
FROM RAW_MDS_NA mds
INNER JOIN TAGS ON mds.objectId = TAGS.order_id
""".strip()

MUNGED_MERGE_SQL = (
    "SELECT DISTINCT objectId, orderId, orderDate, saddr, marketplaceId FROM INPUT"
)

factory.set_step_config(
    "CradleDataLoading_munged",
    job_type="munged",
    data_sources_spec=DataSourcesSpecificationConfig(
        start_date="2024-01-01T00:00:00",
        end_date="2026-01-01T00:00:00",
        data_sources=[
            DataSourceConfig(
                data_source_name="RAW_MDS_NA",
                data_source_type="MDS",
                mds_data_source_properties=MdsDataSourceConfig(
                    service_name="FORTRESS_RETAIL",
                    region="NA",
                    org_id=1,
                    output_schema=[
                        {"field_name": "objectId", "field_type": "STRING"},
                        {"field_name": "orderId", "field_type": "STRING"},
                        {"field_name": "orderDate", "field_type": "STRING"},
                        {"field_name": "saddr", "field_type": "STRING"},
                        {"field_name": "marketplaceId", "field_type": "STRING"},
                    ],
                ),
            ),
            DataSourceConfig(
                data_source_name="TAGS",
                data_source_type="EDX",
                edx_data_source_properties=EdxDataSourceConfig(
                    edx_arn=EDX_TAGS_ARN,
                    schema_overrides=[
                        {"field_name": "order_id", "field_type": "STRING"},
                    ],
                ),
            ),
        ],
    ),
    transform_spec=TransformSpecificationConfig(
        transform_sql=MUNGED_SQL,
        job_split_options=JobSplitOptionsConfig(
            split_job=True, days_per_split=34, merge_sql=MUNGED_MERGE_SQL
        ),
    ),
    output_spec=OutputSpecificationConfig(
        output_schema=["objectId", "orderId", "orderDate", "saddr", "marketplaceId"]
    ),
    cradle_job_spec=CradleJobSpecificationConfig(
        # Original notebook uses 'BRP-ML-Payment-Generate-Data' (no permission)
        cradle_account="BRP-ML-Abuse",
        cluster_type="LARGE",
    ),
)
print("✅ CradleDataLoading_munged configured")

# Normal/good addresses: MDS LEFT JOIN ANDES fraud-tags
NORMAL_SQL = """
SELECT mds.objectId, mds.orderDate, mds.saddr, mds.marketplaceId,
       mds.daysSinceFirstCompletedOrderForCustomerId
FROM RAW_MDS_NA mds
LEFT JOIN FRAUD_TAGS ON mds.objectId = FRAUD_TAGS.objectId
WHERE (FRAUD_TAGS.IS_FRD = '0' OR FRAUD_TAGS.IS_FRD IS NULL)
  AND (FRAUD_TAGS.HAS_CB != '1' OR FRAUD_TAGS.HAS_CB IS NULL)
  AND CAST(mds.daysSinceFirstCompletedOrderForCustomerId AS INT) >= 60
""".strip()

NORMAL_MERGE_SQL = "SELECT DISTINCT objectId, orderDate, saddr, marketplaceId, daysSinceFirstCompletedOrderForCustomerId FROM INPUT"

factory.set_step_config(
    "CradleDataLoading_normal",
    job_type="normal",
    data_sources_spec=DataSourcesSpecificationConfig(
        start_date="2024-01-01T00:00:00",
        end_date="2026-01-01T00:00:00",
        data_sources=[
            DataSourceConfig(
                data_source_name="RAW_MDS_NA",
                data_source_type="MDS",
                mds_data_source_properties=MdsDataSourceConfig(
                    service_name="FORTRESS_RETAIL",
                    region="NA",
                    org_id=1,
                    output_schema=[
                        {"field_name": "objectId", "field_type": "STRING"},
                        {"field_name": "orderDate", "field_type": "STRING"},
                        {"field_name": "saddr", "field_type": "STRING"},
                        {"field_name": "marketplaceId", "field_type": "STRING"},
                        {
                            "field_name": "daysSinceFirstCompletedOrderForCustomerId",
                            "field_type": "STRING",
                        },
                    ],
                ),
            ),
            DataSourceConfig(
                data_source_name="FRAUD_TAGS",
                data_source_type="ANDES",
                andes_data_source_properties=AndesDataSourceConfig(
                    provider="26b27bde-3847-49c6-a07c-0289c17d9c33",
                    table_name="fraud-tags-na",
                ),
            ),
        ],
    ),
    transform_spec=TransformSpecificationConfig(
        transform_sql=NORMAL_SQL,
        job_split_options=JobSplitOptionsConfig(
            split_job=True, days_per_split=20, merge_sql=NORMAL_MERGE_SQL
        ),
    ),
    output_spec=OutputSpecificationConfig(
        output_schema=[
            "objectId",
            "orderDate",
            "saddr",
            "marketplaceId",
            "daysSinceFirstCompletedOrderForCustomerId",
        ]
    ),
    cradle_job_spec=CradleJobSpecificationConfig(
        # Original notebook uses 'BRP-ML-Payment-Generate-Data' (no permission)
        cradle_account="BRP-ML-Abuse",
        cluster_type="LARGE",
    ),
)
print("✅ CradleDataLoading_normal configured")

# --- Phase 4: Sampling ---

factory.set_step_config(
    "TabularPreprocessing_sampling",
    job_type="sampling",
    processing_entry_point="tabular_preprocessing_sampling_munged.py",
    use_large_processing_instance=True,
    output_format="Parquet",
)
print("✅ TabularPreprocessing_sampling configured")

factory.set_step_config(
    "StratifiedSampling_sampling",
    job_type="sampling",
    strata_column="marketplaceId",
    sampling_strategy="external_proportional",
    sampling_multiplier=5,
    allow_replacement=True,
    sampling_filter_column="__cohort__",
    sampling_filter_value="good",
    use_large_processing_instance=True,
)
print("✅ StratifiedSampling_sampling configured")

# --- Phase 5: LLM Scoring ---

BEDROCK_PROMPT = (
    "You are an expert at analyzing shipping addresses. "
    "Rate the strangeness of the following shipping address on a scale of 1-5, "
    "where 1 is a normal address and 5 is extremely strange/suspicious.\n\n"
    "Address: {saddr}\n\n"
    "Respond with only the integer rating."
)

factory.set_step_config(
    "BedrockProcessing_scoring",
    job_type="sampling",
    processing_entry_point="bedrock_processing.py",
    use_large_processing_instance=True,
    bedrock_primary_model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    bedrock_inference_profile_arn="arn:aws:bedrock:us-east-1:601857636239:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0",
    bedrock_use_structured_output=True,
    bedrock_use_converse_api=False,
    bedrock_concurrency_mode="concurrent",
    bedrock_max_concurrent_workers=10,
    bedrock_rate_limit_per_second=15,
    bedrock_max_tokens=64,
    bedrock_temperature=0.5,
    bedrock_output_column_prefix="llm_",
    bedrock_user_prompt_template=BEDROCK_PROMPT,
    bedrock_system_prompt="You are a shipping address quality analyst.",
    bedrock_input_placeholders=["saddr"],
    bedrock_validation_schema={
        "strangeness_rating": {
            "type": "integer",
            "description": "Strangeness rating from 1 (normal) to 5 (very strange)",
        }
    },
)
print("✅ BedrockProcessing_scoring configured")

# --- Phase 6: Training Prep ---

factory.set_step_config(
    "TabularPreprocessing_training",
    job_type="training",
    processing_entry_point="tabular_preprocessing_training_munged.py",
    use_large_processing_instance=True,
    output_format="CSV",
)
print("✅ TabularPreprocessing_training configured")

# --- Phase 7: Training ---

factory.set_step_config(
    "PyTorchTraining",
    training_entry_point="pytorch_training.py",
    training_instance_type="ml.g5.12xlarge",
    training_instance_count=1,
    training_volume_size=500,
    skip_hyperparameters_s3_uri=False,
)
print("✅ PyTorchTraining configured")

# --- Phase 7b: Calibration Path ---

CALIBRATION_SQL = "SELECT saddr, marketplaceId, orderDate FROM RAW_MDS_NA"

factory.set_step_config(
    "CradleDataLoading_calibration",
    job_type="calibration",
    data_sources_spec=DataSourcesSpecificationConfig(
        start_date="2026-05-20T00:00:00",
        end_date="2026-05-22T00:00:00",
        data_sources=[
            DataSourceConfig(
                data_source_name="RAW_MDS_NA",
                data_source_type="MDS",
                mds_data_source_properties=MdsDataSourceConfig(
                    service_name="FORTRESS_RETAIL",
                    region="NA",
                    org_id=1,
                    output_schema=[
                        {"field_name": "saddr", "field_type": "STRING"},
                        {"field_name": "marketplaceId", "field_type": "STRING"},
                        {"field_name": "orderDate", "field_type": "STRING"},
                    ],
                ),
            ),
        ],
    ),
    transform_spec=TransformSpecificationConfig(transform_sql=CALIBRATION_SQL),
    output_spec=OutputSpecificationConfig(
        output_schema=["saddr", "marketplaceId", "orderDate"]
    ),
    cradle_job_spec=CradleJobSpecificationConfig(
        # Original notebook uses 'BRP-ML-Payment-Generate-Data' (no permission)
        cradle_account="BRP-ML-Abuse",
        cluster_type="MEDIUM",
    ),
)
print("✅ CradleDataLoading_calibration configured")

factory.set_step_config(
    "TabularPreprocessing_calibration",
    job_type="calibration",
    processing_entry_point="tabular_preprocessing.py",
    use_large_processing_instance=True,
    output_format="Parquet",
)
print("✅ TabularPreprocessing_calibration configured")

factory.set_step_config(
    "PyTorchModelInference_calibration",
    job_type="calibration",
    processing_entry_point="pytorch_model_inference.py",
    processing_source_dir="dockers",
    use_large_processing_instance=True,
    processing_framework_version="2.1.2",
    id_name="saddr",
    label_name="__placeholder__",
)
print("✅ PyTorchModelInference_calibration configured")

factory.set_step_config(
    "PercentileModelCalibration_calibration",
    job_type="calibration",
    processing_entry_point="percentile_model_calibration.py",
    score_field="prob_class_1",
    use_large_processing_instance=False,
)
print("✅ PercentileModelCalibration_calibration configured")

# --- Phase 8: Deployment ---

factory.set_step_config(
    "Package",
    processing_entry_point="package.py",
    use_large_processing_instance=False,
)
print("✅ Package configured")

factory.set_step_config(
    "Payload",
    processing_entry_point="payload.py",
    use_large_processing_instance=False,
    field_defaults={
        "saddr": "Quadra SQN 412 Bloco C Apt 405|||Brasilia|||DF|||70000-000|||BR"
    },
    expected_tps=5,
    max_latency_in_millisecond=100,
    source_model_inference_content_types=["application/json"],
    source_model_inference_response_types=["application/json"],
)
print("✅ Payload configured")

factory.set_step_config(
    "Registration",
    framework="pytorch",
    inference_entry_point="pytorch_inference_handler.py",
    inference_instance_type="ml.m5.2xlarge",
    model_domain="FORTRESS_RETAIL",
    model_objective="BRMungedAddressModel",
    model_owner="amzn1.abacus.team.xkuchlojq7lt4nir5uma",
    source_model_inference_content_types=["application/json"],
    source_model_inference_response_types=["application/json"],
    source_model_inference_input_variable_list=[["saddr", "TEXT"]],
    source_model_inference_output_variable_list={
        "legacy-score": "NUMERIC",
        "score-percentile": "NUMERIC",
    },
)
print("✅ Registration configured")

# ============================================================================
# Step 7: Generate
# ============================================================================

print("\n" + "=" * 60)
pending = factory.get_pending_steps()
if pending:
    print(f"⚠️  Still pending: {pending}")
    print("Cannot generate — configure remaining steps first.")
    sys.exit(1)

print("Generating final configurations...")
configs = factory.generate_all_configs()
print(f"✅ Generated {len(configs)} configuration instances")

# ============================================================================
# Step 8: Save
# ============================================================================

output_dir = Path(__file__).parent / "pipeline_configs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "config_NA.json"

merge_and_save_configs(configs, str(output_path))
print(f"\n✅ Saved to: {output_path}")
