# Plan: Munged Address Detection — MODS Pipeline Migration

**Date**: 2026-05-27 (updated 2026-05-28)
**Status**: 🟡 Sprint 2 COMPLETE, Sprint 3 next
**Source**: FZ 29d16 (DAG synthesis) + children (29d16a–h, 29d14b–g) from AmazonBuyerAbuseSlipboxAgent
**Owner**: bjjin
**Effort**: ~5-8 days (Tier C migration)
**Project location**: `projects/munged_address_pytorch/`

## Overview

Migrate Munged Address Detection model training from ad-hoc SageMaker notebooks to a fully automated Cursus/MODS pipeline with PIPER scheduling.

**Model**: DistilBERT multilingual binary classifier (87.9% accuracy, 0.90 AUC)
**Pipeline**: 16 nodes, 16 edges across 8 phases
**Key insight**: Framework code changes DONE (8/8). Custom scripts DONE (2/2). EdxUploading step DONE. Simplified training (597L), inference (386L), handler (328L) DONE + aligned end-to-end. Remaining: ETLM job + config_NA.json `specific` section + validation.

## Architecture Summary

```
CradleDataLoading_tagging → EdxUploading_tagging (SINK — uploads to EDX, no Kale)
CradleDataLoading_munged (reads EDX tag_file, Cradle waits natively) ─────────┐
CradleDataLoading_normal (parallel) ──────────────────────────────────────────┼→ TabularPreprocessing_sampling
                                                                                → StratifiedSampling_sampling
                                                                                → BedrockProcessing_scoring
                                                                                → TabularPreprocessing_training
                                                                                → PyTorchTraining
                                                                                → Package + Payload → Registration
                                                                                + Calibration path (parallel)
```

**Note**: Uses `EdxUploading` (NEW step, FZ 18g3i) instead of `DataUploading` (Andes-only, requires Kale ~2 weeks). No DAG edge from SINK to downstream — Cradle reads the EDX ARN statically from config and waits for data readiness natively. Resolution score: 1.00/1.00 (validated in FZ 29d16g).

## Sprint Plan

### Sprint 1: ETLM + Core Infrastructure (Day 1-2)

#### Task 1.1: Create ETLM job for MO munged orders
**Ref**: FZ 29d16a1

- [ ] Create Datanet EXTRACT job `Extract_Munged_MO_Orders_NA`
- [ ] SQL:
  ```sql
  SELECT DISTINCT a.order_id
  FROM trmsbrm.d_brm_mo_tt_orders a
  JOIN trmsbrm.d_mo_tickets b ON a.case_id = b.case_id
  WHERE b.mo_how = 'MNG' AND a.marketplace_id IN (1,7,526970,771770)
  ```
- [ ] Output: EDX `trms-abuse-analytics/munged-address/mo-tagged-orders/["munged_mo_na"]`
- [ ] Verify output in DataCentral
- [ ] Schedule: daily or on-demand before pipeline

#### Task 1.2: Create `config_NA.json` skeleton
**Ref**: FZ 29d16

- [x] Create `projects/munged_address_pytorch/pipeline_configs/config_NA.json` ✅ (skeleton exists)
- [x] Structure: `{"configuration": {"shared": {...}, "specific": {...}}}` — documented in FZ 29d14k ✅
- [x] Add shared config: pipeline_name, pipeline_s3_loc, aws_region ✅
- [x] Created `generate_config_na.py` — DAGConfigFactory-based script to generate full config (FZ 29d14l) ✅
  - Uses `DAGConfigFactory(dag)` → `set_base_config()` → `set_step_config()` × 16 → `generate_all_configs()` → `merge_and_save_configs()`
  - All alignment requirements from FZ 29d14i embedded (job_types, schema field name, FIELD_DEFAULTS)
- [ ] Run `generate_config_na.py` on SageMaker (requires Cursus package imports) to produce final config_NA.json

#### Task 1.3: Write `hyperparameters_NA.json`
**Ref**: FZ 29d16d

- [x] Create `projects/munged_address_pytorch/dockers/hyperparams/hyperparameters_NA.json` ✅ DONE
- [x] Added explicit `address_column`/`address_delimiter` for inference alignment ✅ (FZ 29d14f)
- [x] Fields:
  ```json
  {
    "model_class": "bert",
    "tokenizer": "distilbert-base-multilingual-cased",
    "text_name": "shippingAddress",
    "label_name": "__tag__",
    "id_name": "orderDate",
    "tab_field_list": [], "cat_field_list": [], "full_field_list": [],
    "is_binary": true, "num_classes": 2,
    "max_sen_len": 128, "max_epochs": 2,
    "lr": 2e-5, "batch_size": 64, "weight_decay": 0.01,
    "optimizer": "AdamW", "warmup_steps": 500,
    "class_weights": [1.0, 3.0], "fp16": true,
    "early_stop_metric": "val/f1_score", "early_stop_patience": 2,
    "gradient_clip_val": 1.0, "metric_choices": ["auroc", "f1_score"]
  }
  ```

#### Task 1.4: Write inference handler + model inference + training + requirements
**Ref**: FZ 29d14b, 29d14c, 29d14g

- [x] Create `pytorch_inference_handler.py` ✅ DONE (328L simplified, FZ 29d14g)
  - Secure PyPI + multi-worker-safe install (file lock from rnr pattern)
  - Loads: model.pth + hyperparameters.json + calibration/percentile_score.pkl
  - API: `{"saddr":"..."}` → `{"predictions":[{"legacy-score":X,"score-percentile":Y}]}`
  - Backward-compatible with original + implements `score-percentile` (was TODO)
- [x] Create `pytorch_model_inference.py` ✅ DONE (386L simplified, FZ 29d14c)
  - Lightning Trainer.test() with ddp_spawn for multi-GPU ProcessingStep
  - File-based gather with index-aware reconstruction
  - Output: `predictions.csv` (matches calibration priority list #3)
- [x] Create `pytorch_training.py` ✅ DONE (597L simplified, FZ 29d14b)
  - Lightning DDP for multi-GPU TrainingStep (torchrun-managed)
  - 6 multi-GPU fixes: install race, tokenizer race, val gather, file save, ONNX unwrap, class_weights
  - Output: model.pth + hyperparameters.json + model_artifacts.pth + model.onnx
- [x] Create `requirements-secure.txt` ✅ DONE
- [x] End-to-end alignment verified across all 6 pairs (FZ 29d14f) ✅

---

### Sprint 2: Custom Processing Scripts (Day 2-3) ✅ COMPLETE

#### Task 2.1: Write `tabular_preprocessing_sampling_munged.py`
**Ref**: FZ 29d16b (Script 1)

- [x] Create `projects/munged_address_pytorch/dockers/scripts/tabular_preprocessing_sampling_munged.py` (~130 lines) ✅ DONE
- [x] Functionality: read DATA + DATA_SECONDARY, dedup on (saddr, marketplace), add __cohort__, emit reference_counts.json ✅
- [x] Output schema verified: aligns with StratifiedSampling input (FZ 29d14d) ✅

#### Task 2.2: Write `tabular_preprocessing_training_munged.py`
**Ref**: FZ 29d16b (Script 2)

- [x] Create `projects/munged_address_pytorch/dockers/scripts/tabular_preprocessing_training_munged.py` (~110 lines) ✅ DONE
- [x] Functionality: label flip (bad→1, good+score>3→1), extract shippingAddress from saddr|||, stratified split ✅
- [x] Output verified: train/val/test dirs with columns [shippingAddress, __tag__, orderDate, marketplaceId] (FZ 29d14f Pair 1) ✅

---

### Sprint 3: Config — Steps Needing No Custom Scripts (Day 3-4) 🟡 SCRIPTED (pending execution)

**Note**: All 16 step configs are now scripted in `generate_config_na.py`. Run on SageMaker to produce final JSON. Remaining: Cradle SQL queries (complex nested objects) and ETLM job.

#### Task 3.1: Configure CradleDataLoading nodes (4 entries)
**Ref**: FZ 29d16a, 29d16a1

- [ ] `CradleDataLoading_tagging` — ANDES (FAP) + EDX (MO from ETLM), UNION in SQL
- [ ] `CradleDataLoading_munged` — MDS + EDX tag_file join, split_job=34d, cluster=XLARGE
- [ ] `CradleDataLoading_normal` — MDS + ANDES (fraud-tags) LEFT JOIN, auto_tag equivalent, cluster=LARGE
- [ ] `CradleDataLoading_calibration` — MDS simple SELECT, 2 days, no split

#### Task 3.2: Configure EdxUploading_tagging
**Ref**: FZ 29d16h (supersedes 29d16f)

- [x] EdxUploading step created in the dev repo (FZ 18g3i) ✅ DONE
- [x] DAG updated: `EdxUploading_tagging` replaces `DataUploading_tagging` ✅ DONE
- [x] Alignment verified: resolution score 1.00/1.00 (FZ 29d16g) ✅ DONE
- [ ] Add to config_NA.json:
  ```json
  "EdxUploading_tagging": {
    "__model_type__": "EdxUploadingConfig",
    "edx_provider": "trms-abuse-analytics",
    "edx_subject": "munged-address",
    "edx_dataset": "munged-address-tags",
    "edx_manifest_key": "munged_na"
  }
  ```
- [ ] Verify EDX dataset exists at DataCentral (or create — no Kale needed)
- [ ] ~~Andes table~~ NOT NEEDED ~~Kale attestation~~ NOT NEEDED

#### Task 3.3: Configure StratifiedSampling_sampling
**Ref**: FZ 29d9a3

- [ ] `strategy=external_proportional`, `sampling_multiplier=5`
- [ ] `allow_replacement=true`
- [ ] `sampling_filter_column=__cohort__`, `sampling_filter_value=good`

#### Task 3.4: Configure BedrockProcessing_scoring
**Ref**: FZ 29d16c

- [ ] Self-contained mode (prompt in config)
- [ ] `bedrock_primary_model_id=anthropic.claude-3-5-haiku-20241022-v1:0`
- [ ] `bedrock_use_structured_output=true`
- [ ] `bedrock_concurrency_mode=concurrent`, workers=10, rate=15/sec
- [ ] Full prompt template (verbatim from Nb11/12)

#### Task 3.5: Configure TabularPreprocessing_sampling + _training
- [ ] `_sampling`: `processing_entry_point=tabular_preprocessing_sampling_munged.py`
- [ ] `_training`: `processing_entry_point=tabular_preprocessing_training_munged.py`

#### Task 3.6: Configure PyTorchTraining
**Ref**: FZ 29d16d

- [ ] `training_entry_point=pytorch_training.py` (generic)
- [ ] `training_instance_type=ml.g5.12xlarge`
- [ ] `skip_hyperparameters_s3_uri=false`

#### Task 3.7: Configure calibration path (3 entries)
- [ ] `TabularPreprocessing_calibration` — generic script, job_type=calibration
- [ ] `PyTorchModelInference_calibration` — load model, score calibration data
- [ ] `PercentileModelCalibration_calibration` — standard percentile mapping

#### Task 3.8: Configure Package + Payload + Registration
**Ref**: FZ 29d16e

- [ ] `Package`: `processing_source_dir=dockers/scripts`
- [ ] `Payload`: `field_defaults={"saddr":"..."}`, TPS=5, latency≤100ms
- [ ] `Registration`: domain=FORTRESS_RETAIL, objective=BRMungedAddressModel, input=[["saddr","TEXT"]]

---

### Sprint 4: Validation (Day 4-5)

#### Task 4.1: DAG compatibility check
- [ ] Run `PipelineDAGCompiler.validate_dag_compatibility(dag)`
- [ ] Fix any resolution failures

#### Task 4.2: Alignment verification
- [ ] Run `/cursus-verify-alignment` for TabularPreprocessing_sampling
- [ ] Run `/cursus-verify-alignment` for TabularPreprocessing_training
- [ ] Run `/cursus-verify-alignment` for BedrockProcessing_scoring

#### Task 4.3: Dry run compile
- [ ] `dag_compiler.compile_with_report(dag)` → produces pipeline.json
- [ ] Verify all 16 steps created, 16 dependencies wired (no edge from SINK node)

#### Task 4.4: Sandbox integration test
- [ ] Execute pipeline with subset data (100 addresses per cohort)
- [ ] Verify end-to-end flow: Cradle → TabPreproc → Sampling → Bedrock → Training → Package → Registration
- [ ] Expected: model.tar.gz produced, MIMS registration succeeds in gamma

#### Task 4.5: Numerical equivalence
- [ ] Compare model AUC: Cursus pipeline vs notebook (target: within ±0.01 of 0.88)
- [ ] Compare label distribution: confirm ~25K positive, ~74K negative

---

### Sprint 5: MODS Template Migration (Day 6-8)

#### Task 5.1: Copy project to BuyerAbuseModsTemplate
- [ ] Copy `projects/munged_address_pytorch/` → `BuyerAbuseModsTemplate/src/.../munged_address_pytorch/`
- [ ] Adapt imports (flat Docker-style)
- [ ] Include all dockers/ (scripts, processing, hyperparams)

#### Task 5.2: Add @MODSTemplate decorator
- [ ] Wrap pipeline class with MODS template decorator
- [ ] Configure pipeline metadata (name, version, description)

#### Task 5.3: Wire PIPER schedule
- [ ] Training: every 120 days (per model_metadata.json)
- [ ] Calibration: every 7 days
- [ ] ETLM: daily (MO orders to EDX)

#### Task 5.4: Production deployment (NA first)
- [ ] Register pipeline in PIPER
- [ ] Test in gamma environment
- [ ] Deploy to production (NA region)
- [ ] Monitor first execution

#### Task 5.5: EU/FE expansion (follow-up)
- [ ] Create config_EU.json, config_FE.json (marketplace ID changes only)
- [ ] Create ETLM jobs for EU/FE regions
- [ ] Deploy to EU, FE

---

## Project File Layout

```
projects/munged_address_pytorch/
├── __init__.py                                         ✅
├── munged_address_pytorch_na.py                        ✅ (DAG: 16 nodes, 16 edges)
├── generate_config_na.py                               ✅ (DAGConfigFactory-based, 291L)
├── dockers/
│   ├── __init__.py                                     ✅
│   ├── requirements-secure.txt                         ✅ training deps
│   ├── requirements-gpu-secure.txt                     ✅ GPU inference deps
│   ├── pytorch_training.py                             ✅ 597L simplified (FZ 29d14b)
│   ├── pytorch_inference_handler.py                    ✅ 328L simplified (FZ 29d14g)
│   ├── pytorch_model_inference.py                      ✅ 386L simplified (FZ 29d14c)
│   ├── hyperparams/
│   │   └── hyperparameters_NA.json                     ✅ 37 fields
│   └── scripts/
│       ├── __init__.py                                 ✅
│       ├── package.py                                  ✅ COPY from framework
│       ├── payload.py                                  ✅ COPY from framework
│       ├── percentile_model_calibration.py             ✅ COPY from framework
│       ├── tabular_preprocessing_sampling_munged.py    ✅ ~130L custom
│       └── tabular_preprocessing_training_munged.py    ✅ ~110L custom
└── pipeline_configs/
    └── config_NA.json                                  🟡 skeleton (needs specific section)
```

**Convention notes** (from existing projects: bsm_pytorch, rnr_pytorch_bedrock, names2risk_pytorch):
- `dockers/requirements-secure.txt` — installed at TRAINING time by `pytorch_training.py`
- `dockers/pytorch_inference_handler.py` — inference handler bundled into `model.tar.gz/code/` by Package step
- `dockers/pytorch_model_inference.py` — scoring script for `PyTorchModelInference_calibration` step
- `dockers/scripts/` — processing scripts (generic copies + custom overrides via `processing_entry_point`)
- `pipeline_configs/` — Cursus DAG compiler reads config from here

## Deliverables

| # | Deliverable | Type | Location | Status |
|---|-------------|------|----------|--------|
| 1 | ETLM job (NA) | Datanet job | DataCentral | 🔴 |
| 2 | config_NA.json | JSON config | `pipeline_configs/` | 🟡 `generate_config_na.py` written, needs execution |
| 2b | generate_config_na.py | Config generator (291L) | project root | ✅ |
| 3 | hyperparameters_NA.json | JSON config (37 fields) | `dockers/hyperparams/` | ✅ |
| 4 | requirements-secure.txt | Training deps | `dockers/` | ✅ |
| 5 | pytorch_inference_handler.py | Inference handler (328L) | `dockers/` | ✅ |
| 6 | pytorch_model_inference.py | Model scoring for calibration (386L) | `dockers/` | ✅ |
| 7 | pytorch_training.py | Training script (597L) | `dockers/` | ✅ |
| 8 | tabular_preprocessing_sampling_munged.py | Custom script (~130L) | `dockers/scripts/` | ✅ |
| 9 | tabular_preprocessing_training_munged.py | Custom script (~110L) | `dockers/scripts/` | ✅ |
| 10 | package.py, payload.py, percentile_model_calibration.py | Generic copies | `dockers/scripts/` | ✅ |
| 11 | DAG validation report | Test output | — | 🔴 |
| 12 | Numerical equivalence report | Test output | — | 🔴 |
| 13 | MODS template package | Brazil package | BuyerAbuseModsTemplate | 🔴 |

## Dependencies

| Dependency | Status | Blocks |
|-----------|--------|--------|
| BedrockProcessing self-contained mode | ✅ Done | Sprint 3.4 |
| StratifiedSampling external_proportional | ✅ Done | Sprint 3.3 |
| TabularPreprocessing DATA_SECONDARY | ✅ Done | Sprint 2.1 |
| Spec gaps (BedrockProcessing→TabPreproc, PyTorch→Package) | ✅ Done | Sprint 4 |
| EdxUploading step created + registered | ✅ Done (FZ 18g3i) | Sprint 3.2 |
| EdxUploading DAG resolution validated (score 1.00) | ✅ Done (FZ 29d16g) | Sprint 3.2 |
| MDS service "FORTRESS_RETAIL" accessible from BRP-ML-Abuse account | ⚠️ Verify | Sprint 3.1 |
| ETLM job created + EDX dataset accessible | 🔴 Sprint 1.1 | Sprint 3.1 |
| Bedrock model access (Claude 3.5 Haiku) | ⚠️ Verify | Sprint 3.4 |
| ~~Kale attestation for Andes table~~ | ~~Not needed~~ | ~~Eliminated by EdxUploading~~ |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| MDS service not accessible from BRP-ML-Abuse Cradle account | Blocks data loading | Request access via SAIS admin, or use notebook owner's account temporarily |
| Bedrock Claude 3.5 Haiku throttle rate insufficient for 1.2M addresses | Extends runtime beyond 24h | Use adaptive rate limiting; consider Nova Lite + Converse API as fallback |
| `brm_mo_tt_orders` Andes table missing `mo_how` column | Cannot simplify MO query | Use ETLM approach (confirmed solution in FZ 29d16a1) |
| Numerical divergence from notebook (AUC difference) | Blocks production deployment | Debug: check data splits, label distribution, tokenization differences |

## Verification Criteria

- [ ] DAG compiles without errors (all 16 nodes resolved)
- [ ] All 5 alignment checks PASS for custom steps
- [ ] Pipeline produces model.tar.gz with correct structure
- [ ] Model AUC ≥ 0.87 on test set (within ±0.01 of notebook's 0.88)
- [ ] MIMS registration succeeds in gamma
- [ ] Load test passes (TPS=5, latency≤100ms on ml.m5.xlarge)
- [ ] Calibration produces valid percentile mapping
