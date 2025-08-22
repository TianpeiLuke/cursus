---
tags:
  - analysis
  - metrics
  - codebase
  - documentation
  - statistics
keywords:
  - lines of code
  - code metrics
  - project analysis
  - Python files
  - markdown files
  - test coverage
  - source code analysis
  - documentation metrics
topics:
  - codebase analysis
  - project metrics
  - code statistics
  - documentation analysis
language: python
date of note: 2025-08-22
---

# Cursus Project Line Count Analysis

## Executive Summary

**Generated on:** 2025-08-22

### Project Scale Overview
- **Total Python codebase:** 108494 lines of code
- **Total project content:** Over 895.8K words of documentation
- **File count:** 1068 total files analyzed (Python + Markdown)

### Key Metrics
- **Python files in src/cursus:** 307 files, 65589 lines of code
- **Python files in test:** 218 files, 42905 lines of code
- **Markdown files in slipbox:** 543 files, 230431 lines, 895835 words

### Quality Indicators
- **Test-to-source ratio:** 65% (test LOC / source LOC)
- **Documentation-to-code ratio:** 3.5:1 (documentation lines / source code lines)

---

## Methodology

The analysis was performed using automated scripts that:
- Count non-empty, non-comment lines for Python files
- Count all lines and words for markdown files
- Provide individual file breakdowns and category totals
- Exclude system files and focus on project content

## Detailed Results


---

CURSUS PROJECT LINE COUNT ANALYSIS

---


## PYTHON FILES IN src/cursus PACKAGE


---

### Individual Python files in src/cursus:

```
src/cursus/__init__.py                                           90 lines
src/cursus/__version__.py                                         8 lines
src/cursus/api/__init__.py                                       23 lines
src/cursus/api/dag/__init__.py                                   23 lines
src/cursus/api/dag/base_dag.py                                   60 lines
src/cursus/api/dag/edge_types.py                                220 lines
src/cursus/api/dag/enhanced_dag.py                              273 lines
src/cursus/cli/__init__.py                                        3 lines
src/cursus/cli/__main__.py                                        6 lines
src/cursus/cli/alignment_cli.py                                 992 lines
src/cursus/cli/builder_test_cli.py                              533 lines
src/cursus/cli/catalog_cli.py                                   545 lines
src/cursus/cli/validation_cli.py                                179 lines
src/cursus/core/__init__.py                                     141 lines
src/cursus/core/assembler/__init__.py                            11 lines
src/cursus/core/assembler/pipeline_assembler.py                 334 lines
src/cursus/core/assembler/pipeline_template_base.py             308 lines
src/cursus/core/base/__init__.py                                 65 lines
src/cursus/core/base/builder_base.py                            619 lines
src/cursus/core/base/config_base.py                             322 lines
src/cursus/core/base/contract_base.py                           266 lines
src/cursus/core/base/enums.py                                    36 lines
src/cursus/core/base/hyperparameters_base.py                    228 lines
src/cursus/core/base/specification_base.py                      519 lines
src/cursus/core/compiler/__init__.py                             49 lines
src/cursus/core/compiler/config_resolver.py                     527 lines
src/cursus/core/compiler/dag_compiler.py                        431 lines
src/cursus/core/compiler/dynamic_template.py                    641 lines
src/cursus/core/compiler/exceptions.py                           78 lines
src/cursus/core/compiler/name_generator.py                       73 lines
src/cursus/core/compiler/validation.py                          249 lines
src/cursus/core/config_fields/__init__.py                       224 lines
src/cursus/core/config_fields/circular_reference_tracker.py     153 lines
src/cursus/core/config_fields/config_class_detector.py          140 lines
src/cursus/core/config_fields/config_class_store.py              99 lines
src/cursus/core/config_fields/config_field_categorizer.py       266 lines
src/cursus/core/config_fields/config_merger.py                  266 lines
src/cursus/core/config_fields/constants.py                       68 lines
src/cursus/core/config_fields/cradle_config_factory.py          518 lines
src/cursus/core/config_fields/tier_registry.py                  141 lines
src/cursus/core/config_fields/type_aware_config_serializer.py    471 lines
src/cursus/core/deps/__init__.py                                 40 lines
src/cursus/core/deps/dependency_resolver.py                     451 lines
src/cursus/core/deps/factory.py                                  40 lines
src/cursus/core/deps/property_reference.py                      166 lines
src/cursus/core/deps/registry_manager.py                        161 lines
src/cursus/core/deps/semantic_matcher.py                        183 lines
src/cursus/core/deps/specification_registry.py                   80 lines
src/cursus/mods/__init__.py                                       4 lines
src/cursus/mods/compiler/__init__.py                              7 lines
src/cursus/mods/compiler/mods_dag_compiler.py                   265 lines
src/cursus/pipeline_catalog/__init__.py                          82 lines
src/cursus/pipeline_catalog/mods_pipelines/__init__.py           53 lines
src/cursus/pipeline_catalog/mods_pipelines/dummy_mods_e2e_basic.py    340 lines
src/cursus/pipeline_catalog/mods_pipelines/pytorch_mods_e2e_standard.py    348 lines
src/cursus/pipeline_catalog/mods_pipelines/pytorch_mods_training_basic.py    292 lines
src/cursus/pipeline_catalog/mods_pipelines/xgb_mods_e2e_comprehensive.py    349 lines
src/cursus/pipeline_catalog/mods_pipelines/xgb_mods_training_calibrated.py    306 lines
src/cursus/pipeline_catalog/mods_pipelines/xgb_mods_training_evaluation.py    227 lines
src/cursus/pipeline_catalog/mods_pipelines/xgb_mods_training_simple.py    288 lines
src/cursus/pipeline_catalog/pipelines/__init__.py                52 lines
src/cursus/pipeline_catalog/pipelines/dummy_e2e_basic.py        268 lines
src/cursus/pipeline_catalog/pipelines/pytorch_e2e_standard.py    269 lines
src/cursus/pipeline_catalog/pipelines/pytorch_training_basic.py    245 lines
src/cursus/pipeline_catalog/pipelines/xgb_e2e_comprehensive.py    274 lines
src/cursus/pipeline_catalog/pipelines/xgb_training_calibrated.py    214 lines
src/cursus/pipeline_catalog/pipelines/xgb_training_evaluation.py    262 lines
src/cursus/pipeline_catalog/pipelines/xgb_training_simple.py    262 lines
src/cursus/pipeline_catalog/shared_dags/__init__.py             116 lines
src/cursus/pipeline_catalog/shared_dags/dummy/__init__.py        11 lines
src/cursus/pipeline_catalog/shared_dags/dummy/e2e_basic_dag.py    121 lines
src/cursus/pipeline_catalog/shared_dags/enhanced_metadata.py    370 lines
src/cursus/pipeline_catalog/shared_dags/pytorch/__init__.py      16 lines
src/cursus/pipeline_catalog/shared_dags/pytorch/standard_e2e_dag.py    128 lines
src/cursus/pipeline_catalog/shared_dags/pytorch/training_dag.py    109 lines
src/cursus/pipeline_catalog/shared_dags/registry_sync.py        473 lines
src/cursus/pipeline_catalog/shared_dags/xgboost/__init__.py      26 lines
src/cursus/pipeline_catalog/shared_dags/xgboost/complete_e2e_dag.py    132 lines
src/cursus/pipeline_catalog/shared_dags/xgboost/simple_dag.py    101 lines
src/cursus/pipeline_catalog/shared_dags/xgboost/training_with_calibration_dag.py    114 lines
src/cursus/pipeline_catalog/shared_dags/xgboost/training_with_evaluation_dag.py    114 lines
src/cursus/pipeline_catalog/utils.py                            142 lines
src/cursus/pipeline_catalog/utils/__init__.py                    17 lines
src/cursus/pipeline_catalog/utils/catalog_registry.py           504 lines
src/cursus/pipeline_catalog/utils/connection_traverser.py       437 lines
src/cursus/pipeline_catalog/utils/recommendation_engine.py      518 lines
src/cursus/pipeline_catalog/utils/registry_validator.py         576 lines
src/cursus/pipeline_catalog/utils/tag_discovery.py              450 lines
src/cursus/processing/__init__.py                                69 lines
src/cursus/processing/bert_tokenize_processor.py                 43 lines
src/cursus/processing/bsm_dataloader.py                          36 lines
src/cursus/processing/bsm_datasets.py                           162 lines
src/cursus/processing/bsm_processor.py                          137 lines
src/cursus/processing/categorical_label_processor.py             30 lines
src/cursus/processing/cs_processor.py                            72 lines
src/cursus/processing/gensim_tokenize_processor.py               56 lines
src/cursus/processing/multiclass_label_processor.py              55 lines
src/cursus/processing/numerical_binning_processor.py            271 lines
src/cursus/processing/numerical_imputation_processor.py         106 lines
src/cursus/processing/processors.py                              39 lines
src/cursus/processing/risk_table_processor.py                   176 lines
src/cursus/steps/__init__.py                                     30 lines
src/cursus/steps/builders/__init__.py                            39 lines
src/cursus/steps/builders/builder_batch_transform_step.py       218 lines
src/cursus/steps/builders/builder_cradle_data_loading_step.py    464 lines
src/cursus/steps/builders/builder_currency_conversion_step.py    269 lines
src/cursus/steps/builders/builder_dummy_training_step.py        363 lines
src/cursus/steps/builders/builder_model_calibration_step.py     390 lines
src/cursus/steps/builders/builder_package_step.py               281 lines
src/cursus/steps/builders/builder_payload_step.py               254 lines
src/cursus/steps/builders/builder_pytorch_model_step.py         198 lines
src/cursus/steps/builders/builder_pytorch_training_step.py      335 lines
src/cursus/steps/builders/builder_registration_step.py          209 lines
src/cursus/steps/builders/builder_risk_table_mapping_step.py    356 lines
src/cursus/steps/builders/builder_tabular_preprocessing_step.py    261 lines
src/cursus/steps/builders/builder_xgboost_model_eval_step.py    257 lines
src/cursus/steps/builders/builder_xgboost_model_step.py         196 lines
src/cursus/steps/builders/builder_xgboost_training_step.py      413 lines
src/cursus/steps/builders/s3_utils.py                           150 lines
src/cursus/steps/configs/__init__.py                             84 lines
src/cursus/steps/configs/config_batch_transform_step.py          72 lines
src/cursus/steps/configs/config_cradle_data_loading_step.py     624 lines
src/cursus/steps/configs/config_currency_conversion_step.py     156 lines
src/cursus/steps/configs/config_dummy_training_step.py          105 lines
src/cursus/steps/configs/config_model_calibration_step.py       258 lines
src/cursus/steps/configs/config_package_step.py                  76 lines
src/cursus/steps/configs/config_payload_step.py                 439 lines
src/cursus/steps/configs/config_processing_step_base.py         274 lines
src/cursus/steps/configs/config_pytorch_model_step.py            75 lines
src/cursus/steps/configs/config_pytorch_training_step.py         58 lines
src/cursus/steps/configs/config_registration_step.py            285 lines
src/cursus/steps/configs/config_risk_table_mapping_step.py       84 lines
src/cursus/steps/configs/config_tabular_preprocessing_step.py    160 lines
src/cursus/steps/configs/config_xgboost_model_eval_step.py      138 lines
src/cursus/steps/configs/config_xgboost_model_step.py           134 lines
src/cursus/steps/configs/config_xgboost_training_step.py        138 lines
src/cursus/steps/configs/utils.py                               304 lines
src/cursus/steps/contracts/__init__.py                           42 lines
src/cursus/steps/contracts/contract_validator.py                204 lines
src/cursus/steps/contracts/cradle_data_loading_contract.py       54 lines
src/cursus/steps/contracts/currency_conversion_contract.py       72 lines
src/cursus/steps/contracts/dummy_training_contract.py            27 lines
src/cursus/steps/contracts/mims_registration_contract.py         52 lines
src/cursus/steps/contracts/model_calibration_contract.py         60 lines
src/cursus/steps/contracts/package_contract.py                   39 lines
src/cursus/steps/contracts/payload_contract.py                   49 lines
src/cursus/steps/contracts/pytorch_training_contract.py          84 lines
src/cursus/steps/contracts/risk_table_mapping_contract.py        64 lines
src/cursus/steps/contracts/tabular_preprocess_contract.py        50 lines
src/cursus/steps/contracts/training_script_contract.py          118 lines
src/cursus/steps/contracts/xgboost_model_eval_contract.py        58 lines
src/cursus/steps/contracts/xgboost_training_contract.py          92 lines
src/cursus/steps/hyperparams/__init__.py                         14 lines
src/cursus/steps/hyperparams/hyperparameters_bsm.py             178 lines
src/cursus/steps/hyperparams/hyperparameters_xgboost.py         133 lines
src/cursus/steps/registry/__init__.py                            64 lines
src/cursus/steps/registry/builder_registry.py                   432 lines
src/cursus/steps/registry/exceptions.py                          20 lines
src/cursus/steps/registry/hyperparameter_registry.py             45 lines
src/cursus/steps/registry/step_names.py                         335 lines
src/cursus/steps/registry/step_type_test_variants.py            499 lines
src/cursus/steps/scripts/__init__.py                              7 lines
src/cursus/steps/scripts/currency_conversion.py                 258 lines
src/cursus/steps/scripts/dummy_training.py                      215 lines
src/cursus/steps/scripts/model_calibration.py                   918 lines
src/cursus/steps/scripts/package.py                             259 lines
src/cursus/steps/scripts/payload.py                             425 lines
src/cursus/steps/scripts/risk_table_mapping.py                  399 lines
src/cursus/steps/scripts/tabular_preprocessing.py               207 lines
src/cursus/steps/scripts/xgboost_model_evaluation.py            366 lines
src/cursus/steps/scripts/xgboost_training.py                    455 lines
src/cursus/steps/specs/__init__.py                               72 lines
src/cursus/steps/specs/batch_transform_calibration_spec.py       40 lines
src/cursus/steps/specs/batch_transform_testing_spec.py           40 lines
src/cursus/steps/specs/batch_transform_training_spec.py          40 lines
src/cursus/steps/specs/batch_transform_validation_spec.py        40 lines
src/cursus/steps/specs/cradle_data_loading_calibration_spec.py     39 lines
src/cursus/steps/specs/cradle_data_loading_spec.py               43 lines
src/cursus/steps/specs/cradle_data_loading_testing_spec.py       39 lines
src/cursus/steps/specs/cradle_data_loading_training_spec.py      39 lines
src/cursus/steps/specs/cradle_data_loading_validation_spec.py     39 lines
src/cursus/steps/specs/currency_conversion_calibration_spec.py     35 lines
src/cursus/steps/specs/currency_conversion_testing_spec.py       35 lines
src/cursus/steps/specs/currency_conversion_training_spec.py      35 lines
src/cursus/steps/specs/currency_conversion_validation_spec.py     35 lines
src/cursus/steps/specs/dummy_training_spec.py                    46 lines
src/cursus/steps/specs/model_calibration_calibration_spec.py     52 lines
src/cursus/steps/specs/model_calibration_spec.py                 55 lines
src/cursus/steps/specs/model_calibration_testing_spec.py         52 lines
src/cursus/steps/specs/model_calibration_training_spec.py        52 lines
src/cursus/steps/specs/model_calibration_validation_spec.py      52 lines
src/cursus/steps/specs/package_spec.py                           54 lines
src/cursus/steps/specs/payload_spec.py                           36 lines
src/cursus/steps/specs/pytorch_model_spec.py                     32 lines
src/cursus/steps/specs/pytorch_training_spec.py                  44 lines
src/cursus/steps/specs/registration_spec.py                      40 lines
src/cursus/steps/specs/risk_table_mapping_calibration_spec.py     65 lines
src/cursus/steps/specs/risk_table_mapping_testing_spec.py        65 lines
src/cursus/steps/specs/risk_table_mapping_training_spec.py       65 lines
src/cursus/steps/specs/risk_table_mapping_validation_spec.py     65 lines
src/cursus/steps/specs/tabular_preprocessing_calibration_spec.py     36 lines
src/cursus/steps/specs/tabular_preprocessing_spec.py             36 lines
src/cursus/steps/specs/tabular_preprocessing_testing_spec.py     35 lines
src/cursus/steps/specs/tabular_preprocessing_training_spec.py     36 lines
src/cursus/steps/specs/tabular_preprocessing_validation_spec.py     35 lines
src/cursus/steps/specs/xgboost_model_eval_spec.py                54 lines
src/cursus/steps/specs/xgboost_model_spec.py                     32 lines
src/cursus/steps/specs/xgboost_training_spec.py                  53 lines
src/cursus/validation/__init__.py                                97 lines
src/cursus/validation/alignment/__init__.py                      28 lines
src/cursus/validation/alignment/alignment_reporter.py           635 lines
src/cursus/validation/alignment/alignment_scorer.py             417 lines
src/cursus/validation/alignment/alignment_utils.py               74 lines
src/cursus/validation/alignment/analyzers/__init__.py            10 lines
src/cursus/validation/alignment/analyzers/builder_analyzer.py    273 lines
src/cursus/validation/alignment/analyzers/config_analyzer.py    261 lines
src/cursus/validation/alignment/builder_config_alignment.py     392 lines
src/cursus/validation/alignment/contract_spec_alignment.py      201 lines
src/cursus/validation/alignment/core_models.py                  139 lines
src/cursus/validation/alignment/dependency_classifier.py         97 lines
src/cursus/validation/alignment/discovery/__init__.py             6 lines
src/cursus/validation/alignment/discovery/contract_discovery.py    255 lines
src/cursus/validation/alignment/enhanced_reporter.py            487 lines
src/cursus/validation/alignment/file_resolver.py                223 lines
src/cursus/validation/alignment/framework_patterns.py           331 lines
src/cursus/validation/alignment/level3_validation_config.py     129 lines
src/cursus/validation/alignment/loaders/__init__.py               7 lines
src/cursus/validation/alignment/loaders/contract_loader.py      131 lines
src/cursus/validation/alignment/loaders/specification_loader.py    314 lines
src/cursus/validation/alignment/orchestration/__init__.py         6 lines
src/cursus/validation/alignment/orchestration/validation_orchestrator.py    288 lines
src/cursus/validation/alignment/patterns/__init__.py             11 lines
src/cursus/validation/alignment/patterns/file_resolver.py       207 lines
src/cursus/validation/alignment/patterns/pattern_recognizer.py    213 lines
src/cursus/validation/alignment/processors/__init__.py            6 lines
src/cursus/validation/alignment/processors/spec_file_processor.py    187 lines
src/cursus/validation/alignment/property_path_validator.py      583 lines
src/cursus/validation/alignment/script_analysis_models.py       105 lines
src/cursus/validation/alignment/script_contract_alignment.py    568 lines
src/cursus/validation/alignment/smart_spec_selector.py          230 lines
src/cursus/validation/alignment/spec_dependency_alignment.py    302 lines
src/cursus/validation/alignment/static_analysis/__init__.py      13 lines
src/cursus/validation/alignment/static_analysis/builder_analyzer.py    215 lines
src/cursus/validation/alignment/static_analysis/import_analyzer.py    253 lines
src/cursus/validation/alignment/static_analysis/path_extractor.py    291 lines
src/cursus/validation/alignment/static_analysis/script_analyzer.py    615 lines
src/cursus/validation/alignment/step_type_detection.py          141 lines
src/cursus/validation/alignment/step_type_enhancement_router.py    173 lines
src/cursus/validation/alignment/step_type_enhancers/__init__.py     21 lines
src/cursus/validation/alignment/step_type_enhancers/base_enhancer.py    205 lines
src/cursus/validation/alignment/step_type_enhancers/createmodel_enhancer.py    641 lines
src/cursus/validation/alignment/step_type_enhancers/processing_enhancer.py    463 lines
src/cursus/validation/alignment/step_type_enhancers/registermodel_enhancer.py    158 lines
src/cursus/validation/alignment/step_type_enhancers/training_enhancer.py    494 lines
src/cursus/validation/alignment/step_type_enhancers/transform_enhancer.py    244 lines
src/cursus/validation/alignment/step_type_enhancers/utility_enhancer.py    224 lines
src/cursus/validation/alignment/testability_validator.py        581 lines
src/cursus/validation/alignment/unified_alignment_tester.py     486 lines
src/cursus/validation/alignment/utils.py                        148 lines
src/cursus/validation/alignment/validators/__init__.py            9 lines
src/cursus/validation/alignment/validators/contract_spec_validator.py    104 lines
src/cursus/validation/alignment/validators/dependency_validator.py    313 lines
src/cursus/validation/alignment/validators/legacy_validators.py    178 lines
src/cursus/validation/alignment/validators/script_contract_validator.py    653 lines
src/cursus/validation/alignment/workflow_integration.py         403 lines
src/cursus/validation/builders/__init__.py                       66 lines
src/cursus/validation/builders/base_test.py                     476 lines
src/cursus/validation/builders/builder_reporter.py              707 lines
src/cursus/validation/builders/example_enhanced_usage.py        140 lines
src/cursus/validation/builders/example_usage.py                  56 lines
src/cursus/validation/builders/generic_test.py                  102 lines
src/cursus/validation/builders/integration_tests.py              38 lines
src/cursus/validation/builders/interface_tests.py               227 lines
src/cursus/validation/builders/mock_factory.py                 1011 lines
src/cursus/validation/builders/registry_discovery.py            305 lines
src/cursus/validation/builders/sagemaker_step_type_validator.py    276 lines
src/cursus/validation/builders/scoring.py                       458 lines
src/cursus/validation/builders/specification_tests.py            43 lines
src/cursus/validation/builders/step_creation_tests.py           345 lines
src/cursus/validation/builders/step_info_detector.py             94 lines
src/cursus/validation/builders/test_factory.py                   90 lines
src/cursus/validation/builders/universal_test.py                672 lines
src/cursus/validation/builders/variants/__init__.py               3 lines
src/cursus/validation/builders/variants/createmodel_integration_tests.py    617 lines
src/cursus/validation/builders/variants/createmodel_interface_tests.py    387 lines
src/cursus/validation/builders/variants/createmodel_specification_tests.py    432 lines
src/cursus/validation/builders/variants/createmodel_test.py     412 lines
src/cursus/validation/builders/variants/processing_integration_tests.py    461 lines
src/cursus/validation/builders/variants/processing_interface_tests.py    184 lines
src/cursus/validation/builders/variants/processing_pattern_b_test_runner.py    252 lines
src/cursus/validation/builders/variants/processing_specification_tests.py    323 lines
src/cursus/validation/builders/variants/processing_step_creation_tests.py     52 lines
src/cursus/validation/builders/variants/processing_test.py      311 lines
src/cursus/validation/builders/variants/training_integration_tests.py    574 lines
src/cursus/validation/builders/variants/training_interface_tests.py    425 lines
src/cursus/validation/builders/variants/training_specification_tests.py    434 lines
src/cursus/validation/builders/variants/training_test.py        406 lines
src/cursus/validation/builders/variants/transform_integration_tests.py    643 lines
src/cursus/validation/builders/variants/transform_interface_tests.py    381 lines
src/cursus/validation/builders/variants/transform_specification_tests.py    494 lines
src/cursus/validation/builders/variants/transform_test.py       444 lines
src/cursus/validation/interface/__init__.py                      11 lines
src/cursus/validation/interface/interface_standard_validator.py    371 lines
src/cursus/validation/naming/__init__.py                          9 lines
src/cursus/validation/naming/naming_standard_validator.py       485 lines
src/cursus/validation/shared/chart_utils.py                     307 lines
src/cursus/validation/simple_integration.py                     269 lines

```

**TOTAL LINES OF CODE in src/cursus package: 65589**


## PYTHON TEST FILES IN test FOLDER


---

### Individual Python test files:

```
test/__init__.py                                                  0 lines
test/analyze_test_coverage.py                                   427 lines
test/api/dag/__init__.py                                          0 lines
test/api/dag/test_base_dag.py                                    96 lines
test/api/dag/test_edge_types.py                                 412 lines
test/circular_imports/__init__.py                                 5 lines
test/circular_imports/run_circular_import_test.py                22 lines
test/circular_imports/test_circular_imports.py                  344 lines
test/cli/__init__.py                                              5 lines
test/cli/run_cli_tests.py                                        42 lines
test/cli/test_alignment_cli.py                                  731 lines
test/cli/test_builder_test_cli.py                               614 lines
test/cli/test_validation_cli.py                                 365 lines
test/core/assembler/__init__.py                                   0 lines
test/core/assembler/test_pipeline_assembler.py                  331 lines
test/core/assembler/test_pipeline_builder_template.py           438 lines
test/core/base/__init__.py                                        3 lines
test/core/base/test_all_base.py                                 147 lines
test/core/base/test_builder_base.py                             312 lines
test/core/base/test_config_base.py                              208 lines
test/core/base/test_contract_base.py                            338 lines
test/core/base/test_enums.py                                    286 lines
test/core/base/test_hyperparameters_base.py                     249 lines
test/core/base/test_specification_base.py                       386 lines
test/core/compiler/__init__.py                                    1 lines
test/core/compiler/test_config_resolver.py                      227 lines
test/core/compiler/test_dag_compiler.py                         473 lines
test/core/compiler/test_dynamic_template.py                     317 lines
test/core/compiler/test_enhanced_config_resolver.py             209 lines
test/core/compiler/test_exceptions.py                           167 lines
test/core/compiler/test_fill_execution_document.py              365 lines
test/core/compiler/test_name_generator.py                        62 lines
test/core/compiler/test_validation.py                           305 lines
test/core/config_fields/__init__.py                               3 lines
test/core/config_fields/run_all_tests.py                        207 lines
test/core/config_fields/test_bug_fixes_consolidated.py          386 lines
test/core/config_fields/test_circular_reference_consolidated.py    353 lines
test/core/config_fields/test_circular_reference_tracker.py      217 lines
test/core/config_fields/test_config_class_detector.py           320 lines
test/core/config_fields/test_config_class_store.py              158 lines
test/core/config_fields/test_config_field_categorizer.py        234 lines
test/core/config_fields/test_config_merger.py                   337 lines
test/core/config_fields/test_constants.py                       209 lines
test/core/config_fields/test_integration.py                     268 lines
test/core/config_fields/test_tier_registry.py                   199 lines
test/core/config_fields/test_type_aware_deserialization.py      465 lines
test/core/config_fields/test_type_aware_serialization.py        121 lines
test/core/deps/__init__.py                                        1 lines
test/core/deps/test_dependency_resolver.py                      591 lines
test/core/deps/test_factory.py                                  295 lines
test/core/deps/test_global_state_isolation.py                   211 lines
test/core/deps/test_helpers.py                                   38 lines
test/core/deps/test_property_reference.py                       264 lines
test/core/deps/test_registry_manager.py                         310 lines
test/core/deps/test_semantic_matcher.py                         255 lines
test/core/deps/test_specification_registry.py                   492 lines
test/core/run_core_tests.py                                     522 lines
test/integration/__init__.py                                      8 lines
test/integration/test_job_type_integration.py                   186 lines
test/integration/test_registry_manager_pipeline_integration.py     99 lines
test/integration/test_script_contract_integration.py            123 lines
test/integration/test_step_specification_integration.py         151 lines
test/pipeline_catalog/shared_dags/__init__.py                     6 lines
test/pipeline_catalog/shared_dags/test_enhanced_metadata_core.py    354 lines
test/pipeline_catalog/shared_dags/test_enhanced_metadata_integration.py    368 lines
test/pipeline_catalog/shared_dags/test_enhanced_metadata_models.py    149 lines
test/pipeline_catalog/shared_dags/test_enhanced_metadata.py      31 lines
test/pipeline_catalog/shared_dags/test_registry_sync_core.py    413 lines
test/pipeline_catalog/shared_dags/test_registry_sync_integration.py    411 lines
test/pipeline_catalog/shared_dags/test_registry_sync_models.py    151 lines
test/pipeline_catalog/shared_dags/test_registry_sync.py          26 lines
test/pipeline_catalog/test_indexer.py                           284 lines
test/pipeline_catalog/test_phase1_implementation.py             512 lines
test/pipeline_catalog/test_utils.py                             102 lines
test/pipeline_catalog/utils/__init__.py                           6 lines
test/pipeline_catalog/utils/test_catalog_registry.py            469 lines
test/pipeline_catalog/utils/test_connection_traverser.py        375 lines
test/pipeline_catalog/utils/test_recommendation_engine_core.py    341 lines
test/pipeline_catalog/utils/test_recommendation_engine_integration.py    333 lines
test/pipeline_catalog/utils/test_recommendation_engine_models.py     63 lines
test/pipeline_catalog/utils/test_recommendation_engine.py        27 lines
test/pipeline_catalog/utils/test_registry_validator_core.py     389 lines
test/pipeline_catalog/utils/test_registry_validator_integration.py    407 lines
test/pipeline_catalog/utils/test_registry_validator_models.py    176 lines
test/pipeline_catalog/utils/test_registry_validator.py           31 lines
test/pipeline_catalog/utils/test_tag_discovery.py               512 lines
test/steps/builders/__init__.py                                   0 lines
test/steps/builders/generate_processing_reports.py               35 lines
test/steps/builders/generate_simple_reports.py                  271 lines
test/steps/builders/generate_step_reports.py                    355 lines
test/steps/builders/run_createmodel_tests.py                    208 lines
test/steps/builders/run_processing_tests.py                     117 lines
test/steps/builders/run_training_tests.py                       208 lines
test/steps/builders/run_transform_tests.py                      208 lines
test/steps/builders/test_createmodel_step_builders.py           441 lines
test/steps/builders/test_processing_step_builders.py            454 lines
test/steps/builders/test_real_builders.py                       172 lines
test/steps/builders/test_registry_integration.py                124 lines
test/steps/builders/test_training_step_builders.py              435 lines
test/steps/builders/test_transform_step_builders.py             390 lines
test/steps/configs/__init__.py                                    0 lines
test/steps/configs/test_config_inheritance.py                    93 lines
test/steps/configs/test_config_loading.py                       140 lines
test/steps/configs/test_field_sources.py                        127 lines
test/steps/configs/test_utils_basic_serialization.py            116 lines
test/steps/configs/test_utils_flattened_structure.py            323 lines
test/steps/configs/test_utils.py                                442 lines
test/steps/registry/__init__.py                                   0 lines
test/steps/registry/mock_modules.py                              41 lines
test/steps/registry/test_builder_registry.py                     65 lines
test/steps/registry/test_exceptions.py                           67 lines
test/steps/registry/test_hyperparameter_registry.py              96 lines
test/steps/registry/test_step_builder_discovery.py              147 lines
test/steps/registry/test_step_names.py                          274 lines
test/steps/scripts/__init__.py                                    0 lines
test/steps/scripts/alignment_validation/generate_validation_scripts.py    254 lines
test/steps/scripts/alignment_validation/run_alignment_validation.py    405 lines
test/steps/scripts/alignment_validation/run_visualization_validation.py    271 lines
test/steps/scripts/alignment_validation/validate_currency_conversion.py    206 lines
test/steps/scripts/alignment_validation/validate_dummy_training.py    205 lines
test/steps/scripts/alignment_validation/validate_model_calibration.py    234 lines
test/steps/scripts/alignment_validation/validate_package.py     205 lines
test/steps/scripts/alignment_validation/validate_payload.py     205 lines
test/steps/scripts/alignment_validation/validate_risk_table_mapping.py    205 lines
test/steps/scripts/alignment_validation/validate_tabular_preprocessing.py    205 lines
test/steps/scripts/alignment_validation/validate_xgboost_model_evaluation.py    205 lines
test/steps/scripts/alignment_validation/validate_xgboost_training.py    205 lines
test/steps/scripts/alignment_validation/validation_summary.py    133 lines
test/steps/scripts/regenerate_alignment_visualizations.py        86 lines
test/steps/scripts/test_currency_conversion.py                  500 lines
test/steps/scripts/test_dummy_training.py                       260 lines
test/steps/scripts/test_model_calibration.py                    497 lines
test/steps/scripts/test_package.py                              139 lines
test/steps/scripts/test_payload.py                              338 lines
test/steps/scripts/test_risk_table_mapping.py                   154 lines
test/steps/scripts/test_tabular_preprocessing.py                164 lines
test/steps/scripts/test_xgboost_model_evaluation.py             380 lines
test/steps/scripts/test_xgboost_training.py                     420 lines
test/steps/specs/test_node_type_validation.py                   274 lines
test/steps/specs/test_output_spec_aliases.py                    262 lines
test/steps/specs/test_step_name_consistency.py                  221 lines
test/steps/test_sagemaker_step_type_implementation.py           172 lines
test/validation/__init__.py                                       3 lines
test/validation/alignment/__init__.py                            12 lines
test/validation/alignment/__pycache__/__init__.py                 0 lines
test/validation/alignment/analyzers/__init__.py                   3 lines
test/validation/alignment/analyzers/__pycache__/__init__.py       0 lines
test/validation/alignment/analyzers/test_builder_analyzer.py    191 lines
test/validation/alignment/analyzers/test_config_analyzer.py     209 lines
test/validation/alignment/discovery/__init__.py                   0 lines
test/validation/alignment/loaders/__init__.py                     0 lines
test/validation/alignment/orchestration/__init__.py               0 lines
test/validation/alignment/patterns/__init__.py                    0 lines
test/validation/alignment/processors/__init__.py                  0 lines
test/validation/alignment/reporter/__init__.py                    3 lines
test/validation/alignment/reporter/__pycache__/__init__.py        0 lines
test/validation/alignment/reporter/test_alignment_report.py     124 lines
test/validation/alignment/reporter/test_validation_result.py    103 lines
test/validation/alignment/run_all_alignment_tests.py            297 lines
test/validation/alignment/run_builder_config_tests.py            53 lines
test/validation/alignment/script_contract/__init__.py             3 lines
test/validation/alignment/script_contract/__pycache__/__init__.py      0 lines
test/validation/alignment/script_contract/test_argument_validation.py    366 lines
test/validation/alignment/script_contract/test_script_contract_path_validation.py    249 lines
test/validation/alignment/script_contract/test_script_contract_validator.py    338 lines
test/validation/alignment/script_contract/test_testability_validation.py    254 lines
test/validation/alignment/static_analysis/__init__.py             0 lines
test/validation/alignment/step_type_enhancers/__init__.py         4 lines
test/validation/alignment/step_type_enhancers/__pycache__/__init__.py      0 lines
test/validation/alignment/step_type_enhancers/test_base_enhancer.py    216 lines
test/validation/alignment/step_type_enhancers/test_training_enhancer.py    265 lines
test/validation/alignment/test_alignment_integration.py         219 lines
test/validation/alignment/test_alignment_scorer.py              253 lines
test/validation/alignment/test_builder_argument_debug.py        177 lines
test/validation/alignment/test_builder_argument_integration.py    152 lines
test/validation/alignment/test_builder_config_alignment.py      276 lines
test/validation/alignment/test_enhanced_argument_validation.py    210 lines
test/validation/alignment/test_framework_patterns.py            201 lines
test/validation/alignment/test_sagemaker_property_path_validation.py    259 lines
test/validation/alignment/test_step_type_detection.py           133 lines
test/validation/alignment/test_step_type_enhancement_router.py    227 lines
test/validation/alignment/test_step_type_enhancement_system_comprehensive.py    106 lines
test/validation/alignment/test_unified_alignment_tester_visualization.py    389 lines
test/validation/alignment/test_visualization_integration_complete.py    394 lines
test/validation/alignment/test_workflow_integration.py          381 lines
test/validation/alignment/unified_tester/__init__.py              3 lines
test/validation/alignment/unified_tester/__pycache__/__init__.py      0 lines
test/validation/alignment/unified_tester/test_full_validation.py    441 lines
test/validation/alignment/unified_tester/test_level_validation.py    246 lines
test/validation/alignment/utils/__init__.py                       3 lines
test/validation/alignment/utils/__pycache__/__init__.py           0 lines
test/validation/alignment/utils/test_alignment_issue.py         106 lines
test/validation/alignment/utils/test_alignment_level.py          62 lines
test/validation/alignment/utils/test_core_models.py             201 lines
test/validation/alignment/utils/test_path_reference.py           86 lines
test/validation/alignment/utils/test_script_analysis_models.py    311 lines
test/validation/alignment/utils/test_severity_level.py           64 lines
test/validation/alignment/utils/test_step_type_detection.py     211 lines
test/validation/alignment/utils/test_utility_functions.py       142 lines
test/validation/alignment/validators/__init__.py                  0 lines
test/validation/builders/test_pattern_scoring.py                308 lines
test/validation/interface/__init__.py                             3 lines
test/validation/interface/test_interface_violation.py            59 lines
test/validation/interface/test_validator_core.py                190 lines
test/validation/interface/test_validator_integration.py         116 lines
test/validation/naming/__init__.py                                5 lines
test/validation/naming/run_all_tests.py                          60 lines
test/validation/naming/test_builder_class_name_validation.py     55 lines
test/validation/naming/test_canonical_step_name_validation.py     68 lines
test/validation/naming/test_class_validation.py                  91 lines
test/validation/naming/test_config_class_name_validation.py      55 lines
test/validation/naming/test_file_naming_validation.py           111 lines
test/validation/naming/test_logical_name_validation.py           80 lines
test/validation/naming/test_naming_violation.py                  60 lines
test/validation/naming/test_registry_validation.py              118 lines
test/validation/naming/test_validator_basic.py                   27 lines
test/validation/test_step_type_enhancement_system.py            292 lines
test/validation/test_unified_alignment_tester.py                261 lines

```

**TOTAL LINES OF CODE in test folder: 42905**


## MARKDOWN FILES IN slipbox FOLDER


---

### Individual markdown files:

```
slipbox/0_developer_guide/adding_new_pipeline_step.md                      94 lines      478 words
slipbox/0_developer_guide/alignment_rules.md                              374 lines     1084 words
slipbox/0_developer_guide/best_practices.md                               575 lines     2077 words
slipbox/0_developer_guide/common_pitfalls.md                              927 lines     2640 words
slipbox/0_developer_guide/component_guide.md                               62 lines      359 words
slipbox/0_developer_guide/config_field_manager_guide.md                   360 lines     1404 words
slipbox/0_developer_guide/creation_process.md                             570 lines     1913 words
slipbox/0_developer_guide/design_principles.md                            423 lines     2566 words
slipbox/0_developer_guide/developer_guide_review/developer_guide_review.md    100 lines      423 words
slipbox/0_developer_guide/example.md                                      807 lines     2228 words
slipbox/0_developer_guide/hyperparameter_class.md                         182 lines      616 words
slipbox/0_developer_guide/prerequisites.md                                 91 lines      533 words
slipbox/0_developer_guide/README.md                                       110 lines      646 words
slipbox/0_developer_guide/sagemaker_property_path_reference_database.md    638 lines     1427 words
slipbox/0_developer_guide/script_contract.md                              422 lines     1538 words
slipbox/0_developer_guide/script_testability_implementation.md            342 lines     1089 words
slipbox/0_developer_guide/standardization_rules.md                       1009 lines     3953 words
slipbox/0_developer_guide/step_builder_registry_guide.md                  385 lines     1532 words
slipbox/0_developer_guide/step_builder_registry_usage.md                  188 lines      748 words
slipbox/0_developer_guide/step_builder.md                                1046 lines     3780 words
slipbox/0_developer_guide/step_specification.md                           467 lines     1614 words
slipbox/0_developer_guide/three_tier_config_design.md                     417 lines     1611 words
slipbox/0_developer_guide/validation_checklist.md                         296 lines     1725 words
slipbox/1_design/adaptive_configuration_management_system_revised.md      280 lines     1100 words
slipbox/1_design/adaptive_fluent_proxy_integration.md                     942 lines     2966 words
slipbox/1_design/adaptive_specification_integration.md                    879 lines     3112 words
slipbox/1_design/agentic_workflow_design.md                               715 lines     3222 words
slipbox/1_design/alignment_validation_data_structures.md                 1001 lines     3638 words
slipbox/1_design/alignment_validation_visualization_integration_design.md    662 lines     2661 words
slipbox/1_design/circular_reference_tracker.md                            346 lines     1463 words
slipbox/1_design/config_driven_design.md                                  642 lines     2564 words
slipbox/1_design/config_field_categorization_consolidated.md              620 lines     2399 words
slipbox/1_design/config_field_manager_refactoring.md                       74 lines      418 words
slipbox/1_design/config_manager_three_tier_implementation.md              790 lines     3152 words
slipbox/1_design/config_merger.md                                         462 lines     1526 words
slipbox/1_design/config_registry.md                                       353 lines      999 words
slipbox/1_design/config_resolution_enhancements.md                        557 lines     2466 words
slipbox/1_design/config_tiered_design.md                                  795 lines     3154 words
slipbox/1_design/config_types_format.md                                   278 lines     1058 words
slipbox/1_design/config.md                                                508 lines     1676 words
slipbox/1_design/cradle_data_load_config_helper_design.md                 308 lines     1127 words
slipbox/1_design/createmodel_step_alignment_validation_patterns.md        680 lines     2057 words
slipbox/1_design/createmodel_step_builder_patterns.md                     674 lines     2465 words
slipbox/1_design/dag_to_template.md                                       446 lines     1652 words
slipbox/1_design/default_values_provider_revised.md                       403 lines     1616 words
slipbox/1_design/dependency_resolution_system.md                          401 lines     1445 words
slipbox/1_design/dependency_resolver.md                                   720 lines     2581 words
slipbox/1_design/design_evolution.md                                      460 lines     2091 words
slipbox/1_design/design_principles.md                                     561 lines     1859 words
slipbox/1_design/documentation_yaml_frontmatter_standard.md               323 lines     1042 words
slipbox/1_design/dynamic_template_system.md                               510 lines     1736 words
slipbox/1_design/enhanced_dependency_validation_design.md                 549 lines     2284 words
slipbox/1_design/enhanced_property_reference.md                           221 lines      898 words
slipbox/1_design/enhanced_universal_step_builder_tester_design.md        1059 lines     3720 words
slipbox/1_design/environment_variable_contract_enforcement.md             805 lines     2721 words
slipbox/1_design/essential_inputs_notebook_design_revised.md              292 lines     1264 words
slipbox/1_design/expanded_pipeline_catalog_mods_integration.md            620 lines     2135 words
slipbox/1_design/feature_group_registry_revised.md                        254 lines     1213 words
slipbox/1_design/field_derivation_engine_revised.md                       374 lines     1595 words
slipbox/1_design/flexible_file_resolver_design.md                         566 lines     2457 words
slipbox/1_design/fluent_api.md                                            633 lines     1893 words
slipbox/1_design/global_vs_local_objects.md                               223 lines      956 words
slipbox/1_design/hybrid_design.md                                         428 lines     1777 words
slipbox/1_design/job_type_variant_handling.md                             557 lines     1991 words
slipbox/1_design/level1_script_contract_alignment_design.md               794 lines     3031 words
slipbox/1_design/level2_contract_specification_alignment_design.md        749 lines     2641 words
slipbox/1_design/level2_property_path_validation_implementation.md        422 lines     1875 words
slipbox/1_design/level3_specification_dependency_alignment_design.md      712 lines     2550 words
slipbox/1_design/level4_builder_configuration_alignment_design.md         788 lines     2399 words
slipbox/1_design/mcp_agentic_workflow_agent_integration.md                863 lines     2376 words
slipbox/1_design/mcp_agentic_workflow_implementation.md                   796 lines     2118 words
slipbox/1_design/mcp_agentic_workflow_master_design.md                    288 lines     1319 words
slipbox/1_design/mcp_agentic_workflow_performance.md                      891 lines     2417 words
slipbox/1_design/mcp_agentic_workflow_security_operations.md              419 lines     1077 words
slipbox/1_design/mcp_agentic_workflow_server_architecture.md              888 lines     2075 words
slipbox/1_design/mcp_agentic_workflow_validation_framework.md             474 lines     1198 words
slipbox/1_design/mcp_knowledge_transfer_design.md                        1043 lines     3066 words
slipbox/1_design/mims_registration_integration.md                         137 lines      701 words
slipbox/1_design/model_evaluation_path_handling.md                        111 lines      511 words
slipbox/1_design/mods_dag_compiler_design.md                              709 lines     2520 words
slipbox/1_design/multi_developer_workspace_management_system.md           590 lines     2334 words
slipbox/1_design/packaging_step_improvements.md                           160 lines      617 words
slipbox/1_design/pipeline_assembler.md                                    419 lines     1576 words
slipbox/1_design/pipeline_catalog_design.md                               238 lines     1214 words
slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md             792 lines     2893 words
slipbox/1_design/pipeline_compiler.md                                     443 lines     1643 words
slipbox/1_design/pipeline_dag.md                                          501 lines     1655 words
slipbox/1_design/pipeline_registry.md                                     356 lines     1142 words
slipbox/1_design/pipeline_script_functionality_core_engine_design.md      334 lines     1319 words
slipbox/1_design/pipeline_script_functionality_data_management_design.md    495 lines     1710 words
slipbox/1_design/pipeline_script_functionality_jupyter_integration_design.md    904 lines     2634 words
slipbox/1_design/pipeline_script_functionality_reporting_design.md        482 lines     1236 words
slipbox/1_design/pipeline_script_functionality_system_integration_design.md    934 lines     2526 words
slipbox/1_design/pipeline_script_functionality_testing_master_design.md    233 lines     1289 words
slipbox/1_design/pipeline_script_functionality_testing_modes_design.md    719 lines     2496 words
slipbox/1_design/pipeline_script_functionality_testing_system_design.md   1156 lines     5045 words
slipbox/1_design/pipeline_script_functionality_usage_examples_design.md   1015 lines     2465 words
slipbox/1_design/pipeline_template_base.md                                368 lines     1293 words
slipbox/1_design/pipeline_template_builder_v1.md                          575 lines     2173 words
slipbox/1_design/pipeline_template_builder_v2.md                          598 lines     1718 words
slipbox/1_design/processing_step_alignment_validation_patterns.md         535 lines     1575 words
slipbox/1_design/processing_step_builder_patterns.md                      598 lines     2086 words
slipbox/1_design/README.md                                                868 lines     5233 words
slipbox/1_design/registermodel_step_alignment_validation_patterns.md      566 lines     1628 words
slipbox/1_design/registry_based_step_name_generation.md                   531 lines     1936 words
slipbox/1_design/registry_manager.md                                      587 lines     1721 words
slipbox/1_design/registry_single_source_of_truth.md                       446 lines     2194 words
slipbox/1_design/sagemaker_step_type_aware_unified_alignment_tester_design.md    762 lines     3229 words
slipbox/1_design/sagemaker_step_type_classification_design.md             488 lines     1759 words
slipbox/1_design/sagemaker_step_type_universal_builder_tester_design.md    540 lines     3087 words
slipbox/1_design/script_contract.md                                       551 lines     2040 words
slipbox/1_design/script_integration_testing_system_design.md              758 lines     3153 words
slipbox/1_design/script_testability_refactoring.md                        658 lines     2391 words
slipbox/1_design/simplified_config_field_categorization.md                301 lines      992 words
slipbox/1_design/smart_proxy.md                                           761 lines     2544 words
slipbox/1_design/specification_driven_design.md                           435 lines     1472 words
slipbox/1_design/specification_registry.md                                587 lines     1662 words
slipbox/1_design/standardization_rules.md                                1914 lines     6307 words
slipbox/1_design/step_builder_patterns_summary.md                         478 lines     1683 words
slipbox/1_design/step_builder_registry_design.md                          155 lines      763 words
slipbox/1_design/step_builder.md                                          871 lines     3314 words
slipbox/1_design/step_config_resolver.md                                  402 lines     1486 words
slipbox/1_design/step_contract.md                                         459 lines     1220 words
slipbox/1_design/step_specification.md                                    254 lines      797 words
slipbox/1_design/step_type_enhancement_system_design.md                   412 lines     1385 words
slipbox/1_design/training_step_alignment_validation_patterns.md           695 lines     2066 words
slipbox/1_design/training_step_builder_patterns.md                        455 lines     1449 words
slipbox/1_design/training_step_improvements.md                            320 lines     1444 words
slipbox/1_design/transform_step_alignment_validation_patterns.md          621 lines     1840 words
slipbox/1_design/transform_step_builder_patterns.md                       450 lines     1691 words
slipbox/1_design/two_level_alignment_validation_system_design.md          777 lines     2780 words
slipbox/1_design/two_level_standardization_validation_system_design.md    854 lines     2809 words
slipbox/1_design/type_aware_serializer.md                                 346 lines     1310 words
slipbox/1_design/unified_alignment_tester_architecture.md                 690 lines     3033 words
slipbox/1_design/unified_alignment_tester_design.md                       116 lines      665 words
slipbox/1_design/unified_alignment_tester_master_design.md                361 lines     2101 words
slipbox/1_design/universal_step_builder_test_scoring.md                   738 lines     3247 words
slipbox/1_design/universal_step_builder_test.md                           632 lines     3149 words
slipbox/1_design/utility_step_alignment_validation_patterns.md            703 lines     2137 words
slipbox/1_design/validation_engine.md                                     889 lines     2748 words
slipbox/1_design/zettelkasten_dag_metadata_integration.md                1375 lines     3999 words
slipbox/1_design/zettelkasten_knowledge_management_principles.md          415 lines     2361 words
slipbox/1_design/zettelkasten_pipeline_catalog_utilities.md               677 lines     2240 words
slipbox/2_project_planning/2025-07-04_contract_alignment_implementation_summary.md    390 lines     1658 words
slipbox/2_project_planning/2025-07-04_job_type_variant_solution.md        309 lines     1120 words
slipbox/2_project_planning/2025-07-04_phase1_solution_summary.md          266 lines      994 words
slipbox/2_project_planning/2025-07-04_phase1_step_specification_solution.md    240 lines      793 words
slipbox/2_project_planning/2025-07-04_script_specification_alignment_plan.md    433 lines     1847 words
slipbox/2_project_planning/2025-07-04_script_specification_alignment_prevention_plan.md    569 lines     2350 words
slipbox/2_project_planning/2025-07-04_specification_driven_xgboost_pipeline_plan.md    685 lines     2985 words
slipbox/2_project_planning/2025-07-05_alignment_validation_implementation_plan.md    856 lines     3158 words
slipbox/2_project_planning/2025-07-05_corrected_alignment_architecture_plan.md    315 lines     1432 words
slipbox/2_project_planning/2025-07-05_corrected_alignment_understanding_summary.md    316 lines     1169 words
slipbox/2_project_planning/2025-07-05_phase2_contract_key_alignment_summary.md    210 lines      919 words
slipbox/2_project_planning/2025-07-05_property_path_alignment_fixes_summary.md    136 lines      573 words
slipbox/2_project_planning/2025-07-06_pytorch_training_alignment_implementation_summary.md    235 lines     1044 words
slipbox/2_project_planning/2025-07-06_training_alignment_project_status.md    244 lines     1123 words
slipbox/2_project_planning/2025-07-07_dependency_resolver_benefits.md     382 lines     1457 words
slipbox/2_project_planning/2025-07-07_phase5_training_step_modernization_summary.md    285 lines     1524 words
slipbox/2_project_planning/2025-07-07_phase6_2_registration_step_implementation_summary.md    215 lines     1022 words
slipbox/2_project_planning/2025-07-07_phase6_model_steps_implementation_summary.md    244 lines      881 words
slipbox/2_project_planning/2025-07-07_project_status_update.md            193 lines     1309 words
slipbox/2_project_planning/2025-07-07_specification_driven_architecture_analysis.md    372 lines     1778 words
slipbox/2_project_planning/2025-07-07_specification_driven_step_builder_plan.md    604 lines     3754 words
slipbox/2_project_planning/2025-07-07_step_name_consistency_implementation_plan.md    390 lines     1470 words
slipbox/2_project_planning/2025-07-07_step_name_consistency_implementation_status.md    228 lines     1064 words
slipbox/2_project_planning/2025-07-08_comprehensive_dependency_matching_analysis.md    253 lines     1774 words
slipbox/2_project_planning/2025-07-08_dependency_resolution_alias_support_plan.md    268 lines     1248 words
slipbox/2_project_planning/2025-07-08_phase1_dependency_resolver_implementation.md    263 lines      892 words
slipbox/2_project_planning/2025-07-08_phase1_registry_manager_implementation.md    337 lines      999 words
slipbox/2_project_planning/2025-07-08_phase1_semantic_matcher_implementation.md    194 lines      680 words
slipbox/2_project_planning/2025-07-08_remove_global_singletons.md         755 lines     3300 words
slipbox/2_project_planning/2025-07-08_script_specification_alignment_prevention_plan.md    360 lines     1351 words
slipbox/2_project_planning/2025-07-09_abstract_pipeline_template_design.md    704 lines     2381 words
slipbox/2_project_planning/2025-07-09_pipeline_template_base_design.md    493 lines     1857 words
slipbox/2_project_planning/2025-07-09_pipeline_template_modernization_plan.md    721 lines     2991 words
slipbox/2_project_planning/2025-07-09_simplify_pipeline_assembler.md      354 lines     1968 words
slipbox/2_project_planning/2025-07-09_simplify_pipeline_builder_template.md    354 lines     1966 words
slipbox/2_project_planning/2025-07-12_mims_payload_path_handling_fix.md    121 lines      627 words
slipbox/2_project_planning/2025-07-16_dummy_training_job_arguments_implementation.md    196 lines      784 words
slipbox/2_project_planning/2025-07-16_script_contract_job_arguments_enhancement_plan.md    672 lines     2523 words
slipbox/2_project_planning/2025-07-16_smart_proxy_implementation_plan.md    228 lines      952 words
slipbox/2_project_planning/2025-07-17_config_field_categorization_refactoring_plan.md    763 lines     4412 words
slipbox/2_project_planning/2025-07-18_fix_config_types_format.md          301 lines     1105 words
slipbox/2_project_planning/2025-07-18_simplified_xgboost_config_implementation_plan.md    308 lines     1225 words
slipbox/2_project_planning/2025-07-23_essential_inputs_implementation_plan.md    479 lines     2274 words
slipbox/2_project_planning/2025-07-24_phase1_implementation_summary.md    142 lines      760 words
slipbox/2_project_planning/2025-07-24_phase2_implementation_summary.md    187 lines      827 words
slipbox/2_project_planning/2025-07-25_cradle_data_load_config_tiered_implementation_plan.md    544 lines     2307 words
slipbox/2_project_planning/2025-07-25_simplified_xgboost_config_implementation_plan.md    444 lines     2149 words
slipbox/2_project_planning/2025-08-07_universal_step_builder_test_enhancement_plan.md    592 lines     2623 words
slipbox/2_project_planning/2025-08-07_validation_tools_implementation_plan.md    430 lines     2028 words
slipbox/2_project_planning/2025-08-09_mcp_knowledge_transfer_implementation_plan.md    370 lines     1654 words
slipbox/2_project_planning/2025-08-09_two_level_alignment_validation_implementation_plan.md    416 lines     1886 words
slipbox/2_project_planning/2025-08-10_alignment_validation_refactoring_plan.md   1508 lines     7077 words
slipbox/2_project_planning/2025-08-11_code_alignment_standardization_plan.md   1395 lines     8264 words
slipbox/2_project_planning/2025-08-12_fluent_api_implementation_plan.md    664 lines     2929 words
slipbox/2_project_planning/2025-08-12_property_path_validation_level2_implementation_plan.md    786 lines     2608 words
slipbox/2_project_planning/2025-08-13_phase3_completion_summary.md        262 lines     1221 words
slipbox/2_project_planning/2025-08-13_sagemaker_step_type_aware_unified_alignment_tester_implementation_plan.md    985 lines     3960 words
slipbox/2_project_planning/2025-08-13_script_integration_testing_implementation_plan.md   1203 lines     4861 words
slipbox/2_project_planning/2025-08-14_simplified_universal_step_builder_test_plan.md    752 lines     3150 words
slipbox/2_project_planning/2025-08-15_alignment_validation_visualization_integration_plan.md    635 lines     3478 words
slipbox/2_project_planning/2025-08-15_sagemaker_step_type_variants_4level_validation_implementation.md    522 lines     2776 words
slipbox/2_project_planning/2025-08-15_universal_step_builder_test_overhaul_implementation_plan.md    601 lines     3436 words
slipbox/2_project_planning/2025-08-17_multi_developer_workspace_management_implementation_plan.md   1367 lines     4537 words
slipbox/2_project_planning/2025-08-19_mods_pipeline_dag_compiler_implementation_plan.md    280 lines     1306 words
slipbox/2_project_planning/2025-08-19_pipeline_catalog_implementation_plan.md    230 lines     1296 words
slipbox/2_project_planning/2025-08-20_mods_pipeline_catalog_integration_implementation_plan.md    441 lines     2087 words
slipbox/2_project_planning/2025-08-20_pipeline_catalog_zettelkasten_refactoring_plan.md    623 lines     3226 words
slipbox/2_project_planning/2025-08-21_enhanced_dag_metadata_adoption_plan.md    495 lines     2474 words
slipbox/2_project_planning/2025-08-21_model_calibration_job_type_variant_expansion_plan.md    753 lines     3458 words
slipbox/2_project_planning/2025-08-21_pipeline_script_functionality_data_flow_phase_plan.md    625 lines     1873 words
slipbox/2_project_planning/2025-08-21_pipeline_script_functionality_foundation_phase_plan.md    906 lines     2645 words
slipbox/2_project_planning/2025-08-21_pipeline_script_functionality_jupyter_integration_phase_plan.md   1725 lines     4733 words
slipbox/2_project_planning/2025-08-21_pipeline_script_functionality_s3_integration_phase_plan.md   1050 lines     3039 words
slipbox/2_project_planning/2025-08-21_pipeline_script_functionality_testing_implementation_plan.md   1407 lines     4845 words
slipbox/2_project_planning/2025-08-21_pipeline_script_functionality_testing_master_implementation_plan.md    223 lines     1305 words
slipbox/2_project_planning/cursus_packaging_plan.md                       394 lines     1380 words
slipbox/2_project_planning/phase1_solution_summary.md                     266 lines      994 words
slipbox/2_project_planning/phase1_step_specification_solution.md          240 lines      793 words
slipbox/2_project_planning/script_specification_alignment_prevention_plan.md    572 lines     2385 words
slipbox/2_project_planning/specification_driven_xgboost_pipeline_plan.md    738 lines     3225 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_alignment_check.md     93 lines      532 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_implementation_plan_v1.md    493 lines     1547 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_implementation_plan_v2.md    732 lines     2371 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_implementation_plan_v3.md    762 lines     2622 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_implementation_plan_v4.md    782 lines     2744 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_implementation_plan_v5.md    184 lines      778 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_implementation_verification.md    204 lines      621 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validation_code_example_v2.md    275 lines     1103 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validation_code_example.md    364 lines     1526 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validation_example_v1.md    230 lines      907 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validation_example_v2.md    153 lines      712 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validation_example_v3.md    161 lines      844 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validation_example.md    230 lines      904 words
slipbox/3_llm_developer/developer_demo/dummy_training/dummy_training_validator_assessment.md    221 lines     1176 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_implementation_plan_v2.md    735 lines     2749 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_implementation_plan_v3_continuation.md    637 lines     2076 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_implementation_plan_v3.md    699 lines     2849 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_implementation_plan.md    729 lines     2533 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_validation_report_v2.md    201 lines     1101 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_validation_report_v3.md    220 lines     1199 words
slipbox/3_llm_developer/developer_demo/model_calibration/model_calibration_validation_report.md    257 lines     1210 words
slipbox/3_llm_developer/developer_demo/model_calibration/pipeline_step_planner_prompt.md    606 lines     2418 words
slipbox/3_llm_developer/developer_demo/model_calibration/pipeline_step_programmer_prompt.md    757 lines     2830 words
slipbox/3_llm_developer/developer_demo/model_calibration/plan_validator_prompt.md    533 lines     3057 words
slipbox/3_llm_developer/developer_demo/pipeline_step_planner_prompt.md    120 lines      704 words
slipbox/3_llm_developer/developer_demo/README.md                          106 lines      809 words
slipbox/3_llm_developer/developer_demo/risk_table_mapping/risk_table_mapping_alignment_check.md    118 lines      569 words
slipbox/3_llm_developer/developer_demo/risk_table_mapping/risk_table_mapping_validation_report.md    208 lines     1108 words
slipbox/3_llm_developer/developer_prompt_templates/planner_prompt_template.md    135 lines      652 words
slipbox/3_llm_developer/developer_prompt_templates/README.md              221 lines     1261 words
slipbox/3_llm_developer/developer_prompt_templates/step1_initial_planner_prompt_template.md   1515 lines     5226 words
slipbox/3_llm_developer/developer_prompt_templates/step2_plan_validator_prompt_template.md    629 lines     3272 words
slipbox/3_llm_developer/developer_prompt_templates/step3_revision_planner_prompt_template.md    263 lines     1155 words
slipbox/3_llm_developer/developer_prompt_templates/step4_programmer_prompt_template.md   1506 lines     5070 words
slipbox/3_llm_developer/developer_prompt_templates/step5a_two_level_validation_agent_prompt_template.md    603 lines     3103 words
slipbox/3_llm_developer/developer_prompt_templates/step5b_two_level_standardization_validation_agent_prompt_template.md    651 lines     2704 words
slipbox/3_llm_developer/developer_prompt_templates/step6_code_refinement_programmer_prompt_template.md    654 lines     2987 words
slipbox/3_llm_developer/developer_prompt_templates/two_level_validation_report_format.md    525 lines     2488 words
slipbox/3_llm_developer/developer_prompt_templates/validator_prompt_template.md    382 lines     1910 words
slipbox/3_llm_developer/notebook_digest_prompts/claude_notebook_analyzer_template.md     94 lines      532 words
slipbox/3_llm_developer/notebook_digest_prompts/example_digest_output.md    160 lines     1070 words
slipbox/3_llm_developer/notebook_digest_prompts/jupyter_to_sagemaker_pipeline_analyzer.md     57 lines      348 words
slipbox/3_llm_developer/notebook_digest_prompts/README.md                  62 lines      328 words
slipbox/3_llm_developer/notebook_digest_prompts/usage_guide.md             56 lines      299 words
slipbox/3_llm_developer/notebook_digests/model_training_pda_digest.md     224 lines     1476 words
slipbox/4_analysis/alignment_tester_robustness_analysis.md                581 lines     2433 words
slipbox/4_analysis/circular_reference_necessity_analysis.md               332 lines     1496 words
slipbox/4_analysis/config_field_categorization_comparison.md              218 lines     1013 words
slipbox/4_analysis/cursus_line_count_analysis_2025-08-22.md              1222 lines     4830 words
slipbox/4_analysis/dynamic_pipeline_template_design_principles_compliance_analysis.md    188 lines     1143 words
slipbox/4_analysis/fluent_api_dag_compiler_integration_analysis.md        513 lines     2046 words
slipbox/4_analysis/fluent_api_data_structure_reuse_analysis.md            530 lines     1807 words
slipbox/4_analysis/fluent_api_user_input_collection_analysis.md          1009 lines     3300 words
slipbox/4_analysis/level3_path_mapping_test_responsibility_analysis.md    296 lines     1886 words
slipbox/4_analysis/pipeline_design_philosophy_developer_perspective_comparison.md    740 lines     2813 words
slipbox/4_analysis/pipeline_design_philosophy_lean_product_analysis.md    428 lines     2575 words
slipbox/4_analysis/pipeline_design_philosophy_user_perspective_comparison.md    753 lines     3115 words
slipbox/4_analysis/pipeline_migration_analysis.md                         360 lines     1456 words
slipbox/4_analysis/pytorch_training_step_builder_failure_analysis.md      433 lines     1886 words
slipbox/4_analysis/sagemaker_pipeline_pain_point_analysis.md              297 lines     2040 words
slipbox/4_analysis/step_builder_local_override_patterns_analysis.md       440 lines     2213 words
slipbox/4_analysis/step_builder_methods_comprehensive_analysis.md         317 lines     1769 words
slipbox/4_analysis/step_builder_methods_top_pain_points_analysis.md       424 lines     2433 words
slipbox/4_analysis/two_level_validation_pain_point_solution_analysis.md    541 lines     2794 words
slipbox/4_analysis/unified_alignment_tester_coverage_analysis.md          361 lines     2478 words
slipbox/4_analysis/unified_alignment_tester_pain_points_analysis.md       930 lines     6383 words
slipbox/4_analysis/unified_step_builder_testers_implementation_analysis.md    202 lines     1032 words
slipbox/4_analysis/unified_testers_comparative_analysis.md                721 lines     3658 words
slipbox/4_analysis/validation_system_complexity_analysis.md               476 lines     2397 words
slipbox/4_analysis/xgboost_evaluation_notebook_field_dependency_analysis.md    382 lines     2041 words
slipbox/4_analysis/xgboost_pipeline_field_dependency_table.md             280 lines     2983 words
slipbox/api/dag/base_dag.md                                               269 lines     1186 words
slipbox/api/dag/edge_types.md                                             281 lines     1043 words
slipbox/api/dag/enhanced_dag.md                                           370 lines     1225 words
slipbox/api/dag/README.md                                                 133 lines      595 words
slipbox/cli/builder_test_cli.md                                           317 lines     1324 words
slipbox/cli/catalog_cli.md                                                491 lines     1551 words
slipbox/cli/cli_entry_point.md                                            278 lines     1096 words
slipbox/cli/cli_main_interface.md                                         224 lines     1017 words
slipbox/cli/README.md                                                     396 lines     1431 words
slipbox/cli/user_guide.md                                                 713 lines     2238 words
slipbox/cli/validation_cli.md                                             375 lines     1586 words
slipbox/core/assembler/pipeline_assembler.md                              374 lines     1390 words
slipbox/core/assembler/pipeline_builder_template.md                       396 lines     1303 words
slipbox/core/assembler/pipeline_examples.md                               316 lines     1044 words
slipbox/core/assembler/pipeline_template_base.md                          343 lines     1051 words
slipbox/core/assembler/README.md                                          249 lines      956 words
slipbox/core/assembler/template_implementation.md                         318 lines     1075 words
slipbox/core/base/builder_base.md                                         637 lines     2132 words
slipbox/core/base/config_base.md                                          593 lines     2331 words
slipbox/core/base/contract_base.md                                        308 lines     1048 words
slipbox/core/base/contract_validator.md                                   546 lines     1800 words
slipbox/core/base/enums.md                                                240 lines      829 words
slipbox/core/base/hyperparameters_base.md                                 445 lines     1535 words
slipbox/core/base/README.md                                               168 lines      667 words
slipbox/core/base/specification_base.md                                   460 lines     1408 words
slipbox/core/compiler/config_resolver.md                                  124 lines      441 words
slipbox/core/compiler/dag_compiler.md                                     139 lines      418 words
slipbox/core/compiler/dynamic_template.md                                 104 lines      453 words
slipbox/core/compiler/exceptions.md                                       151 lines      485 words
slipbox/core/compiler/mods_dag_compiler.md                                177 lines      613 words
slipbox/core/compiler/name_generator.md                                   119 lines      431 words
slipbox/core/compiler/README.md                                           354 lines     1045 words
slipbox/core/compiler/validation.md                                       149 lines      470 words
slipbox/core/config_field/circular_reference_tracker.md                   544 lines     1548 words
slipbox/core/config_field/config_constants.md                             474 lines     1457 words
slipbox/core/config_field/config_field_categorizer.md                     470 lines     1526 words
slipbox/core/config_field/config_merger.md                                650 lines     2054 words
slipbox/core/config_field/default_values_provider.md                      213 lines      803 words
slipbox/core/config_field/essential_input_models.md                       229 lines     1065 words
slipbox/core/config_field/field_derivation_engine.md                      273 lines     1086 words
slipbox/core/config_field/README.md                                       295 lines     1077 words
slipbox/core/config_field/tier_registry.md                                108 lines      592 words
slipbox/core/config_field/type_aware_config_serializer.md                 762 lines     2489 words
slipbox/core/deps/dependency_resolver.md                                  431 lines     1663 words
slipbox/core/deps/factory.md                                              270 lines      813 words
slipbox/core/deps/property_reference.md                                   595 lines     2374 words
slipbox/core/deps/README.md                                               198 lines      827 words
slipbox/core/deps/registry_manager.md                                     378 lines     1166 words
slipbox/core/deps/semantic_matcher.md                                     408 lines     1522 words
slipbox/core/deps/specification_registry.md                               299 lines      951 words
slipbox/examples/mods_pipeline_bsm_pytorch.md                              58 lines      380 words
slipbox/examples/mods_pipeline_xgboost_end_to_end_simple.md                82 lines      500 words
slipbox/examples/mods_pipeline_xgboost_end_to_end.md                       89 lines      513 words
slipbox/examples/README.md                                                134 lines      610 words
slipbox/ml/lightning_models/dist_utils.md                                  81 lines      396 words
slipbox/ml/lightning_models/pl_bert_classification.md                      83 lines      505 words
slipbox/ml/lightning_models/pl_bert.md                                     65 lines      414 words
slipbox/ml/lightning_models/pl_lstm.md                                     96 lines      532 words
slipbox/ml/lightning_models/pl_model_plots.md                              92 lines      413 words
slipbox/ml/lightning_models/pl_multimodal_bert.md                          88 lines      586 words
slipbox/ml/lightning_models/pl_multimodal_cnn.md                          107 lines      659 words
slipbox/ml/lightning_models/pl_multimodal_cross_attn.md                   101 lines      704 words
slipbox/ml/lightning_models/pl_multimodal_gate_fusion.md                   97 lines      707 words
slipbox/ml/lightning_models/pl_multimodal_moe.md                           97 lines      705 words
slipbox/ml/lightning_models/pl_tab_ae.md                                   60 lines      364 words
slipbox/ml/lightning_models/pl_text_cnn.md                                108 lines      632 words
slipbox/ml/lightning_models/pl_train.md                                   141 lines      511 words
slipbox/ml/lightning_models/README.md                                      88 lines      467 words
slipbox/ml/notebooks/create_config_xgb_w_eval.md                          194 lines      841 words
slipbox/ml/processing/bert_tokenize_processor.md                           73 lines      341 words
slipbox/ml/processing/bsm_processor.md                                     74 lines      361 words
slipbox/ml/processing/categorical_label_processor.md                       53 lines      264 words
slipbox/ml/processing/cs_processor.md                                      86 lines      452 words
slipbox/ml/processing/df_category_risk.md                                  75 lines      441 words
slipbox/ml/processing/gensim_tokenize_processor.md                         69 lines      340 words
slipbox/ml/processing/multiclass_label_processor.md                        56 lines      287 words
slipbox/ml/processing/numerical_binning_processor.md                       68 lines      357 words
slipbox/ml/processing/numerical_imputation_processor.md                    55 lines      307 words
slipbox/ml/processing/processors.md                                        79 lines      429 words
slipbox/ml/processing/README.md                                            64 lines      354 words
slipbox/ml/processing/risk_table_processor.md                              76 lines      422 words
slipbox/mods/compiler/mods_dag_compiler.md                                364 lines     1424 words
slipbox/mods/compiler/README.md                                           361 lines     1268 words
slipbox/mods/README.md                                                    181 lines      724 words
slipbox/pipeline_catalog/mods_pipelines/README.md                         502 lines     1880 words
slipbox/pipeline_catalog/pipelines/README.md                              365 lines     1429 words
slipbox/pipeline_catalog/README.md                                        235 lines      957 words
slipbox/pipeline_catalog/shared_dags/README.md                            512 lines     1756 words
slipbox/pipeline_catalog/utils.md                                         385 lines     1316 words
slipbox/pipeline_catalog/utils/README.md                                  593 lines     2272 words
slipbox/README.md                                                         231 lines     1191 words
slipbox/steps/builders/batch_transform_step.md                            154 lines      695 words
slipbox/steps/builders/currency_conversion_step.md                        190 lines      944 words
slipbox/steps/builders/data_load_step_cradle.md                           242 lines     1187 words
slipbox/steps/builders/hyperparameter_prep_step.md                        125 lines      476 words
slipbox/steps/builders/mims_packaging_step.md                             180 lines      909 words
slipbox/steps/builders/mims_payload_step.md                               230 lines     1119 words
slipbox/steps/builders/mims_registration_step.md                          203 lines      913 words
slipbox/steps/builders/model_eval_step_xgboost.md                         171 lines      794 words
slipbox/steps/builders/model_step_pytorch.md                              173 lines      707 words
slipbox/steps/builders/model_step_xgboost.md                              173 lines      710 words
slipbox/steps/builders/README.md                                          196 lines      876 words
slipbox/steps/builders/risk_table_map_step.md                             147 lines      665 words
slipbox/steps/builders/tabular_preprocessing_step.md                      177 lines      915 words
slipbox/steps/builders/training_step_pytorch.md                           181 lines      768 words
slipbox/steps/builders/training_step_xgboost.md                           173 lines      828 words
slipbox/steps/contracts/cradle_data_loading_contract.md                   207 lines      889 words
slipbox/steps/contracts/currency_conversion_contract.md                   264 lines     1094 words
slipbox/steps/contracts/dummy_training_contract.md                        239 lines      948 words
slipbox/steps/contracts/hyperparameter_prep_contract.md                   254 lines      945 words
slipbox/steps/contracts/mims_package_contract.md                          263 lines     1116 words
slipbox/steps/contracts/mims_payload_contract.md                          307 lines     1156 words
slipbox/steps/contracts/mims_registration_contract.md                     292 lines     1113 words
slipbox/steps/contracts/model_evaluation_contract.md                      274 lines     1153 words
slipbox/steps/contracts/pytorch_train_contract.md                         274 lines      898 words
slipbox/steps/contracts/README.md                                         200 lines      832 words
slipbox/steps/contracts/risk_table_mapping_contract.md                    266 lines     1142 words
slipbox/steps/contracts/tabular_preprocess_contract.md                    227 lines      979 words
slipbox/steps/contracts/xgboost_train_contract.md                         289 lines     1102 words
slipbox/steps/scripts/contract_utils_doc.md                                75 lines      364 words
slipbox/steps/scripts/currency_conversion_doc.md                          108 lines      526 words
slipbox/steps/scripts/dummy_training_doc.md                               144 lines      751 words
slipbox/steps/scripts/mims_package_doc.md                                 132 lines      681 words
slipbox/steps/scripts/mims_payload_doc.md                                 183 lines      836 words
slipbox/steps/scripts/model_calibration_doc.md                            379 lines     1564 words
slipbox/steps/scripts/model_evaluation_xgb_doc.md                         159 lines      772 words
slipbox/steps/scripts/MODS_MIMS_Model_Registration.md                     152 lines      658 words
slipbox/steps/scripts/risk_table_mapping_doc.md                           158 lines      794 words
slipbox/steps/scripts/tabular_preprocess_doc.md                           164 lines      760 words
slipbox/steps/specs/data_loading_training_spec.md                         167 lines      740 words
slipbox/steps/specs/README.md                                              95 lines      391 words
slipbox/test/2025-08-15_step_builder_test_execution_summary.md            312 lines     1365 words
slipbox/test/2025-08-15_universal_step_builder_comprehensive_test_report.md    172 lines      989 words
slipbox/test/base_classes_test_fixes_summary.md                            73 lines      371 words
slipbox/test/base_classes_test_report.md                                  251 lines     1316 words
slipbox/test/base_specifications_consolidation_complete_report.md         167 lines      628 words
slipbox/test/base_specifications_consolidation_report.md                  241 lines      893 words
slipbox/test/builders_test_report_2025_08_16.md                           328 lines     1824 words
slipbox/test/builders_test_score_summary_2025_08_16.md                    274 lines     1416 words
slipbox/test/circular_import_analysis_report.md                           513 lines     2467 words
slipbox/test/circular_import_fix_summary.md                               214 lines      923 words
slipbox/test/compiler_test_coverage_analysis_report.md                    318 lines     1502 words
slipbox/test/config_fields_test_coverage_report.md                        281 lines     1673 words
slipbox/test/config_fields_test_improvements_summary.md                   277 lines     1060 words
slipbox/test/core_package_comprehensive_test_analysis.md                  262 lines     1290 words
slipbox/test/core_package_test_coverage_redundancy_report.md              513 lines     2184 words
slipbox/test/createmodel_step_builder_failure_analysis.md                 263 lines     1260 words
slipbox/test/deps_test_coverage_analysis_report.md                        277 lines     1303 words
slipbox/test/level1_alignment_validation_consolidated_report_2025_08_11.md    435 lines     2057 words
slipbox/test/level2_alignment_validation_consolidated_report_2025_08_11.md    438 lines     2572 words
slipbox/test/level3_alignment_validation_consolidated_report_2025_08_11.md    648 lines     3783 words
slipbox/test/level3_false_positive_fix_analysis.md                        117 lines      657 words
slipbox/test/level4_alignment_validation_consolidated_report_2025_08_11.md    995 lines     5550 words
slipbox/test/pipeline_scripts_testability_report.md                       113 lines      619 words
slipbox/test/processing_builders_test_report.md                           161 lines      943 words
slipbox/test/redundant_test_removal_summary.md                             93 lines      451 words
slipbox/test/sagemaker_step_type_verification_report.md                   174 lines      913 words
slipbox/test/universal_builder_test_analysis_report.md                    316 lines     1517 words
slipbox/test/universal_builder_test_enhancement_report.md                 519 lines     1769 words
slipbox/test/universal_builder_test_sagemaker_enhancement_report.md       207 lines     1038 words
slipbox/test/universal_processing_builder_test_analysis_report.md         459 lines     1799 words
slipbox/test/xgboost_model_eval_false_positives_fix_report.md             177 lines      769 words
slipbox/validation/alignment/alignment_reporter.md                        336 lines     1207 words
slipbox/validation/alignment/alignment_scorer.md                          446 lines     1417 words
slipbox/validation/alignment/alignment_utils.md                           757 lines     2026 words
slipbox/validation/alignment/analyzers/builder_analyzer.md                531 lines     1502 words
slipbox/validation/alignment/analyzers/config_analyzer.md                 484 lines     1581 words
slipbox/validation/alignment/builder_config_alignment.md                  488 lines     1554 words
slipbox/validation/alignment/contract_spec_alignment.md                   469 lines     1517 words
slipbox/validation/alignment/core_models.md                               452 lines     1350 words
slipbox/validation/alignment/dependency_classifier.md                     455 lines     1281 words
slipbox/validation/alignment/discovery/contract_discovery.md              349 lines     1196 words
slipbox/validation/alignment/enhanced_reporter.md                         357 lines     1351 words
slipbox/validation/alignment/file_resolver.md                             277 lines      878 words
slipbox/validation/alignment/framework_patterns.md                        345 lines     1412 words
slipbox/validation/alignment/level3_validation_config.md                  304 lines     1137 words
slipbox/validation/alignment/loaders/contract_loader.md                   383 lines     1278 words
slipbox/validation/alignment/loaders/specification_loader.md              369 lines     1463 words
slipbox/validation/alignment/orchestration/validation_orchestrator.md     428 lines     1430 words
slipbox/validation/alignment/patterns/file_resolver.md                    257 lines      988 words
slipbox/validation/alignment/patterns/pattern_recognizer.md               407 lines     1378 words
slipbox/validation/alignment/processors/spec_file_processor.md            409 lines     1392 words
slipbox/validation/alignment/property_path_validator.md                   546 lines     1565 words
slipbox/validation/alignment/README.md                                    255 lines     1012 words
slipbox/validation/alignment/script_analysis_models.md                    362 lines     1354 words
slipbox/validation/alignment/script_contract_alignment.md                 689 lines     2418 words
slipbox/validation/alignment/smart_spec_selector.md                       243 lines     1062 words
slipbox/validation/alignment/spec_dependency_alignment.md                 481 lines     1621 words
slipbox/validation/alignment/static_analysis/builder_analyzer.md          329 lines     1244 words
slipbox/validation/alignment/static_analysis/import_analyzer.md           375 lines     1255 words
slipbox/validation/alignment/static_analysis/path_extractor.md            414 lines     1336 words
slipbox/validation/alignment/static_analysis/script_analyzer.md           429 lines     1403 words
slipbox/validation/alignment/step_type_detection.md                       269 lines     1189 words
slipbox/validation/alignment/step_type_enhancement_router.md              304 lines     1156 words
slipbox/validation/alignment/step_type_enhancers/base_enhancer.md         436 lines     1494 words
slipbox/validation/alignment/step_type_enhancers/createmodel_enhancer.md    388 lines     1350 words
slipbox/validation/alignment/step_type_enhancers/processing_enhancer.md    496 lines     1556 words
slipbox/validation/alignment/step_type_enhancers/registermodel_enhancer.md    459 lines     1471 words
slipbox/validation/alignment/step_type_enhancers/training_enhancer.md     582 lines     1845 words
slipbox/validation/alignment/step_type_enhancers/transform_enhancer.md    510 lines     1625 words
slipbox/validation/alignment/step_type_enhancers/utility_enhancer.md      544 lines     1711 words
slipbox/validation/alignment/testability_validator.md                     377 lines     1465 words
slipbox/validation/alignment/unified_alignment_tester.md                  487 lines     1496 words
slipbox/validation/alignment/utils.md                                     413 lines     1427 words
slipbox/validation/alignment/validators/contract_spec_validator.md        430 lines     1456 words
slipbox/validation/alignment/validators/dependency_validator.md           313 lines     1286 words
slipbox/validation/alignment/validators/legacy_validators.md              394 lines     1435 words
slipbox/validation/alignment/validators/script_contract_validator.md      358 lines     1508 words
slipbox/validation/alignment/workflow_integration.md                      404 lines     1222 words
slipbox/validation/builders/base_test.md                                  519 lines     1708 words
slipbox/validation/builders/builder_reporter.md                           531 lines     1626 words
slipbox/validation/builders/example_enhanced_usage.md                     273 lines      999 words
slipbox/validation/builders/example_usage.md                              321 lines     1170 words
slipbox/validation/builders/generic_test.md                               403 lines     1377 words
slipbox/validation/builders/integration_tests.md                          302 lines     1184 words
slipbox/validation/builders/interface_tests.md                            476 lines     1622 words
slipbox/validation/builders/mock_factory.md                               604 lines     1850 words
slipbox/validation/builders/README.md                                     394 lines     1319 words
slipbox/validation/builders/registry_discovery.md                         497 lines     1620 words
slipbox/validation/builders/sagemaker_step_type_validator.md              443 lines     1534 words
slipbox/validation/builders/scoring.md                                    449 lines     1372 words
slipbox/validation/builders/specification_tests.md                        421 lines     1551 words
slipbox/validation/builders/step_creation_tests.md                        576 lines     1918 words
slipbox/validation/builders/step_info_detector.md                         532 lines     1657 words
slipbox/validation/builders/test_factory.md                               473 lines     1516 words
slipbox/validation/builders/universal_test.md                             292 lines     1050 words
slipbox/validation/builders/variants/createmodel_integration_tests.md     463 lines     1311 words
slipbox/validation/builders/variants/createmodel_interface_tests.md       437 lines     1361 words
slipbox/validation/builders/variants/createmodel_specification_tests.md    447 lines     1345 words
slipbox/validation/builders/variants/createmodel_test.md                  421 lines     1134 words
slipbox/validation/builders/variants/processing_integration_tests.md      519 lines     1509 words
slipbox/validation/builders/variants/processing_interface_tests.md        470 lines     1450 words
slipbox/validation/builders/variants/processing_pattern_b_test_runner.md    452 lines     1358 words
slipbox/validation/builders/variants/processing_specification_tests.md    637 lines     1758 words
slipbox/validation/builders/variants/processing_step_creation_tests.md    466 lines     1486 words
slipbox/validation/builders/variants/processing_test.md                   583 lines     1741 words
slipbox/validation/builders/variants/training_integration_tests.md        724 lines     1803 words
slipbox/validation/builders/variants/training_interface_tests.md          775 lines     2122 words
slipbox/validation/builders/variants/training_specification_tests.md      197 lines      786 words
slipbox/validation/builders/variants/training_test.md                     305 lines     1130 words
slipbox/validation/builders/variants/transform_integration_tests.md       298 lines     1178 words
slipbox/validation/builders/variants/transform_interface_tests.md         322 lines     1187 words
slipbox/validation/builders/variants/transform_specification_tests.md     375 lines     1409 words
slipbox/validation/builders/variants/transform_test.md                    331 lines     1270 words
slipbox/validation/interface/interface_standard_validator.md              445 lines     1388 words
slipbox/validation/naming/naming_standard_validator.md                    601 lines     1752 words
slipbox/validation/README.md                                              189 lines      829 words
slipbox/validation/shared/chart_utils.md                                  480 lines     1461 words
slipbox/validation/simple_integration.md                                  433 lines     1303 words

```

**TOTAL LINES in slipbox markdown files: 230431**


**TOTAL WORDS in slipbox markdown files: 895835**



---

SUMMARY

---

- **Python files in src/cursus:     307 files, 65589 lines of code**
- **Python files in test:           218 files, 42905 lines of code**
- **Markdown files in slipbox:      543 files, 230431 lines, 895835 words**

- **GRAND TOTAL Python LOC:         108494 lines**

---


## Key Insights

### Source Code Distribution
- The cursus project contains a substantial codebase with **65589 lines** of Python source code
- The code is well-organized across **307 Python files** in the main source package
- Major components include pipeline catalog, step builders, validation framework, and core utilities

### Test Coverage
- Comprehensive test suite with **42905 lines** of test code across **218 test files**
- Test-to-source ratio: approximately **65%** (test LOC / source LOC)
- Tests cover all major components including core functionality, builders, validation, and integration

### Documentation Quality
- Extensive documentation with **543 markdown files** containing **230431 lines** and **895835 words**
- Documentation-to-code ratio: approximately **3.5:1** (documentation lines / source code lines)
- Comprehensive coverage including design documents, developer guides, analysis reports, and API documentation

### Project Scale
- **Total Python codebase:** 108494 lines of code
- **Total project content:** Over 895.8K words of documentation
- **File count:** 1068 total files analyzed (Python + Markdown)

## Recommendations

1. **Code Maintenance**: With 65589+ lines of source code, consider implementing automated code quality checks
2. **Test Strategy**: Strong test coverage ratio suggests good testing practices - maintain this standard
3. **Documentation**: Excellent documentation coverage - continue maintaining this comprehensive approach
4. **Monitoring**: Regular analysis of these metrics can help track project growth and complexity

## Technical Notes

- Analysis excludes empty lines and comment-only lines for Python files
- All lines counted for markdown files to capture full documentation scope
- Generated using automated scripts for consistency and reproducibility
- Date-stamped for historical tracking of project evolution

---

*This analysis was generated automatically using the cursus project line counting tools.*
