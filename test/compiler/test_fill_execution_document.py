"""
Unit tests for the execution document filling functionality in DynamicPipelineTemplate.

These tests verify that the DynamicPipelineTemplate correctly populates execution documents
with Cradle data loading requests and registration configurations.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.pipeline_dag.base_dag import PipelineDAG
from src.pipeline_api.dynamic_template import DynamicPipelineTemplate


class TestFillExecutionDocument(unittest.TestCase):
    """Tests for the fill_execution_document method in DynamicPipelineTemplate."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DAG with data loading and registration nodes
        self.dag = PipelineDAG()
        self.dag.add_node("CradleDataLoading-Training")
        self.dag.add_node("CradleDataLoading-Calibration")
        self.dag.add_node("XGBoostTraining")
        self.dag.add_node("Registration-NA")
        self.dag.add_edge("CradleDataLoading-Training", "XGBoostTraining")
        self.dag.add_edge("XGBoostTraining", "Registration-NA")
        self.dag.add_edge("CradleDataLoading-Calibration", "Registration-NA")
        
        # Create template execution document
        self.execution_document = {
            "PIPELINE_STEP_CONFIGS": {
                "CradleDataLoading-Training": {
                    "STEP_CONFIG": {
                        "dataSources": {
                            "dataSources": [
                                {
                                    "dataSourceName": "PLACEHOLDER",
                                    "dataSourceType": "MDS",
                                    "mdsDataSourceProperties": {
                                        "serviceName": "PLACEHOLDER",
                                        "orgId": "PLACEHOLDER",
                                        "region": "NA",
                                        "useHourlyEdxDataSet": False,
                                        "outputSchema": []
                                    }
                                }
                            ],
                            "startDate": "PLACEHOLDER",
                            "endDate": "PLACEHOLDER"
                        },
                        "transformSpecification": {
                            "transformSQL": "PLACEHOLDER",
                            "jobSplitOptions": {
                                "splitJob": False,
                                "daysPerSplit": 1,
                                "mergeSql": "PLACEHOLDER"
                            }
                        },
                        "outputSpecification": {
                            "outputSchema": [],
                            "outputPath": "PLACEHOLDER",
                            "outputFormat": "PARQUET",
                            "outputSaveMode": "ERRORIFEXISTS",
                            "outputFileCount": 0,
                            "keepDotInOutputSchema": True
                        },
                        "cradleJobSpecification": {
                            "clusterType": "LARGE",
                            "cradleAccount": "PLACEHOLDER",
                            "extraSparkJobArguments": "",
                            "jobRetryCount": 0
                        }
                    },
                    "STEP_TYPE": [
                        "WORKFLOW_INPUT",
                        "CradleDataLoadingStep"
                    ]
                },
                "CradleDataLoading-Calibration": {
                    "STEP_CONFIG": {
                        "dataSources": {
                            "dataSources": [
                                {
                                    "dataSourceName": "PLACEHOLDER",
                                    "dataSourceType": "MDS",
                                    "mdsDataSourceProperties": {
                                        "serviceName": "PLACEHOLDER",
                                        "orgId": "PLACEHOLDER",
                                        "region": "NA",
                                        "useHourlyEdxDataSet": False,
                                        "outputSchema": []
                                    }
                                }
                            ],
                            "startDate": "PLACEHOLDER",
                            "endDate": "PLACEHOLDER"
                        },
                        "transformSpecification": {
                            "transformSQL": "PLACEHOLDER",
                            "jobSplitOptions": {
                                "splitJob": False,
                                "daysPerSplit": 1,
                                "mergeSql": "PLACEHOLDER"
                            }
                        },
                        "outputSpecification": {
                            "outputSchema": [],
                            "outputPath": "PLACEHOLDER",
                            "outputFormat": "PARQUET",
                            "outputSaveMode": "ERRORIFEXISTS",
                            "outputFileCount": 0,
                            "keepDotInOutputSchema": True
                        },
                        "cradleJobSpecification": {
                            "clusterType": "LARGE",
                            "cradleAccount": "PLACEHOLDER",
                            "extraSparkJobArguments": "",
                            "jobRetryCount": 0
                        }
                    },
                    "STEP_TYPE": [
                        "WORKFLOW_INPUT",
                        "CradleDataLoadingStep"
                    ]
                },
                "XGBoostTraining": {
                    "STEP_CONFIG": {},
                    "STEP_TYPE": "TRAINING_STEP"
                },
                "Registration-NA": {
                    "STEP_CONFIG": {
                        "model_domain": "The domain to register your model in (this is where you will find your model on DAWS)",
                        "model_objective": "The objective to register your model in (this is where you will find your model on DAWS)",
                        "source_model_inference_content_types": "Provide a list of types (application/json and text/csv are currently supported) for the content. Ex) ['text/csv']",
                        "source_model_inference_response_types": "Provide a list of types (application/json and text/csv are currently supported) for the response. Ex) ['application/json']",
                        "source_model_inference_input_variable_list": "Provide a dictionary mapping the variable name to the variable type (variable types supported are 'TEXT' and 'NUMERIC') for both input and output vars. Ex) {'INVAR': 'TEXT'}",
                        "source_model_inference_output_variable_list": "Provide a dictionary mapping the variable name to the variable type (variable types supported are 'TEXT' and 'NUMERIC') for both input and output vars. Ex) {'OUTVAR': 'NUMERIC'}",
                        "model_registration_region": "The region that your SageMaker model will be registered in, valid values are NA/EU/FE",
                        "model_owner": "The team id of the team that owns this model. This can be found at https://permissions.amazon.com/ by searching for your team name to get the team id.",
                        "source_model_inference_image_arn": "The image arn for your trained model to be registered. This can be found by calling model.image_uri on a created model or estimator.model_uri/estimator.image_uri on an Estimator object.",
                        "source_model_region": "Optional: The region your model was trained in and the model artifacts are located in. Default is 'us-east-1'",
                        "source_model_environment_variable_map": "Optional: You can provide a dictionary of environment variables to be associated with your model. Ex) {'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'}",
                        "load_testing_info_map": "Optional: You can provide a dictionary of load testing parameters. All the parameters listed are required if providing the load testing map. Ex) {'samplePayloadS3Bucket': 'mods-personal-bucket', 'samplePayloadS3Key': 'sample_payload.csv', 'expectedTPS': '2000', 'maxLatencyInMillisecond': '50', 'instanceTypeList': ['ml.m5.xlarge', 'ml.m5d.2xlarge']}"
                    },
                    "STEP_TYPE": [
                        "PROCESSING_STEP",
                        "MimsModelRegistrationProcessingStep"
                    ]
                }
            },
            "PIPELINE_ADDITIONAL_PARAMS": {}
        }
        
        # Create mock pipeline metadata
        self.cradle_requests = {
            "CradleDataLoading-Training": {
                "dataSources": {
                    "dataSources": [
                        {
                            "dataSourceName": "RAW_MDS",
                            "dataSourceType": "MDS",
                            "mdsDataSourceProperties": {
                                "serviceName": "test-service",
                                "orgId": "test-org",
                                "region": "NA",
                                "useHourlyEdxDataSet": False,
                                "outputSchema": [
                                    {
                                        "fieldName": "objectId",
                                        "fieldType": "STRING"
                                    }
                                ]
                            }
                        }
                    ],
                    "startDate": "2025-07-01T00:00:00",
                    "endDate": "2025-07-31T00:00:00"
                },
                "transformSpecification": {
                    "transformSQL": "SELECT * FROM RAW_MDS",
                    "jobSplitOptions": {
                        "splitJob": False,
                        "daysPerSplit": 1,
                        "mergeSql": "SELECT * FROM INPUT"
                    }
                },
                "outputSpecification": {
                    "outputSchema": ["objectId"],
                    "outputPath": "s3://test-bucket/output/training",
                    "outputFormat": "PARQUET",
                    "outputSaveMode": "ERRORIFEXISTS",
                    "outputFileCount": 0,
                    "keepDotInOutputSchema": True
                },
                "cradleJobSpecification": {
                    "clusterType": "LARGE",
                    "cradleAccount": "TEST-ACCOUNT",
                    "extraSparkJobArguments": "",
                    "jobRetryCount": 0
                }
            },
            "CradleDataLoading-Calibration": {
                "dataSources": {
                    "dataSources": [
                        {
                            "dataSourceName": "RAW_MDS_CALIB",
                            "dataSourceType": "MDS",
                            "mdsDataSourceProperties": {
                                "serviceName": "test-service-calib",
                                "orgId": "test-org",
                                "region": "NA",
                                "useHourlyEdxDataSet": False,
                                "outputSchema": [
                                    {
                                        "fieldName": "objectId",
                                        "fieldType": "STRING"
                                    }
                                ]
                            }
                        }
                    ],
                    "startDate": "2025-06-01T00:00:00",
                    "endDate": "2025-06-30T00:00:00"
                },
                "transformSpecification": {
                    "transformSQL": "SELECT * FROM RAW_MDS_CALIB",
                    "jobSplitOptions": {
                        "splitJob": False,
                        "daysPerSplit": 1,
                        "mergeSql": "SELECT * FROM INPUT"
                    }
                },
                "outputSpecification": {
                    "outputSchema": ["objectId"],
                    "outputPath": "s3://test-bucket/output/calibration",
                    "outputFormat": "PARQUET",
                    "outputSaveMode": "ERRORIFEXISTS",
                    "outputFileCount": 0,
                    "keepDotInOutputSchema": True
                },
                "cradleJobSpecification": {
                    "clusterType": "LARGE",
                    "cradleAccount": "TEST-ACCOUNT",
                    "extraSparkJobArguments": "",
                    "jobRetryCount": 0
                }
            }
        }
        
        self.registration_configs = {
            "Registration-NA": {
                "model_domain": "fraud",
                "model_objective": "buyer-abuse",
                "source_model_inference_content_types": ["application/json", "text/csv"],
                "source_model_inference_response_types": ["application/json"],
                "source_model_inference_input_variable_list": {"BUYER_ID": "TEXT", "ORDER_ID": "TEXT"},
                "source_model_inference_output_variable_list": {"RISK_SCORE": "NUMERIC", "VERDICT": "TEXT"},
                "model_registration_region": "NA",
                "model_owner": "test-team-id",
                "source_model_inference_image_arn": "test-image-uri:latest",
                "source_model_region": "us-east-1",
                "source_model_environment_variable_map": {
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_REGION": "us-east-1",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                },
                "load_testing_info_map": {
                    "sample_payload_s3_bucket": "test-bucket",
                    "sample_payload_s3_key": "test-payload.csv",
                    "expected_tps": "100",
                    "max_latency_in_millisecond": "100",
                    "instance_type_list": ["ml.m5.xlarge"],
                    "max_acceptable_error_rate": "0.01"
                }
            }
        }
        
        self.model_name = "test-model"

    def mock_template(self, mock_dag=None, mock_config_path=None):
        """Create a mocked DynamicPipelineTemplate."""
        # Set default values for mocks
        if not mock_dag:
            mock_dag = self.dag
        if not mock_config_path:
            mock_config_path = "mock_config.json"
            
        # Create a mocked template
        template = MagicMock(spec=DynamicPipelineTemplate)
        template.pipeline_metadata = {
            'cradle_loading_requests': self.cradle_requests,
            'registration_configs': self.registration_configs,
            'model_name': self.model_name
        }
        
        # Create mock configs - important for the test
        mock_registration_cfg = MagicMock()
        mock_registration_cfg.__class__.__name__ = "RegistrationConfig"
        mock_payload_cfg = MagicMock()
        mock_payload_cfg.__class__.__name__ = "PayloadConfig"
        mock_package_cfg = MagicMock()
        mock_package_cfg.__class__.__name__ = "PackageConfig"
        
        # Set configs attribute
        template.configs = {
            "registration": mock_registration_cfg,
            "payload": mock_payload_cfg,
            "package": mock_package_cfg
        }
        
        # Set up necessary methods for the test
        template.fill_execution_document = DynamicPipelineTemplate.fill_execution_document.__get__(template)
        template._fill_cradle_configurations = DynamicPipelineTemplate._fill_cradle_configurations.__get__(template)
        template._fill_registration_configurations = DynamicPipelineTemplate._fill_registration_configurations.__get__(template)
        template._find_registration_step_nodes = MagicMock(return_value=["Registration-NA"])
        template._create_config_map = MagicMock(return_value={})
        template._has_required_registration_fields = MagicMock(return_value=True)
        
        template.logger = MagicMock()
        
        return template
        
    def test_fill_execution_document(self):
        """Test fill_execution_document populates the execution document with metadata."""
        # Create a mocked template
        template = self.mock_template()
        
        # Call the method under test
        result = template.fill_execution_document(self.execution_document)
        
        # Verify Cradle data loading configurations were populated
        cradle_step_1 = result["PIPELINE_STEP_CONFIGS"]["CradleDataLoading-Training"]
        cradle_step_2 = result["PIPELINE_STEP_CONFIGS"]["CradleDataLoading-Calibration"]
        
        # Verify Training configuration
        self.assertEqual(cradle_step_1["STEP_CONFIG"]["dataSources"]["dataSources"][0]["dataSourceName"], "RAW_MDS")
        self.assertEqual(cradle_step_1["STEP_CONFIG"]["dataSources"]["startDate"], "2025-07-01T00:00:00")
        self.assertEqual(cradle_step_1["STEP_CONFIG"]["outputSpecification"]["outputPath"], "s3://test-bucket/output/training")
        
        # Verify Calibration configuration
        self.assertEqual(cradle_step_2["STEP_CONFIG"]["dataSources"]["dataSources"][0]["dataSourceName"], "RAW_MDS_CALIB")
        self.assertEqual(cradle_step_2["STEP_CONFIG"]["dataSources"]["startDate"], "2025-06-01T00:00:00")
        self.assertEqual(cradle_step_2["STEP_CONFIG"]["outputSpecification"]["outputPath"], "s3://test-bucket/output/calibration")
        
        # Verify Registration configuration was populated
        reg_step = result["PIPELINE_STEP_CONFIGS"]["Registration-NA"]
        
        self.assertEqual(reg_step["STEP_CONFIG"]["model_domain"], "fraud")
        self.assertEqual(reg_step["STEP_CONFIG"]["model_objective"], "buyer-abuse")
        self.assertEqual(reg_step["STEP_CONFIG"]["model_registration_region"], "NA")
        self.assertEqual(reg_step["STEP_CONFIG"]["model_owner"], "test-team-id")
        self.assertEqual(reg_step["STEP_CONFIG"]["source_model_inference_image_arn"], "test-image-uri:latest")
        
        # Verify environment variables
        self.assertEqual(
            reg_step["STEP_CONFIG"]["source_model_environment_variable_map"]["SAGEMAKER_PROGRAM"],
            "inference.py"
        )
        
        # Verify load testing info
        self.assertEqual(
            reg_step["STEP_CONFIG"]["load_testing_info_map"]["instance_type_list"],
            ["ml.m5.xlarge"]
        )
    
    def test_fill_execution_document_with_missing_sections(self):
        """Test fill_execution_document handles missing sections gracefully."""
        # Create execution document with missing PIPELINE_STEP_CONFIGS
        incomplete_doc = {
            "PIPELINE_ADDITIONAL_PARAMS": {}
        }
        
        # Create a mocked template
        template = self.mock_template()
        
        # Call the method under test
        result = template.fill_execution_document(incomplete_doc)
        
        # Verify warning was logged
        template.logger.warning.assert_called_with("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
        
        # Verify document was returned unchanged
        self.assertEqual(result, incomplete_doc)
    
    def test_fill_execution_document_with_missing_steps(self):
        """Test fill_execution_document handles missing step entries gracefully."""
        # Create execution document with missing steps
        incomplete_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "XGBoostTraining": {
                    "STEP_CONFIG": {},
                    "STEP_TYPE": "TRAINING_STEP"
                }
            },
            "PIPELINE_ADDITIONAL_PARAMS": {}
        }
        
        # Create a mocked template
        template = self.mock_template()
        
        # Call the method under test
        result = template.fill_execution_document(incomplete_doc)
        
        # Verify warnings were logged for missing steps
        template.logger.warning.assert_any_call("Cradle step 'CradleDataLoading-Training' not found in execution document")
        template.logger.warning.assert_any_call("Cradle step 'CradleDataLoading-Calibration' not found in execution document")
        
        # Verify document still has XGBoostTraining
        self.assertIn("XGBoostTraining", result["PIPELINE_STEP_CONFIGS"])
        
    def test_find_registration_step_nodes(self):
        """Test _find_registration_step_nodes finds registration steps correctly."""
        # Use the mock template instead of creating a real one
        template = self.mock_template()
        
        # Set up the config map to return
        mock_registration_cfg = MagicMock()
        mock_registration_cfg.__class__.__name__ = "RegistrationConfig"
        config_map = {
            "CradleDataLoading-Training": MagicMock(),
            "XGBoostTraining": MagicMock(),
            "Registration-NA": mock_registration_cfg
        }
        template._create_config_map.return_value = config_map
            
        # Call the method under test - use the real method
        template._find_registration_step_nodes = DynamicPipelineTemplate._find_registration_step_nodes.__get__(template)
        nodes = template._find_registration_step_nodes()
        
        # Verify result
        self.assertIn("Registration-NA", nodes)


if __name__ == '__main__':
    unittest.main()
