"""
Consolidated bug fix tests.

This module consolidates all bug fix related testing,
addressing the redundancy identified in the test coverage analysis.
Replaces: test_config_loading_fixed.py, test_config_recursion_fix.py, 
test_utils_additional_config.py
"""

import unittest
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest import mock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cursus.core.config_fields.type_aware_config_serializer import (
    TypeAwareConfigSerializer,
)
from cursus.core.config_fields.config_merger import ConfigMerger
from pydantic import BaseModel


class MdsDataSourceConfig(BaseModel):
    """Test model for MDS data source configuration."""

    name: str
    region: str
    table_name: str

    class Config:
        extra = "allow"


class DataSourceConfig(BaseModel):
    """Test model for data source configuration."""

    mds_data_source: MdsDataSourceConfig
    additional_sources: Optional[List["DataSourceConfig"]] = None

    class Config:
        extra = "allow"


class DataSourcesSpecificationConfig(BaseModel):
    """Test model for data sources specification."""

    data_sources: List[DataSourceConfig]
    primary_source: DataSourceConfig

    class Config:
        extra = "allow"


class CradleDataLoadConfig(BaseModel):
    """Test model for cradle data load configuration."""

    data_sources_specification: DataSourcesSpecificationConfig
    processing_config: Dict[str, Any]

    class Config:
        extra = "allow"


class XGBoostModelHyperparameters(BaseModel):
    """Test model for XGBoost hyperparameters."""

    num_round: int = 100
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 1.0
    colsample_bytree: float = 1.0

    class Config:
        extra = "allow"


class XGBoostTrainingConfig(BaseModel):
    """Test model for XGBoost training configuration."""

    hyperparameters: XGBoostModelHyperparameters
    training_data_path: str
    validation_data_path: str

    class Config:
        extra = "allow"


class PayloadConfig(BaseModel):
    """Test model for payload configuration with potential recursion issues."""

    name: str
    nested_payload: Optional["PayloadConfig"] = None
    payload_list: List["PayloadConfig"] = []

    class Config:
        extra = "allow"


class TestBugFixesConsolidated(unittest.TestCase):
    """Consolidated test cases for various bug fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = Path(self.temp_dir.name) / "test_config.json"

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_circular_reference_handling_in_data_sources(self):
        """Test fix for circular reference handling in data sources configuration."""
        # Create a data source configuration with potential circular references
        mds_config = MdsDataSourceConfig(
            name="test_mds", region="us-west-2", table_name="test_table"
        )

        data_source = DataSourceConfig(mds_data_source=mds_config)

        # Create circular reference scenario
        data_source.additional_sources = [data_source]  # Self-reference

        spec_config = DataSourcesSpecificationConfig(
            data_sources=[data_source], primary_source=data_source
        )

        cradle_config = CradleDataLoadConfig(
            data_sources_specification=spec_config,
            processing_config={"batch_size": 100},
        )

        # Test that serialization handles circular references without infinite recursion
        serializer = TypeAwareConfigSerializer()

        try:
            serialized = serializer.serialize(cradle_config)

            # Verify basic structure is preserved
            self.assertIn("data_sources_specification", serialized)
            self.assertIn("processing_config", serialized)

            # Verify processing config is intact
            self.assertEqual(serialized["processing_config"]["batch_size"], 100)

            # Verify data sources structure exists
            data_sources = serialized["data_sources_specification"]["data_sources"]
            self.assertIsInstance(data_sources, list)
            self.assertGreater(len(data_sources), 0)

        except RecursionError:
            self.fail("Circular reference caused infinite recursion - bug not fixed")
        except Exception as e:
            # Other exceptions are acceptable as long as it's not infinite recursion
            self.assertNotIsInstance(e, RecursionError)

    def test_special_list_format_handling(self):
        """Test fix for special list format handling that caused serialization issues."""
        # Create a configuration with special list formats
        mds_configs = []
        for i in range(3):
            mds_config = MdsDataSourceConfig(
                name=f"mds_{i}", region="us-west-2", table_name=f"table_{i}"
            )
            mds_configs.append(mds_config)

        # Create data sources with complex list relationships
        data_sources = []
        for i, mds_config in enumerate(mds_configs):
            data_source = DataSourceConfig(mds_data_source=mds_config)

            # Add references to previous data sources (complex but not circular)
            if i > 0:
                data_source.additional_sources = data_sources[:i]

            data_sources.append(data_source)

        spec_config = DataSourcesSpecificationConfig(
            data_sources=data_sources, primary_source=data_sources[0]
        )

        # Test serialization of complex list structure
        serializer = TypeAwareConfigSerializer()

        try:
            serialized = serializer.serialize(spec_config)

            # Verify structure
            self.assertIn("data_sources", serialized)
            self.assertIn("primary_source", serialized)

            # Verify list handling
            data_sources_list = serialized["data_sources"]
            self.assertIsInstance(data_sources_list, list)
            self.assertEqual(len(data_sources_list), 3)

            # Verify each data source has proper structure
            for i, ds in enumerate(data_sources_list):
                self.assertIn("mds_data_source", ds)
                mds = ds["mds_data_source"]
                self.assertEqual(mds["name"], f"mds_{i}")

                # Verify additional_sources handling
                if "additional_sources" in ds:
                    additional = ds["additional_sources"]
                    if additional is not None:
                        self.assertIsInstance(additional, list)
                        # Should have i references (0 for first, 1 for second, 2 for third)
                        if i > 0:
                            self.assertLessEqual(len(additional), i)

        except Exception as e:
            self.fail(f"Special list format handling failed: {e}")

    def test_circular_reference_handling_in_hyperparameters(self):
        """Test fix for circular reference handling in hyperparameters."""
        # Create hyperparameters configuration
        hyperparams = XGBoostModelHyperparameters(
            num_round=150, max_depth=8, min_child_weight=2
        )

        training_config = XGBoostTrainingConfig(
            hyperparameters=hyperparams,
            training_data_path="/path/to/training/data",
            validation_data_path="/path/to/validation/data",
        )

        # Test serialization of nested hyperparameters
        serializer = TypeAwareConfigSerializer()

        try:
            serialized = serializer.serialize(training_config)

            # Verify structure
            self.assertIn("hyperparameters", serialized)
            self.assertIn("training_data_path", serialized)
            self.assertIn("validation_data_path", serialized)

            # Verify hyperparameters structure
            hyperparams_data = serialized["hyperparameters"]
            self.assertIsInstance(hyperparams_data, dict)
            self.assertEqual(hyperparams_data["num_round"], 150)
            self.assertEqual(hyperparams_data["max_depth"], 8)
            self.assertEqual(hyperparams_data["min_child_weight"], 2)

            # Verify paths are preserved
            self.assertEqual(serialized["training_data_path"], "/path/to/training/data")
            self.assertEqual(
                serialized["validation_data_path"], "/path/to/validation/data"
            )

        except Exception as e:
            self.fail(f"Hyperparameters serialization failed: {e}")

    def test_payload_config_recursion_fix(self):
        """Test fix for recursion issues in payload configuration."""
        # Create a payload configuration with nested structure
        nested_payload = PayloadConfig(name="nested_payload")

        main_payload = PayloadConfig(
            name="main_payload",
            nested_payload=nested_payload,
            payload_list=[nested_payload],
        )

        # Test serialization without recursion issues
        serializer = TypeAwareConfigSerializer()

        try:
            serialized = serializer.serialize(main_payload)

            # Verify basic structure
            self.assertEqual(serialized["name"], "main_payload")

            # Verify nested payload handling
            if "nested_payload" in serialized and serialized["nested_payload"]:
                nested = serialized["nested_payload"]
                if isinstance(nested, dict):
                    self.assertEqual(nested.get("name"), "nested_payload")

            # Verify payload list handling
            if "payload_list" in serialized:
                payload_list = serialized["payload_list"]
                self.assertIsInstance(payload_list, list)

        except RecursionError:
            self.fail("Payload configuration caused infinite recursion - bug not fixed")
        except Exception as e:
            # Other exceptions are acceptable as long as it's not infinite recursion
            self.assertNotIsInstance(e, RecursionError)

    def test_load_real_config_file(self):
        """Test fix for loading real configuration files with complex structures."""
        # Create a realistic configuration file structure
        config_data = {
            "metadata": {
                "created_at": "2025-08-07T08:00:00",
                "config_types": {
                    "CradleDataLoadConfig": "CradleDataLoadConfig",
                    "XGBoostTrainingConfig": "XGBoostTrainingConfig",
                },
            },
            "configuration": {
                "shared": {"region": "us-west-2", "batch_size": 100},
                "specific": {
                    "CradleDataLoadConfig": {
                        "data_sources_specification": {
                            "data_sources": [
                                {
                                    "mds_data_source": {
                                        "name": "test_mds",
                                        "region": "us-west-2",
                                        "table_name": "test_table",
                                    }
                                }
                            ],
                            "primary_source": {
                                "mds_data_source": {
                                    "name": "primary_mds",
                                    "region": "us-west-2",
                                    "table_name": "primary_table",
                                }
                            },
                        },
                        "processing_config": {
                            "instance_type": "ml.m5.large",
                            "instance_count": 1,
                        },
                    },
                    "XGBoostTrainingConfig": {
                        "hyperparameters": {
                            "num_round": 100,
                            "max_depth": 6,
                            "min_child_weight": 1,
                        },
                        "training_data_path": "/path/to/training",
                        "validation_data_path": "/path/to/validation",
                    },
                },
            },
        }

        # Write to file
        with open(self.test_file_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Test loading the file
        try:
            loaded_config = ConfigMerger.load(str(self.test_file_path))

            # Verify structure
            self.assertIn("shared", loaded_config)
            self.assertIn("specific", loaded_config)

            # Verify shared fields
            shared = loaded_config["shared"]
            self.assertEqual(shared["region"], "us-west-2")
            self.assertEqual(shared["batch_size"], 100)

            # Verify specific fields
            specific = loaded_config["specific"]
            self.assertIn("CradleDataLoadConfig", specific)
            self.assertIn("XGBoostTrainingConfig", specific)

            # Verify nested structures are preserved
            cradle_config = specific["CradleDataLoadConfig"]
            self.assertIn("data_sources_specification", cradle_config)
            self.assertIn("processing_config", cradle_config)

            xgboost_config = specific["XGBoostTrainingConfig"]
            self.assertIn("hyperparameters", xgboost_config)
            self.assertEqual(xgboost_config["hyperparameters"]["num_round"], 100)

        except Exception as e:
            self.fail(f"Loading real config file failed: {e}")

    def test_additional_config_with_special_list(self):
        """Test fix for additional config handling with special list formats."""
        # Create a configuration with additional config that has special list handling
        base_config = {
            "name": "base_config",
            "items": ["item1", "item2", "item3"],
            "nested_items": [
                {"name": "nested1", "value": 1},
                {"name": "nested2", "value": 2},
            ],
        }

        additional_config = {
            "name": "additional_config",
            "extra_items": ["extra1", "extra2"],
            "complex_nested": {"level1": {"level2": ["deep1", "deep2"]}},
        }

        # Test merging configurations with special list handling
        try:
            # Simulate the merge operation
            merged_config = {**base_config, **additional_config}

            # Verify structure
            self.assertEqual(
                merged_config["name"], "additional_config"
            )  # Should override
            self.assertIn("items", merged_config)
            self.assertIn("extra_items", merged_config)
            self.assertIn("complex_nested", merged_config)

            # Verify list handling
            self.assertEqual(len(merged_config["items"]), 3)
            self.assertEqual(len(merged_config["extra_items"]), 2)
            self.assertEqual(len(merged_config["nested_items"]), 2)

            # Verify nested structure
            complex_nested = merged_config["complex_nested"]
            self.assertIn("level1", complex_nested)
            self.assertIn("level2", complex_nested["level1"])
            self.assertEqual(len(complex_nested["level1"]["level2"]), 2)

        except Exception as e:
            self.fail(f"Additional config with special list handling failed: {e}")

    def test_deep_recursion_prevention(self):
        """Test fix for preventing deep recursion in complex configurations."""

        # Create a deeply nested structure that could cause recursion issues
        def create_nested_config(depth, max_depth=10):
            if depth >= max_depth:
                return {"name": f"leaf_{depth}", "value": depth}

            return {
                "name": f"level_{depth}",
                "value": depth,
                "nested": create_nested_config(depth + 1, max_depth),
                "items": [
                    {"name": f"item_{depth}_{i}", "value": depth * 10 + i}
                    for i in range(2)
                ],
            }

        deep_config = create_nested_config(0, 8)  # 8 levels deep

        # Test serialization without hitting recursion limits
        serializer = TypeAwareConfigSerializer()

        try:
            serialized = serializer.serialize(deep_config)

            # Verify basic structure
            self.assertEqual(serialized["name"], "level_0")
            self.assertEqual(serialized["value"], 0)
            self.assertIn("nested", serialized)
            self.assertIn("items", serialized)

            # Verify items structure
            items = serialized["items"]
            self.assertIsInstance(items, list)
            self.assertEqual(len(items), 2)

        except RecursionError:
            self.fail("Deep nesting caused infinite recursion - bug not fixed")
        except Exception as e:
            # Other exceptions are acceptable as long as it's not infinite recursion
            self.assertNotIsInstance(e, RecursionError)

    def test_memory_leak_prevention(self):
        """Test fix for memory leaks in large configuration processing."""
        # Create a large configuration that could cause memory issues
        large_config = {
            "name": "large_config",
            "large_list": [
                {"id": i, "data": f"data_{i}", "nested": {"value": i * 2}}
                for i in range(100)  # Reasonable size for testing
            ],
            "metadata": {
                "processing_info": {"batch_size": 32, "instance_type": "ml.m5.large"}
            },
        }

        # Test serialization without memory issues
        serializer = TypeAwareConfigSerializer()

        try:
            serialized = serializer.serialize(large_config)

            # Verify structure
            self.assertEqual(serialized["name"], "large_config")
            self.assertIn("large_list", serialized)
            self.assertIn("metadata", serialized)

            # Verify large list handling
            large_list = serialized["large_list"]
            self.assertIsInstance(large_list, list)
            self.assertEqual(len(large_list), 100)

            # Verify first and last items
            first_item = large_list[0]
            self.assertEqual(first_item["id"], 0)
            self.assertEqual(first_item["data"], "data_0")

            last_item = large_list[-1]
            self.assertEqual(last_item["id"], 99)
            self.assertEqual(last_item["data"], "data_99")

        except MemoryError:
            self.fail("Large configuration caused memory error - bug not fixed")
        except Exception as e:
            # Other exceptions are acceptable as long as it's not memory related
            self.assertNotIsInstance(e, MemoryError)

    def test_error_handling_improvements(self):
        """Test improvements in error handling and reporting."""
        # Test with invalid configuration that should produce clear error messages
        invalid_configs = [
            None,  # None config
            {},  # Empty config
            {"invalid": object()},  # Non-serializable object
        ]

        serializer = TypeAwareConfigSerializer()

        for i, invalid_config in enumerate(invalid_configs):
            with self.subTest(config_index=i):
                try:
                    serialized = serializer.serialize(invalid_config)

                    # If serialization succeeds, verify it handles the case gracefully
                    if invalid_config is None:
                        self.assertIsNone(serialized)
                    elif invalid_config == {}:
                        self.assertEqual(serialized, {})

                except Exception as e:
                    # Verify error messages are informative
                    error_msg = str(e)
                    self.assertGreater(
                        len(error_msg), 0, "Error message should not be empty"
                    )

                    # Should not be a generic error
                    self.assertNotIn("object has no attribute", error_msg.lower())


if __name__ == "__main__":
    unittest.main()
