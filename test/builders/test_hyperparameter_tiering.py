"""
Test for the hyperparameter tiering functionality.

This test demonstrates how the field classification into three tiers works
and how derived classes can inherit and extend the base hyperparameters.
"""

import unittest
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters
from src.pipeline_steps.hyperparameters_bsm import BSMModelHyperparameters

class TestHyperparameterTiering(unittest.TestCase):
    def test_field_categorization(self):
        """Test that fields are properly categorized into tiers."""
        # Create base hyperparameters with essential fields
        base_hyperparam = ModelHyperparameters(
            full_field_list=["field1", "field2", "label"],
            cat_field_list=["field1"],
            tab_field_list=["field2"],
            id_name="id",
            label_name="label",
            multiclass_categories=[0, 1],
        )
        
        # Get field categories
        categories = base_hyperparam.categorize_fields()
        
        # Verify essential fields are correctly categorized
        essential_fields = sorted(categories['essential'])
        self.assertIn("full_field_list", essential_fields)
        self.assertIn("cat_field_list", essential_fields)
        self.assertIn("tab_field_list", essential_fields)
        self.assertIn("id_name", essential_fields)
        self.assertIn("label_name", essential_fields)
        self.assertIn("multiclass_categories", essential_fields)
        
        # Verify system fields are correctly categorized
        system_fields = sorted(categories['system'])
        self.assertIn("model_class", system_fields)
        self.assertIn("device", system_fields)
        self.assertIn("batch_size", system_fields)
        self.assertIn("class_weights", system_fields)
        self.assertIn("metric_choices", system_fields)
        
        # Verify derived fields are correctly categorized
        derived_fields = sorted(categories['derived'])
        self.assertIn("is_binary", derived_fields)
        self.assertIn("num_classes", derived_fields)
        self.assertIn("input_tab_dim", derived_fields)
        
        # Verify derived field values are correct
        self.assertEqual(base_hyperparam.num_classes, 2)
        self.assertEqual(base_hyperparam.is_binary, True)
        self.assertEqual(base_hyperparam.input_tab_dim, 1)  # One tabular field
        
        # Test the string representation (should show fields by tier)
        str_repr = str(base_hyperparam)
        self.assertIn("Essential User Inputs", str_repr)
        self.assertIn("System Inputs", str_repr)
        self.assertIn("Derived Fields", str_repr)
    
    def test_xgboost_hyperparams_categorization(self):
        """Test that XGBoost hyperparameters are properly categorized and derived."""
        # Create XGBoost hyperparameters with essential fields
        xgb_hyperparam = XGBoostModelHyperparameters(
            full_field_list=["field1", "field2", "label"],
            cat_field_list=["field1"],
            tab_field_list=["field2"],
            id_name="id",
            label_name="label",
            multiclass_categories=[0, 1],
            num_round=100,
            max_depth=6,
            min_child_weight=1.0,
        )
        
        # Get field categories
        categories = xgb_hyperparam.categorize_fields()
        
        # Verify XGBoost-specific essential fields are correctly categorized
        essential_fields = sorted(categories['essential'])
        self.assertIn("num_round", essential_fields)
        self.assertIn("max_depth", essential_fields)
        
        # Verify XGBoost-specific system fields are correctly categorized
        system_fields = sorted(categories['system'])
        self.assertIn("min_child_weight", system_fields)
        
        # Verify XGBoost-specific system fields are correctly categorized
        system_fields = sorted(categories['system'])
        self.assertIn("booster", system_fields)
        self.assertIn("eta", system_fields)
        self.assertIn("tree_method", system_fields)
        
        # Verify XGBoost-specific derived fields are correctly categorized
        derived_fields = sorted(categories['derived'])
        self.assertIn("objective", derived_fields)
        self.assertIn("eval_metric", derived_fields)
        
        # Verify derived field values are correct for binary classification
        self.assertEqual(xgb_hyperparam.objective, "binary:logistic")
        self.assertEqual(xgb_hyperparam.eval_metric, ['logloss', 'auc'])
        
        # Create multiclass XGBoost hyperparameters
        xgb_multiclass = XGBoostModelHyperparameters(
            full_field_list=["field1", "field2", "label"],
            cat_field_list=["field1"],
            tab_field_list=["field2"],
            id_name="id",
            label_name="label",
            multiclass_categories=[0, 1, 2],  # 3 classes
            num_round=100,
            max_depth=6,
            min_child_weight=1.0,
        )
        
        # Verify derived field values are correct for multiclass
        self.assertEqual(xgb_multiclass.num_classes, 3)
        self.assertEqual(xgb_multiclass.is_binary, False)
        self.assertEqual(xgb_multiclass.objective, "multi:softmax")
        self.assertEqual(xgb_multiclass.eval_metric, ['mlogloss', 'merror'])
    
    def test_inheritance(self):
        """Test that hyperparameter inheritance works correctly."""
        # Create base hyperparameters
        base_hyperparam = ModelHyperparameters(
            full_field_list=["field1", "field2", "label"],
            cat_field_list=["field1"],
            tab_field_list=["field2"],
            id_name="id",
            label_name="label",
            multiclass_categories=[0, 1],
            batch_size=64,  # Override a system input
        )
        
        # Create XGBoost hyperparameters from base
        xgb_hyperparam = XGBoostModelHyperparameters.from_base_hyperparam(
            base_hyperparam,
            num_round=200,
            max_depth=8,
            min_child_weight=2.0,
            eta=0.1,  # Override a system input
        )
        
        # Verify inherited fields
        self.assertEqual(xgb_hyperparam.full_field_list, ["field1", "field2", "label"])
        self.assertEqual(xgb_hyperparam.batch_size, 64)  # Inherited override
        
        # Verify overridden fields
        self.assertEqual(xgb_hyperparam.eta, 0.1)
        
        # Verify derived fields are calculated correctly
        self.assertEqual(xgb_hyperparam.is_binary, True)
        self.assertEqual(xgb_hyperparam.objective, "binary:logistic")
        
        # Verify XGBoost-specific fields
        self.assertEqual(xgb_hyperparam.num_round, 200)
        self.assertEqual(xgb_hyperparam.max_depth, 8)
        self.assertEqual(xgb_hyperparam.min_child_weight, 2.0)

    def test_bsm_hyperparams_categorization(self):
        """Test that BSM hyperparameters are properly categorized and derived."""
        # Create BSM hyperparameters with essential fields
        bsm_hyperparam = BSMModelHyperparameters(
            full_field_list=["field1", "field2", "dialogue", "label"],
            cat_field_list=["field1"],
            tab_field_list=["field2"],
            id_name="id",
            label_name="label",
            multiclass_categories=[0, 1],
            tokenizer="bert-base-uncased",
            text_name="dialogue",
        )
        
        # Get field categories
        categories = bsm_hyperparam.categorize_fields()
        
        # Verify essential fields are correctly categorized
        essential_fields = sorted(categories['essential'])
        self.assertIn("tokenizer", essential_fields)
        self.assertIn("text_name", essential_fields)
        
        # Verify system fields are correctly categorized
        system_fields = sorted(categories['system'])
        self.assertIn("model_class", system_fields)
        self.assertIn("lr_decay", system_fields)
        self.assertIn("max_sen_len", system_fields)
        
        # Verify derived fields are correctly categorized
        derived_fields = sorted(categories['derived'])
        self.assertIn("model_config_dict", derived_fields)
        self.assertIn("tokenizer_config", derived_fields)
        
        # Verify derived field values are correct
        self.assertEqual(bsm_hyperparam.model_config_dict["num_layers"], bsm_hyperparam.num_layers)
        self.assertEqual(bsm_hyperparam.tokenizer_config["name"], "bert-base-uncased")
        
        # Verify trainer config method works correctly
        trainer_config = bsm_hyperparam.get_trainer_config()
        self.assertEqual(trainer_config["max_epochs"], bsm_hyperparam.max_epochs)
        self.assertEqual(trainer_config["precision"], 32)  # Default is fp16=False so 32-bit precision


if __name__ == "__main__":
    unittest.main()
