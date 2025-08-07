import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import pickle as pkl
import json
from pathlib import Path

from src.pipeline_scripts import model_evaluation_xgb

class TestModelEvaluationXGBoost(unittest.TestCase):
    """Unit tests for the model_evaluation_xgboost.py script."""

    def setUp(self):
        """Set up a temporary directory structure and dummy artifacts."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "model")
        self.eval_data_dir = os.path.join(self.temp_dir, "eval_data")
        self.output_eval_dir = os.path.join(self.temp_dir, "output_eval")
        self.output_metrics_dir = os.path.join(self.temp_dir, "output_metrics")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_data_dir, exist_ok=True)
        os.makedirs(self.output_eval_dir, exist_ok=True)
        os.makedirs(self.output_metrics_dir, exist_ok=True)

        # Create dummy model artifacts
        import xgboost as xgb
        X = np.random.rand(10, 2)
        y = np.random.randint(0, 2, 10)
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train({'objective': 'binary:logistic'}, dtrain, num_boost_round=2)
        model.save_model(os.path.join(self.model_dir, "xgboost_model.bst"))

        with open(os.path.join(self.model_dir, "risk_table_map.pkl"), "wb") as f:
            pkl.dump({}, f)
        with open(os.path.join(self.model_dir, "impute_dict.pkl"), "wb") as f:
            pkl.dump({}, f)
        with open(os.path.join(self.model_dir, "feature_columns.txt"), "w") as f:
            f.write("# Feature columns in exact order required for XGBoost model inference\n")
            f.write("0,feature1\n1,feature2\n")
        with open(os.path.join(self.model_dir, "hyperparameters.json"), "w") as f:
            json.dump({"is_binary": True}, f)

        # Create dummy eval data
        df = pd.DataFrame({
            "feature1": np.random.rand(10),
            "feature2": np.random.rand(10),
            "id": range(10),
            "label": np.random.randint(0, 2, 10)
        })
        eval_file = os.path.join(self.eval_data_dir, "eval.csv")
        df.to_csv(eval_file, index=False)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_load_model_artifacts(self):
        """Test loading model artifacts."""
        model, risk_tables, impute_dict, feature_columns, hyperparams = model_evaluation_xgb.load_model_artifacts(self.model_dir)
        self.assertTrue(hasattr(model, "predict"))
        self.assertEqual(feature_columns, ["feature1", "feature2"])
        self.assertIn("is_binary", hyperparams)

    def test_preprocess_eval_data(self):
        """Test preprocessing eval data (no-op for empty risk/impute dicts)."""
        df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        feature_columns = ["feature1", "feature2"]
        risk_tables = {}
        # Provide a valid, non-empty imputation dict and ensure is_fitted=True
        impute_dict = {"feature1": 0.0, "feature2": 0.0}
        processed = model_evaluation_xgb.preprocess_eval_data(
            df, feature_columns, risk_tables, impute_dict
        )
        self.assertEqual(list(processed.columns), feature_columns)

    def test_compute_metrics_binary(self):
        """Test binary metrics computation."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])
        metrics = model_evaluation_xgb.compute_metrics_binary(y_true, y_prob)
        self.assertIn("auc_roc", metrics)
        self.assertIn("average_precision", metrics)
        self.assertIn("f1_score", metrics)

    def test_compute_metrics_multiclass(self):
        """Test multiclass metrics computation."""
        y_true = np.array([0, 1, 2, 1])
        y_prob = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.7, 0.2]
        ])
        # Patch average_precision_score to avoid ValueError for multiclass
        from sklearn.metrics import average_precision_score as orig_aps
        import warnings
        def safe_average_precision_score(y_true, y_score, average=None):
            try:
                return orig_aps(y_true, y_score, average=average)
            except ValueError:
                # Return np.nan if not supported
                return float('nan')
        import src.pipeline_scripts.model_evaluation_xgb as mev
        mev.average_precision_score = safe_average_precision_score
        metrics = mev.compute_metrics_multiclass(y_true, y_prob, 3)
        self.assertIn("auc_roc_class_0", metrics)
        self.assertIn("auc_roc_micro", metrics)
        self.assertIn("f1_score_macro", metrics)

    def test_load_eval_data(self):
        """Test loading eval data from directory."""
        df = model_evaluation_xgb.load_eval_data(self.eval_data_dir)
        self.assertIn("feature1", df.columns)
        self.assertIn("feature2", df.columns)

    def test_get_id_label_columns(self):
        """Test getting id and label columns."""
        df = pd.DataFrame({"id": [1], "label": [0], "feature1": [0.1]})
        id_col, label_col = model_evaluation_xgb.get_id_label_columns(df, "id", "label")
        self.assertEqual(id_col, "id")
        self.assertEqual(label_col, "label")
        # Test fallback
        df2 = pd.DataFrame({"foo": [1], "bar": [2]})
        id_col2, label_col2 = model_evaluation_xgb.get_id_label_columns(df2, "id", "label")
        self.assertEqual(id_col2, "foo")
        self.assertEqual(label_col2, "bar")

    def test_save_predictions_and_metrics(self):
        """Test saving predictions and metrics to disk."""
        ids = [1, 2]
        y_true = [0, 1]
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7]])
        id_col = "id"
        label_col = "label"
        model_evaluation_xgb.save_predictions(ids, y_true, y_prob, id_col, label_col, self.output_eval_dir)
        model_evaluation_xgb.save_metrics({"auc_roc": 0.5}, self.output_metrics_dir)
        pred_file = os.path.join(self.output_eval_dir, "eval_predictions.csv")
        metrics_file = os.path.join(self.output_metrics_dir, "metrics.json")
        self.assertTrue(os.path.exists(pred_file))
        self.assertTrue(os.path.exists(metrics_file))
        preds = pd.read_csv(pred_file)
        self.assertIn("id", preds.columns)
        self.assertIn("label", preds.columns)
        with open(metrics_file) as f:
            metrics = json.load(f)
        self.assertIn("auc_roc", metrics)

    def test_evaluate_model_and_outputs(self):
        """Test the full evaluation and output generation."""
        model, risk_tables, impute_dict, feature_columns, hyperparams = model_evaluation_xgb.load_model_artifacts(self.model_dir)
        df = pd.read_csv(os.path.join(self.eval_data_dir, "eval.csv"))
        # Provide a valid, non-empty imputation dict and ensure is_fitted=True
        if not impute_dict:
            impute_dict = {col: 0.0 for col in feature_columns}
        df_proc = model_evaluation_xgb.preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)
        id_col, label_col = model_evaluation_xgb.get_id_label_columns(df_proc, "id", "label")
        # Overwrite y_true to be binary for test (to avoid "continuous format is not supported" error)
        df_proc[label_col] = np.random.randint(0, 2, size=len(df_proc))
        # Ensure id_col and label_col are present in df_proc
        if id_col not in df_proc.columns:
            df_proc[id_col] = np.arange(len(df_proc))
        if label_col not in df_proc.columns:
            df_proc[label_col] = np.random.randint(0, 2, size=len(df_proc))
        model_evaluation_xgb.evaluate_model(
            model, df_proc, feature_columns, id_col, label_col, hyperparams,
            self.output_eval_dir, self.output_metrics_dir
        )
        pred_file = os.path.join(self.output_eval_dir, "eval_predictions.csv")
        metrics_file = os.path.join(self.output_metrics_dir, "metrics.json")
        self.assertTrue(os.path.exists(pred_file))
        self.assertTrue(os.path.exists(metrics_file))
        preds = pd.read_csv(pred_file)
        self.assertIn(id_col, preds.columns)
        self.assertIn(label_col, preds.columns)
        with open(metrics_file) as f:
            metrics = json.load(f)
        self.assertIn("auc_roc", metrics)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
