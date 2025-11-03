#!/usr/bin/env python3
"""
Feature Selection Script

A comprehensive feature selection script that implements multiple statistical and 
machine learning-based feature selection methods. Follows the cursus framework's 
testability patterns and provides a self-contained solution.

Author: Cursus Framework
Date: 2025-10-25
"""

import os
import sys
import argparse
import json
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression, SelectKBest,
    chi2, f_classif, f_regression, RFE
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


# -------------------------------------------------------------------------
# Logging setup - Updated for CloudWatch compatibility
# -------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility"""
    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            # StreamHandler with stdout for CloudWatch
            logging.StreamHandler(sys.stdout)
        ],
    )

    # Configure our module's logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # Allow propagation to root logger

    # Force flush stdout
    sys.stdout.flush()

    return logger


# Initialize logger
logger = setup_logging()


# -------------------------------------------------------------------------
# Container path constants
# -------------------------------------------------------------------------
CONTAINER_PATHS = {
    "INPUT_DATA": "/opt/ml/processing/input",
    "OUTPUT_DATA": "/opt/ml/processing/output",
}


# -------------------------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------------------------
def load_selected_features(selected_features_dir: str) -> List[str]:
    """
    Load pre-computed selected features from training job.
    
    Args:
        selected_features_dir: Directory containing selected_features.json
        
    Returns:
        List of selected feature names
    """
    selected_features_file = os.path.join(selected_features_dir, 'selected_features.json')
    
    if not os.path.exists(selected_features_file):
        raise FileNotFoundError(f"Selected features file not found: {selected_features_file}")
    
    try:
        with open(selected_features_file, 'r') as f:
            data = json.load(f)
        
        selected_features = data.get('selected_features', [])
        logger.info(f"Loaded {len(selected_features)} pre-selected features from {selected_features_file}")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
        
    except Exception as e:
        logger.error(f"Error loading selected features: {e}")
        raise


def load_single_split_data(input_data_dir: str, job_type: str) -> Dict[str, pd.DataFrame]:
    """
    Load single split data for non-training job types.
    
    Args:
        input_data_dir: Directory containing job_type subdirectory
        job_type: Type of job (validation, testing, etc.)
        
    Returns:
        Dictionary with single split DataFrame
    """
    split_dir = os.path.join(input_data_dir, job_type)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Look for the processed data file
    data_file = os.path.join(split_dir, f"{job_type}_processed_data.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed data file not found: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {job_type} split: {df.shape}")
        return {job_type: df}
        
    except Exception as e:
        logger.error(f"Error loading {job_type} data: {e}")
        raise


def load_preprocessed_data(input_data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load train/val/test splits from tabular preprocessing output.
    
    Args:
        input_data_dir: Directory containing train/val/test subdirectories
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(input_data_dir, split_name)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        # Look for the processed data file (following tabular_preprocessing pattern)
        data_file = os.path.join(split_dir, f"{split_name}_processed_data.csv")
        if not os.path.exists(data_file):
            logger.warning(f"Processed data file not found: {data_file}")
            continue
        
        try:
            splits[split_name] = pd.read_csv(data_file)
            logger.info(f"Loaded {split_name} split: {splits[split_name].shape}")
        except Exception as e:
            logger.error(f"Error loading {split_name} data: {e}")
            raise
    
    if not splits:
        raise FileNotFoundError("No valid data splits found in input directory")
    
    return splits


def save_selected_data(splits: Dict[str, pd.DataFrame], selected_features: List[str], 
                      target_variable: str, output_dir: str) -> None:
    """
    Save feature-selected splits in format expected by XGBoost training.
    
    Args:
        splits: Dictionary of DataFrames by split name
        selected_features: List of selected feature names
        target_variable: Target column name
        output_dir: Output directory path
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Features to keep (selected features + target)
    columns_to_keep = selected_features + [target_variable]
    
    for split_name, df in splits.items():
        # Create split subdirectory
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Verify all required columns exist
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in {split_name} split: {missing_cols}")
            raise ValueError(f"Missing columns in {split_name} split: {missing_cols}")
        
        # Filter to selected features + target
        selected_df = df[columns_to_keep].copy()
        
        # Save in same format as tabular preprocessing output
        output_file = os.path.join(split_dir, f"{split_name}_processed_data.csv")
        selected_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {split_name} split with {len(selected_features)} features: {output_file}")


# -------------------------------------------------------------------------
# Feature Selection Methods - Statistical
# -------------------------------------------------------------------------
def variance_threshold_selection(X: pd.DataFrame, threshold: float = 0.01) -> Dict[str, Any]:
    """
    Remove features with low variance.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying variance threshold selection with threshold={threshold}")
    
    variances = X.var()
    selected_features = variances[variances > threshold].index.tolist()
    
    logger.info(f"Variance threshold selected {len(selected_features)} out of {len(X.columns)} features")
    
    return {
        "method": "variance_threshold",
        "selected_features": selected_features,
        "scores": variances.to_dict(),
        "threshold": threshold,
        "n_selected": len(selected_features)
    }


def correlation_based_selection(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Remove highly correlated features, keeping those with higher target correlation.
    
    Args:
        X: Feature matrix
        y: Target variable
        threshold: Correlation threshold
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying correlation-based selection with threshold={threshold}")
    
    # Compute feature correlations
    corr_matrix = X.corr().abs()
    
    # Compute target correlations
    target_corr = X.corrwith(y).abs()
    
    # Find highly correlated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index 
                       if not pd.isna(upper_tri.loc[row, col]) and upper_tri.loc[row, col] > threshold]
    
    # Remove features with lower target correlation from each pair
    features_to_remove = set()
    for feat1, feat2 in high_corr_pairs:
        if target_corr[feat1] > target_corr[feat2]:
            features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat1)
    
    selected_features = [f for f in X.columns if f not in features_to_remove]
    
    logger.info(f"Correlation-based selection removed {len(features_to_remove)} features, "
                f"selected {len(selected_features)} out of {len(X.columns)} features")
    
    return {
        "method": "correlation_based",
        "selected_features": selected_features,
        "scores": target_corr.to_dict(),
        "removed_features": list(features_to_remove),
        "high_corr_pairs": high_corr_pairs,
        "threshold": threshold,
        "n_selected": len(selected_features)
    }


def mutual_info_selection(X: pd.DataFrame, y: pd.Series, k: int = 10, random_state: int = 42) -> Dict[str, Any]:
    """
    Select features based on mutual information with target.
    
    Args:
        X: Feature matrix
        y: Target variable
        k: Number of features to select
        random_state: Random seed
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying mutual information selection with k={k}")
    
    # Determine if classification or regression
    is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
    
    try:
        if is_classification:
            mi_scores = mutual_info_classif(X, y, random_state=random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=random_state)
        
        # Select top k features
        selector = SelectKBest(score_func=lambda X, y: mi_scores, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        scores = dict(zip(X.columns, mi_scores))
        
        logger.info(f"Mutual information selected {len(selected_features)} features "
                    f"(classification={is_classification})")
        
        return {
            "method": "mutual_information",
            "selected_features": selected_features,
            "scores": scores,
            "k": k,
            "n_selected": len(selected_features),
            "is_classification": is_classification
        }
    except Exception as e:
        logger.error(f"Error in mutual information selection: {e}")
        # Return empty result on error
        return {
            "method": "mutual_information",
            "selected_features": [],
            "scores": {},
            "k": k,
            "n_selected": 0,
            "is_classification": is_classification,
            "error": str(e)
        }


def chi2_selection(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Dict[str, Any]:
    """
    Select features using chi-square test (for non-negative features).
    
    Args:
        X: Feature matrix (non-negative values)
        y: Target variable
        k: Number of features to select
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying chi-square selection with k={k}")
    
    try:
        # Ensure non-negative values
        X_nonneg = X.copy()
        X_nonneg[X_nonneg < 0] = 0
        
        # Apply chi-square test
        selector = SelectKBest(score_func=chi2, k=min(k, len(X.columns)))
        selector.fit(X_nonneg, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        scores = dict(zip(X.columns, selector.scores_))
        
        logger.info(f"Chi-square selected {len(selected_features)} features")
        
        return {
            "method": "chi2",
            "selected_features": selected_features,
            "scores": scores,
            "k": k,
            "n_selected": len(selected_features)
        }
    except Exception as e:
        logger.error(f"Error in chi-square selection: {e}")
        return {
            "method": "chi2",
            "selected_features": [],
            "scores": {},
            "k": k,
            "n_selected": 0,
            "error": str(e)
        }


def f_classif_selection(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Dict[str, Any]:
    """
    Select features using ANOVA F-test.
    
    Args:
        X: Feature matrix
        y: Target variable
        k: Number of features to select
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying F-test selection with k={k}")
    
    try:
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        score_func = f_classif if is_classification else f_regression
        selector = SelectKBest(score_func=score_func, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        scores = dict(zip(X.columns, selector.scores_))
        
        logger.info(f"F-test selected {len(selected_features)} features "
                    f"(classification={is_classification})")
        
        return {
            "method": "f_test",
            "selected_features": selected_features,
            "scores": scores,
            "k": k,
            "n_selected": len(selected_features),
            "is_classification": is_classification
        }
    except Exception as e:
        logger.error(f"Error in F-test selection: {e}")
        return {
            "method": "f_test",
            "selected_features": [],
            "scores": {},
            "k": k,
            "n_selected": 0,
            "error": str(e)
        }


# -------------------------------------------------------------------------
# Feature Selection Methods - ML-Based
# -------------------------------------------------------------------------
def rfe_selection(X: pd.DataFrame, y: pd.Series, estimator_type: str = 'rf', 
                 n_features: int = 10, random_state: int = 42) -> Dict[str, Any]:
    """
    Recursive Feature Elimination with various estimators.
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator_type: Type of estimator ('rf', 'svm', 'linear')
        n_features: Number of features to select
        random_state: Random seed
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying RFE selection with estimator={estimator_type}, n_features={n_features}")
    
    try:
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        # Select estimator
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(n_estimators=50, random_state=random_state) if is_classification else RandomForestRegressor(n_estimators=50, random_state=random_state)
        elif estimator_type == 'svm':
            estimator = SVC(kernel='linear', random_state=random_state) if is_classification else SVR(kernel='linear')
        elif estimator_type == 'linear':
            estimator = LogisticRegression(random_state=random_state, max_iter=1000) if is_classification else LinearRegression()
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        # Apply RFE
        selector = RFE(estimator=estimator, n_features_to_select=min(n_features, len(X.columns)))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        rankings = dict(zip(X.columns, selector.ranking_))
        
        logger.info(f"RFE selected {len(selected_features)} features")
        
        return {
            "method": f"rfe_{estimator_type}",
            "selected_features": selected_features,
            "scores": {f: 1.0/rank for f, rank in rankings.items()},  # Convert ranking to score
            "rankings": rankings,
            "estimator_type": estimator_type,
            "n_features": n_features,
            "n_selected": len(selected_features),
            "is_classification": is_classification
        }
    except Exception as e:
        logger.error(f"Error in RFE selection: {e}")
        return {
            "method": f"rfe_{estimator_type}",
            "selected_features": [],
            "scores": {},
            "rankings": {},
            "estimator_type": estimator_type,
            "n_features": n_features,
            "n_selected": 0,
            "is_classification": is_classification,
            "error": str(e)
        }


def feature_importance_selection(X: pd.DataFrame, y: pd.Series, method: str = 'random_forest',
                               n_features: int = 10, random_state: int = 42) -> Dict[str, Any]:
    """
    Select features based on model feature importance.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Method for importance ('random_forest', 'xgboost', 'extra_trees')
        n_features: Number of features to select
        random_state: Random seed
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying feature importance selection with method={method}, n_features={n_features}")
    
    try:
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        # Select model
        if method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=random_state) if is_classification else RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif method == 'extra_trees':
            model = ExtraTreesClassifier(n_estimators=100, random_state=random_state) if is_classification else ExtraTreesRegressor(n_estimators=100, random_state=random_state)
        elif method == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(n_estimators=100, random_state=random_state) if is_classification else xgb.XGBRegressor(n_estimators=100, random_state=random_state)
            except ImportError:
                logger.warning("XGBoost not available, falling back to random forest")
                model = RandomForestClassifier(n_estimators=100, random_state=random_state) if is_classification else RandomForestRegressor(n_estimators=100, random_state=random_state)
                method = 'random_forest'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit model and get importance
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Select top features
        feature_importance_pairs = list(zip(X.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [pair[0] for pair in feature_importance_pairs[:min(n_features, len(X.columns))]]
        scores = dict(feature_importance_pairs)
        
        logger.info(f"Feature importance selected {len(selected_features)} features")
        
        return {
            "method": f"importance_{method}",
            "selected_features": selected_features,
            "scores": scores,
            "method_used": method,
            "n_features": n_features,
            "n_selected": len(selected_features),
            "is_classification": is_classification
        }
    except Exception as e:
        logger.error(f"Error in feature importance selection: {e}")
        return {
            "method": f"importance_{method}",
            "selected_features": [],
            "scores": {},
            "method_used": method,
            "n_features": n_features,
            "n_selected": 0,
            "is_classification": is_classification,
            "error": str(e)
        }


def lasso_selection(X: pd.DataFrame, y: pd.Series, alpha: float = 0.01, random_state: int = 42) -> Dict[str, Any]:
    """
    Select features using LASSO regularization.
    
    Args:
        X: Feature matrix
        y: Target variable
        alpha: Regularization strength
        random_state: Random seed
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying LASSO selection with alpha={alpha}")
    
    try:
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        if is_classification:
            # Use Logistic Regression with L1 penalty
            model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear', random_state=random_state, max_iter=1000)
            model.fit(X_scaled, y)
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        else:
            # Use LASSO regression
            if alpha == 'auto':
                model = LassoCV(cv=5, random_state=random_state)
            else:
                model = Lasso(alpha=alpha, random_state=random_state)
            model.fit(X_scaled, y)
            coefficients = model.coef_
        
        # Select features with non-zero coefficients
        selected_features = [X.columns[i] for i, coef in enumerate(coefficients) if abs(coef) > 1e-6]
        scores = dict(zip(X.columns, np.abs(coefficients)))
        
        logger.info(f"LASSO selected {len(selected_features)} features")
        
        return {
            "method": "lasso",
            "selected_features": selected_features,
            "scores": scores,
            "alpha": alpha if alpha != 'auto' else getattr(model, 'alpha_', alpha),
            "n_selected": len(selected_features),
            "is_classification": is_classification
        }
    except Exception as e:
        logger.error(f"Error in LASSO selection: {e}")
        return {
            "method": "lasso",
            "selected_features": [],
            "scores": {},
            "alpha": alpha,
            "n_selected": 0,
            "is_classification": is_classification,
            "error": str(e)
        }


def permutation_importance_selection(X: pd.DataFrame, y: pd.Series, estimator_type: str = 'rf',
                                   n_features: int = 10, random_state: int = 42) -> Dict[str, Any]:
    """
    Select features using permutation importance.
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator_type: Type of estimator
        n_features: Number of features to select
        random_state: Random seed
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Applying permutation importance selection with estimator={estimator_type}, n_features={n_features}")
    
    try:
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        # Select estimator
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(n_estimators=50, random_state=random_state) if is_classification else RandomForestRegressor(n_estimators=50, random_state=random_state)
        else:
            # Default to random forest
            estimator = RandomForestClassifier(n_estimators=50, random_state=random_state) if is_classification else RandomForestRegressor(n_estimators=50, random_state=random_state)
        
        # Fit estimator
        estimator.fit(X, y)
        
        # Compute permutation importance
        perm_importance = permutation_importance(estimator, X, y, n_repeats=5, random_state=random_state)
        
        # Select top features
        feature_importance_pairs = list(zip(X.columns, perm_importance.importances_mean))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [pair[0] for pair in feature_importance_pairs[:min(n_features, len(X.columns))]]
        scores = dict(feature_importance_pairs)
        
        logger.info(f"Permutation importance selected {len(selected_features)} features")
        
        return {
            "method": f"permutation_{estimator_type}",
            "selected_features": selected_features,
            "scores": scores,
            "estimator_type": estimator_type,
            "n_features": n_features,
            "n_selected": len(selected_features),
            "is_classification": is_classification
        }
    except Exception as e:
        logger.error(f"Error in permutation importance selection: {e}")
        return {
            "method": f"permutation_{estimator_type}",
            "selected_features": [],
            "scores": {},
            "estimator_type": estimator_type,
            "n_features": n_features,
            "n_selected": 0,
            "is_classification": is_classification,
            "error": str(e)
        }


# -------------------------------------------------------------------------
# Ensemble Selection Logic
# -------------------------------------------------------------------------
def combine_selection_results(method_results: List[Dict[str, Any]], 
                            combination_strategy: str = 'voting',
                            final_k: int = 10) -> Dict[str, Any]:
    """
    Combine results from multiple feature selection methods.
    
    Args:
        method_results: List of results from different methods
        combination_strategy: Strategy for combination ('voting', 'ranking', 'scoring')
        final_k: Final number of features to select
        
    Returns:
        Combined selection results
    """
    if not method_results:
        return {"selected_features": [], "scores": {}, "method_contributions": {}}
    
    # Collect all features and their scores from all methods
    all_features = set()
    method_scores = {}
    method_selections = {}
    
    for result in method_results:
        method_name = result["method"]
        selected_features = result["selected_features"]
        scores = result["scores"]
        
        all_features.update(selected_features)
        method_scores[method_name] = scores
        method_selections[method_name] = set(selected_features)
    
    all_features = list(all_features)
    
    if combination_strategy == 'voting':
        # Count how many methods selected each feature
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for selections in method_selections.values() if feature in selections)
            feature_votes[feature] = votes
        
        # Sort by votes and select top k
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in sorted_features[:final_k]]
        combined_scores = feature_votes
        
    elif combination_strategy == 'ranking':
        # Combine rankings from all methods
        feature_ranks = {f: [] for f in all_features}
        
        for method_name, scores in method_scores.items():
            # Convert scores to rankings
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (feature, score) in enumerate(sorted_scores):
                if feature in all_features:
                    feature_ranks[feature].append(rank + 1)
        
        # Average rankings (lower is better)
        avg_ranks = {}
        for feature, ranks in feature_ranks.items():
            if ranks:
                avg_ranks[feature] = np.mean(ranks)
            else:
                avg_ranks[feature] = len(all_features)  # Worst possible rank
        
        # Sort by average rank and select top k
        sorted_features = sorted(avg_ranks.items(), key=lambda x: x[1])
        selected_features = [f for f, _ in sorted_features[:final_k]]
        combined_scores = {f: 1.0/rank for f, rank in avg_ranks.items()}  # Convert to scores
        
    elif combination_strategy == 'scoring':
        # Normalize and combine scores from all methods
        normalized_scores = {}
        
        for method_name, scores in method_scores.items():
            # Normalize scores to 0-1 range
            score_values = list(scores.values())
            if score_values:
                min_score = min(score_values)
                max_score = max(score_values)
                score_range = max_score - min_score
                
                if score_range > 0:
                    normalized_scores[method_name] = {
                        f: (s - min_score) / score_range for f, s in scores.items()
                    }
                else:
                    normalized_scores[method_name] = {f: 1.0 for f in scores.keys()}
        
        # Average normalized scores
        combined_scores = {}
        for feature in all_features:
            scores_for_feature = []
            for method_scores in normalized_scores.values():
                if feature in method_scores:
                    scores_for_feature.append(method_scores[feature])
            
            if scores_for_feature:
                combined_scores[feature] = np.mean(scores_for_feature)
            else:
                combined_scores[feature] = 0.0
        
        # Sort by combined score and select top k
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in sorted_features[:final_k]]
    
    else:
        raise ValueError(f"Unknown combination strategy: {combination_strategy}")
    
    # Calculate method contributions
    method_contributions = {}
    for method_name, selections in method_selections.items():
        contribution = len(set(selected_features) & selections) / len(selected_features) if selected_features else 0
        method_contributions[method_name] = contribution
    
    return {
        "selected_features": selected_features,
        "scores": combined_scores,
        "method_contributions": method_contributions,
        "combination_strategy": combination_strategy,
        "n_methods": len(method_results),
        "n_selected": len(selected_features)
    }


# -------------------------------------------------------------------------
# Feature Selection Pipeline
# -------------------------------------------------------------------------
def apply_feature_selection_pipeline(splits: Dict[str, pd.DataFrame], target_variable: str,
                                   methods: List[str], method_configs: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Apply feature selection pipeline using training data, then apply to all splits.
    
    Args:
        splits: Dictionary of train/val/test DataFrames
        target_variable: Target column name
        methods: List of feature selection methods to apply
        method_configs: Configuration for each method
        
    Returns:
        Dictionary with selection results and metadata
    """
    # Use training data for feature selection
    if 'train' not in splits:
        raise ValueError("Training data not found in splits")
    
    train_df = splits['train']
    
    # Separate features and target
    if target_variable not in train_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in training data columns")
    
    feature_columns = [col for col in train_df.columns if col != target_variable]
    X_train = train_df[feature_columns]
    y_train = train_df[target_variable]
    
    logger.info(f"Starting feature selection on {len(feature_columns)} features")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Target variable: {target_variable}")
    
    # Apply each selection method
    method_results = []
    for method in methods:
        logger.info(f"Applying {method} feature selection...")
        start_time = time.time()
        
        try:
            if method == 'variance':
                result = variance_threshold_selection(X_train, **method_configs.get(method, {}))
            elif method == 'correlation':
                result = correlation_based_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'mutual_info':
                result = mutual_info_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'chi2':
                result = chi2_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'f_test':
                result = f_classif_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'rfe':
                result = rfe_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'importance':
                result = feature_importance_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'lasso':
                result = lasso_selection(X_train, y_train, **method_configs.get(method, {}))
            elif method == 'permutation':
                result = permutation_importance_selection(X_train, y_train, **method_configs.get(method, {}))
            else:
                logger.warning(f"Unknown method: {method}, skipping...")
                continue
            
            # Add processing time
            result['processing_time'] = time.time() - start_time
            method_results.append(result)
            
            logger.info(f"{method} selected {result['n_selected']} features in {result['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in {method} selection: {e}")
            # Continue with other methods
            continue
    
    if not method_results:
        raise RuntimeError("No feature selection methods completed successfully")
    
    # Combine results from multiple methods
    final_k = method_configs.get('final_k', min(50, len(feature_columns) // 2))
    combination_strategy = method_configs.get('combination_strategy', 'voting')
    
    logger.info(f"Combining results using {combination_strategy} strategy, selecting top {final_k} features")
    combined_result = combine_selection_results(method_results, combination_strategy, final_k)
    
    logger.info(f"Final selection: {len(combined_result['selected_features'])} features")
    logger.info(f"Selected features: {combined_result['selected_features']}")
    
    return {
        'selected_features': combined_result['selected_features'],
        'method_results': method_results,
        'combined_result': combined_result,
        'original_features': feature_columns,
        'target_variable': target_variable,
        'n_original_features': len(feature_columns),
        'n_selected_features': len(combined_result['selected_features'])
    }


# -------------------------------------------------------------------------
# Output Generation
# -------------------------------------------------------------------------
def save_selection_results(selection_results: Dict[str, Any], 
                          selected_features_dir: str) -> None:
    """
    Save feature selection results and metadata to selected_features channel.
    
    Args:
        selection_results: Results from feature selection pipeline
        selected_features_dir: Directory for all feature selection output files
    """
    # Ensure output directory exists
    os.makedirs(selected_features_dir, exist_ok=True)
    
    # Save selected features metadata
    selected_features_data = {
        "selected_features": selection_results['selected_features'],
        "selection_metadata": {
            "n_original_features": selection_results['n_original_features'],
            "n_selected_features": selection_results['n_selected_features'],
            "selection_ratio": selection_results['n_selected_features'] / selection_results['n_original_features'],
            "methods_used": [result['method'] for result in selection_results['method_results']],
            "combination_strategy": selection_results['combined_result']['combination_strategy'],
            "target_variable": selection_results['target_variable']
        },
        "method_contributions": selection_results['combined_result']['method_contributions']
    }
    
    with open(os.path.join(selected_features_dir, 'selected_features.json'), 'w') as f:
        json.dump(selected_features_data, f, indent=2, sort_keys=True)
    
    # Save detailed feature scores
    feature_scores_data = []
    for feature in selection_results['original_features']:
        row = {
            'feature_name': feature,
            'combined_score': selection_results['combined_result']['scores'].get(feature, 0),
            'selected': feature in selection_results['selected_features']
        }
        
        # Add scores from individual methods
        for result in selection_results['method_results']:
            method_name = result['method']
            row[f'{method_name}_score'] = result['scores'].get(feature, 0)
        
        feature_scores_data.append(row)
    
    # Sort by combined score
    feature_scores_data.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Save as CSV
    feature_scores_df = pd.DataFrame(feature_scores_data)
    feature_scores_df.to_csv(os.path.join(selected_features_dir, 'feature_scores.csv'), index=False)
    
    # Save selection summary report (merged into same directory)
    summary_report = {
        "selection_summary": {
            "total_features": selection_results['n_original_features'],
            "selected_features": selection_results['n_selected_features'],
            "selection_methods": [result['method'] for result in selection_results['method_results']],
            "combination_strategy": selection_results['combined_result']['combination_strategy'],
            "processing_time": sum(result.get('processing_time', 0) for result in selection_results['method_results']),
            "target_variable": selection_results['target_variable']
        },
        "method_performance": {
            result['method']: {
                "n_selected": result['n_selected'],
                "processing_time": result.get('processing_time', 0)
            } for result in selection_results['method_results']
        },
        "feature_statistics": {
            "avg_score": np.mean(list(selection_results['combined_result']['scores'].values())),
            "score_std": np.std(list(selection_results['combined_result']['scores'].values())),
            "min_score": min(selection_results['combined_result']['scores'].values()),
            "max_score": max(selection_results['combined_result']['scores'].values())
        }
    }
    
    with open(os.path.join(selected_features_dir, 'feature_selection_report.json'), 'w') as f:
        json.dump(summary_report, f, indent=2, sort_keys=True)
    
    logger.info(f"Saved all feature selection results to {selected_features_dir}")


# -------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str], 
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace
) -> None:
    """
    Main function for feature selection processing.
    
    Args:
        input_paths: Dictionary of input paths with logical names
            - "processed_data": Directory containing train/val/test splits from tabular preprocessing
        output_paths: Dictionary of output paths with logical names
            - "processed_data": Directory for feature-selected train/val/test splits (XGBoost input format)
            - "selected_features": Directory for selected_features.json and feature_scores.csv
            - "selection_report": Directory for feature_selection_report.json
        environ_vars: Dictionary of environment variables
            - "FEATURE_SELECTION_METHODS": Comma-separated list of methods
            - "LABEL_FIELD": Target column name (standard across framework)
            - "N_FEATURES_TO_SELECT": Number/percentage of features to select
            - "CORRELATION_THRESHOLD": Threshold for correlation filtering
            - "VARIANCE_THRESHOLD": Threshold for variance filtering
            - "RANDOM_STATE": Random seed for reproducibility
            - "COMBINATION_STRATEGY": Method combination strategy
        job_args: Command line arguments
            - job_type: Must be "training" to process all splits
    """
    try:
        logger.info("====== STARTING FEATURE SELECTION ======")
        
        # Extract configuration from environment variables
        methods_str = environ_vars.get("FEATURE_SELECTION_METHODS", "variance,correlation,mutual_info,rfe")
        methods = [method.strip() for method in methods_str.split(",")]
        
        target_variable = environ_vars.get("LABEL_FIELD")
        if not target_variable:
            raise ValueError("LABEL_FIELD environment variable must be set")
        
        n_features_to_select = int(environ_vars.get("N_FEATURES_TO_SELECT", 10))
        correlation_threshold = float(environ_vars.get("CORRELATION_THRESHOLD", 0.95))
        variance_threshold = float(environ_vars.get("VARIANCE_THRESHOLD", 0.01))
        random_state = int(environ_vars.get("RANDOM_STATE", 42))
        combination_strategy = environ_vars.get("COMBINATION_STRATEGY", "voting")
        
        logger.info(f"Configuration:")
        logger.info(f"  Methods: {methods}")
        logger.info(f"  Target variable: {target_variable}")
        logger.info(f"  Number of features to select: {n_features_to_select}")
        logger.info(f"  Combination strategy: {combination_strategy}")
        
        # Set up method configurations
        method_configs = {
            "variance": {"threshold": variance_threshold},
            "correlation": {"threshold": correlation_threshold},
            "mutual_info": {"k": n_features_to_select, "random_state": random_state},
            "chi2": {"k": n_features_to_select},
            "f_test": {"k": n_features_to_select},
            "rfe": {"estimator_type": "rf", "n_features": n_features_to_select, "random_state": random_state},
            "importance": {"method": "random_forest", "n_features": n_features_to_select, "random_state": random_state},
            "lasso": {"alpha": 0.01, "random_state": random_state},
            "permutation": {"estimator_type": "rf", "n_features": n_features_to_select, "random_state": random_state},
            "final_k": n_features_to_select,
            "combination_strategy": combination_strategy
        }
        
        # Handle different job types
        if job_args.job_type == "training":
            # Training mode: Run full feature selection pipeline
            logger.info("Running in TRAINING mode: performing full feature selection")
            
            # Load preprocessed data (all splits)
            input_data_dir = input_paths["processed_data"]
            logger.info(f"Loading data from {input_data_dir}")
            splits = load_preprocessed_data(input_data_dir)
            
            # Apply feature selection pipeline
            logger.info("Starting feature selection pipeline...")
            selection_results = apply_feature_selection_pipeline(splits, target_variable, methods, method_configs)
            
            # Save feature-selected data
            output_data_dir = output_paths["processed_data"]
            logger.info(f"Saving selected data to {output_data_dir}")
            save_selected_data(splits, selection_results['selected_features'], target_variable, output_data_dir)
            
            # Save selection results and metadata to selected_features channel
            selected_features_dir = output_paths.get("selected_features", output_data_dir)
            
            logger.info(f"Saving all results to {selected_features_dir}")
            save_selection_results(selection_results, selected_features_dir)
            
        else:
            # Non-training mode: Use pre-computed selected features
            logger.info(f"Running in {job_args.job_type.upper()} mode: using pre-computed selected features")
            
            # Load pre-computed selected features
            if "selected_features" not in input_paths:
                raise ValueError(f"For non-training job type '{job_args.job_type}', selected_features input path must be provided")
            
            selected_features_input_dir = input_paths["selected_features"]
            logger.info(f"Loading pre-computed selected features from {selected_features_input_dir}")
            selected_features = load_selected_features(selected_features_input_dir)
            
            # Load single split data
            input_data_dir = input_paths["processed_data"]
            logger.info(f"Loading {job_args.job_type} data from {input_data_dir}")
            splits = load_single_split_data(input_data_dir, job_args.job_type)
            
            # Apply feature filtering (no computation, just filtering)
            logger.info(f"Applying pre-computed feature selection to {job_args.job_type} data")
            output_data_dir = output_paths["processed_data"]
            save_selected_data(splits, selected_features, target_variable, output_data_dir)
            
            # Create minimal selection results for consistency
            selection_results = {
                'selected_features': selected_features,
                'method_results': [],  # No methods run in non-training mode
                'combined_result': {
                    'selected_features': selected_features,
                    'scores': {f: 1.0 for f in selected_features},  # Dummy scores
                    'method_contributions': {},
                    'combination_strategy': 'pre_computed',
                    'n_methods': 0,
                    'n_selected': len(selected_features)
                },
                'original_features': [],  # Not available in non-training mode
                'target_variable': target_variable,
                'n_original_features': 0,  # Not available in non-training mode
                'n_selected_features': len(selected_features)
            }
            
            # Save minimal metadata (copy from training job artifacts)
            selected_features_dir = output_paths.get("selected_features", output_data_dir)
            
            logger.info(f"Copying all metadata to {selected_features_dir}")
            
            # Copy all files from input selected_features directory
            import shutil
            os.makedirs(selected_features_dir, exist_ok=True)
            
            # Copy files from input selected_features directory
            for filename in ['selected_features.json', 'feature_scores.csv', 'feature_selection_report.json']:
                src_file = os.path.join(selected_features_input_dir, filename)
                dst_file = os.path.join(selected_features_dir, filename)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied {filename} from training job")
                else:
                    logger.warning(f"File {filename} not found in training artifacts")
            
            # If report file wasn't found, create minimal one
            report_file = os.path.join(selected_features_dir, 'feature_selection_report.json')
            if not os.path.exists(report_file):
                minimal_report = {
                    "selection_summary": {
                        "total_features": 0,  # Not available in non-training mode
                        "selected_features": len(selected_features),
                        "selection_methods": ["pre_computed"],
                        "combination_strategy": "pre_computed",
                        "processing_time": 0,  # No processing in non-training mode
                        "target_variable": target_variable,
                        "job_type": job_args.job_type,
                        "mode": "non_training"
                    },
                    "method_performance": {},  # No methods run
                    "feature_statistics": {
                        "selected_feature_count": len(selected_features)
                    }
                }
                
                with open(report_file, 'w') as f:
                    json.dump(minimal_report, f, indent=2, sort_keys=True)
                logger.info("Created minimal selection report")
            
            logger.info(f"Applied pre-computed feature selection: {len(selected_features)} features")
        
        logger.info("====== FEATURE SELECTION COMPLETED SUCCESSFULLY ======")
        
    except Exception as e:
        logger.error(f"FATAL ERROR in feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# -------------------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Feature selection script starting...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Feature Selection Script")
    parser.add_argument(
        "--job_type",
        type=str,
        default="training",
        choices=["training", "validation", "testing"],
        help="Type of job (training/validation/testing)"
    )
    args = parser.parse_args()
    
    # Define input and output paths using container defaults
    input_paths = {
        "processed_data": CONTAINER_PATHS["INPUT_DATA"]
    }
    
    # For non-training jobs, add selected_features input path
    if args.job_type != "training":
        input_paths["selected_features"] = CONTAINER_PATHS["INPUT_DATA"] + "/selected_features"
    
    output_paths = {
        "processed_data": CONTAINER_PATHS["OUTPUT_DATA"],
        "selected_features": CONTAINER_PATHS["OUTPUT_DATA"] + "/selected_features"
    }
    
    # Collect environment variables
    environ_vars = {
        "FEATURE_SELECTION_METHODS": os.environ.get("FEATURE_SELECTION_METHODS", "variance,correlation,mutual_info,rfe"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD"),
        "N_FEATURES_TO_SELECT": os.environ.get("N_FEATURES_TO_SELECT", "10"),
        "CORRELATION_THRESHOLD": os.environ.get("CORRELATION_THRESHOLD", "0.95"),
        "VARIANCE_THRESHOLD": os.environ.get("VARIANCE_THRESHOLD", "0.01"),
        "RANDOM_STATE": os.environ.get("RANDOM_STATE", "42"),
        "COMBINATION_STRATEGY": os.environ.get("COMBINATION_STRATEGY", "voting")
    }
    
    try:
        logger.info(f"Starting feature selection with:")
        logger.info(f"  Input directory: {input_paths['processed_data']}")
        logger.info(f"  Output directory: {output_paths['processed_data']}")
        logger.info(f"  Job type: {args.job_type}")
        
        # Call the main function
        main(input_paths, output_paths, environ_vars, args)
        
        logger.info("Feature selection script completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Exception during feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
