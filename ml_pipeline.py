"""
Machine Learning Training, Validation, and Model Evaluation Pipeline
Follows strict ML best practices:
- Pre-split train/test data before feature transformations.
- Compares Baseline (Ridge), Random Forest, Gradient Boosting, and MLP Regressors.
- Evaluates R2, RMSE, MAE, MAPE, and 5-fold Cross-Validation.
- Exports trained models, encoders, scalers, and diagnostic metrics to models/ directory.
"""

import json
import os
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature definitions
BEAM_NUMERICAL_FEATURES = [
    "n_stories", "story_num", "story_height", "bay_x_count", "bay_y_count",
    "bay_x_length", "bay_y_length", "span", "dead_load", "live_load",
    "zone_z", "response_reduction_r", "importance_factor_i", "wind_speed"
]
BEAM_CATEGORICAL_FEATURES = ["beam_location", "seismic_zone", "soil_type"]

COLUMN_NUMERICAL_FEATURES = [
    "n_stories", "story_num", "story_height", "bay_x_count", "bay_y_count",
    "bay_x_length", "bay_y_length", "trib_area_m2", "dead_load", "live_load",
    "zone_z", "response_reduction_r", "importance_factor_i", "wind_speed"
]
COLUMN_CATEGORICAL_FEATURES = ["column_location", "seismic_zone", "soil_type"]


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer with StandardScaler and OneHotEncoder."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    target_name: str,
) -> Tuple[Dict, Pipeline]:
    """
    Trains and benchmarks multiple regressors on the pre-split dataset.
    Returns evaluation metrics dictionary and the best performing model pipeline.
    """
    candidate_models = {
        "Baseline_Ridge": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42),
        "MLP_NeuralNet": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42, early_stopping=True),
    }

    results = {}
    best_r2 = -float("inf")
    best_pipeline = None
    best_model_name = None

    print(f"\n--- Benchmarking Models for Target: {target_name} ---")

    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])

        # 5-fold cross validation on training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)

        # Fit on training set
        pipeline.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred)) * 100.0

        print(f"[{name}] Test R2: {r2:.4f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}% | CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        results[name] = {
            "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_mape_percent": round(mape, 2),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
        }

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipeline
            best_model_name = name

    print(f"-> Selected Best Model for {target_name}: {best_model_name} (R2 = {best_r2:.4f})")
    results["best_model_selected"] = best_model_name
    return results, best_pipeline


def train_beam_models(df_beams: pd.DataFrame) -> Dict:
    """Trains regression models for Beam Section Modulus (Zp_req) and Beam Weight."""
    X = df_beams[BEAM_NUMERICAL_FEATURES + BEAM_CATEGORICAL_FEATURES]
    y_zp = df_beams["target_zp_req_cm3"]
    y_weight = df_beams["target_beam_weight_kg_m"]

    # 80/20 Train-Test split BEFORE featurization
    X_train, X_test, y_zp_train, y_zp_test, y_w_train, y_w_test = train_test_split(
        X, y_zp, y_weight, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(BEAM_NUMERICAL_FEATURES, BEAM_CATEGORICAL_FEATURES)

    # 1. Target: Zp_req (cm3)
    results_zp, best_pipe_zp = evaluate_models(
        X_train, y_zp_train, X_test, y_zp_test, preprocessor, "Beam_Zp_req_cm3"
    )
    joblib.dump(best_pipe_zp, f"{MODELS_DIR}/beam_zp_model.joblib")

    # 2. Target: Beam Weight (kg/m)
    results_weight, best_pipe_weight = evaluate_models(
        X_train, y_w_train, X_test, y_w_test, preprocessor, "Beam_Weight_kg_m"
    )
    joblib.dump(best_pipe_weight, f"{MODELS_DIR}/beam_weight_model.joblib")

    # Feature Importance analysis from Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", rf_model)])
    rf_pipe.fit(X_train, y_zp_train)
    
    # Extract feature names after one-hot encoding
    cat_encoder = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(BEAM_CATEGORICAL_FEATURES))
    all_feature_names = BEAM_NUMERICAL_FEATURES + encoded_cat_names
    importances = rf_pipe.named_steps["regressor"].feature_importances_
    
    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    results_zp["top_feature_importances"] = [
        {"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp[:10]
    ]

    metrics = {
        "beam_zp_metrics": results_zp,
        "beam_weight_metrics": results_weight,
    }
    return metrics


def train_column_models(df_columns: pd.DataFrame) -> Dict:
    """Trains regression models for Column Required Area (A_req) and Column Weight."""
    X = df_columns[COLUMN_NUMERICAL_FEATURES + COLUMN_CATEGORICAL_FEATURES]
    y_area = df_columns["target_area_req_cm2"]
    y_weight = df_columns["target_col_weight_kg_m"]

    # 80/20 Train-Test split BEFORE featurization
    X_train, X_test, y_area_train, y_area_test, y_w_train, y_w_test = train_test_split(
        X, y_area, y_weight, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(COLUMN_NUMERICAL_FEATURES, COLUMN_CATEGORICAL_FEATURES)

    # 1. Target: Area_req (cm2)
    results_area, best_pipe_area = evaluate_models(
        X_train, y_area_train, X_test, y_area_test, preprocessor, "Column_Area_req_cm2"
    )
    joblib.dump(best_pipe_area, f"{MODELS_DIR}/column_area_model.joblib")

    # 2. Target: Column Weight (kg/m)
    results_weight, best_pipe_weight = evaluate_models(
        X_train, y_w_train, X_test, y_w_test, preprocessor, "Column_Weight_kg_m"
    )
    joblib.dump(best_pipe_weight, f"{MODELS_DIR}/column_weight_model.joblib")

    # Feature Importance analysis
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", rf_model)])
    rf_pipe.fit(X_train, y_area_train)

    cat_encoder = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(COLUMN_CATEGORICAL_FEATURES))
    all_feature_names = COLUMN_NUMERICAL_FEATURES + encoded_cat_names
    importances = rf_pipe.named_steps["regressor"].feature_importances_

    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    results_area["top_feature_importances"] = [
        {"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp[:10]
    ]

    metrics = {
        "column_area_metrics": results_area,
        "column_weight_metrics": results_weight,
    }
    return metrics


def run_full_pipeline(dataset_dir: str = ".") -> Dict:
    """Executes dataset loading, training, and saves all metrics to models/metrics.json."""
    print("Loading simulated datasets...")
    df_beams = pd.read_csv(f"{dataset_dir}/dataset_beams.csv")
    df_columns = pd.read_csv(f"{dataset_dir}/dataset_columns.csv")

    print(f"Dataset Loaded: Beams shape = {df_beams.shape}, Columns shape = {df_columns.shape}")

    beam_metrics = train_beam_models(df_beams)
    col_metrics = train_column_models(df_columns)

    all_metrics = {
        "beams": beam_metrics,
        "columns": col_metrics,
        "total_beam_samples": len(df_beams),
        "total_column_samples": len(df_columns),
    }

    with open(f"{MODELS_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll models and metrics successfully saved to {MODELS_DIR}/")
    return all_metrics


if __name__ == "__main__":
    run_full_pipeline()
