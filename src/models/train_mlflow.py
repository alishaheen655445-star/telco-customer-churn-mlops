"""
MLflow Training Script - FINAL WORKING VERSION
"""

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import build_preprocessing_pipeline


def run_training():

    # =========================
    # Load data
    # =========================
    print("\nLoading data...")
    df = load_data()
    df = clean_data(df)

    print("\n========== Dataset Shape ==========")
    print(df.shape)

    # =========================
    # Split data
    # =========================
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining started...")

    # =========================
    # MLflow FIX (IMPORTANT)
    # =========================
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Project")

    # =========================
    # Preprocessing
    # =========================
    preprocessor = build_preprocessing_pipeline(X_train)

    # =========================
    # Model
    # =========================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # =========================
    # Train + Log
    # =========================
    with mlflow.start_run():

        pipeline.fit(X_train, y_train)

        print("Training finished ✔")

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # =========================
        # Metrics
        # =========================
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        # =========================
        # Print results
        # =========================
        print("\n========== RESULTS ==========")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc:.4f}")

        # =========================
        # MLflow logging
        # =========================
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        # =========================
        # Save model (FIXED)
        # =========================
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            pip_requirements=[
                "mlflow",
                "scikit-learn",
                "pandas",
                "numpy"
            ]
        )

        print("\nModel logged successfully ✔")


if __name__ == "__main__":
    run_training()