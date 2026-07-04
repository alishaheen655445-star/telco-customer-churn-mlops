"""
Model Training with MLflow.
"""

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import build_preprocessing_pipeline


def train_with_mlflow():
    """
    Train Random Forest model and log everything to MLflow.
    """

    # ==========================
    # Load data
    # ==========================
    dataframe = load_data()
    dataframe = clean_data(dataframe)

    # ==========================
    # Split data
    # ==========================
    X_train, X_test, y_train, y_test = split_data(dataframe)

    # ==========================
    # Build preprocessing
    # ==========================
    preprocessor = build_preprocessing_pipeline(X_train)

    # ==========================
    # Build model
    # ==========================
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),
            ),
        ]
    )

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    # ==========================
    # MLflow
    # ==========================
    mlflow.set_experiment("Telco_Churn_Project")

    with mlflow.start_run():

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="random_forest_model"
        )

    print("\n========== MLflow ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nModel logged successfully!")


if __name__ == "__main__":
    train_with_mlflow()