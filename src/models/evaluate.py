"""
Model Evaluation Module

This module evaluates the trained Random Forest model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import (
    build_preprocessing_pipeline,
)


def evaluate_model():

    # ==========================
    # Load data
    # ==========================

    dataframe = load_data()
    dataframe = clean_data(dataframe)

    # ==========================
    # Split
    # ==========================

    X_train, X_test, y_train, y_test = split_data(dataframe)

    # ==========================
    # Pipeline
    # ==========================

    preprocessor = build_preprocessing_pipeline(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    # ==========================
    # Train
    # ==========================

    model.fit(X_train, y_train)

    # ==========================
    # Predict
    # ==========================

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    # ==========================
    # Metrics
    # ==========================

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities)

    print("\n========== Model Evaluation ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\n========== Classification Report ==========\n")

    print(classification_report(y_test, predictions))

    # ==========================
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    figures_path = Path("reports/figures")
    figures_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        figures_path / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nConfusion Matrix saved successfully!")


if __name__ == "__main__":
    evaluate_model()