"""
Model Training Module

This module trains the first machine learning model.
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import (
    build_preprocessing_pipeline,
)


def train_model():
    """
    Train Random Forest model.
    """

    # =============================
    # Load dataset
    # =============================
    dataframe = load_data()

    # =============================
    # Clean dataset
    # =============================
    dataframe = clean_data(dataframe)

    # =============================
    # Split dataset
    # =============================
    X_train, X_test, y_train, y_test = split_data(dataframe)

    # =============================
    # Build preprocessing pipeline
    # =============================
    preprocessor = build_preprocessing_pipeline(X_train)

    # =============================
    # Create ML pipeline
    # =============================
    model = Pipeline(
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

    # =============================
    # Train model
    # =============================
    print("\nTraining model...\n")

    model.fit(X_train, y_train)

    # =============================
    # Predict
    # =============================
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("========== Training Finished ==========")

    print(f"Accuracy : {accuracy:.4f}")

    return model


if __name__ == "__main__":
    train_model() 