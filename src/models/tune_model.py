"""
Hyperparameter Tuning Module

Uses RandomizedSearchCV to optimize the Random Forest model.
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import (
    build_preprocessing_pipeline,
)


def tune_model():

    # =============================
    # Load dataset
    # =============================
    dataframe = load_data()
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
    # Create pipeline
    # =============================
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # =============================
    # Hyperparameter search space
    # =============================
    param_dist = {

        "classifier__n_estimators": [100, 200, 300],

        "classifier__max_depth": [
            None,
            10,
            20,
            30,
            40,
        ],

        "classifier__min_samples_split": [
            2,
            5,
            10,
        ],

        "classifier__min_samples_leaf": [
            1,
            2,
            4,
        ],

        "classifier__max_features": [
            "sqrt",
            "log2",
        ],

    }

    # =============================
    # Random Search
    # =============================
    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_dist,

        n_iter=15,

        cv=5,

        scoring="f1",

        random_state=42,

        n_jobs=-1,

        verbose=1,

    )

    print("\nSearching for best parameters...\n")

    search.fit(X_train, y_train)

    print("\n========== Best Parameters ==========")

    print(search.best_params_)

    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities)

    print("\n========== Tuned Model ==========")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"ROC AUC  : {auc:.4f}")

    print("\n========== Classification Report ==========\n")

    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    tune_model()