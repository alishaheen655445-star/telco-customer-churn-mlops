"""
Preprocessing Pipeline Module

This module creates a preprocessing pipeline for machine learning.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(X):
    """
    Build preprocessing pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.

    Returns
    -------
    ColumnTransformer
    """

    # Select categorical columns
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Select numerical columns
    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # Pipeline for categorical columns
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Pipeline for numerical columns
    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),

            (
                "numerical",
                numerical_pipeline,
                numerical_features
            )

        ]

    )

    return preprocessor