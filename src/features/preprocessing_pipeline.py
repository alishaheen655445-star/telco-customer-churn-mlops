"""
Preprocessing Pipeline
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(X):

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "cat",
                categorical_pipeline,
                categorical_features
            ),

            (
                "num",
                numerical_pipeline,
                numerical_features
            ),

        ]

    )

    return preprocessor