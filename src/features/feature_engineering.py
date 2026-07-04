"""
Feature Engineering Module
"""

from src.data.preprocessing import load_data, clean_data
from src.features.data_split import split_data
from src.features.preprocessing_pipeline import (
    build_preprocessing_pipeline,
)


def prepare_data():
    """
    Prepare data for machine learning.
    """

    # Load data
    dataframe = load_data()

    # Clean data
    dataframe = clean_data(dataframe)

    # Split data
    X_train, X_test, y_train, y_test = split_data(dataframe)

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(X_train)

    print("\n========== Feature Engineering ==========")

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    print("\n========== Preprocessing Pipeline ==========")
    print(preprocessor)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    prepare_data()