"""
Feature Engineering Module

This module:
1. Loads the dataset.
2. Cleans the data.
3. Analyzes the features.
4. Splits the dataset.
5. Builds the preprocessing pipeline.
"""

from src.data.preprocessing import load_data, clean_data, split_data
from src.utils.feature_analysis import analyze_features
from src.features.preprocessing_pipeline import build_preprocessing_pipeline


def main():
    """
    Main function.
    """

    # ============================
    # Load dataset
    # ============================
    dataframe = load_data()

    print("\n========== Raw Dataset ==========")
    print(f"Shape: {dataframe.shape}")

    print("\nChurn Distribution:")
    print(dataframe["Churn Value"].value_counts())

    # ============================
    # Clean dataset
    # ============================
    dataframe = clean_data(dataframe)

    print("\n========== Clean Dataset ==========")
    print(dataframe.head())

    print("\nDataset Shape:")
    print(dataframe.shape)

    print("\nChurn Distribution After Cleaning:")
    print(dataframe["Churn Value"].value_counts())

    # ============================
    # Analyze features
    # ============================
    analyze_features(dataframe)

    # ============================
    # Split dataset
    # ============================
    X_train, X_test, y_train, y_test = split_data(dataframe)

    print("\n========== Dataset Split ==========")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")

    # ============================
    # Build preprocessing pipeline
    # ============================
    preprocessor = build_preprocessing_pipeline(X_train)

    print("\n========== Preprocessing Pipeline ==========")
    print(preprocessor)


if __name__ == "__main__":
    main()