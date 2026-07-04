"""
Feature analysis module.

Analyze dataset features before feature engineering.
"""

import pandas as pd


def analyze_features(dataframe: pd.DataFrame) -> None:
    """
    Analyze every feature in the dataset.
    """

    print("\n========== Feature Analysis ==========\n")

    for column in dataframe.columns:

        print(f"Feature: {column}")
        print(f"Data Type: {dataframe[column].dtype}")
        print(f"Unique Values: {dataframe[column].nunique()}")

        if dataframe[column].nunique() <= 10:
            print("Values:")
            print(dataframe[column].value_counts())

        print("-" * 50)