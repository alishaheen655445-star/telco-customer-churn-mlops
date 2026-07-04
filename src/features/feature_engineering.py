"""
Feature engineering module.

Prepare cleaned data for machine learning.
"""

from src.data.preprocessing import load_data
from src.data.preprocessing import clean_data

from src.utils.feature_analysis import analyze_features


def main():
    """
    Run feature engineering pipeline.
    """

    # Load dataset
    dataframe = load_data()

    # Clean dataset
    dataframe = clean_data(dataframe)

    # Analyze features
    analyze_features(dataframe)


if __name__ == "__main__":
    main()