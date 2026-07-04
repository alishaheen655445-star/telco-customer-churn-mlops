"""
Data Splitting Module

Splits the cleaned dataset into training and testing sets.
"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(dataframe: pd.DataFrame):
    """
    Split dataset into train and test sets.
    """

    # Features
    X = dataframe.drop(columns=["Churn Value"])

    # Target
    y = dataframe["Churn Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test