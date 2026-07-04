"""
Data preprocessing module.

This module provides functions for:
- Loading the dataset
- Inspecting the dataset
- Cleaning the dataset
"""

from pathlib import Path
import pandas as pd

# Dataset path
DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")


def load_data() -> pd.DataFrame:
    """
    Load dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_excel(DATA_PATH)


def inspect_data(dataframe: pd.DataFrame) -> None:
    """
    Print dataset information.
    """

    print("\n========== First Five Rows ==========")
    print(dataframe.head())

    print("\n========== Dataset Shape ==========")
    print(dataframe.shape)

    print("\n========== Data Types ==========")
    print(dataframe.dtypes)

    print("\n========== Missing Values ==========")
    print(dataframe.isnull().sum())

    print("\n========== Duplicate Rows ==========")
    print(dataframe.duplicated().sum())


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset.
    """

    dataframe = dataframe.copy()

    # Convert Total Charges to numeric
    dataframe["Total Charges"] = pd.to_numeric(
        dataframe["Total Charges"],
        errors="coerce"
    )

    # Remove rows where Total Charges is missing
    dataframe = dataframe.dropna(subset=["Total Charges"])

    # Remove unnecessary columns
    dataframe = dataframe.drop(
        columns=[
            "CustomerID",
            "Count",
            "Lat Long",
            "Churn Label",
            "Churn Score",
            "Churn Reason",
        ]
    )

    dataframe.reset_index(drop=True, inplace=True)

    print("\n========== Dataset Shape After Cleaning ==========")
    print(dataframe.shape)

    return dataframe


if __name__ == "__main__":

    df = load_data()

    inspect_data(df)

    df = clean_data(df)

    print(df.head())