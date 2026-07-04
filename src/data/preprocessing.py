"""
Data preprocessing module.

This module provides functions for:
1. Loading the dataset.
2. Inspecting the dataset.
3. Cleaning the dataset.
"""

from pathlib import Path
import pandas as pd

# Path to the raw dataset
DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")


def load_data() -> pd.DataFrame:
    """
    Load the dataset from the Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_excel(DATA_PATH)

    return dataframe


def inspect_data(dataframe: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.
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
    Clean the dataset by removing unnecessary columns
    and fixing data types.
    """

    # Remove columns that are not useful for machine learning
    columns_to_drop = [
        "CustomerID",
        "Count",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "Lat Long",
    ]

    dataframe = dataframe.drop(columns=columns_to_drop)

    # Convert Total Charges to numeric
    dataframe["Total Charges"] = pd.to_numeric(
        dataframe["Total Charges"],
        errors="coerce"
    )

    # Remove rows containing missing values
    dataframe = dataframe.dropna()

    # Reset dataframe index
    dataframe = dataframe.reset_index(drop=True)

    return dataframe


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Display dataset information
    inspect_data(df)

    # Clean dataset
    df = clean_data(df)

    print("\n========== Dataset Shape After Cleaning ==========")
    print(df.shape)

    print("\n========== Remaining Columns ==========")
    print(df.columns)

    print("\n========== Missing Values After Cleaning ==========")
    print(df.isnull().sum())