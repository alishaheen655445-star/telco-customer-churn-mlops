"""
Data preprocessing module.

This module provides functions for loading and inspecting
the Telco Customer Churn dataset.
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

    print("\n========== Dataset Shape ==========")
    print(dataframe.shape)

    print("\n========== Data Types ==========")
    print(dataframe.dtypes)

    print("\n========== Missing Values ==========")
    print(dataframe.isnull().sum())

    print("\n========== Duplicate Rows ==========")
    print(dataframe.duplicated().sum())