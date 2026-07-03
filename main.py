"""
Main entry point of the project.
"""

from src.data.preprocessing import load_data, inspect_data


def main() -> None:
    """
    Run the data loading and inspection process.
    """

    dataframe = load_data()

    print("========== First Five Rows ==========")
    print(dataframe.head())

    inspect_data(dataframe)


if __name__ == "__main__":
    main()