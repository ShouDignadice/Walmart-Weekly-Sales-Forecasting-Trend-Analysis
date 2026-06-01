from ast import List
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

def load_data(data_dir: Path) -> pd.DataFrame:

    df = pd.read_csv(data_dir / "Walmart_Sales.csv")

    return df

def get_correlation_columns() -> List[str]:

    return [
        "Weekly_Sales",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment"
    ]

def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:

    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

def calculate_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:

    correlation_matrix = df[columns].corr()

    return correlation_matrix

def get_weekly_sales_correlation(correlation_matrix: pd.DataFrame) -> pd.Series:

    weekly_sales_corr = correlation_matrix["Weekly_Sales"].drop("Weekly_Sales")

    return weekly_sales_corr

def sort_correlation_by_strength(correlation_values: pd.Series) -> pd.Series:

    sorted_corr = correlation_values.sort_values(
        key=abs,
        ascending=False
    )

    return sorted_corr

def display_results(sorted_corr: pd.Series) -> None:

    print("Correlation ranked by strength:")
    print(sorted_corr.to_string())

def main() -> None:

    df = load_data(DATA_DIR)

    correlation_columns = get_correlation_columns()

    validate_columns(df, correlation_columns)

    correlation_matrix = calculate_correlation_matrix(df, correlation_columns)

    weekly_sales_corr = get_weekly_sales_correlation(correlation_matrix)

    sorted_corr = sort_correlation_by_strength(weekly_sales_corr)

    display_results(sorted_corr)

if __name__ == "__main__":
        main()