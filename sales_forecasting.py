from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.arima.model import ARIMA

pd.options.display.float_format = "{:,.2f}".format

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
PLOTS_DIR = OUTPUT_DIR / 'plots'

def load_data(data_dir: Path) -> pd.DataFrame:

    df = pd.read_csv(data_dir / 'Walmart_Sales.csv')

    return df

def prepare_time_series(df: pd.DataFrame) -> pd.Series:

    time_series_df = df.copy()

    time_series_df["Date"] = pd.to_datetime(time_series_df["Date"], dayfirst=True)

    weekly_sales = (
        time_series_df.groupby("Date")["Weekly_Sales"]
        .sum()
        .sort_index()
        .asfreq("W-FRI")
    )

    return weekly_sales

def check_time_series(weekly_sales: pd.Series) -> None:

    print("Time series shape:")
    print(weekly_sales.shape)

    print("\nDate range:")
    print(f"start date: {weekly_sales.index.min().date()}")
    print(f"end date: {weekly_sales.index.max().date()}")

def format_sales_in_millions(value: float, position: int) -> str:

    return f"{value / 1_000_000:.0f}M"

def plot_weekly_sales_time_series(weekly_sales: pd.Series) -> None:

    plt.figure(figsize=(12, 8))

    plt.plot(weekly_sales.index, weekly_sales.values)

    plt.title("Total Walmart Weekly Sales Across All Stores")
    plt.xlabel("Date")
    plt.ylabel("Total Weekly Sales")

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_sales_in_millions))

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "Weekly_sales_time_series.png", dpi=300)

    plt.close()

def split_time_series(weekly_sales: pd.Series, test_weeks: int = 12) -> tuple[pd.Series, pd.Series]:

    if test_weeks <= 0 or test_weeks >= len(weekly_sales):
        raise ValueError("test_weeks must be greater than 0 and smaller than time series.")

    train_sales = weekly_sales.iloc[:-test_weeks].copy()
    test_sales = weekly_sales.iloc[-test_weeks:].copy()

    return train_sales, test_sales

def build_arima_forecast(
        train_sales: pd.Series,
        test_sales: pd.Series,
        order: tuple[int, int, int] = (1, 1, 1)
) -> tuple[object, pd.Series]:

    model = ARIMA(train_sales, order=order)

    fitted_model = model.fit()

    forecast_values = fitted_model.forecast(steps=len(test_sales))

    forecast = pd.Series(forecast_values.to_numpy(), index=test_sales.index, name="ARIMA_Forecast")

    return fitted_model, forecast

def main() -> None:

    sales_df = load_data(DATA_DIR)
    weekly_sales = prepare_time_series(sales_df)
    check_time_series(weekly_sales)
    plot_weekly_sales_time_series(weekly_sales)

    train_sales, test_sales = split_time_series(weekly_sales, test_weeks=12)

    fitted_arima, arima_forecast = build_arima_forecast(train_sales, test_sales, order=(1, 1, 1))

    comparison = pd.DataFrame({"Actual_Sales": test_sales, "Arima_Forecast": arima_forecast})

    print("\nARIMA Forecast Comparison:")
    print(comparison.to_string(float_format=lambda value: f"{value:,.2f}"))

    #print("First five weeks:")
    #print(weekly_sales.head())

    #print("\nLast five weeks:")
    #print(weekly_sales.tail())

    #print("\nNumber of weeks:")
    #print(len(weekly_sales))

    #print("\nMissing sales values:")
    #print(weekly_sales.isna().sum())

if __name__ == '__main__':
    main()