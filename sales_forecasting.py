from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

pd.options.display.float_format = "{:,.2f}".format

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
PLOTS_DIR = OUTPUT_DIR / 'plots'

def load_data(data_dir: Path) -> pd.DataFrame:

    df = pd.read_csv(data_dir / 'Walmart_Sales.csv')

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    weekly_sales = df.groupby("Date")["Weekly_Sales"].sum().sort_index().asfreq("W-FRI")

    return weekly_sales

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

def difference_series(series: pd.Series, d: int) ->pd.Series:

    differenced_series = series.copy()

    for _ in range(d):
        differenced_series = differenced_series.diff().dropna()

    return differenced_series

def suggest_differencing_order(train_sales: pd.Series, max_d: int = 2,) -> int:

    for d in range(max_d + 1):
        candidate_series =difference_series(train_sales, d)

        adf_result = adfuller(candidate_series, autolag="AIC",)

        p_value = adf_result[1]

        #print(f"ADF p-value for d={d}: {p_value:.6f}")

        if p_value < 0.05:
            return d

    return max_d

def plot_acf_chart(train_sales: pd.Series, d: int, lags: int = 52,) -> None:

    stationary_sales = difference_series(train_sales, d)

    max_lags = min(lags, len(stationary_sales) // 2 -1,)

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_acf(stationary_sales, lags=max_lags, zero=False, ax=ax,)

    ax.set_title(f"ACF of Training Sales After d={d} Differencing")

    fig.tight_layout()

    fig.savefig(PLOTS_DIR / f"acf_d{d}.png", dpi=300)

    plt.close(fig)

def plot_pacf_chart(train_sales: pd.Series, d: int, lags: int = 52,) -> None:

    stationary_sales = difference_series(train_sales, d)

    max_lags = min(lags, len(stationary_sales) // 2 -1,)

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_pacf(stationary_sales, lags=max_lags, zero=False, ax=ax,)

    ax.set_title(f"PACF of Training Sales After d={d} Differencing")

    fig.tight_layout()

    fig.savefig(PLOTS_DIR / f"pacf_d{d}.png", dpi=300)

    plt.close(fig)

def build_arima_forecast(
        train_sales: pd.Series,
        test_sales: pd.Series,
        order: tuple[int, int, int] = (1, 1, 1)
) -> tuple[object, pd.Series]:

    fitted_model = ARIMA(train_sales, order=order).fit()

    forecast_values = fitted_model.forecast(steps=len(test_sales))

    forecast = pd.Series(forecast_values.to_numpy(), index=test_sales.index, name="ARIMA_Forecast")

    return forecast

def compare_arima_models(
    train_sales: pd.Series,
    test_sales: pd.Series,
    orders: list[tuple[int, int, int]],
) -> pd.DataFrame:

    model_results = []

    for order in orders:
        fitted_model = ARIMA(
            train_sales,
            order=order,
        ).fit()

        forecast = fitted_model.forecast(
            steps=len(test_sales)
        )

        forecast.index = test_sales.index

        mae = mean_absolute_error(
            test_sales,
            forecast,
        )

        rmse = np.sqrt(
            mean_squared_error(
                test_sales,
                forecast,
            )
        )

        mape = (
            np.mean(
                np.abs(
                    (test_sales - forecast)
                    / test_sales
                )
            )
            * 100
        )

        p = order[0]
        q = order[2]

        ljung_box_result = acorr_ljungbox(
            fitted_model.resid,
            lags=[10],
            model_df=p + q,
            return_df=True,
        )

        ljung_box_p_value = (
            ljung_box_result["lb_pvalue"].iloc[0]
        )

        model_results.append(
            {
                "Order": order,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "AIC": fitted_model.aic,
                "BIC": fitted_model.bic,
                "Ljung_Box_P_Value": ljung_box_p_value,
            }
        )

    results_df = pd.DataFrame(model_results)

    return results_df.sort_values(
        by="RMSE"
    )

def main() -> None:

    weekly_sales = load_data(DATA_DIR)
    train_sales, test_sales = split_time_series(weekly_sales, test_weeks=12)

    d = suggest_differencing_order(weekly_sales)

    plot_weekly_sales_time_series(weekly_sales)

    plot_acf_chart(train_sales, d=d,)
    plot_pacf_chart(train_sales, d=d,)

    candidate_orders = [(1, d, 1), (1, d, 2)]

    model_comparison = compare_arima_models(train_sales, test_sales, candidate_orders)

    print("\nARIMA Model Comparison:")

    print(
        model_comparison.to_string(
            index=False,
            formatters={
                "MAE": lambda value: f"${value:,.2f}",
                "RMSE": lambda value: f"${value:,.2f}",
                "MAPE": lambda value: f"{value:.2f}%",
                "AIC": lambda value: f"{value:,.2f}",
                "BIC": lambda value: f"{value:,.2f}",
            },
        )
    )

    best_order = model_comparison.iloc[0]["Order"]
    print(f"\nBest ARIMA order based on RMSE: {best_order}")

    best_forecast = build_arima_forecast(train_sales, test_sales, best_order,)

    forecast_comparison = pd.DataFrame({"Actual_Sales": test_sales, "ARIMA_Forecast": best_forecast,})

    print("\nBest ARIMA Forecast Comparison:")
    print(forecast_comparison.to_string(float_format=lambda value: f"{value:,.2f}"))

if __name__ == '__main__':
    main()