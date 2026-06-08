from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.arima.model import ARIMA

from forecasting_utils import (
    calculate_ljung_box_p_value,
    evaluate_forecast,
    format_sales_in_millions,
    load_weekly_sales,
    plot_acf_pacf,
    plot_forecast,
    split_time_series,
)

pd.options.display.float_format = "{:,.2f}".format

DATA_PATH = Path("data/Walmart_Sales.csv")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"

TEST_WEEKS = 12
LJUNG_BOX_LAG = 10
CANDIDATE_ORDERS = [
    (1, 0, 1),
    (1, 0, 2),
]

def plot_weekly_sales_time_series(weekly_sales: pd.Series) -> None:

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(weekly_sales.index, weekly_sales.values)
    ax.set_title("Total Walmart Weekly Sales Across All Stores")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Weekly Sales")
    ax.yaxis.set_major_formatter(FuncFormatter(format_sales_in_millions))
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "weekly_sales_time_series.png", dpi=300)
    plt.close(fig)


def build_arima_forecast(
    train_sales: pd.Series,
    test_sales: pd.Series,
    order: tuple[int, int, int],
) -> pd.Series:
    """Fit one ARIMA model and forecast the test period."""
    fitted_model = ARIMA(train_sales, order=order).fit()
    forecast = fitted_model.forecast(steps=len(test_sales))
    forecast.index = test_sales.index
    forecast.name = "ARIMA_Forecast"

    return forecast

def compare_arima_models(
    train_sales: pd.Series,
    test_sales: pd.Series,
    orders: list[tuple[int, int, int]],
) -> pd.DataFrame:

    model_results = []

    for order in orders:
        fitted_model = ARIMA(train_sales, order=order).fit()
        forecast = fitted_model.forecast(steps=len(test_sales))
        forecast.index = test_sales.index

        metrics = evaluate_forecast(test_sales, forecast)
        p, _, q = order

        ljung_box_p_value = calculate_ljung_box_p_value(
            fitted_model,
            model_df=p + q,
            lag=LJUNG_BOX_LAG,
        )

        model_results.append(
            {
                "Order": order,
                **metrics,
                "AIC": fitted_model.aic,
                "BIC": fitted_model.bic,
                "Ljung_Box_P_Value": ljung_box_p_value,
            }
        )

    return pd.DataFrame(model_results).sort_values(by="RMSE")

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    weekly_sales = load_weekly_sales(DATA_PATH)
    train_sales, test_sales = split_time_series(
        weekly_sales,
        test_weeks=TEST_WEEKS,
    )

    plot_weekly_sales_time_series(weekly_sales)
    plot_acf_pacf(
        train_sales,
        plots_dir=PLOTS_DIR,
        file_prefix="arima",
        lags=52,
    )

    model_comparison = compare_arima_models(
        train_sales,
        test_sales,
        CANDIDATE_ORDERS,
    )

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
                "Ljung_Box_P_Value": lambda value: f"{value:.4f}",
            },
        )
    )

    best_order = model_comparison.iloc[0]["Order"]
    print(f"\nBest ARIMA order based on RMSE: {best_order}")

    best_forecast = build_arima_forecast(
        train_sales,
        test_sales,
        best_order,
    )

    forecast_comparison = pd.DataFrame(
        {
            "Actual_Sales": test_sales,
            "ARIMA_Forecast": best_forecast,
        }
    )

    forecast_comparison.to_csv(
        OUTPUT_DIR / "arima_forecast_comparison.csv",
        index_label="Date",
    )

    plot_forecast(
        train_sales,
        test_sales,
        best_forecast,
        model_name="ARIMA",
        output_path=PLOTS_DIR / "arima_forecast_vs_actual.png",
    )

    print("\nBest ARIMA Forecast Comparison:")
    print(
        forecast_comparison.to_string(
            float_format=lambda value: f"{value:,.2f}"
        )
    )

if __name__ == "__main__":
    main()
