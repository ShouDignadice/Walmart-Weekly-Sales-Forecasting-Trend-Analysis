from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting_utils import (
    SALES_SCALE,
    calculate_ljung_box_p_value,
    evaluate_forecast,
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
ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 0, 52)
DIAGNOSTIC_LAGS = 60
LJUNG_BOX_LAG = 10


def fit_and_forecast(
    train_sales: pd.Series,
    forecast_steps: int,
) -> tuple[object, pd.DataFrame]:

    scaled_train_sales = train_sales / SALES_SCALE

    model = SARIMAX(
        scaled_train_sales,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(
        disp=False,
        maxiter=200,
    )

    forecast_result = fitted_model.get_forecast(steps=forecast_steps)
    forecast = forecast_result.predicted_mean * SALES_SCALE
    confidence_interval = forecast_result.conf_int(alpha=0.05) * SALES_SCALE

    forecast_df = pd.DataFrame(
        {
            "SARIMA_Forecast": forecast,
            "Lower_95_CI": confidence_interval.iloc[:, 0],
            "Upper_95_CI": confidence_interval.iloc[:, 1],
        }
    )

    return fitted_model, forecast_df

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    weekly_sales = load_weekly_sales(DATA_PATH)
    train_sales, test_sales = split_time_series(
        weekly_sales,
        test_weeks=TEST_WEEKS,
    )

    plot_acf_pacf(
        train_sales,
        plots_dir=PLOTS_DIR,
        file_prefix="sarima",
        lags=DIAGNOSTIC_LAGS,
        seasonal_period=SEASONAL_ORDER[3],
    )

    fitted_model, forecast_df = fit_and_forecast(
        train_sales,
        forecast_steps=len(test_sales),
    )
    forecast_df.index = test_sales.index

    comparison_df = pd.concat(
        [
            test_sales.rename("Actual_Sales"),
            forecast_df,
        ],
        axis=1,
    )

    metrics = evaluate_forecast(
        test_sales,
        comparison_df["SARIMA_Forecast"],
    )

    p, _, q = ORDER
    seasonal_p, _, seasonal_q, _ = SEASONAL_ORDER
    model_df = p + q + seasonal_p + seasonal_q

    ljung_box_p_value = calculate_ljung_box_p_value(
        fitted_model,
        model_df=model_df,
        lag=LJUNG_BOX_LAG,
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "sarima_forecast_comparison.csv",
        index_label="Date",
    )

    plot_forecast(
        train_sales,
        test_sales,
        comparison_df["SARIMA_Forecast"],
        model_name="SARIMA",
        output_path=PLOTS_DIR / "sarima_forecast_vs_actual.png",
        lower_ci=comparison_df["Lower_95_CI"],
        upper_ci=comparison_df["Upper_95_CI"],
    )

    residual_result = (
        "Residuals resemble white noise."
        if ljung_box_p_value > 0.05
        else "Residual autocorrelation may still remain."
    )

    print(f"SARIMA{ORDER} x {SEASONAL_ORDER}")
    print(f"MAE:  ${metrics['MAE']:,.2f}")
    print(f"RMSE: ${metrics['RMSE']:,.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(
        f"Ljung-Box p-value at lag {LJUNG_BOX_LAG}: "
        f"{ljung_box_p_value:.4f}"
    )
    print(residual_result)
    print("\nForecast Comparison:")
    print(comparison_df.to_string())


if __name__ == "__main__":
    main()
