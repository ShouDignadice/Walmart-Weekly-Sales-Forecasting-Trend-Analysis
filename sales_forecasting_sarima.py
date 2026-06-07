from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

pd.options.display.float_format = "{:,.2f}".format

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"

SEASONAL_PERIOD = 52
SALES_SCALE = 1_000_000

def load_data(data_dir: Path) -> pd.Series:

    df = pd.read_csv(data_dir / "Walmart_Sales.csv")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True,)

    weekly_sales = (
        df.groupby("Date")["Weekly_Sales"]
        .sum()
        .sort_index()
        .asfreq("W-FRI")
    )

    if weekly_sales.isna().any():
        missing_dates = (
            weekly_sales[weekly_sales.isna()]
            .index.strftime("%Y-%m-%d")
            .tolist()
        )

        raise ValueError(
            f"Missing weekly sales values were found for: {missing_dates}"
        )

    return weekly_sales


def format_sales_in_millions(
    value: float,
    position: int,
) -> str:

    return f"${value / SALES_SCALE:.0f}M"


def split_time_series(
    weekly_sales: pd.Series,
    test_weeks: int = 12,
) -> tuple[pd.Series, pd.Series]:

    if test_weeks <= 0 or test_weeks >= len(weekly_sales):
        raise ValueError(
            "test_weeks must be greater than 0 "
            "and smaller than the time series."
        )

    train_sales = weekly_sales.iloc[:-test_weeks].copy()
    test_sales = weekly_sales.iloc[-test_weeks:].copy()

    return train_sales, test_sales


def difference_series(
    series: pd.Series,
    d: int,
) -> pd.Series:

    differenced_series = series.copy()

    for _ in range(d):
        differenced_series = (
            differenced_series
            .diff()
            .dropna()
        )

    return differenced_series


def suggest_differencing_order(
    train_sales: pd.Series,
    max_d: int = 2,
) -> int:

    for d in range(max_d + 1):
        candidate_series = difference_series(
            train_sales,
            d,
        )

        adf_result = adfuller(
            candidate_series,
            autolag="AIC",
        )

        p_value = adf_result[1]

        print(
            f"ADF p-value for d={d}: "
            f"{p_value:.6f}"
        )

        if p_value < 0.05:
            return d

    return max_d


def plot_weekly_sales_time_series(
    weekly_sales: pd.Series,
) -> None:

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        weekly_sales.index,
        weekly_sales.values,
    )

    ax.set_title(
        "Total Walmart Weekly Sales Across All Stores"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Weekly Sales")

    ax.yaxis.set_major_formatter(
        FuncFormatter(format_sales_in_millions)
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "sarima_weekly_sales_time_series.png",
        dpi=300,
    )

    plt.close(fig)


def plot_acf_chart(
    train_sales: pd.Series,
    d: int,
    lags: int = 60,
) -> None:

    stationary_sales = difference_series(
        train_sales,
        d,
    )

    max_lags = min(
        lags,
        len(stationary_sales) // 2 - 1,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    plot_acf(
        stationary_sales,
        lags=max_lags,
        zero=False,
        ax=ax,
    )

    ax.axvline(
        SEASONAL_PERIOD,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_title(
        f"ACF After d={d} Differencing "
        "— Seasonal Lag = 52"
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / f"sarima_acf_d{d}.png",
        dpi=300,
    )

    plt.close(fig)


def plot_pacf_chart(
    train_sales: pd.Series,
    d: int,
    lags: int = 60,
) -> None:

    stationary_sales = difference_series(
        train_sales,
        d,
    )

    max_lags = min(
        lags,
        len(stationary_sales) // 2 - 1,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    plot_pacf(
        stationary_sales,
        lags=max_lags,
        zero=False,
        method="ywm",
        ax=ax,
    )

    ax.axvline(
        SEASONAL_PERIOD,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_title(
        f"PACF After d={d} Differencing "
        "— Seasonal Lag = 52"
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / f"sarima_pacf_d{d}.png",
        dpi=300,
    )

    plt.close(fig)


def fit_sarima_model(
    train_sales: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> object:

    scaled_train_sales = (
        train_sales / SALES_SCALE
    )

    model = SARIMAX(
        scaled_train_sales,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(
        disp=False,
        maxiter=200,
    )

    return fitted_model


def calculate_ljung_box_p_value(
    fitted_model: object,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    lag: int = 10,
) -> float:

    burn_in = int(
        getattr(
            fitted_model,
            "loglikelihood_burn",
            0,
        )
    )

    residuals = (
        fitted_model.resid
        .iloc[burn_in:]
        .dropna()
    )

    p, _, q = order

    seasonal_p, _, seasonal_q, _ = (
        seasonal_order
    )

    model_degrees_of_freedom = (
        p
        + q
        + seasonal_p
        + seasonal_q
    )

    if (
        len(residuals) <= lag
        or lag <= model_degrees_of_freedom
    ):
        return np.nan

    ljung_box_result = acorr_ljungbox(
        residuals,
        lags=[lag],
        model_df=model_degrees_of_freedom,
        return_df=True,
    )

    return float(
        ljung_box_result["lb_pvalue"].iloc[0]
    )


def build_sarima_forecast(
    train_sales: pd.Series,
    test_sales: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> tuple[object, pd.DataFrame]:

    fitted_model = fit_sarima_model(
        train_sales,
        order,
        seasonal_order,
    )

    forecast_result = fitted_model.get_forecast(
        steps=len(test_sales)
    )

    forecast = (
        forecast_result.predicted_mean
        * SALES_SCALE
    )

    confidence_interval = (
        forecast_result.conf_int(alpha=0.05)
        * SALES_SCALE
    )

    forecast.index = test_sales.index
    confidence_interval.index = test_sales.index

    forecast_df = pd.DataFrame(
        {
            "SARIMA_Forecast": forecast,
            "Lower_95_CI": confidence_interval.iloc[:, 0],
            "Upper_95_CI": confidence_interval.iloc[:, 1],
        },
        index=test_sales.index,
    )

    return fitted_model, forecast_df


def compare_sarima_models(
    train_sales: pd.Series,
    test_sales: pd.Series,
    model_candidates: list[
        tuple[
            tuple[int, int, int],
            tuple[int, int, int, int],
        ]
    ],
) -> pd.DataFrame:

    model_results: list[
        dict[str, object]
    ] = []

    for order, seasonal_order in model_candidates:
        print(
            f"Fitting SARIMA{order}"
            f"x{seasonal_order}..."
        )

        try:
            with warnings.catch_warnings(
                record=True
            ) as captured_warnings:

                warnings.simplefilter("always")

                fitted_model = fit_sarima_model(
                    train_sales,
                    order,
                    seasonal_order,
                )

            forecast = (
                fitted_model.forecast(
                    steps=len(test_sales)
                )
                * SALES_SCALE
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
                        (
                            test_sales
                            - forecast
                        )
                        / test_sales
                    )
                )
                * 100
            )

            ljung_box_p_value = (
                calculate_ljung_box_p_value(
                    fitted_model,
                    order,
                    seasonal_order,
                )
            )

            converged = bool(
                fitted_model.mle_retvals.get(
                    "converged",
                    False,
                )
            )

            warning_messages = "; ".join(
                str(warning.message)
                for warning in captured_warnings
            )

            model_results.append(
                {
                    "Order": order,
                    "Seasonal_Order": seasonal_order,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "AIC": fitted_model.aic,
                    "BIC": fitted_model.bic,
                    "Ljung_Box_P_Value": (
                        ljung_box_p_value
                    ),
                    "Converged": converged,
                    "Warnings": warning_messages,
                }
            )

        except Exception as error:
            model_results.append(
                {
                    "Order": order,
                    "Seasonal_Order": seasonal_order,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Ljung_Box_P_Value": np.nan,
                    "Converged": False,
                    "Warnings": (
                        f"Model failed: {error}"
                    ),
                }
            )

    results_df = pd.DataFrame(
        model_results
    )

    valid_results = results_df.dropna(
        subset=["RMSE"]
    )

    if valid_results.empty:
        raise RuntimeError(
            "Every SARIMA candidate failed "
            "to produce a forecast."
        )

    return (
        results_df
        .sort_values(
            by="RMSE",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def plot_sarima_forecast(
    train_sales: pd.Series,
    test_sales: pd.Series,
    forecast_df: pd.DataFrame,
) -> None:

    recent_train_sales = (
        train_sales.iloc[-52:]
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.plot(
        recent_train_sales.index,
        recent_train_sales.values,
        label="Training Sales",
    )

    ax.plot(
        test_sales.index,
        test_sales.values,
        marker="o",
        label="Actual Test Sales",
    )

    ax.plot(
        forecast_df.index,
        forecast_df["SARIMA_Forecast"],
        marker="o",
        label="SARIMA Forecast",
    )

    ax.fill_between(
        forecast_df.index,
        forecast_df["Lower_95_CI"],
        forecast_df["Upper_95_CI"],
        alpha=0.2,
        label="95% Confidence Interval",
    )

    ax.set_title(
        "SARIMA Forecast vs. Actual "
        "Walmart Weekly Sales"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Weekly Sales")

    ax.yaxis.set_major_formatter(
        FuncFormatter(format_sales_in_millions)
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "sarima_forecast_vs_actual.png",
        dpi=300,
    )

    plt.close(fig)


def main() -> None:

    weekly_sales = load_data(
        DATA_DIR
    )

    train_sales, test_sales = (
        split_time_series(
            weekly_sales,
            test_weeks=12,
        )
    )

    d = suggest_differencing_order(
        train_sales
    )

    plot_weekly_sales_time_series(
        weekly_sales
    )

    plot_acf_chart(
        train_sales,
        d=d,
    )

    plot_pacf_chart(
        train_sales,
        d=d,
    )

    model_candidates = [
        (
            (1, d, 1),
            (1, 0, 0, SEASONAL_PERIOD),
        ),
        (
            (1, d, 2),
            (1, 0, 0, SEASONAL_PERIOD),
        ),
        (
            (2, d, 1),
            (1, 0, 0, SEASONAL_PERIOD),
        ),
        (
            (1, d, 1),
            (0, 0, 1, SEASONAL_PERIOD),
        ),
    ]

    model_comparison = (
        compare_sarima_models(
            train_sales,
            test_sales,
            model_candidates,
        )
    )

    model_comparison.to_csv(
        OUTPUT_DIR
        / "sarima_model_comparison.csv",
        index=False,
    )

    print(
        "\nSARIMA Model Comparison:"
    )

    print(
        model_comparison.to_string(
            index=False,
            formatters={
                "MAE": (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"${value:,.2f}"
                ),
                "RMSE": (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"${value:,.2f}"
                ),
                "MAPE": (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.2f}%"
                ),
                "AIC": (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"{value:,.2f}"
                ),
                "BIC": (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"{value:,.2f}"
                ),
                "Ljung_Box_P_Value": (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.4f}"
                ),
            },
        )
    )

    best_row = (
        model_comparison
        .dropna(subset=["RMSE"])
        .iloc[0]
    )

    best_order = best_row["Order"]

    best_seasonal_order = (
        best_row["Seasonal_Order"]
    )

    print(
        "\nBest SARIMA configuration "
        "based on test RMSE: "
        f"SARIMA{best_order}"
        f"x{best_seasonal_order}"
    )

    _, best_forecast_df = (
        build_sarima_forecast(
            train_sales,
            test_sales,
            best_order,
            best_seasonal_order,
        )
    )

    forecast_comparison = pd.concat(
        [
            test_sales.rename(
                "Actual_Sales"
            ),
            best_forecast_df,
        ],
        axis=1,
    )

    forecast_comparison.to_csv(
        OUTPUT_DIR
        / "sarima_forecast_comparison.csv",
        index_label="Date",
    )

    plot_sarima_forecast(
        train_sales,
        test_sales,
        best_forecast_df,
    )

    print(
        "\nBest SARIMA Forecast Comparison:"
    )

    print(
        forecast_comparison.to_string(
            float_format=(
                lambda value:
                f"{value:,.2f}"
            )
        )
    )

if __name__ == "__main__":
    main()