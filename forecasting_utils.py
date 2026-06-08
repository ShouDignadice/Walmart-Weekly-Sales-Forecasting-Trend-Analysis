from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

SALES_SCALE = 1_000_000

def load_weekly_sales(data_path: Path) -> pd.Series:

    sales = pd.read_csv(
        data_path,
        parse_dates=["Date"],
        dayfirst=True,
    )

    return (
        sales.groupby("Date")["Weekly_Sales"]
        .sum()
        .sort_index()
        .asfreq("W-FRI")
    )

def split_time_series(
    weekly_sales: pd.Series,
    test_weeks: int,
) -> tuple[pd.Series, pd.Series]:

    if not 0 < test_weeks < len(weekly_sales):
        raise ValueError(
            "test_weeks must be greater than 0 and smaller than the time series."
        )

    train_sales = weekly_sales.iloc[:-test_weeks].copy()
    test_sales = weekly_sales.iloc[-test_weeks:].copy()

    return train_sales, test_sales


def plot_acf_pacf(
    series: pd.Series,
    plots_dir: Path,
    file_prefix: str,
    lags: int = 52,
    seasonal_period: int | None = None,
) -> None:

    max_lags = min(
        lags,
        len(series) // 2 - 1,
    )

    if max_lags < 1:
        raise ValueError("The series is too short to create ACF and PACF charts.")

    plots_dir.mkdir(parents=True, exist_ok=True)

    chart_settings = [
        ("ACF", plot_acf, "Autocorrelation"),
        ("PACF", plot_pacf, "Partial Autocorrelation"),
    ]

    for chart_name, plot_function, y_label in chart_settings:
        fig, ax = plt.subplots(figsize=(12, 6))

        plot_kwargs = {
            "x": series,
            "lags": max_lags,
            "zero": False,
            "ax": ax,
        }

        if chart_name == "PACF":
            plot_kwargs["method"] = "ywm"

        plot_function(**plot_kwargs)

        if seasonal_period is not None and seasonal_period <= max_lags:
            ax.axvline(
                seasonal_period,
                linestyle="--",
                alpha=0.7,
                label=f"Seasonal lag: {seasonal_period}",
            )
            ax.legend()

        ax.set_title(f"{chart_name} of Training Sales")
        ax.set_xlabel("Lag")
        ax.set_ylabel(y_label)

        fig.tight_layout()
        fig.savefig(
            plots_dir / f"{file_prefix}_{chart_name.lower()}.png",
            dpi=300,
        )
        plt.close(fig)


def evaluate_forecast(
    actual_sales: pd.Series,
    forecast_sales: pd.Series,
) -> dict[str, float]:

    mae = mean_absolute_error(actual_sales, forecast_sales)
    rmse = np.sqrt(mean_squared_error(actual_sales, forecast_sales))
    mape = np.mean(
        np.abs((actual_sales - forecast_sales) / actual_sales)
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

def calculate_ljung_box_p_value(
    fitted_model: Any,
    model_df: int,
    lag: int = 10,
) -> float:

    burn_in = int(getattr(fitted_model, "loglikelihood_burn", 0))
    residuals = pd.Series(fitted_model.resid).iloc[burn_in:].dropna()

    result = acorr_ljungbox(
        residuals,
        lags=[lag],
        model_df=model_df,
        return_df=True,
    )

    return float(result["lb_pvalue"].iloc[0])

def format_sales_in_millions(value: float, _: int) -> str:

    return f"${value / SALES_SCALE:.0f}M"


def plot_forecast(
    train_sales: pd.Series,
    test_sales: pd.Series,
    forecast_sales: pd.Series,
    model_name: str,
    output_path: Path,
    lower_ci: pd.Series | None = None,
    upper_ci: pd.Series | None = None,
    recent_train_weeks: int = 52,
) -> None:

    recent_train_sales = train_sales.iloc[-recent_train_weeks:]

    fig, ax = plt.subplots(figsize=(13, 7))

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
        forecast_sales.index,
        forecast_sales.values,
        marker="o",
        label=f"{model_name} Forecast",
    )

    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            forecast_sales.index,
            lower_ci,
            upper_ci,
            alpha=0.2,
            label="95% Confidence Interval",
        )

    ax.set_title(f"{model_name} Forecast vs. Actual Walmart Weekly Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Weekly Sales")
    ax.yaxis.set_major_formatter(FuncFormatter(format_sales_in_millions))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)