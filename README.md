# Walmart Weekly Sales Forecasting

This project analyzes Walmart weekly sales and compares **ARIMA** and **SARIMA** models for forecasting total sales across all stores. The goal is to identify demand patterns, measure forecast accuracy, and determine whether annual seasonality improves predictions.

## Project Workflow

1. Cleaned and converted the `Date` column to datetime format.
2. Created time features such as year, month, and week of year.
3. Analyzed monthly, holiday, and store-level sales patterns.
4. Aggregated store sales into one total weekly time series.
5. Reserved the final 12 weeks as a holdout test set.
6. Built and compared ARIMA and SARIMA forecasting models.

## Exploratory Findings

- **December** had the highest average store-level weekly sales at approximately **$1.28 million**.
- **November** ranked second at approximately **$1.15 million**.
- **January** had the lowest average at approximately **$924,000**.
- Many of the strongest sales weeks occurred around **Black Friday and Christmas**.
- Average sales varied significantly across stores.
- The recurring November–December increase followed by a January decline suggests strong annual seasonality.

### Correlation with Weekly Sales

| Variable | Correlation |
|---|---:|
| Unemployment | -0.106 |
| CPI | -0.073 |
| Temperature | -0.064 |
| Holiday Flag | 0.037 |
| Fuel Price | 0.009 |

The external variables had weak individual linear relationships with sales. This suggests that recurring seasonality, holiday timing, and store-level differences are more important than any single external variable by itself.

## ARIMA Baseline

The Augmented Dickey-Fuller test indicated that regular differencing was not required, so `d = 0`. ACF and PACF plots were used to identify candidate models.

| Model | MAE | RMSE | MAPE | Ljung-Box p-value |
|---|---:|---:|---:|---:|
| ARIMA `(1,0,1)` | **$1,378,933** | **$1,796,535** | **3.06%** | < 0.01 |
| ARIMA `(1,0,2)` | $1,588,545 | $1,892,971 | 3.51% | 0.02 |

ARIMA `(1,0,1)` was the stronger nonseasonal baseline. However, its forecasts became relatively flat, and the low Ljung-Box p-value showed that predictable time-series structure remained in the residuals.

## SARIMA Results

Because the data is weekly and displays yearly seasonality, the seasonal period was set to `52`.

| Model | MAE | RMSE | MAPE | Ljung-Box p-value |
|---|---:|---:|---:|---:|
| SARIMA `(1,0,1) x (1,0,0,52)` | **$657,878** | **$773,578** | **1.43%** | **0.5076** |
| SARIMA `(1,0,1) x (0,0,1,52)` | $791,020 | $914,409 | 1.73% | 0.0275 |
| SARIMA `(1,0,2) x (1,0,0,52)` | $833,066 | $1,088,442 | 1.82% | 0.7267 |
| SARIMA `(2,0,1) x (1,0,0,52)` | $838,253 | $1,094,020 | 1.83% | 0.5854 |

## Final Model Selection

The final model was:

```text
SARIMA(1,0,1) x (1,0,0,52)
```

It was selected because it:

- Produced the lowest holdout **MAE, RMSE, and MAPE**.
- Converged successfully without a fitting warning.
- Passed the Ljung-Box residual test with a p-value of `0.5076`.
- Captured the annual pattern using a 52-week seasonal term.

Another candidate produced lower AIC and BIC values, but its test RMSE was approximately **$320,442 higher**. Since the project focuses on forecasting unseen sales, holdout accuracy was prioritized over training fit.

## Why SARIMA Outperformed ARIMA

ARIMA models only nonseasonal relationships. SARIMA adds a seasonal component, allowing the model to use sales behavior from approximately the same week one year earlier.

Compared with the best ARIMA model, SARIMA achieved:

| Metric | ARIMA | SARIMA | Improvement |
|---|---:|---:|---:|
| MAE | $1,378,933 | **$657,878** | **52.3% lower** |
| RMSE | $1,796,535 | **$773,578** | **56.9% lower** |
| MAPE | 3.06% | **1.43%** | **53.3% lower** |

SARIMA also removed the significant residual autocorrelation left by ARIMA and followed the week-to-week movement of the test data more closely.

## Forecast vs. Actual Sales

![SARIMA Forecast vs. Actual Walmart Weekly Sales](outputs/plots/sarima_forecast_vs_actual.png)

The selected model achieved a **1.43% MAPE** on the 12-week holdout period, and all displayed actual values fell within the model's 95% forecast intervals.

## Conclusion

Walmart weekly sales show clear holiday-driven and annual seasonality. ARIMA provided a useful baseline, but SARIMA produced substantially more accurate forecasts by modeling the recurring 52-week pattern.

The selected `SARIMA(1,0,1) x (1,0,0,52)` model reduced RMSE by approximately **56.9%** and provided residuals with no significant autocorrelation at the tested lag.

## Future Work

- Refit the selected model on the full dataset and forecast future sales.
- Compare against seasonal-naive forecasts.
- Add external variables through SARIMAX.
- Evaluate models using rolling time-series cross-validation.
- Build separate forecasting models for individual stores.
