## Trend Analysis

Trend analysis was performed to understand Walmart weekly sales behavior before developing forecasting models. After converting the `Date` column into datetime format, additional time-based features were created, including `Year`, `Month`, `Month_Name`, and `Week_Of_Year`.

Sales were analyzed by week, month, holiday period, year, and store to identify recurring seasonal patterns, demand spikes, and differences in store performance.

### Monthly and Seasonal Patterns

The analysis showed that Walmart sales follow a noticeable seasonal pattern, with the strongest sales occurring during the holiday shopping period.

Across the store-level weekly observations:

* December had the highest average weekly sales at approximately **$1.28 million**.
* November had the second-highest average weekly sales at approximately **$1.15 million**.
* January had the lowest average weekly sales at approximately **$924,000**.

The sharp increase in November and December, followed by a decline in January, indicates a recurring holiday-driven sales cycle.

Holiday weeks also produced higher average weekly sales than non-holiday weeks. Many of the highest sales weeks occurred around Black Friday and Christmas, demonstrating the importance of holiday demand in Walmart’s sales performance.

### Store-Level Differences

Average weekly sales varied significantly across stores. This indicates that sales patterns are influenced not only by seasonality and holiday periods, but also by differences in store size, location, customer demand, and overall store performance.

Because store-level behavior varies, the initial forecasting workflow focuses on total weekly sales across all stores. Individual store forecasting can be evaluated separately in a later phase.

### Correlation Analysis

Correlation analysis was performed to measure the linear relationship between `Weekly_Sales` and the available external variables.

| Variable     | Correlation with Weekly Sales |
| ------------ | ----------------------------: |
| Unemployment |                        -0.106 |
| CPI          |                        -0.073 |
| Temperature  |                        -0.064 |
| Holiday Flag |                         0.037 |
| Fuel Price   |                         0.009 |

The external variables showed weak linear correlations with weekly sales.

Unemployment had the strongest negative correlation at approximately `-0.106`, while `Holiday_Flag` had a weak positive correlation of approximately `0.037`. These results suggest that no individual external variable strongly explains weekly sales by itself.

The weak correlation for `Holiday_Flag` does not mean that holidays are unimportant. Correlation measures only a single overall linear relationship, while holiday effects are concentrated around specific events such as Black Friday and Christmas.

Overall, the exploratory analysis suggests that Walmart weekly sales are driven more strongly by recurring seasonality, holiday demand, and store-level differences than by any single external variable.

## Key Findings

* December had the highest average store-level weekly sales at approximately **$1.28 million**.
* January had the lowest average store-level weekly sales at approximately **$924,000**.
* Holiday weeks produced higher average sales than non-holiday weeks.
* Many of the highest sales weeks occurred around Black Friday and Christmas.
* Average weekly sales varied significantly across stores.
* December 2010 produced slightly higher total sales than December 2011 despite higher unemployment and lower CPI.
* The external variables had weak individual linear relationships with weekly sales.
* The recurring November and December increases followed by a January decline provide strong evidence of annual seasonality.

## ARIMA Model Identification

The store-level sales observations were aggregated by date to create a single weekly time series representing total Walmart sales across all stores.

The final 12 weeks were reserved as a holdout test set. All model identification and training were performed using the earlier observations to prevent information from the test period from influencing the model.

The Augmented Dickey-Fuller test was used to evaluate stationarity and suggested that nonseasonal differencing was not required, resulting in:

```text
d = 0
```

ACF and PACF plots were then used to identify possible autoregressive and moving-average terms. Based on these diagnostics, two nonseasonal ARIMA candidates were evaluated:

* ARIMA `(1,0,1)`
* ARIMA `(1,0,2)`

## ARIMA Model Results

Both models were trained on the same training period and evaluated on the same 12-week holdout period using:

* Mean Absolute Error
* Root Mean Squared Error
* Mean Absolute Percentage Error
* Akaike Information Criterion
* Bayesian Information Criterion
* Ljung–Box residual test

| Model           |        MAE |       RMSE |  MAPE |     AIC |     BIC | Ljung–Box p-value |
| --------------- | ---------: | ---------: | ----: | ------: | ------: | ----------------: |
| ARIMA `(1,0,1)` | $1,378,933 | $1,796,535 | 3.06% | 4435.67 | 4447.17 |            < 0.01 |
| ARIMA `(1,0,2)` | $1,588,545 | $1,892,971 | 3.51% | 4426.04 | 4440.42 |              0.02 |

ARIMA `(1,0,1)` produced the strongest out-of-sample forecasting performance. Its forecasts were approximately **3.06% away from actual weekly sales on average**, and it achieved lower MAE and RMSE than ARIMA `(1,0,2)`.

Although ARIMA `(1,0,2)` produced lower AIC and BIC values, indicating a better fit to the training data after accounting for model complexity, it performed worse on the unseen test period.

Because the primary objective of the project is forecasting future sales, out-of-sample MAE, RMSE, and MAPE were prioritized over training fit.

## ARIMA Limitations

The ARIMA `(1,0,1)` forecasts gradually converged toward approximately **$47.2 million in total weekly sales**. This resulted in a relatively flat forecast that did not capture several sharp increases and decreases in the test period.

The Ljung–Box p-values for both models were below the `0.05` threshold. This indicates that statistically significant autocorrelation remained in the residuals at the tested lag.

In other words, the ARIMA models left predictable time-series structure unexplained.

This result is consistent with the earlier trend analysis. Walmart sales contain recurring holiday and annual patterns, while a standard ARIMA model only represents nonseasonal autoregressive, differencing, and moving-average relationships.

Therefore, ARIMA `(1,0,1)` was retained as the strongest **nonseasonal baseline model**, but it was not considered the final forecasting solution.

## Next Phase: SARIMA

The trend analysis, flat ARIMA forecasts, and remaining residual autocorrelation all support evaluating a seasonal ARIMA model.

SARIMA extends ARIMA by adding seasonal parameters:

```text
ARIMA order:       (p, d, q)
Seasonal order:    (P, D, Q, s)
```

Because the dataset contains weekly observations and the identified pattern repeats annually, an initial seasonal period of approximately:

```text
s = 52
```

will be evaluated.

The SARIMA models will be trained and tested using the same holdout period and evaluation metrics as the ARIMA models. Their performance will be compared based on forecast accuracy, residual behavior, and their ability to reproduce recurring seasonal sales movements.
