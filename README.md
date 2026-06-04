## Trend Analysis

Trend analysis was performed to understand Walmart weekly sales behavior before building forecasting models. After converting the Date column into datetime format, additional time-based features were created, including Year, Month, Month_Name, and Week_Of_Year. Sales were then analyzed by week, month, holiday period, and store to identify seasonal patterns and demand spikes.

The analysis showed that Walmart sales peak during the holiday season, especially in November and December. December had the highest average weekly sales at approximately \$1.28M, followed by November at approximately \$1.15M. January had the lowest average weekly sales at approximately \$924K, suggesting a post-holiday sales decline.

Holiday weeks also showed higher average weekly sales than non-holiday weeks, and the top sales weeks were largely concentrated around Black Friday and Christmas. This suggests that holiday shopping periods play a major role in Walmart sales performance.

Store-level analysis showed that average weekly sales vary significantly across locations. This indicates that sales patterns are influenced not only by seasonality and holidays, but also by store-level differences.

Correlation analysis was also performed to evaluate the relationship between weekly sales and external factors such as Holiday_Flag, Temperature, Fuel_Price, CPI, and Unemployment. The results showed weak linear correlations overall, with unemployment having the strongest negative correlation at approximately -0.106 and holiday flag having a weak positive correlation at approximately 0.037. This suggests that individual external factors had limited impact on sales by themselves, while seasonality, holiday demand, and store-level differences appeared to be stronger drivers of sales patterns.

Overall, the trend analysis suggests that Walmart weekly sales are driven mainly by seasonal demand, holiday spikes, and store performance differences rather than any single external factor. These findings support the next phase of the project, where time series forecasting models such as ARIMA and SARIMA will be used to predict future sales patterns.

## Key Findings
* December had the highest average weekly sales at approximately $1.28M.
* January had the lowest average weekly sales at approximately $924K.
* Holiday weeks had higher average weekly sales than non-holiday weeks.
* The highest sales weeks were concentrated around Black Friday and Christmas.
* Store-level analysis showed significant differences in average weekly sales across locations.
* December 2010 had slightly higher total sales than December 2011, despite having higher unemployment and lower CPI.
