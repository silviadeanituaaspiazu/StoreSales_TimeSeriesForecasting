# Linear Regression vs XGBoost

During the experimentation phase, an interesting and counterintuitive insight emerged. A simple Linear Regression model outperformed the advanced XGBoost algorithm, even though it used only 5 features as regressors compared to the 12 provided to XGBoost.

## Model 1: Linear Regression (Baseline)
- **Features (5):** `sales_lag15`, `sales_lag30`, `dcoilwtico`, `store_nbr`, `family`.
- **Result:** Superior predictive performance.

## Model 2: XGBoost (Advanced Model)
- **Features (12):** `sales_lag15`, `sales_lag30`, `sales_lag365`, `dcoilwtico`, `store_nbr`, `store_cluster`, `family`, `month`, `weekday`, `is_holiday`, `city`, `onpromotion`.
- **Result:** Underperformed compared to the baseline.

Several reasons can explain this behavior in the context of time-series forecasting:

1. **The prediction horizon and linear stability:** Linear regression naturally captures and projects global trends stably into the future. 
2. **The incapacity of trees to extrapolate:** Tree-based models cannot predict values outside the range they observed during training. If sales are trending upward, XGBoost hits a "glass ceiling," whereas a linear model simply extends the line.
3. **High dimensionality causing overfitting:** Feeding too many variables (especially high-cardinality categorical features) without explicit time-series structuring causes XGBoost to memorize noise rather than learning generalizable patterns.
