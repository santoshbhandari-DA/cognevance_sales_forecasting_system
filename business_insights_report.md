# Sales Forecasting - Business Insights Report

## 1. Dataset Overview
- Records analysed: **4,565,000**
- Period: **2019-01-01** to **2023-12-31**
- Stores: **50** | Items: **50**
- Total sales: **133,593,174**

## 2. Historical Trends
- Sales changed **5.0%** from 2022 to 2023.
- Strongest month on average: **Mar**; weakest: **Sep**.
- Best weekday: **Wednesday**; weakest weekday: **Saturday**.
- Top store: **store_12** | Top item: **item_20**.
- Promotions lift average sales per row by about **50.2%** compared to non-promo days.
- Price vs sales correlation is **-0.06** - price shows only a weak direct relationship with sales.

## 3. Forecasting Approach
- Data aggregated to daily total sales.
- Features: price, promo share, weekday, month, lag-1/7/14 sales, 7-day rolling average.
- Models: Linear Regression (main), same-day-last-week baseline, Holt-Winters (if available).
- Time-based split: first 80% train, last 20% test (no shuffling).

| Model | MAE | RMSE | R2 | MAPE % |
|---|---|---|---|---|
| Linear Regression | 436.82 | 555.37 | 0.999 | 0.54 |
| Baseline (same day last week) | 1877.69 | 2131.49 | 0.989 | 2.39 |
| Holt-Winters (Time Series) | 66369.02 | 84154.76 | -16.488 | 99.83 |

Best model by MAPE: **Linear Regression**.

## 4. Future Sales Outlook (next 30 days)
- Forecast total: **2,674,092** vs last 30 days actual: **2,275,904** (+17.5%).
- Expected peak day: **2024-01-24** (108,400 units).
- Assumption: price and promo share stay at their last-30-day averages.

## 5. Recommendations
1. Stock up and staff more before **Wednesdays** and in **Mar**.
2. Use promotions in slow periods (**Sep**, **Saturdays**) to flatten demand.
3. Focus inventory planning on top store **store_12** and top item **item_20**.
4. Re-train the model regularly (monthly) with new data to keep accuracy high.
5. Extend the model per store/item for more granular replenishment decisions.

## 6. Charts
See the `charts/` folder (monthly trend, yearly sales, seasonality, weekday, promo, price, actual vs predicted, forecast, dashboard).
