"""
Cognevance - Level 2: Sales Forecasting System
Dataset columns: date, store_id, item_id, sales, price, promo, weekday, month
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LAGS = [1, 7, 14]


# ---------------------------------------------------------------- 1. LOAD + CLEAN
def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["date", "store_id", "item_id", "sales", "price", "promo"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    print("=" * 60)
    print("RAW DATA")
    print("=" * 60)
    print("Shape:", df.shape)
    print(df.head(), "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")

    rows_before = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    df = df.drop_duplicates()

    for col in ["sales", "price", "promo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["sales"])
    df = df[df["sales"] >= 0]

    df["price"] = df.groupby("item_id")["price"].transform(lambda s: s.fillna(s.median()))
    df["price"] = df["price"].fillna(df["price"].median())
    df["promo"] = df["promo"].fillna(0).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek  # 0 = Monday
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    print(f"Rows before cleaning: {rows_before} | after cleaning: {len(df)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Stores: {df['store_id'].nunique()} | Items: {df['item_id'].nunique()}\n")
    return df


# ---------------------------------------------------------------- 2. EDA
def run_eda(df, chart_dir):
    insights = {}

    monthly = df.groupby("year_month")["sales"].sum()
    plt.figure(figsize=(11, 4.5))
    plt.plot(monthly.index, monthly.values, marker="o", color="#1f77b4")
    plt.title("Monthly Total Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/01_monthly_sales_trend.png", dpi=150)
    plt.close()

    yearly = df.groupby("year")["sales"].sum()
    plt.figure(figsize=(7, 4.5))
    plt.bar(yearly.index.astype(str), yearly.values, color="#2ca02c")
    plt.title("Yearly Total Sales")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/02_yearly_sales.png", dpi=150)
    plt.close()

    month_avg = df.groupby(["year", "month"])["sales"].sum().groupby("month").mean()
    plt.figure(figsize=(9, 4.5))
    plt.bar([MONTH_NAMES[m - 1] for m in month_avg.index], month_avg.values, color="#ff7f0e")
    plt.title("Average Sales by Calendar Month (Seasonality)")
    plt.ylabel("Avg monthly sales")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/03_monthly_seasonality.png", dpi=150)
    plt.close()

    wk = df.groupby("weekday")["sales"].mean().reindex(range(7))
    plt.figure(figsize=(9, 4.5))
    plt.bar([d[:3] for d in DAY_NAMES], wk.values, color="#9467bd")
    plt.title("Average Sales per Row by Weekday")
    plt.ylabel("Avg sales")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/04_weekday_sales.png", dpi=150)
    plt.close()

    promo_avg = df.groupby("promo")["sales"].mean()
    plt.figure(figsize=(5.5, 4.5))
    labels = ["No Promo" if p == 0 else "Promo" for p in promo_avg.index]
    plt.bar(labels, promo_avg.values, color=["#7f7f7f", "#d62728"][: len(labels)])
    plt.title("Average Sales: Promo vs No Promo")
    plt.ylabel("Avg sales")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/05_promo_impact.png", dpi=150)
    plt.close()

    sample = df.sample(min(len(df), 5000), random_state=42)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(sample["price"], sample["sales"], alpha=0.3, s=10, color="#17becf")
    plt.title("Price vs Sales")
    plt.xlabel("Price")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/06_price_vs_sales.png", dpi=150)
    plt.close()

    top_stores = df.groupby("store_id")["sales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4.5))
    plt.bar(top_stores.index.astype(str), top_stores.values, color="#8c564b")
    plt.title("Top Stores by Total Sales")
    plt.xlabel("Store ID")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/07_top_stores.png", dpi=150)
    plt.close()

    top_items = df.groupby("item_id")["sales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4.5))
    plt.bar(top_items.index.astype(str), top_items.values, color="#e377c2")
    plt.title("Top Items by Total Sales")
    plt.xlabel("Item ID")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/08_top_items.png", dpi=150)
    plt.close()

    corr = df[["sales", "price", "promo", "weekday", "month"]].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/09_correlation.png", dpi=150)
    plt.close()

    insights["monthly"] = monthly
    insights["yearly"] = yearly
    insights["peak_month_name"] = MONTH_NAMES[int(month_avg.idxmax()) - 1]
    insights["low_month_name"] = MONTH_NAMES[int(month_avg.idxmin()) - 1]
    insights["best_day"] = DAY_NAMES[int(wk.idxmax())]
    insights["worst_day"] = DAY_NAMES[int(wk.idxmin())]
    if 0 in promo_avg.index and 1 in promo_avg.index and promo_avg[0] > 0:
        insights["promo_uplift_pct"] = (promo_avg[1] / promo_avg[0] - 1) * 100
    else:
        insights["promo_uplift_pct"] = np.nan
    insights["price_corr"] = df["price"].corr(df["sales"])
    insights["top_store"] = top_stores.index[0]
    insights["top_item"] = top_items.index[0]
    if len(yearly) >= 2:
        insights["yoy_pct"] = (yearly.iloc[-1] / yearly.iloc[-2] - 1) * 100
        insights["yoy_years"] = (int(yearly.index[-2]), int(yearly.index[-1]))
    else:
        insights["yoy_pct"] = np.nan
        insights["yoy_years"] = None
    return insights


# ---------------------------------------------------------------- 3. DAILY SERIES + FEATURES
def make_daily(df):
    daily = df.groupby("date").agg(
        sales=("sales", "sum"),
        price=("price", "mean"),
        promo=("promo", "mean"),
    )
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily["sales"] = daily["sales"].interpolate().bfill()
    daily["price"] = daily["price"].interpolate().bfill()
    daily["promo"] = daily["promo"].fillna(0)
    daily.index.name = "date"
    return daily


def build_features(daily):
    X = pd.DataFrame(index=daily.index)
    X["price"] = daily["price"]
    X["promo"] = daily["promo"]
    for lag in LAGS:
        X[f"lag_{lag}"] = daily["sales"].shift(lag)
    X["roll_mean_7"] = daily["sales"].shift(1).rolling(7).mean()
    dow = pd.get_dummies(daily.index.dayofweek, prefix="dow").astype(int)
    dow.index = daily.index
    mon = pd.get_dummies(daily.index.month, prefix="mon").astype(int)
    mon.index = daily.index
    return pd.concat([X, dow, mon], axis=1)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


# ---------------------------------------------------------------- 4. MODEL
def train_model(daily, chart_dir, data_dir):
    X = build_features(daily)
    data = pd.concat([daily["sales"].rename("y"), X], axis=1).dropna()
    y = data.pop("y")
    X = data

    split = int(len(X) * 0.8)  # time based split, no shuffling
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    baseline = X_test["lag_7"]
    res = {
        "Linear Regression": metrics(y_test.values, pred),
        "Baseline (same day last week)": metrics(y_test.values, baseline.values),
    }

    hw_pred = None
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        train_series = daily["sales"].loc[y_train.index.min():y_train.index.max()]
        hw = ExponentialSmoothing(
            train_series, trend="add", seasonal="add", seasonal_periods=7
        ).fit()
        hw_pred = hw.forecast(len(y_test)).values
        res["Holt-Winters (Time Series)"] = metrics(y_test.values, hw_pred)
    except Exception as e:
        print("Holt-Winters skipped:", e)

    results_df = pd.DataFrame(res).T
    print("=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)
    print(results_df.round(3), "\n")
    results_df.round(3).to_csv(f"{data_dir}/model_metrics.csv")

    out = pd.DataFrame({"actual": y_test, "predicted_lr": pred})
    if hw_pred is not None:
        out["predicted_holt_winters"] = hw_pred
    out.to_csv(f"{data_dir}/test_predictions.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(y_train.index[-90:], y_train.values[-90:], color="#999999", label="Train (last 90 days)")
    plt.plot(y_test.index, y_test.values, color="#1f77b4", label="Actual (test)")
    plt.plot(y_test.index, pred, color="#d62728", linestyle="--", label="Linear Regression")
    if hw_pred is not None:
        plt.plot(y_test.index, hw_pred, color="#2ca02c", linestyle=":", label="Holt-Winters")
    plt.title("Actual vs Predicted Sales (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Daily total sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/10_actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, pred, alpha=0.5, color="#d62728")
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (scatter)")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/11_actual_vs_predicted_scatter.png", dpi=150)
    plt.close()

    final_model = LinearRegression().fit(X, y)
    return final_model, X.columns, res, out


# ---------------------------------------------------------------- 5. FUTURE FORECAST
def forecast_future(model, columns, daily, horizon, data_dir):
    history = list(daily["sales"].values)
    last_date = daily.index[-1]
    price_assume = daily["price"].tail(30).mean()
    promo_assume = daily["promo"].tail(30).mean()

    rows = []
    for i in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=i)
        feat = {"price": price_assume, "promo": promo_assume}
        for lag in LAGS:
            feat[f"lag_{lag}"] = history[-lag]
        feat["roll_mean_7"] = np.mean(history[-7:])
        feat[f"dow_{d.dayofweek}"] = 1
        feat[f"mon_{d.month}"] = 1
        x = pd.DataFrame([feat]).reindex(columns=columns, fill_value=0)
        yhat = max(float(model.predict(x)[0]), 0)
        history.append(yhat)
        rows.append((d, yhat))

    fc = pd.DataFrame(rows, columns=["date", "forecast_sales"]).set_index("date")
    fc.to_csv(f"{data_dir}/future_forecast.csv")
    return fc


# ---------------------------------------------------------------- 6. DASHBOARD
def make_dashboard(df, daily, fc, test_out, insights, chart_dir):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Sales Forecasting Dashboard", fontsize=18, fontweight="bold")

    m = insights["monthly"]
    ax[0, 0].plot(m.index, m.values, marker="o", color="#1f77b4")
    ax[0, 0].set_title("Monthly Sales Trend")

    y = insights["yearly"]
    ax[0, 1].bar(y.index.astype(str), y.values, color="#2ca02c")
    ax[0, 1].set_title("Yearly Sales")

    wk = df.groupby("weekday")["sales"].mean().reindex(range(7))
    ax[0, 2].bar([d[:3] for d in DAY_NAMES], wk.values, color="#9467bd")
    ax[0, 2].set_title("Avg Sales by Weekday")

    pa = df.groupby("promo")["sales"].mean()
    ax[1, 0].bar(["No Promo" if p == 0 else "Promo" for p in pa.index], pa.values,
                 color=["#7f7f7f", "#d62728"][: len(pa)])
    ax[1, 0].set_title("Promo vs No Promo (avg sales)")

    ax[1, 1].plot(test_out.index, test_out["actual"], label="Actual", color="#1f77b4")
    ax[1, 1].plot(test_out.index, test_out["predicted_lr"], "--", label="Predicted", color="#d62728")
    ax[1, 1].set_title("Actual vs Predicted (Test)")
    ax[1, 1].legend()

    tail = daily["sales"].tail(90)
    ax[1, 2].plot(tail.index, tail.values, label="History (last 90d)", color="#555555")
    ax[1, 2].plot(fc.index, fc["forecast_sales"], label="Forecast", color="#ff7f0e", linewidth=2)
    ax[1, 2].axvline(daily.index[-1], color="k", linestyle=":")
    ax[1, 2].set_title(f"Next {len(fc)} Days Forecast")
    ax[1, 2].legend()

    for a in ax.flat:
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{chart_dir}/12_dashboard.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(tail.index, tail.values, label="History (last 90 days)", color="#555555")
    plt.plot(fc.index, fc["forecast_sales"], label="Forecast", color="#ff7f0e", linewidth=2)
    plt.axvline(daily.index[-1], color="k", linestyle=":")
    plt.title(f"Future Sales Forecast - Next {len(fc)} Days")
    plt.xlabel("Date")
    plt.ylabel("Daily total sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/13_future_forecast.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- 7. BUSINESS REPORT
def write_report(df, daily, fc, results, insights, report_dir):
    horizon = len(fc)
    last_period = daily["sales"].tail(horizon).sum()
    next_period = fc["forecast_sales"].sum()
    growth = (next_period / last_period - 1) * 100 if last_period else np.nan
    peak_day = fc["forecast_sales"].idxmax()
    best_model = min(results, key=lambda k: results[k]["MAPE_%"])

    promo_txt = (
        f"Promotions lift average sales per row by about **{insights['promo_uplift_pct']:.1f}%** "
        "compared to non-promo days."
        if not np.isnan(insights["promo_uplift_pct"])
        else "Promo impact could not be measured (no promo/non-promo split)."
    )
    yoy_txt = ""
    if insights["yoy_years"]:
        a, b = insights["yoy_years"]
        yoy_txt = f"- Sales changed **{insights['yoy_pct']:.1f}%** from {a} to {b}.\n"

    corr = insights["price_corr"]
    price_txt = (
        "higher prices are associated with lower sales" if corr < -0.1
        else "higher prices are associated with higher sales (likely premium items)" if corr > 0.1
        else "price shows only a weak direct relationship with sales"
    )

    metric_rows = "\n".join(
        f"| {name} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.3f} | {m['MAPE_%']:.2f} |"
        for name, m in results.items()
    )

    report = f"""# Sales Forecasting - Business Insights Report

## 1. Dataset Overview
- Records analysed: **{len(df):,}**
- Period: **{df['date'].min().date()}** to **{df['date'].max().date()}**
- Stores: **{df['store_id'].nunique()}** | Items: **{df['item_id'].nunique()}**
- Total sales: **{df['sales'].sum():,.0f}**

## 2. Historical Trends
{yoy_txt}- Strongest month on average: **{insights['peak_month_name']}**; weakest: **{insights['low_month_name']}**.
- Best weekday: **{insights['best_day']}**; weakest weekday: **{insights['worst_day']}**.
- Top store: **{insights['top_store']}** | Top item: **{insights['top_item']}**.
- {promo_txt}
- Price vs sales correlation is **{corr:.2f}** - {price_txt}.

## 3. Forecasting Approach
- Data aggregated to daily total sales.
- Features: price, promo share, weekday, month, lag-1/7/14 sales, 7-day rolling average.
- Models: Linear Regression (main), same-day-last-week baseline, Holt-Winters (if available).
- Time-based split: first 80% train, last 20% test (no shuffling).

| Model | MAE | RMSE | R2 | MAPE % |
|---|---|---|---|---|
{metric_rows}

Best model by MAPE: **{best_model}**.

## 4. Future Sales Outlook (next {horizon} days)
- Forecast total: **{next_period:,.0f}** vs last {horizon} days actual: **{last_period:,.0f}** ({growth:+.1f}%).
- Expected peak day: **{peak_day.date()}** ({fc['forecast_sales'].max():,.0f} units).
- Assumption: price and promo share stay at their last-30-day averages.

## 5. Recommendations
1. Stock up and staff more before **{insights['best_day']}s** and in **{insights['peak_month_name']}**.
2. Use promotions in slow periods (**{insights['low_month_name']}**, **{insights['worst_day']}s**) to flatten demand.
3. Focus inventory planning on top store **{insights['top_store']}** and top item **{insights['top_item']}**.
4. Re-train the model regularly (monthly) with new data to keep accuracy high.
5. Extend the model per store/item for more granular replenishment decisions.

## 6. Charts
See the `charts/` folder (monthly trend, yearly sales, seasonality, weekday, promo, price, actual vs predicted, forecast, dashboard).
"""
    with open(f"{report_dir}/business_insights_report.md", "w", encoding="utf-8") as f:
        f.write(report)


# ---------------------------------------------------------------- MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sales.csv", help="path to raw csv")
    parser.add_argument("--horizon", type=int, default=30, help="days to forecast")
    args = parser.parse_args()

    chart_dir, data_dir, report_dir = "charts", "data", "report"
    for d in (chart_dir, data_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    df = load_and_clean(args.data)
    df.to_csv(f"{data_dir}/cleaned_sales.csv", index=False)

    insights = run_eda(df, chart_dir)
    daily = make_daily(df)
    model, cols, results, test_out = train_model(daily, chart_dir, data_dir)
    fc = forecast_future(model, cols, daily, args.horizon, data_dir)
    make_dashboard(df, daily, fc, test_out, insights, chart_dir)
    write_report(df, daily, fc, results, insights, report_dir)

    print("DONE. Check charts/, report/ and data/ folders.")


if __name__ == "__main__":
    main()