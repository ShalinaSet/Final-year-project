import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
np.random.seed(42)

# 1. LOAD DATA

conn = sqlite3.connect("pub_sales.db")
df = pd.read_sql("SELECT * FROM daily_sales", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# 2. DATA QUALITY CHECKS (Explicitly validate the dataset before any modelling which is important even for synthetic data to demonstrate rigour)

print("DATA QUALITY REPORT")

# 2a. Shape and date range
print(f"Total rows       : {len(df)}")
print(f"Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Columns          : {list(df.columns)}")

# 2b. Missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing.to_string())
if missing.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values detected — will be forward-filled.")

# 2c. Duplicate dates
duplicate_dates = df[df.duplicated(subset=["date"], keep=False)]
print(f"\nDuplicate dates  : {len(duplicate_dates)}")
if len(duplicate_dates) == 0:
    print("No duplicate dates found.")
else:
    print("Duplicate dates detected:")
    print(duplicate_dates[["date", "weekday", "sales_gbp"]])

# 2d. Negative or zero sales
invalid_sales = df[df["sales_gbp"] <= 0]
print(f"\nNegative/zero sales rows: {len(invalid_sales)}")
if len(invalid_sales) == 0:
    print("All sales values are positive.")
else:
    print("Invalid sales values detected:")
    print(invalid_sales[["date", "weekday", "sales_gbp"]])

# 2e. Outlier detection using IQR method
Q1 = df["sales_gbp"].quantile(0.25)
Q3 = df["sales_gbp"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["sales_gbp"] < lower_bound) | (df["sales_gbp"] > upper_bound)]
print(f"\nOutlier detection (IQR method):")
print(f"  Lower bound : £{lower_bound:,.2f}")
print(f"  Upper bound : £{upper_bound:,.2f}")
print(f"  Outliers    : {len(outliers)} rows")
if len(outliers) > 0:
    print("  Outlier dates (likely event days — expected):")
    print(outliers[["date", "weekday", "sales_gbp"]].to_string(index=False))

# 2f. Sales summary statistics
print(f"\nSales summary statistics:")
print(df["sales_gbp"].describe().apply(lambda x: f"£{x:,.2f}").to_string())

# 2g. Weekday distribution
print(f"\nRows per weekday:")
print(df["weekday"].value_counts().sort_index().to_string())

# 2h. Weather distribution
print(f"\nWeather distribution:")
print(df["weather"].value_counts().to_string())

print("\n✓ Data quality check complete.")
print("─" * 45)

# 3. ENSURE COMPLETE DATE RANGE

full_dates = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
df = df.set_index("date").reindex(full_dates)
df.index.name = "date"

df["weekday"]      = df.index.day_name()
df["month"]        = df.index.month
df["weather"]      = df["weather"].fillna("Cloudy")
df["reservations"] = df["reservations"].ffill()
df["sales_gbp"]    = df["sales_gbp"].ffill()

df = df.reset_index().sort_values("date").reset_index(drop=True)


# 4. FEATURE ENGINEERING

# Calendar features
df["is_weekend"] = df["weekday"].isin(["Friday", "Saturday"]).astype(int)
df["is_sunday"]  = (df["weekday"] == "Sunday").astype(int)

event_dates = ["2025-01-01", "2025-10-31", "2025-12-24", "2025-12-25", "2025-12-26"]
df["is_event"] = df["date"].astype(str).isin(event_dates).astype(int)

# Binary weather indicators
df["is_rainy"] = (df["weather"] == "Rainy").astype(int)
df["is_sunny"] = (df["weather"] == "Sunny").astype(int)
df["is_cold"]  = (df["weather"] == "Cold").astype(int)
# Cloudy is the baseline

# Interaction term: rainy on a weekend hurts more than rainy on a quiet day
df["rainy_weekend"] = df["is_rainy"] * df["is_weekend"]

print("FEATURE SUMMARY")
feature_cols = ["is_weekend", "is_sunday", "is_event",
                "is_rainy", "is_sunny", "is_cold", "rainy_weekend"]
print(df[feature_cols].sum().to_string())
print("Features engineered: {feature_cols}")
print("─" * 45)

# 5. TRAIN / TEST SPLIT (TIME-AWARE 80/20)

df = df.set_index("date")
target = df["sales_gbp"].astype(float)

exog_cols = ["is_weekend", "is_sunday", "is_event",
             "is_rainy", "is_sunny", "is_cold", "rainy_weekend"]
exog = df[exog_cols].astype(float)

split = int(len(df) * 0.8)

train_y    = target.iloc[:split]
test_y     = target.iloc[split:]
train_exog = exog.iloc[:split].copy()
test_exog  = exog.iloc[split:].copy()

# Scale event flag to reduce Christmas spike influence
train_exog["is_event"] = train_exog["is_event"] * 0.5
test_exog["is_event"]  = test_exog["is_event"]  * 0.5

print(f"Train size : {len(train_y)} days  ({train_y.index[0].date()} → {train_y.index[-1].date()})")
print(f"Test size  : {len(test_y)} days  ({test_y.index[0].date()} → {test_y.index[-1].date()})")
print("─" * 45)

# 6. METRICS

def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

results = []

# MODEL 1 — Rolling Moving Average (Baseline)

window = 7
ma_preds = []
history = list(train_y)

for t in range(len(test_y)):
    yhat = np.mean(history[-window:])
    ma_preds.append(yhat)
    history.append(test_y.iloc[t])

results.append(("Moving Average (7-day)", mape(test_y, ma_preds), rmse(test_y, ma_preds)))
print("✓ Moving Average complete")

# MODEL 2 — Exponential Smoothing (Holt-Winters)

exp_model = ExponentialSmoothing(
    train_y,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    damped_trend=True
).fit(optimized=True)

exp_preds = exp_model.forecast(len(test_y))
results.append(("Exponential Smoothing", mape(test_y, exp_preds), rmse(test_y, exp_preds)))
print("✓ Exponential Smoothing complete")

# MODEL 3 — SARIMA

sarima_model = SARIMAX(
    train_y,
    order=(1, 1, 0),
    seasonal_order=(1, 1, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarima_preds = sarima_model.forecast(len(test_y))
results.append(("SARIMA (1,1,0)(1,1,0,7)", mape(test_y, sarima_preds), rmse(test_y, sarima_preds)))
print("SARIMA complete")

# MODEL 4 — SARIMAX

sarimax_model = SARIMAX(
    train_y,
    exog=train_exog,
    order=(1, 1, 0),
    seasonal_order=(1, 1, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarimax_preds = sarimax_model.forecast(steps=len(test_y), exog=test_exog)
results.append(("SARIMAX (1,1,0)(1,1,0,7)", mape(test_y, sarimax_preds), rmse(test_y, sarimax_preds)))
print("SARIMAX complete")

# 7. RESULTS TABLE

results_df = pd.DataFrame(
    results, columns=["Model", "MAPE (%)", "RMSE (£)"]
).sort_values("MAPE (%)")

print("MODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv("forecast_model_comparison.csv", index=False)
print("\nSaved: forecast_model_comparison.csv")

# 8. FORECAST vs ACTUAL PLOT

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test_y.index, test_y.values,       label="Actual",                color="steelblue", linewidth=2)
ax.plot(test_y.index, ma_preds,             label="Moving Average",        color="grey",      linewidth=1.2, linestyle="--")
ax.plot(test_y.index, exp_preds.values,     label="Exponential Smoothing", color="orange",    linewidth=1.2)
ax.plot(test_y.index, sarima_preds.values,  label="SARIMA",                color="green",     linewidth=1.2)
ax.plot(test_y.index, sarimax_preds.values, label="SARIMAX (Best)",        color="red",       linewidth=1.5)

ax.set_title("Demand Forecast Comparison — Test Period (Oct–Dec 2025)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Sales (£)")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
plt.savefig("forecast_comparison.png", dpi=150)
plt.show()
print("Saved: forecast_comparison.png")

# 9. RESIDUAL DIAGNOSTICS (SARIMAX)

residuals = test_y.values - np.array(sarimax_preds)

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(range(len(residuals)), residuals, alpha=0.6, color="steelblue", s=30)
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_title("SARIMAX Residuals Over Test Period")
axes[0].set_xlabel("Observation Index (test days)")
axes[0].set_ylabel("Residuals (£)")
axes[0].grid(True, alpha=0.4)

axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_title("Residual Distribution (SARIMAX)")
axes[1].set_xlabel("Residual (£)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.4)

fig2.tight_layout()
plt.savefig("residual_diagnostics.png", dpi=150)
plt.show()
print("Saved: residual_diagnostics.png")

# 10. FORECAST 2026 FOR FRONT-END

print("\nGenerating 2026 forecast...")

dates_2026 = pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")
df_2026 = pd.DataFrame({"date": dates_2026})
df_2026["weekday"] = df_2026["date"].dt.day_name()
df_2026["month"]   = df_2026["date"].dt.month

df_2026["is_weekend"] = df_2026["weekday"].isin(["Friday", "Saturday"]).astype(int)
df_2026["is_sunday"]  = (df_2026["weekday"] == "Sunday").astype(int)
df_2026["is_event"]   = df_2026["date"].astype(str).isin([
    "2026-01-01", "2026-10-31", "2026-12-24", "2026-12-25", "2026-12-26"
]).astype(int) * 0.5

# Assign random weather for 2026 based on season
def get_season(month):
    if month in [12, 1, 2]: return "winter"
    elif month in [3, 4, 5]: return "spring"
    elif month in [6, 7, 8]: return "summer"
    else: return "autumn"

weather_options = ["Sunny", "Cloudy", "Rainy", "Cold"]
weather_probs = {
    "winter": [0.15, 0.40, 0.30, 0.15],
    "spring": [0.50, 0.25, 0.15, 0.10],
    "summer": [0.45, 0.25, 0.20, 0.10],
    "autumn": [0.25, 0.30, 0.30, 0.15]
}

df_2026["season"]  = df_2026["month"].apply(get_season)
df_2026["weather"] = df_2026.apply(
    lambda row: np.random.choice(weather_options, p=weather_probs[row["season"]]), axis=1
)

# Binary weather features (matching training)
df_2026["is_rainy"] = (df_2026["weather"] == "Rainy").astype(int)
df_2026["is_sunny"] = (df_2026["weather"] == "Sunny").astype(int)
df_2026["is_cold"]  = (df_2026["weather"] == "Cold").astype(int)
df_2026["rainy_weekend"] = df_2026["is_rainy"] * df_2026["is_weekend"]

# Match column order exactly to training exog
df_2026_exog = df_2026[exog_cols].astype(float)
df_2026_exog["is_event"] = df_2026_exog["is_event"] * 0.5

forecast_2026 = sarimax_model.forecast(steps=len(df_2026_exog), exog=df_2026_exog)

df_2026["forecast_sales_gbp"] = np.round(forecast_2026.values, 2)
df_2026[["date", "weekday", "month", "forecast_sales_gbp"]].to_excel(
    "forecast_2026.xlsx", index=False
)
print("Saved: forecast_2026.xlsx")
print("Done")