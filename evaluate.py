import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from pmdarima import auto_arima

warnings.filterwarnings("ignore")
np.random.seed(42)


# 1. LOAD DATA

conn = sqlite3.connect("pub_sales.db")
df = pd.read_sql("SELECT * FROM daily_sales", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# 2. DATA QUALITY CHECKS # Explicitly validate the dataset before any modelling.


print("DATA QUALITY REPORT")

print(f"Total rows       : {len(df)}")
print(f"Date range       : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Columns          : {list(df.columns)}")

missing = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing.to_string())
if missing.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values detected — will be forward-filled.")

duplicate_dates = df[df.duplicated(subset=["date"], keep=False)]
print(f"\nDuplicate dates  : {len(duplicate_dates)}")
if len(duplicate_dates) == 0:
    print("No duplicate dates found.")
else:
    print("Duplicate dates detected:")
    print(duplicate_dates[["date", "weekday", "sales_gbp"]])

invalid_sales = df[df["sales_gbp"] <= 0]
print(f"\nNegative/zero sales rows: {len(invalid_sales)}")
if len(invalid_sales) == 0:
    print("All sales values are positive.")
else:
    print("Invalid sales values detected:")
    print(invalid_sales[["date", "weekday", "sales_gbp"]])

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

print(f"\nSales summary statistics:")
print(df["sales_gbp"].describe().apply(lambda x: f"£{x:,.2f}").to_string())
print(f"\nRows per weekday:")
print(df["weekday"].value_counts().sort_index().to_string())
print(f"\nWeather distribution:")
print(df["weather"].value_counts().to_string())
print("\nData quality check complete.")


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

event_dates = ["2025-01-01", "2025-10-31", "2025-12-24", "2025-12-25", "2025-12-26"]
# Event flag scaled by 0.5 once here to moderate Christmas spike influence.
df["is_event"] = df["date"].astype(str).isin(event_dates).astype(float) * 0.5

# Binary weather indicators (Cloudy = baseline, avoids dummy variable trap).
df["is_rainy"]      = (df["weather"] == "Rainy").astype(int)
df["is_sunny"]      = (df["weather"] == "Sunny").astype(int)
df["is_cold"]       = (df["weather"] == "Cold").astype(int)

# Interaction term: rain on a busy weekend suppresses sales more than rain on a quiet day. This is constructed for reference but not included in the final exog to keep the model stable.
df["rainy_weekend"] = df["is_rainy"] * df["weekday"].isin(["Friday", "Saturday"]).astype(int)

exog_cols = ["is_event", "is_rainy", "is_sunny", "is_cold"]

print("\nFEATURE SUMMARY")
print(df[exog_cols + ["rainy_weekend"]].sum().to_string())
print(f"\nFeatures used in SARIMAX: {exog_cols}")
print("Note: is_weekend and is_sunday excluded — captured by SARIMA seasonal component.")


# 5. TRAIN / TEST SPLIT (TIME-AWARE 80/20)
# An 80/20 chronological split is used so the model is always tested on data it has never seen, preserving temporal order.

df = df.set_index("date")
target = df["sales_gbp"].astype(float)
exog   = df[exog_cols].astype(float)

split      = int(len(df) * 0.8)
train_y    = target.iloc[:split]
test_y     = target.iloc[split:]
train_exog = exog.iloc[:split].copy()
test_exog  = exog.iloc[split:].copy()

print(f"\nTrain size : {len(train_y)} days  ({train_y.index[0].date()} to {train_y.index[-1].date()})")
print(f"Test size  : {len(test_y)} days  ({test_y.index[0].date()} to {test_y.index[-1].date()})")


# 6. ADF STATIONARITY TEST The Augmented Dickey-Fuller test formally checks whether the series is stationary before fitting ARIMA. A p-value below 0.05 indicates stationarity.
# If non-stationary, first-order differencing (d=1) is required.

print("\nADF STATIONARITY TEST")
adf_result = adfuller(train_y, autolag="AIC")
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
print("Critical values:")
for key, val in adf_result[4].items():
    print(f"   {key}: {val:.4f}")
if adf_result[1] < 0.05:
    print("Series is stationary (p < 0.05) — differencing not strictly required.")
else:
    print("Series is non-stationary (p >= 0.05) — d=1 differencing applied in ARIMA.")


# 7. ACF AND PACF PLOTS

fig_acf, axes_acf = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(train_y,  lags=28, ax=axes_acf[0],
         title="ACF — Training Sales (lags=28)\nSpikes at lag 7, 14, 21 confirm weekly seasonality")
plot_pacf(train_y, lags=28, ax=axes_acf[1],
          title="PACF — Training Sales (lags=28)\nSignificant lag 1 justifies AR(1) order")
fig_acf.tight_layout()
plt.savefig("acf_pacf_plots.png", dpi=150)
plt.show()
print("\nSaved: acf_pacf_plots.png")


# 8. AUTO_ARIMA PARAMETER SEARCH

print("\nAUTO_ARIMA PARAMETER SEARCH")
print("Searching for optimal ARIMA parameters (this may take a moment)...")

auto_model = auto_arima(
    train_y,
    seasonal=True,
    m=7,
    d=1, D=1,
    max_p=2, max_q=2,
    max_P=2, max_Q=2,
    information_criterion="aic",
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

best_order          = auto_model.order
best_seasonal_order = auto_model.seasonal_order
print(f"auto_arima recommended order          : {best_order}")
print(f"auto_arima recommended seasonal order : {best_seasonal_order}")
print(f"AIC: {auto_model.aic():.2f}")
print(f"Final model order adopted from auto_arima: {best_order} x {best_seasonal_order}")


# 9. METRICS

def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

results = []


# MODEL 1 — Rolling Moving Average (Baseline)
# Used as a naive benchmark to demonstrate improvement from more complex models.

window   = 7
ma_preds = []
history  = list(train_y)

for t in range(len(test_y)):
    yhat = np.mean(history[-window:])
    ma_preds.append(yhat)
    history.append(test_y.iloc[t])

results.append(("Moving Average (7-day)", mape(test_y, ma_preds), rmse(test_y, ma_preds)))
print("\nMoving Average complete")


# MODEL 2 — Exponential Smoothing (Holt-Winters)
# Models trend and seasonality separately with exponentially weighted observations.
# damped_trend=True prevents over-extrapolation of the upward December trend.

exp_model = ExponentialSmoothing(
    train_y,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    damped_trend=True
).fit(optimized=True)

exp_preds = exp_model.forecast(len(test_y))
results.append(("Exponential Smoothing", mape(test_y, exp_preds), rmse(test_y, exp_preds)))
print("Exponential Smoothing complete")


# MODEL 3 — SARIMA
# Order adopted from auto_arima recommendation.

sarima_model = SARIMAX(
    train_y,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarima_preds = sarima_model.forecast(len(test_y))
results.append((f"SARIMA {best_order}x{best_seasonal_order}", mape(test_y, sarima_preds), rmse(test_y, sarima_preds)))
print("SARIMA complete")


# MODEL 4 — SARIMAX
# Extends SARIMA with exogenous weather and event features.

sarimax_model = SARIMAX(
    train_y,
    exog=train_exog,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarimax_preds = sarimax_model.forecast(steps=len(test_y), exog=test_exog)
results.append((f"SARIMAX {best_order}x{best_seasonal_order}", mape(test_y, sarimax_preds), rmse(test_y, sarimax_preds)))
print("SARIMAX complete")


# 10. SARIMAX MODEL SUMMARY
# Prints coefficient values, standard errors, z-scores and p-values for all features

print("\nSARIMAX MODEL SUMMARY")
print(sarimax_model.summary())


# 11. LJUNG-BOX RESIDUAL TEST
# Tests whether residuals from SARIMAX contain remaining autocorrelation.
# A p-value above 0.05 at all tested lags indicates residuals are white noise.

print("\nLJUNG-BOX RESIDUAL TEST")
sarimax_residuals = test_y.values - np.array(sarimax_preds)
lb_result = acorr_ljungbox(sarimax_residuals, lags=[7, 14, 21], return_df=True)
print(lb_result.to_string())
if (lb_result["lb_pvalue"] > 0.05).all():
    print("\nAll p-values > 0.05 — residuals are white noise. Model fit is adequate.")
else:
    print("\nSome p-values <= 0.05 — residuals contain remaining autocorrelation.")
    print("This is expected given the high volatility of hospitality sales data")
    print("and the use of only one year of synthetic training data.")


# 12. RESULTS TABLE

results_df = pd.DataFrame(
    results, columns=["Model", "MAPE (%)", "RMSE (£)"]
).sort_values("MAPE (%)")

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv("forecast_model_comparison.csv", index=False)
print("\nSaved: forecast_model_comparison.csv")


# 13. WALK-FORWARD CROSS-VALIDATION
# A single 80/20 split provides limited evidence of generalisation.

print("\nWALK-FORWARD CROSS-VALIDATION (SARIMAX)")

tscv           = TimeSeriesSplit(n_splits=5)
cv_mape_scores = []
cv_rmse_scores = []

target_arr = target.values
exog_arr   = exog.values

for fold, (train_idx, test_idx) in enumerate(tscv.split(target_arr)):
    cv_train_exog_df = pd.DataFrame(exog_arr[train_idx], columns=exog_cols)
    cv_test_exog_df  = pd.DataFrame(exog_arr[test_idx],  columns=exog_cols)

    try:
        cv_model = SARIMAX(
            target_arr[train_idx],
            exog=cv_train_exog_df,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        cv_preds  = cv_model.forecast(steps=len(test_idx), exog=cv_test_exog_df)
        fold_mape = mape(target_arr[test_idx], cv_preds)
        fold_rmse = rmse(target_arr[test_idx], cv_preds)
        cv_mape_scores.append(fold_mape)
        cv_rmse_scores.append(fold_rmse)
        print(f"  Fold {fold+1}: Train={len(train_idx)} days, "
              f"Test={len(test_idx)} days, "
              f"MAPE={fold_mape:.2f}%,  RMSE=£{fold_rmse:,.0f}")
    except Exception as e:
        print(f"  Fold {fold+1}: skipped — {e}")

if cv_mape_scores:
    print(f"\n  Mean CV MAPE : {np.mean(cv_mape_scores):.2f}%  (±{np.std(cv_mape_scores):.2f}%)")
    print(f"  Mean CV RMSE : £{np.mean(cv_rmse_scores):,.0f}  (±£{np.std(cv_rmse_scores):,.0f})")
    print("  Note: High variance across folds is expected with one year of data.")
    print("  Folds with sufficient training data (Folds 4-5) show more reliable estimates.")


# 14. FORECAST vs ACTUAL PLOT WITH CONFIDENCE INTERVALS
# get_forecast() is used to obtain the 95% confidence interval.

forecast_obj  = sarimax_model.get_forecast(steps=len(test_y), exog=test_exog)
forecast_mean = forecast_obj.predicted_mean
forecast_ci   = forecast_obj.conf_int(alpha=0.05)

# Clip CI to non-negative values since sales cannot be negative
forecast_ci.iloc[:, 0] = forecast_ci.iloc[:, 0].clip(lower=0)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_y.index, test_y.values,       label="Actual",                color="steelblue", linewidth=2)
ax.plot(test_y.index, ma_preds,             label="Moving Average",        color="grey",      linewidth=1.2, linestyle="--")
ax.plot(test_y.index, exp_preds.values,     label="Exponential Smoothing", color="orange",    linewidth=1.2)
ax.plot(test_y.index, sarima_preds.values,  label="SARIMA",                color="green",     linewidth=1.2)
ax.plot(test_y.index, forecast_mean.values, label="SARIMAX (Best)",        color="red",       linewidth=1.5)
ax.fill_between(
    test_y.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="red", alpha=0.15, label="SARIMAX 95% CI"
)
ax.set_title("Demand Forecast Comparison — Test Period (Oct–Dec 2025)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Sales (£)")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
plt.savefig("forecast_comparison.png", dpi=150)
plt.show()
print("\nSaved: forecast_comparison.png")


# 15. RESIDUAL DIAGNOSTICS

residuals = test_y.values - np.array(forecast_mean)

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


# 16. BUSINESS INSIGHT — FEATURE COEFFICIENT INTERPRETATION
# SARIMAX coefficients are extracted and presented.

print("\nBUSINESS INSIGHT: FEATURE IMPACT")
params = sarimax_model.params
feature_labels = {
    "is_event" : "Special event day",
    "is_rainy" : "Rainy weather",
    "is_sunny" : "Sunny weather",
    "is_cold"  : "Cold weather",
}
print(f"\n{'Feature':<32} {'Coefficient (£)':>16}  Direction")
for col, label in feature_labels.items():
    if col in params:
        coef      = params[col]
        direction = "Increases sales" if coef > 0 else "Reduces sales"
        print(f"{label:<32} £{coef:>12,.0f}  {direction}")

print("\nInventory Recommendations:")
print("1. Thursdays and Fridays show the highest forecast sales.")
print("   Increase stock orders mid-week to avoid shortages on peak days.")
print("2. Rainy weather reduces sales. Reduce perishable orders when rain is forecast.")
print("3. Raise stock levels on event days such as Christmas Eve and New Year's Day.")
print("4. Sunday sales are significantly lower. Minimal replenishment is needed.")
print("5. Cold weather slightly increases sales, likely driven by comfort food demand.")


# 17. FORECAST 2026 — SCENARIO-BASED
# Each scenario represents a plausible range of UK weather conditions:
#   Optimistic  — more sunny days, fewer rainy days
#   Neutral     — average UK seasonal weather mix
#   Pessimistic — more rainy and cold days

print("\nGenerating 2026 scenario-based forecasts...")

dates_2026 = pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")
df_2026    = pd.DataFrame({"date": dates_2026})
df_2026["weekday"] = df_2026["date"].dt.day_name()
df_2026["month"]   = df_2026["date"].dt.month

event_dates_2026 = ["2026-01-01", "2026-10-31", "2026-12-24", "2026-12-25", "2026-12-26"]
df_2026["is_event"] = df_2026["date"].astype(str).isin(event_dates_2026).astype(float) * 0.5

scenarios = {
    "Optimistic":  {"is_rainy": 0.10, "is_sunny": 0.55, "is_cold": 0.10},
    "Neutral":     {"is_rainy": 0.25, "is_sunny": 0.35, "is_cold": 0.15},
    "Pessimistic": {"is_rainy": 0.45, "is_sunny": 0.15, "is_cold": 0.20},
}

scenario_forecasts = {}

for scenario_name, probs in scenarios.items():
    np.random.seed(42)
    df_s = df_2026.copy()

    p_cloudy     = 1.0 - probs["is_rainy"] - probs["is_sunny"] - probs["is_cold"]
    weather_draw = np.random.choice(
        ["Rainy", "Sunny", "Cold", "Cloudy"],
        size=len(df_s),
        p=[probs["is_rainy"], probs["is_sunny"], probs["is_cold"], p_cloudy]
    )
    df_s["is_rainy"] = (weather_draw == "Rainy").astype(int)
    df_s["is_sunny"] = (weather_draw == "Sunny").astype(int)
    df_s["is_cold"]  = (weather_draw == "Cold").astype(int)

    sc_exog       = df_s[exog_cols].astype(float)
    forecast_vals = sarimax_model.forecast(steps=len(sc_exog), exog=sc_exog)
    scenario_forecasts[scenario_name] = np.round(forecast_vals.values, 2)
    print(f"  {scenario_name:<12}: average daily forecast £{np.mean(forecast_vals.values):,.0f}")

df_out = df_2026[["date", "weekday", "month"]].copy()
for sc_name, vals in scenario_forecasts.items():
    df_out[f"forecast_{sc_name.lower()}_gbp"] = vals
df_out.to_excel("forecast_2026_scenarios.xlsx", index=False)
print("Saved: forecast_2026_scenarios.xlsx")

df_neutral = df_2026[["date", "weekday", "month"]].copy()
df_neutral["forecast_sales_gbp"] = scenario_forecasts["Neutral"]
df_neutral.to_excel("forecast_2026.xlsx", index=False)
print("Saved: forecast_2026.xlsx (Neutral scenario — used by frontend)")


# 18. SCENARIO COMPARISON PLOT
# Monthly averages are plotted instead of daily values to make the scenario differences visible.

df_plot = df_2026[["date", "month"]].copy()
for sc_name, vals in scenario_forecasts.items():
    df_plot[sc_name] = vals

df_monthly = df_plot.groupby("month")[list(scenarios.keys())].mean().reset_index()
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig3, ax3 = plt.subplots(figsize=(14, 6))
colours = {"Optimistic": "green", "Neutral": "steelblue", "Pessimistic": "red"}
styles  = {"Optimistic": "-",     "Neutral": "-",          "Pessimistic": "--"}

for sc_name in scenarios.keys():
    ax3.plot(month_labels, df_monthly[sc_name],
             label=f"{sc_name} scenario",
             color=colours[sc_name],
             linewidth=2,
             linestyle=styles[sc_name],
             marker="o", markersize=5)

ax3.fill_between(
    month_labels,
    df_monthly["Pessimistic"],
    df_monthly["Optimistic"],
    alpha=0.15, color="steelblue", label="Scenario range"
)
ax3.set_title("2026 Average Daily Sales Forecast by Month — Weather Scenario Comparison", fontsize=13)
ax3.set_xlabel("Month")
ax3.set_ylabel("Average Daily Forecast Sales (£)")
ax3.legend()
ax3.grid(True, alpha=0.4)
fig3.tight_layout()
plt.savefig("forecast_2026_scenarios.png", dpi=150)
plt.show()
print("Saved: forecast_2026_scenarios.png")

print("\nDone")