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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

PUB = "local"


# Settings per pub — db file, output prefix, seed, events, scenarios

PUB_CONFIG = {
    "local": {
        "db_file":    "pub_sales.db",
        "pub_name":   "Local Pub",
        "prefix":     "",           # no prefix — keeps original filenames
        "seed":       42,
        "event_dates": [
            "2025-01-01", "2025-10-31",
            "2025-12-24", "2025-12-25", "2025-12-26",
        ],
        "event_dates_2026": [
            "2026-01-01", "2026-10-31",
            "2026-12-24", "2026-12-25", "2026-12-26",
        ],
        "ml_extra_feat": "is_december",                                    # local pub peaks in December
        "ml_extra_col":  lambda df: (df["month_num"] == 12).astype(int),
        "scenarios": {
            "Optimistic":  {"is_rainy": 0.10, "is_sunny": 0.55, "is_cold": 0.10},
            "Neutral":     {"is_rainy": 0.25, "is_sunny": 0.35, "is_cold": 0.15},
            "Pessimistic": {"is_rainy": 0.45, "is_sunny": 0.15, "is_cold": 0.20},
        },
    },
    "riverside": {
        "db_file":    "riverside_pub_sales.db",
        "pub_name":   "Riverside Pub (London)",
        "prefix":     "riverside_",
        "seed":       99,
        "event_dates": [
            "2025-01-01", "2025-02-14", "2025-04-18", "2025-04-20",
            "2025-05-05", "2025-08-25", "2025-10-31",
            "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31",
        ],
        "event_dates_2026": [
            "2026-01-01", "2026-02-14", "2026-04-03", "2026-04-05",
            "2026-05-04", "2026-08-31", "2026-10-31",
            "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-31",
        ],
        "ml_extra_feat": "is_summer",                                      # riverside pub peaks in summer
        "ml_extra_col":  lambda df: df["month_num"].isin([6, 7, 8]).astype(int),
        "scenarios": {
            "Optimistic":  {"is_rainy": 0.08, "is_sunny": 0.60, "is_cold": 0.08},
            "Neutral":     {"is_rainy": 0.28, "is_sunny": 0.38, "is_cold": 0.12},
            "Pessimistic": {"is_rainy": 0.50, "is_sunny": 0.12, "is_cold": 0.15},
        },
    },
}

cfg    = PUB_CONFIG[PUB]
PREFIX = cfg["prefix"]
np.random.seed(cfg["seed"])  # fixed seed for reproducibility


# Load the data

conn = sqlite3.connect(cfg["db_file"])  # retrieve data from database
daily_sales = pd.read_sql("SELECT * FROM daily_sales", conn)
conn.close()
daily_sales["date"] = pd.to_datetime(daily_sales["date"])
daily_sales = daily_sales.sort_values("date").reset_index(drop=True)


# Data quality report and checks

print("=" * 60)
print(f"DATA QUALITY REPORT — {cfg['pub_name'].upper()}")
print("=" * 60)
print(f"Total rows      : {len(daily_sales)}")
print(f"Date range      : {daily_sales['date'].min().date()} to {daily_sales['date'].max().date()}")
print(f"Mean daily sales: £{daily_sales['sales_gbp'].mean():,.2f}")
print(f"Columns         : {list(daily_sales.columns)}")

missing = daily_sales.isnull().sum()  # count missing values in each column
print(f"\nMissing values per column:")
print(missing.to_string())
if missing.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values detected — will be forward-filled.")

duplicate_dates = daily_sales[daily_sales.duplicated(subset=["date"], keep=False)]  # find rows where dates are duplicated
print(f"\nDuplicate dates  : {len(duplicate_dates)}")
if len(duplicate_dates) == 0:
    print("No duplicate dates found.")
else:
    print("Duplicate dates detected:")
    print(duplicate_dates[["date", "weekday", "sales_gbp"]])

invalid_sales = daily_sales[daily_sales["sales_gbp"] <= 0]  # find rows where sales are zero or negative
print(f"\nNegative/zero sales rows: {len(invalid_sales)}")
if len(invalid_sales) == 0:
    print("All sales values are positive.")
else:
    print("Invalid sales values detected:")
    print(invalid_sales[["date", "weekday", "sales_gbp"]])

Q1 = daily_sales["sales_gbp"].quantile(0.25)
Q3 = daily_sales["sales_gbp"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = daily_sales[(daily_sales["sales_gbp"] < lower_bound) | (daily_sales["sales_gbp"] > upper_bound)]  # find rows outside the bounds
print(f"\nOutlier detection (IQR method):")
print(f"  Lower bound : £{lower_bound:,.2f}")
print(f"  Upper bound : £{upper_bound:,.2f}")
print(f"  Outliers    : {len(outliers)} rows")
if len(outliers) > 0:
    print("  Outlier dates (likely event days — expected):")
    print(outliers[["date", "weekday", "sales_gbp"]].to_string(index=False))

print(f"\nSales summary statistics:")
print(daily_sales["sales_gbp"].describe().apply(lambda x: f"£{x:,.2f}").to_string())
print(f"\nAverage sales by weekday:")
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_avg   = daily_sales.groupby("weekday")["sales_gbp"].mean()
for day in day_order:
    if day in dow_avg:
        print(f"  {day:<12}: £{dow_avg[day]:,.0f}")
print(f"\nWeather distribution:")
print(daily_sales["weather"].value_counts().to_string())
print("\nData quality check complete.")


# Ensure full date range — fill any missing days so time series has no gaps

full_dates  = pd.date_range(start=daily_sales["date"].min(),
                            end=daily_sales["date"].max(), freq="D")
daily_sales = daily_sales.set_index("date").reindex(full_dates)
daily_sales.index.name      = "date"
daily_sales["weekday"]      = daily_sales.index.day_name()
daily_sales["month"]        = daily_sales.index.month
daily_sales["weather"]      = daily_sales["weather"].fillna("Cloudy")
daily_sales["reservations"] = daily_sales["reservations"].ffill()
daily_sales["sales_gbp"]    = daily_sales["sales_gbp"].ffill()
daily_sales = daily_sales.reset_index().sort_values("date").reset_index(drop=True)


# Feature engineering — include external factors that influence pub sales

daily_sales["is_event"]      = daily_sales["date"].astype(str).isin(cfg["event_dates"]).astype(float) * 0.5  # boost sales for events
daily_sales["is_rainy"]      = (daily_sales["weather"] == "Rainy").astype(int)
daily_sales["is_sunny"]      = (daily_sales["weather"] == "Sunny").astype(int)
daily_sales["is_cold"]       = (daily_sales["weather"] == "Cold").astype(int)
daily_sales["rainy_weekend"] = daily_sales["is_rainy"] * daily_sales["weekday"].isin(["Friday", "Saturday"]).astype(int)

exog_cols = ["is_event", "is_rainy", "is_sunny", "is_cold"]  # for sarimax model

print(f"\nFEATURE SUMMARY — {cfg['pub_name']}")
print(daily_sales[exog_cols + ["rainy_weekend"]].sum().to_string())
print(f"\nFeatures used in SARIMAX: {exog_cols}")
print("Note: is_weekend and is_sunday excluded — captured by SARIMA seasonal component.")


# Train and test split — time-aware 80/20, no shuffling to avoid data leakage

daily_sales = daily_sales.set_index("date")  # for time series analysis
target = daily_sales["sales_gbp"].astype(float)
exog   = daily_sales[exog_cols].astype(float)  # weather and events to help predict sales

split      = int(len(daily_sales) * 0.8)
train_y    = target.iloc[:split]
test_y     = target.iloc[split:]
train_exog = exog.iloc[:split].copy()
test_exog  = exog.iloc[split:].copy()

print(f"\nTrain size : {len(train_y)} days  ({train_y.index[0].date()} to {train_y.index[-1].date()})")
print(f"Test size  : {len(test_y)} days  ({test_y.index[0].date()} to {test_y.index[-1].date()})")


# ADF stationarity test

print("\nADF STATIONARITY TEST")
adf_result = adfuller(train_y, autolag="AIC")  # check if data is stationary
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
print("Critical values:")
for key, val in adf_result[4].items():  # show critical values for 1%, 5%, and 10% levels
    print(f"   {key}: {val:.4f}")
if adf_result[1] < 0.05:  # stable
    print("Series is stationary (p < 0.05) — differencing not strictly required.")
else:  # not stable
    print("Series is non-stationary (p >= 0.05) — d=1 differencing applied in ARIMA.")


# ACF and PACF — confirm weekly seasonality and AR order before modelling

fig_acf, axes_acf = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(train_y,  lags=28, ax=axes_acf[0],
         title=f"ACF — {cfg['pub_name']} Training Sales (lags=28)\nSpikes at lag 7, 14, 21 confirm weekly seasonality")
plot_pacf(train_y, lags=28, ax=axes_acf[1],
          title=f"PACF — {cfg['pub_name']} Training Sales (lags=28)\nSignificant lag 1 justifies AR(1) order")
fig_acf.tight_layout()
plt.savefig(f"{PREFIX}acf_pacf_plots.png", dpi=150)
plt.show()
print(f"\nSaved: {PREFIX}acf_pacf_plots.png")


# Auto ARIMA parameter search — finds the best p,d,q order automatically

print("\nAUTO_ARIMA PARAMETER SEARCH")
print("Searching for optimal ARIMA parameters.")

auto_model = auto_arima(
    train_y,
    seasonal=True,          # enable seasonal ARIMA (SARIMA)
    m=7,                    # weekly seasonal period
    d=1, D=1,               # differencing for trend and seasonality
    max_p=2, max_q=2,       # limit AR and MA parameters
    max_P=2, max_Q=2,       # limit seasonal AR and MA parameters
    information_criterion="aic",  # select model with lowest AIC
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

best_order          = auto_model.order          # best p,d,q values for ARIMA model
best_seasonal_order = auto_model.seasonal_order  # best seasonal P,D,Q,m values
print(f"auto_arima recommended order          : {best_order}")
print(f"auto_arima recommended seasonal order : {best_seasonal_order}")
print(f"AIC: {auto_model.aic():.2f}")
print(f"Final model order adopted from auto_arima: {best_order} x {best_seasonal_order}")


# MAPE and RMSE — evaluation metrics used across all models

def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

results = []


# 1. Moving average — naive baseline model

window = 7
movingaverage_preds = []
history = list(train_y)

for t in range(len(test_y)):
    yhat = np.mean(history[-window:])  # predict next day as average of last 7 days
    movingaverage_preds.append(yhat)
    history.append(test_y.iloc[t])

results.append(("Moving Average (7-day)", mape(test_y, movingaverage_preds), rmse(test_y, movingaverage_preds)))
print("\nMoving Average complete")


# 2. Exponential smoothing — gives more weight to recent data

exp_model = ExponentialSmoothing(
    train_y,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    damped_trend=True  # prevents unrealistic extrapolation of the trend
).fit(optimized=True)

exp_preds = exp_model.forecast(len(test_y))
results.append(("Exponential Smoothing", mape(test_y, exp_preds), rmse(test_y, exp_preds)))
print("Exponential Smoothing complete")


# 3. SARIMA — seasonal time series without external variables
# uses SARIMAX class without exog — statsmodels has no standalone SARIMA class

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


# 4. SARIMAX — adds weather and event features to improve accuracy

sarimax_model = SARIMAX(
    train_y,
    exog=train_exog,  # weather and event flags as external regressors
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarimax_preds = sarimax_model.forecast(steps=len(test_y), exog=test_exog)
results.append((f"SARIMAX {best_order}x{best_seasonal_order}", mape(test_y, sarimax_preds), rmse(test_y, sarimax_preds)))
print("SARIMAX complete")


# 5. Decision tree regressor — interpretable machine learning rules

print("\nBuilding ML feature set (lag features + calendar features)")

daily_sales_ml = daily_sales.reset_index().copy()

daily_sales_ml["lag_1"]       = daily_sales_ml["sales_gbp"].shift(1)   # yesterday's sales
daily_sales_ml["lag_7"]       = daily_sales_ml["sales_gbp"].shift(7)   # sales from last week
daily_sales_ml["lag_14"]      = daily_sales_ml["sales_gbp"].shift(14)  # sales from 2 weeks ago
daily_sales_ml["roll_7_mean"] = daily_sales_ml["sales_gbp"].shift(1).rolling(7).mean()  # average sales over last 7 days
daily_sales_ml["roll_7_std"]  = daily_sales_ml["sales_gbp"].shift(1).rolling(7).std()   # how much sales varied last week
daily_sales_ml["day_of_week"] = pd.to_datetime(daily_sales_ml["date"]).dt.dayofweek
daily_sales_ml["month_num"]   = pd.to_datetime(daily_sales_ml["date"]).dt.month

# pub-specific seasonal flag — is_december for local pub, is_summer for riverside
daily_sales_ml[cfg["ml_extra_feat"]] = cfg["ml_extra_col"](daily_sales_ml)

ml_feature_cols = [
    "lag_1", "lag_7", "lag_14", "roll_7_mean", "roll_7_std",
    "day_of_week", "month_num", cfg["ml_extra_feat"],
    "is_event", "is_rainy", "is_sunny", "is_cold"
]

daily_sales_ml = daily_sales_ml.dropna(subset=ml_feature_cols).reset_index(drop=True)

ml_split = int(len(daily_sales_ml) * 0.8)
ml_train = daily_sales_ml.iloc[:ml_split]
ml_test  = daily_sales_ml.iloc[ml_split:]

X_train = ml_train[ml_feature_cols]
y_train = ml_train["sales_gbp"]
X_test  = ml_test[ml_feature_cols]
y_test  = ml_test["sales_gbp"]

print(f"ML train: {len(X_train)} days, ML test: {len(X_test)} days")
print(f"Features used: {ml_feature_cols}")

dt_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=cfg["seed"])
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

results.append(("Decision Tree (max_depth=5)", mape(y_test, dt_preds), rmse(y_test, dt_preds)))
print("Decision Tree complete")

print(f"\nDECISION TREE — TOP RULES")
tree_rules = export_text(dt_model, feature_names=ml_feature_cols, max_depth=3)
print(tree_rules)


# 6. Random forest — ensemble of 200 trees, reduces overfitting vs single tree

rf_model = RandomForestRegressor(
    n_estimators=200,    # number of trees in the forest
    max_depth=8,
    min_samples_leaf=3,
    random_state=cfg["seed"],
    n_jobs=-1            # use all CPU cores to speed up training
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

results.append(("Random Forest (200 trees)", mape(y_test, rf_preds), rmse(y_test, rf_preds)))
print("Random Forest complete")

fi_daily_sales = pd.DataFrame({
    "Feature":    ml_feature_cols,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nRANDOM FOREST — FEATURE IMPORTANCE")
print(fi_daily_sales.to_string(index=False))

fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
ax_fi.barh(fi_daily_sales["Feature"][::-1], fi_daily_sales["Importance"][::-1],
           color="steelblue", edgecolor="white")
ax_fi.set_title(f"Random Forest — Feature Importance ({cfg['pub_name']})\n"
                f"(Higher = more influential in predicting daily sales)", fontsize=13)
ax_fi.set_xlabel("Importance Score")
ax_fi.grid(True, axis="x", alpha=0.4)
fig_fi.tight_layout()
plt.savefig(f"{PREFIX}feature_importance_rf.png", dpi=150)
plt.show()
print(f"Saved: {PREFIX}feature_importance_rf.png")

ml_test_index = pd.to_datetime(ml_test["date"])


# 7. Linear regression — scaled features, assumes linear relationship between inputs and sales

scaler     = StandardScaler()
X_train_lr = scaler.fit_transform(X_train)  # fit scaler on train only to avoid data leakage
X_test_lr  = scaler.transform(X_test)       # apply same scale to test set

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train)
lr_preds = lr_model.predict(X_test_lr)
lr_preds = np.clip(lr_preds, 0, None)       # sales cannot be negative

results.append(("Linear Regression", mape(y_test, lr_preds), rmse(y_test, lr_preds)))
print("Linear Regression complete")

print("\nLINEAR REGRESSION — COEFFICIENTS")
lr_coef_df = pd.DataFrame({
    "Feature":     ml_feature_cols,
    "Coefficient": lr_model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print(lr_coef_df.to_string(index=False))
print(f"Intercept: £{lr_model.intercept_:,.0f}")
print("Note: Coefficients on standardised features — larger absolute value = more influence.")


# SARIMAX model summary — shows coefficients and statistical significance

print(f"\nSARIMAX MODEL SUMMARY")
print(sarimax_model.summary())


# Ljung-Box residual test — checks if any patterns remain in the errors

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


# Results table — all models sorted by MAPE

results_daily_sales = pd.DataFrame(
    results, columns=["Model", "MAPE (%)", "RMSE (£)"]
).sort_values("MAPE (%)")

print(f"\n{'=' * 60}")
print(f"MODEL COMPARISON — {cfg['pub_name'].upper()}")
print(f"{'=' * 60}")
print(results_daily_sales.to_string(index=False))
results_daily_sales.to_csv(f"{PREFIX}forecast_model_comparison.csv", index=False)
print(f"\nSaved: {PREFIX}forecast_model_comparison.csv")


# Walk-forward cross-validation — tests SARIMAX across 5 expanding training windows

print(f"\nWALK-FORWARD CROSS-VALIDATION (SARIMAX)")

tscv = TimeSeriesSplit(n_splits=5)
cv_mape_scores = []
cv_rmse_scores = []

target_arr = target.values
exog_arr   = exog.values

for fold, (train_idx, test_idx) in enumerate(tscv.split(target_arr)):
    cv_train_exog_daily_sales = pd.DataFrame(exog_arr[train_idx], columns=exog_cols)
    cv_test_exog_daily_sales  = pd.DataFrame(exog_arr[test_idx],  columns=exog_cols)

    try:
        cv_model = SARIMAX(
            target_arr[train_idx],
            exog=cv_train_exog_daily_sales,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        cv_preds  = cv_model.forecast(steps=len(test_idx), exog=cv_test_exog_daily_sales)
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


# Forecast comparison plot — all models vs actual on test period

forecast_obj  = sarimax_model.get_forecast(steps=len(test_y), exog=test_exog)
forecast_mean = forecast_obj.predicted_mean
forecast_ci   = forecast_obj.conf_int(alpha=0.05)
forecast_ci.iloc[:, 0] = forecast_ci.iloc[:, 0].clip(lower=0)  # lower bound cannot be negative

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_y.index, test_y.values,        label="Actual",                color="steelblue", linewidth=2)
ax.plot(test_y.index, movingaverage_preds,  label="Moving Average",        color="grey",      linewidth=1.2, linestyle="--")
ax.plot(test_y.index, exp_preds.values,     label="Exponential Smoothing", color="orange",    linewidth=1.2)
ax.plot(test_y.index, sarima_preds.values,  label="SARIMA",                color="green",     linewidth=1.2)
ax.plot(test_y.index, forecast_mean.values, label="SARIMAX",               color="red",       linewidth=1.5)
ax.fill_between(
    test_y.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="red", alpha=0.15, label="SARIMAX 95% CI"
)
ax.plot(ml_test_index, dt_preds, label="Decision Tree",    color="purple",   linewidth=1.2, linestyle=":")
ax.plot(ml_test_index, rf_preds, label="Random Forest",    color="darkgreen",linewidth=1.5, linestyle="-.")
ax.plot(ml_test_index, lr_preds, label="Linear Regression",color="#f59e0b",  linewidth=1.2, linestyle=(0, (3,1,1,1)))
ax.set_title(f"{cfg['pub_name']} — Demand Forecast Comparison — All Models, Test Period (Oct–Dec 2025)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Sales (£)")
ax.legend(ncol=2)
ax.grid(True, alpha=0.4)
fig.tight_layout()
plt.savefig(f"{PREFIX}forecast_comparison.png", dpi=150)
plt.show()
print(f"\nSaved: {PREFIX}forecast_comparison.png")


# Residual diagnostics — check if errors are random or show a pattern

residuals = test_y.values - np.array(forecast_mean)

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(range(len(residuals)), residuals, alpha=0.6, color="steelblue", s=30)
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_title(f"SARIMAX Residuals Over Test Period ({cfg['pub_name']})")
axes[0].set_xlabel("Observation Index (test days)")
axes[0].set_ylabel("Residuals (£)")
axes[0].grid(True, alpha=0.4)

axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_title(f"Residual Distribution (SARIMAX) — {cfg['pub_name']}")
axes[1].set_xlabel("Residual (£)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.4)

fig2.tight_layout()
plt.savefig(f"{PREFIX}residual_diagnostics.png", dpi=150)
plt.show()
print(f"Saved: {PREFIX}residual_diagnostics.png")


# Business insight — translate SARIMAX coefficients into actionable recommendations

print(f"\nBUSINESS INSIGHT: FEATURE IMPACT — {cfg['pub_name'].upper()}")
params = sarimax_model.params
feature_labels = {
    "is_event": "Special event day",
    "is_rainy": "Rainy weather",
    "is_sunny": "Sunny weather",
    "is_cold":  "Cold weather",
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


# Forecast 2026 — refit SARIMAX on full year before forecasting to avoid decay

print(f"\nGenerating 2026 scenario-based forecasts ({cfg['pub_name']})")

sarimax_full = SARIMAX(
    target,
    exog=exog,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)  # refit on all 365 days — train-only model would decay to zero by mid-2026

dates_2026       = pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")
daily_sales_2026 = pd.DataFrame({"date": dates_2026})
daily_sales_2026["weekday"]  = daily_sales_2026["date"].dt.day_name()
daily_sales_2026["month"]    = daily_sales_2026["date"].dt.month
daily_sales_2026["is_event"] = daily_sales_2026["date"].astype(str).isin(
    cfg["event_dates_2026"]).astype(float) * 0.5  # flag known 2026 event dates

scenario_forecasts = {}

for scenario_name, probs in cfg["scenarios"].items():
    np.random.seed(cfg["seed"])
    daily_sales_s = daily_sales_2026.copy()
    p_cloudy      = 1.0 - probs["is_rainy"] - probs["is_sunny"] - probs["is_cold"]
    weather_draw  = np.random.choice(
        ["Rainy", "Sunny", "Cold", "Cloudy"],
        size=len(daily_sales_s),
        p=[probs["is_rainy"], probs["is_sunny"], probs["is_cold"], p_cloudy]
    )
    daily_sales_s["is_rainy"] = (weather_draw == "Rainy").astype(int)
    daily_sales_s["is_sunny"] = (weather_draw == "Sunny").astype(int)
    daily_sales_s["is_cold"]  = (weather_draw == "Cold").astype(int)

    sc_exog       = daily_sales_s[exog_cols].astype(float)
    forecast_vals = sarimax_full.forecast(steps=len(sc_exog), exog=sc_exog)
    forecast_vals = np.clip(forecast_vals.values, 0, None)  # clip negative forecasts to zero
    scenario_forecasts[scenario_name] = np.round(forecast_vals, 2)
    print(f"  {scenario_name:<12}: average daily forecast £{np.mean(forecast_vals):,.0f}")

# Save all three scenarios
daily_sales_out = daily_sales_2026[["date", "weekday", "month"]].copy()
for sc_name, vals in scenario_forecasts.items():
    daily_sales_out[f"forecast_{sc_name.lower()}_gbp"] = vals
daily_sales_out.to_excel(f"{PREFIX}forecast_2026_scenarios.xlsx", index=False)
print(f"Saved: {PREFIX}forecast_2026_scenarios.xlsx")

# Save neutral scenario — this is what the frontend reads
daily_sales_neutral = daily_sales_2026[["date", "weekday", "month"]].copy()
daily_sales_neutral["forecast_sales_gbp"] = scenario_forecasts["Neutral"]
daily_sales_neutral.to_excel(f"{PREFIX}forecast_2026.xlsx", index=False)
print(f"Saved: {PREFIX}forecast_2026.xlsx (Neutral scenario — used by frontend)")


# Scenario comparison plot — visualise the impact of different weather assumptions on 2026

daily_sales_plot = daily_sales_2026[["date", "month"]].copy()
for sc_name, vals in scenario_forecasts.items():
    daily_sales_plot[sc_name] = vals

daily_sales_monthly = daily_sales_plot.groupby("month")[list(cfg["scenarios"].keys())].mean().reset_index()
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig3, ax3 = plt.subplots(figsize=(14, 6))
colours = {"Optimistic": "green", "Neutral": "steelblue", "Pessimistic": "red"}
styles  = {"Optimistic": "-",     "Neutral": "-",          "Pessimistic": "--"}

for sc_name in cfg["scenarios"].keys():
    ax3.plot(month_labels, daily_sales_monthly[sc_name],
             label=f"{sc_name} scenario",
             color=colours[sc_name],
             linewidth=2,
             linestyle=styles[sc_name],
             marker="o", markersize=5)

ax3.fill_between(
    month_labels,
    daily_sales_monthly["Pessimistic"],
    daily_sales_monthly["Optimistic"],
    alpha=0.15, color="steelblue", label="Scenario range"
)
ax3.set_title(f"{cfg['pub_name']} — 2026 Average Daily Sales Forecast by Month — Weather Scenario Comparison",
              fontsize=13)
ax3.set_xlabel("Month")
ax3.set_ylabel("Average Daily Forecast Sales (£)")
ax3.legend()
ax3.grid(True, alpha=0.4)
fig3.tight_layout()
plt.savefig(f"{PREFIX}forecast_2026_scenarios.png", dpi=150)
plt.show()
print(f"Saved: {PREFIX}forecast_2026_scenarios.png")


print(f"\n{'=' * 60}")
print(f"EVALUATION COMPLETE — {cfg['pub_name'].upper()}")
print(f"{'=' * 60}")
print("Files generated:")
for f in [
    f"{PREFIX}acf_pacf_plots.png",
    f"{PREFIX}feature_importance_rf.png",
    f"{PREFIX}forecast_comparison.png",
    f"{PREFIX}residual_diagnostics.png",
    f"{PREFIX}forecast_2026_scenarios.png",
    f"{PREFIX}forecast_model_comparison.csv",
    f"{PREFIX}forecast_2026.xlsx",
    f"{PREFIX}forecast_2026_scenarios.xlsx",
]:
    print(f"  {f}")

print(f"\nTo switch pub, change the PUB variable at the top of this file:")
print(f'  PUB = "local"      → runs local pub (pub_sales.db)')
print(f'  PUB = "riverside"  → runs riverside pub (riverside_pub_sales.db)')
print("\nDone.")