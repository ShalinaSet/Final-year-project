# Pub Forecast

## Smart Inventory Management with Demand Forecasting for London Pubs

**BSc Computer Science Final Year Project**  
**Author:** Hsu Hnin Set @ Shalina Set  
**Student ID:** 001343697

---

## Project Overview

This project implements a demand forecasting and inventory support system for pubs using a combination of statistical and machine learning models. The system compares seven forecasting models and uses their outputs to generate stock order recommendations through a Streamlit dashboard.

The project covers two contrasting pub datasets:

- **Local Pub** — shaped mainly by weekday office trade and Christmas peaks  
- **Riverside Pub (London)** — more sensitive to weather, tourism, and seasonal variation  

This allows a comparative evaluation of how different forecasting models behave across different pub trading profiles.

The final system includes:

- model benchmarking using **MAPE** and **RMSE**
- 2026 scenario-based forecasting
- stock order recommendations
- an interactive Streamlit frontend
- downloadable PDF reports

---

## Repository Structure

### Core files

- `Dataset 2025.py` — generates the local pub dataset  
- `dataset_riverside_pub_2025.py` — generates the riverside pub dataset  
- `evaluate.py` — runs all seven models; switch pub at the top of the file  
- `app.py` — launches the Streamlit frontend  

### Database

- `Database/pub_sales.db` — local pub SQLite database  
- `Database/riverside_pub_sales.db` — riverside pub SQLite database  

### Excel files

- `Excel files/forecast_2026.xlsx` — local pub neutral forecast  
- `Excel files/forecast_2026_scenarios.xlsx` — local pub all three scenarios  
- `Excel files/forecast_model_comparison.csv` — local pub model comparison  
- `Excel files/riverside_forecast_2026.xlsx` — riverside pub neutral forecast  
- `Excel files/riverside_forecast_2026_scenarios.xlsx` — riverside pub all three scenarios  
- `Excel files/riverside_forecast_model_comparison.csv` — riverside pub model comparison  

### Results

- `Results/acf_pacf_plots.png`
- `Results/feature_importance_rf.png`
- `Results/forecast_comparison.png`
- `Results/residual_diagnostics.png`
- `Results/forecast_2026_scenarios.png`
- `Results/riverside_acf_pacf_plots.png`
- `Results/riverside_feature_importance_rf.png`
- `Results/riverside_forecast_comparison.png`
- `Results/riverside_residual_diagnostics.png`
- `Results/riverside_forecast_2026_scenarios.png`

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Streamlit
- Recommended IDE: PyCharm or VS Code

### Install dependencies

Run the following in your terminal:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn pmdarima
pip install streamlit plotly openpyxl reportlab

``` 

---

## Quick Start

### 1. Generate the datasets

Run both dataset generators once:

```bash
python "Dataset 2025.py"
python dataset_riverside_pub_2025.py
```

These generate:

- `Database/pub_sales.db`
- `Database/riverside_pub_sales.db`

If the databases already exist, they do not need to be regenerated.

### 2. Run model evaluation

Open `evaluate.py` and set the `PUB` variable at the top:

```python
PUB = "local"
# or
PUB = "riverside"
```

Then run:

```bash
python evaluate.py
```

Run it once for each pub so that all forecast files and model comparison outputs are generated.

This script will:

- print a full data quality report
- run all seven forecasting models
- perform ADF stationarity testing
- run auto ARIMA parameter search
- produce a model comparison table
- run walk-forward cross-validation
- generate charts, CSV files, and Excel forecast files

### 3. Launch the Streamlit app

After generating the required files for both pubs, run:

```bash
streamlit run app.py
```

The app will usually open at:

```text
http://localhost:8501
```

The first load may take some time while cached models are built.

---

## Switching Between Pubs

### In `evaluate.py`

Change the `PUB` variable:

```python
PUB = "local"
PUB = "riverside"
```

### In `app.py`

Go to the **Settings** tab and use the pub selector.

- selecting **Local Pub** loads the local dataset and outputs
- selecting **Riverside Pub** loads the riverside dataset and outputs

---

## Forecasting Models Included

The system compares seven forecasting models:

1. **Moving Average (7-day)** — naive baseline  
2. **Exponential Smoothing** — weighted recent data with damped trend  
3. **SARIMA** — seasonal time series model without exogenous variables  
4. **SARIMAX** — seasonal time series model with weather and event features  
5. **Decision Tree** — interpretable machine learning model  
6. **Random Forest** — ensemble model with strongest MAPE performance  
7. **Linear Regression** — baseline linear model using scaled features  

### Production model

The production model is **SARIMAX**.

It was selected because:

- Random Forest achieved the best MAPE on both datasets
- SARIMAX provides interpretable coefficients
- SARIMAX can quantify the effect of weather and events on sales
- this makes it more suitable for decision support in a hospitality setting

---

## Latest Model Results

### Local Pub

- Random Forest — **MAPE 15.05%** | **RMSE £2,967**
- Decision Tree — **MAPE 18.40%** | **RMSE £3,636**
- SARIMAX — **MAPE 20.18%** | **RMSE £3,081**
- Linear Regression — **MAPE 20.24%** | **RMSE £2,953**
- Exponential Smoothing — **MAPE 23.79%** | **RMSE £4,472**
- SARIMA — **MAPE 24.18%** | **RMSE £4,392**
- Moving Average — **MAPE 133.77%** | **RMSE £8,545**

### Riverside Pub

- Random Forest — **MAPE 15.52%** | **RMSE £3,968**
- Decision Tree — **MAPE 16.61%** | **RMSE £3,980**
- Linear Regression — **MAPE 22.09%** | **RMSE £4,116**
- SARIMAX — **MAPE 26.43%** | **RMSE £5,475**
- Exponential Smoothing — **MAPE 28.58%** | **RMSE £5,885**
- SARIMA — **MAPE 29.97%** | **RMSE £6,334**
- Moving Average — **MAPE 59.48%** | **RMSE £7,363**

---

## 2026 Scenario Forecasting

The system generates forecasts under three weather assumptions:

- **Optimistic**
- **Neutral**
- **Pessimistic**

### Local Pub — Average Daily Forecast (2026)

- Optimistic: **£22,123**
- Neutral: **£21,652**
- Pessimistic: **£21,029**

### Riverside Pub — Average Daily Forecast (2026)

- Optimistic: **£40,779**
- Neutral: **£39,781**
- Pessimistic: **£38,680**

These forecasts are generated using the full-year refitted SARIMAX model rather than the 80/20 split model, which avoids forecast decay across 2026.

---

## Streamlit App Pages

### Dashboard

- weather/forecast card for the selected date
- forecast sales and estimated reservations
- historical 2025 vs forecast 2026 sales chart
- weekly pattern bar chart
- model performance summary
- Random Forest feature importance
- Decision Tree key rules

### Inventory Management

- date range picker
- daily forecast bar chart
- stock order recommendations
- red / amber / green stock alerts
- stock order summary table

### Reports & Data

- full 2026 forecast table
- month filter
- CSV download
- MAPE and RMSE comparison charts
- feature importance chart

### Settings

- pub selector
- theme toggle
- PDF report generator
- About section

---

## Inventory Recommendation Logic

The app converts forecast sales into stock recommendations using item-specific sales ratios.

Tracked stock categories include:

- Lager (kegs)
- Ale (kegs)
- Wine (cases)
- Spirits (bottles)
- Soft drinks (cases)
- Food ingredients (£)

Each category uses:

- a sales ratio to estimate required stock
- a reorder point to determine urgency
- a maximum stock assumption
- a stock status rule:
  - **HIGH** — order urgently
  - **MEDIUM** — order soon
  - **LOW** — stock sufficient

---

## Important Notes

1. Do not run the dataset generators repeatedly unless you intentionally want to recreate the databases.  
2. Always run `evaluate.py` before launching `app.py`, because the app depends on the generated Excel and CSV files.  
3. Run `evaluate.py` for both pubs before using the pub switcher in the app.  
4. Fixed seeds are used for reproducibility:
   - Local pub: `42`
   - Riverside pub: `99`
5. The 2026 forecast is generated by refitting SARIMAX on all 365 days of 2025, which is necessary to avoid unrealistic forecast decay.  
6. If you are using PyCharm, running `evaluate.py` from the terminal is recommended if `matplotlib` windows block execution.

---

## Technical Stack

- **Language** — Python
- **Database** — SQLite (`sqlite3`)
- **Time-series modelling** — `statsmodels`
- **Auto ARIMA** — `pmdarima`
- **Machine learning** — `scikit-learn`
- **Data handling** — `pandas`, `numpy`
- **Static charts** — `matplotlib`
- **Interactive frontend** — `Streamlit`
- **Interactive charts** — `Plotly`
- **Excel export** — `openpyxl`
- **PDF export** — `reportlab`

---

## Reproducibility

The project is reproducible provided that:

1. the database generation scripts are run first  
2. `evaluate.py` is run for both pubs  
3. the generated output files are present in the correct folders  
4. the same fixed random seeds are preserved  

---

## Author

**Hsu Hnin Set @ Shalina Set**  
**Student ID:** 001343697
