import pandas as pd
import numpy as np
import sqlite3
import os

EXCEL_FOLDER = "Excel files"
DATABASE_FOLDER = "Database"
RESULTS_FOLDER  = "Results"

os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(EXCEL_FOLDER,    exist_ok=True)

db_path = os.path.join(DATABASE_FOLDER, "riverside_pub_sales.db")

np.random.seed(99)  # fixed seed for reproducibility

print("Generating riverside pub dataset (London, 2025)...")
print("Seed: 99 | Year: 2025 | 365 days")
print("-" * 50)

dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")

# Sales rules
sales_rules = {
    1: {
        "Monday":    (4000,  5500),
        "Tuesday":   (5000,  6500),
        "Wednesday": (9000,  10500),
        "Thursday":  (14000, 17000),
        "Friday":    (15500, 17500),
        "Saturday":  (4500,  6500),
        "Sunday":    (3500,  5500),
    },
    2: {
        "Monday":    (4000,  5500),
        "Tuesday":   (5500,  7000),
        "Wednesday": (10000, 11500),
        "Thursday":  (14000, 17000),
        "Friday":    (15500, 18000),
        "Saturday":  (4500,  6500),
        "Sunday":    (3500,  6000),
    },
    3: {
        "Monday":    (4500,  5500),
        "Tuesday":   (5500,  7000),
        "Wednesday": (11000, 12500),
        "Thursday":  (14500, 17500),
        "Friday":    (16000, 19500),
        "Saturday":  (6000,  9000),
        "Sunday":    (5000,  8000),
    },
    4: {
        "Monday":    (5500,  6500),
        "Tuesday":   (7000,  9000),
        "Wednesday": (11500, 14500),
        "Thursday":  (14500, 18500),
        "Friday":    (17000, 19500),
        "Saturday":  (6000,  8000),
        "Sunday":    (4500,  7000),
    },
    5: {
        "Monday":    (5500,  7000),
        "Tuesday":   (7500,  10000),
        "Wednesday": (12500, 15500),
        "Thursday":  (14500, 19000),
        "Friday":    (17000, 20500),
        "Saturday":  (7000,  9000),
        "Sunday":    (5500,  7500),
    },
    6: {
        "Monday":    (5500,  7000),
        "Tuesday":   (7500,  10000),
        "Wednesday": (13000, 16000),
        "Thursday":  (15000, 21000),
        "Friday":    (17000, 23000),
        "Saturday":  (7000,  9000),
        "Sunday":    (6000,  8000),
    },
    7: {
        "Monday":    (6000,  7500),
        "Tuesday":   (8000,  10500),
        "Wednesday": (14000, 18000),
        "Thursday":  (18000, 25000),
        "Friday":    (20000, 27000),
        "Saturday":  (7500,  9500),
        "Sunday":    (6500,  8500),
    },
    8: {
        "Monday":    (5500,  7000),
        "Tuesday":   (7500,  10000),
        "Wednesday": (12500, 15500),
        "Thursday":  (14500, 19000),
        "Friday":    (17000, 20500),
        "Saturday":  (7500,  9000),
        "Sunday":    (6500,  8500),
    },
    9: {
        "Monday":    (5500,  7000),
        "Tuesday":   (7500,  10000),
        "Wednesday": (12500, 15500),
        "Thursday":  (14500, 19000),
        "Friday":    (17000, 20500),
        "Saturday":  (6500,  8500),
        "Sunday":    (5500,  7500),
    },
    10: {
        "Monday":    (5500,  7000),
        "Tuesday":   (7500,  10000),
        "Wednesday": (12500, 15500),
        "Thursday":  (14500, 19000),
        "Friday":    (17000, 20500),
        "Saturday":  (6500,  8500),
        "Sunday":    (5500,  7500),
    },
    11: {
        "Monday":    (5500,  7500),
        "Tuesday":   (7500,  10000),
        "Wednesday": (13000, 17500),
        "Thursday":  (16000, 23000),
        "Friday":    (18000, 25000),
        "Saturday":  (7000,  9000),
        "Sunday":    (6000,  8000),
    },
    12: {
        "Monday":    (6000,  8500),
        "Tuesday":   (8000,  11000),
        "Wednesday": (14000, 20000),
        "Thursday":  (18000, 30000),
        "Friday":    (20000, 32000),
        "Saturday":  (9000,  12000),
        "Sunday":    (7000,  10000),
    },
}

# Weather multipliers
weather_multiplier = {
    "Sunny": 1.15,
    "Cloudy": 0.95,
    "Cold":   1.00,
    "Rainy":  0.70
}

# Probabilistic weather per season
weather_probs = { # Sunny, Cloudy, Rainy, Cold
    "winter": [0.15, 0.40, 0.30, 0.15],
    "spring": [0.50, 0.25, 0.15, 0.10],
    "summer": [0.45, 0.25, 0.20, 0.10],
    "autumn": [0.25, 0.30, 0.30, 0.15]
}

weather_options = ["Sunny", "Cloudy", "Rainy", "Cold"]

# event dates
event_multipliers = {
    "2025-01-01": 1.25,   # New Year's Day
    "2025-02-14": 1.30,   # Valentine's Day
    "2025-03-17": 1.20,   # St Patrick's Day
    "2025-04-20": 1.15,   # Easter Sunday
    "2025-05-01": 1.20,   # Early May bank holiday Thursday
    "2025-05-08": 1.20,   # Bank holiday Thursday
    "2025-05-26": 1.15,   # Spring bank holiday
    "2025-08-25": 1.20,   # Summer bank holiday
    "2025-10-31": 1.20,   # Halloween
    "2025-11-05": 1.15,   # Bonfire Night
    "2025-12-04": 1.20,   # First Christmas party Thursday of December
    "2025-12-11": 1.25,   # Christmas party peak Thursday
    "2025-12-12": 1.25,   # Christmas party peak Friday
    "2025-12-18": 1.30,   # Last big Thursday before Christmas
    "2025-12-19": 1.30,   # Last big Friday before Christmas
    "2025-12-24": 1.40,   # Christmas Eve
    "2025-12-25": 1.45,   # Christmas Day
    "2025-12-26": 1.25,   # Boxing Day
    "2025-12-31": 1.45,   # New Year's Eve — biggest night
}

# generate data
records = []

for date in dates:
    month    = date.month
    weekday  = date.strftime("%A")
    date_str = str(date.date())

    low, high  = sales_rules[month][weekday]
    base_sales = np.random.randint(low, high)

    # season
    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    else:
        season = "autumn"

    # correct weather selection
    weather = np.random.choice(weather_options, p=weather_probs[season])

    # correct multiplier usage
    sales = base_sales * weather_multiplier[weather]

    # event multiplier
    if date_str in event_multipliers:
        sales *= event_multipliers[date_str]

    noise = np.random.uniform(0.97, 1.03)
    sales = round(sales * noise, 2)

    base_res = int(sales / 55)
    if weekday in ["Thursday", "Friday"] and weather == "Sunny":
        base_res = int(base_res * 1.4)
    elif weekday in ["Saturday", "Sunday"] and weather == "Sunny":
        base_res = int(base_res * 1.2)

    reservations = min(280, max(10, base_res + np.random.randint(-3, 8)))

    records.append({
        "date": date_str,
        "weekday": weekday,
        "weather": weather,
        "reservations": reservations,
        "sales_gbp": sales,
    })

riverside_sales = pd.DataFrame(records)

conn = sqlite3.connect(db_path)
riverside_sales.to_sql("daily_sales", conn, if_exists="replace", index=False)
conn.close()

riverside_sales.to_excel(os.path.join(EXCEL_FOLDER, "riverside_pub_sales_2025.xlsx"), index=False)

print("\nDataset generated successfully!")
print(f"Total rows      : {len(riverside_sales)}")
print(f"Date range      : {riverside_sales['date'].min()} to {riverside_sales['date'].max()}")
print(f"Mean daily sales: £{riverside_sales['sales_gbp'].mean():,.2f}")
print(f"Min daily sales : £{riverside_sales['sales_gbp'].min():,.2f}")
print(f"Max daily sales : £{riverside_sales['sales_gbp'].max():,.2f}")
print("\nDone.")