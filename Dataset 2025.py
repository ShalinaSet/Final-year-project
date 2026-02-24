import pandas as pd
import numpy as np
import sqlite3
import os

if os.path.exists("pub_sales.db"):
    print("✓ pub_sales.db already exists — skipping regeneration.")
    print("  If you want to regenerate, manually delete pub_sales.db first.")
    exit()

# Fixed seed — ensures identical dataset every time this script runs
np.random.seed(42)

# Create date range
dates = pd.date_range(start="2025-01-01", end="2025-12-31")

# Sales rules by month and days
rules = {
    1: {"Monday": (2000,3000), "Tuesday": (3000,6000), "Wednesday": (4500,8000),
        "Thursday": (8000,12000), "Friday": (4000,7000), "Saturday": (2000,3500),
        "Sunday": (800,1500)},
    2: {"Monday": (2500,3500), "Tuesday": (4000,7000), "Wednesday": (7000,10000),
        "Thursday": (10000,17000), "Friday": (6000,11000), "Saturday": (3400,4500),
        "Sunday": (1000,1800)},
    3: {"Monday": (3000,4000), "Tuesday": (4500,8000), "Wednesday": (8000,13000),
        "Thursday": (14000,19000), "Friday": (8000,12000), "Saturday": (3600,4800),
        "Sunday": (1200,2000)},
    4: {"Monday": (3300,4400), "Tuesday": (6000,9000), "Wednesday": (9000,14000),
        "Thursday": (17000,21000), "Friday": (9000,13000), "Saturday": (3800,4800),
        "Sunday": (1200,2200)},
    5: {"Monday": (3300,4500), "Tuesday": (6000,10000), "Wednesday": (10000,14000),
        "Thursday": (17000,22000), "Friday": (9000,13000), "Saturday": (3900,5000),
        "Sunday": (1300,2300)},
    6: {"Monday": (3300,4500), "Tuesday": (6000,10000), "Wednesday": (10000,14000),
        "Thursday": (17500,23000), "Friday": (9000,13000), "Saturday": (3900,5000),
        "Sunday": (1400,2500)},
    7: {"Monday": (3500,4800), "Tuesday": (8000,12000), "Wednesday": (12000,16000),
        "Thursday": (20000,24000), "Friday": (9000,14000), "Saturday": (4000,5000),
        "Sunday": (1500,2800)},
    8: {"Monday": (4000,5000), "Tuesday": (9000,12000), "Wednesday": (12000,17000),
        "Thursday": (20000,25000), "Friday": (10000,14000), "Saturday": (4500,5000),
        "Sunday": (1500,2800)},
    9: {"Monday": (4000,5000), "Tuesday": (10000,13000), "Wednesday": (13000,17000),
        "Thursday": (22000,26000), "Friday": (11000,14000), "Saturday": (5000,6000),
        "Sunday": (1400,2500)},
    10: {"Monday": (4000,5000), "Tuesday": (11000,13000), "Wednesday": (13000,18000),
         "Thursday": (23000,27000), "Friday": (12000,15000), "Saturday": (5000,6000),
         "Sunday": (1400,2400)},
    11: {"Monday": (5000,6000), "Tuesday": (13000,16000), "Wednesday": (15000,19000),
         "Thursday": (24000,28000), "Friday": (14000,18000), "Saturday": (5000,6500),
         "Sunday": (1500,2500)},
    12: {"Monday": (6000,7500), "Tuesday": (15000,18000), "Wednesday": (19000,20000),
         "Thursday": (28000,29000), "Friday": (15000,21000), "Saturday": (6000,7500),
         "Sunday": (2000,3500)}
}

# Weather multipliers
weather_multiplier = {
    "Sunny": 1.05,
    "Cloudy": 1.00,
    "Cold":   1.10,
    "Rainy":  0.70
}

# Probabilistic weather per season
weather_probs = {
    "winter": [0.15, 0.40, 0.30, 0.15],  # Sunny, Cloudy, Rainy, Cold
    "spring": [0.50, 0.25, 0.15, 0.10],
    "summer": [0.45, 0.25, 0.20, 0.10],
    "autumn": [0.25, 0.30, 0.30, 0.15]
}

weather_options = ["Sunny", "Cloudy", "Rainy", "Cold"]

# Event boosts
event_boosts = {
    "2025-01-01": 1.15,  # New Year's Day
    "2025-10-31": 1.10,  # Halloween
    "2025-12-24": 1.20,  # Christmas Eve
    "2025-12-25": 1.25,  # Christmas Day
    "2025-12-26": 1.15   # Boxing Day
}

# Reservation trends
month_weights = {
    1: 0.07, 2: 0.07, 3: 0.08, 4: 0.09, 5: 0.10, 6: 0.10,
    7: 0.10, 8: 0.10, 9: 0.08, 10: 0.07, 11: 0.07, 12: 0.17
}

weekday_weights = {
    "Monday": 0.8, "Tuesday": 0.9, "Wednesday": 1.0,
    "Thursday": 1.1, "Friday": 1.3, "Saturday": 1.3, "Sunday": 0.5
}

# Generate dataset
data = []

for date in dates:
    month = date.month
    weekday = date.day_name()

    low, high = rules[month][weekday]
    base_sales = np.random.uniform(low, high)

    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    else:
        season = "autumn"

    weather = np.random.choice(weather_options, p=weather_probs[season])
    sales = base_sales * weather_multiplier[weather]

    date_str = date.strftime("%Y-%m-%d")
    if date_str in event_boosts:
        sales *= event_boosts[date_str]

    reservations = int(
        (sales / 85) * month_weights[month] * weekday_weights[weekday]
        + np.random.randint(3, 15)
    )
    reservations = min(reservations, 220)

    data.append([date, weekday, month, reservations, weather, round(sales, 2)])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["date", "weekday", "month", "reservations", "weather", "sales_gbp"])

print(f"Total rows generated : {len(df)}")
print(f"\nWeekday distribution :\n{df['weekday'].value_counts().sort_index()}")
print(f"\nSales summary:")
print(df["sales_gbp"].describe().apply(lambda x: f"£{x:,.2f}"))

# Save to SQLite
conn = sqlite3.connect("pub_sales.db")
df.to_sql("daily_sales", conn, if_exists="replace", index=False)
conn.close()
print("\n Dataset saved to 'pub_sales.db'")

# Save to Excel
df.to_excel("daily_sales_2025.xlsx", index=False)
print("Dataset saved to 'daily_sales_2025.xlsx'")
print("Dataset is now locked. Do not run this script again unless you intentionally want to regenerate — delete pub_sales.db first.")