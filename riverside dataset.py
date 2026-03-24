import pandas as pd
import numpy as np
import sqlite3
import os

if os.path.exists("riverside_pub_sales.db"):
    print("riverside_pub_sales.db already exists — skipping generation.")
    print("Delete the file manually if you want to regenerate.")
    exit()

np.random.seed(99)

print("Generating riverside pub dataset (London, 2025)...")

dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")

# sales rule


sales_rules = {
    1: {
        "Monday":    (6000,  9000),
        "Tuesday":   (6000,  9000),
        "Wednesday": (6500,  9500),
        "Thursday":  (9000,  13000),
        "Friday":    (12000, 17000),
        "Saturday":  (14000, 19000),
        "Sunday":    (13000, 18000),
    },
    2: {
        "Monday":    (6500,  9500),
        "Tuesday":   (6500,  9500),
        "Wednesday": (7000,  10000),
        "Thursday":  (9500,  13500),
        "Friday":    (12500, 17500),
        "Saturday":  (14500, 19500),
        "Sunday":    (13500, 18500),
    },
    3: {
        "Monday":    (7000,  9000),
        "Tuesday":   (7000,  10500),
        "Wednesday": (7500,  11000),
        "Thursday":  (10500, 15000),
        "Friday":    (14000, 19000),
        "Saturday":  (16000, 22000),
        "Sunday":    (15000, 21000),
    },
    4: {
        "Monday":    (6500,  10000),
        "Tuesday":   (8000,  12000),
        "Wednesday": (8500,  12500),
        "Thursday":  (12000, 17000),
        "Friday":    (16000, 22000),
        "Saturday":  (19000, 26000),
        "Sunday":    (18000, 25000),
    },
    5: {
        "Monday":    (7000,  10000),
        "Tuesday":   (9000,  13000),
        "Wednesday": (9500,  13500),
        "Thursday":  (13000, 18000),
        "Friday":    (17000, 23000),
        "Saturday":  (21000, 28000),
        "Sunday":    (20000, 27000),
    },
    6: {
        "Monday":    (6500, 10000),
        "Tuesday":   (10000, 15000),
        "Wednesday": (11000, 16000),
        "Thursday":  (15000, 21000),
        "Friday":    (20000, 27000),
        "Saturday":  (25000, 33000),
        "Sunday":    (24000, 32000),
    },
    7: {
        "Monday":    (7000, 10000),
        "Tuesday":   (11000, 16000),
        "Wednesday": (12000, 17000),
        "Thursday":  (16000, 22000),
        "Friday":    (21000, 29000),
        "Saturday":  (27000, 36000),
        "Sunday":    (26000, 35000),
    },
    8: {
        "Monday":    (7000, 10500),
        "Tuesday":   (10500, 15500),
        "Wednesday": (11500, 16500),
        "Thursday":  (15500, 21500),
        "Friday":    (20500, 28000),
        "Saturday":  (26000, 34000),
        "Sunday":    (25000, 33000),
    },
    9: {
        "Monday":    (6000,  10000),
        "Tuesday":   (9000,  13500),
        "Wednesday": (9500,  14000),
        "Thursday":  (13500, 19000),
        "Friday":    (18000, 24000),
        "Saturday":  (22000, 29000),
        "Sunday":    (21000, 28000),
    },
    10: {
        "Monday":    (7500,  10000),
        "Tuesday":   (8500,  12500),
        "Wednesday": (9000,  13000),
        "Thursday":  (12500, 17500),
        "Friday":    (17000, 23000),
        "Saturday":  (20000, 27000),
        "Sunday":    (19000, 26000),
    },
    11: {
        "Monday":    (8500,  10000),
        "Tuesday":   (7500,  11000),
        "Wednesday": (8000,  11500),
        "Thursday":  (11000, 15500),
        "Friday":    (15500, 21000),
        "Saturday":  (18000, 24000),
        "Sunday":    (17000, 23000),
    },
    12: {
        "Monday":    (9000, 14000),
        "Tuesday":   (10000, 15000),
        "Wednesday": (11000, 16000),
        "Thursday":  (15000, 21000),
        "Friday":    (22000, 27000),
        "Saturday":  (25000, 32000),
        "Sunday":    (25000, 30000),
    },
}

# London weather — more rain than first pub, less cold
weather_by_month = {
    1:  ["Cloudy"] * 12 + ["Rainy"] * 10 + ["Cold"] * 7 + ["Sunny"] * 2,
    2:  ["Cloudy"] * 11 + ["Rainy"] * 9  + ["Cold"] * 6 + ["Sunny"] * 2,
    3:  ["Cloudy"] * 12 + ["Rainy"] * 8  + ["Cold"] * 3 + ["Sunny"] * 8,
    4:  ["Cloudy"] * 10 + ["Rainy"] * 8  + ["Sunny"] * 10 + ["Cold"] * 2,
    5:  ["Cloudy"] * 9  + ["Rainy"] * 7  + ["Sunny"] * 14,
    6:  ["Sunny"] * 16  + ["Cloudy"] * 8 + ["Rainy"] * 6,
    7:  ["Sunny"] * 18  + ["Cloudy"] * 8 + ["Rainy"] * 5,
    8:  ["Sunny"] * 17  + ["Cloudy"] * 8 + ["Rainy"] * 6,
    9:  ["Cloudy"] * 12 + ["Sunny"] * 10 + ["Rainy"] * 8,
    10: ["Cloudy"] * 13 + ["Rainy"] * 10 + ["Sunny"] * 5 + ["Cold"] * 3,
    11: ["Cloudy"] * 12 + ["Rainy"] * 11 + ["Cold"] * 5 + ["Sunny"] * 2,
    12: ["Cloudy"] * 11 + ["Rainy"] * 9  + ["Cold"] * 8 + ["Sunny"] * 3,
}

# weather multipliers
weather_multipliers = {
    "Sunny":  1.18,   # terrace packed, great for riverside
    "Cloudy": 1.00,   # baseline
    "Rainy":  0.62,   # terrace empty, much bigger hit than local pub
    "Cold":   0.88,   # cold but covered areas still used
}

# events date
event_multipliers = {
    "2025-01-01": 1.40,   # New Year's Day
    "2025-02-14": 1.35,   # Valentine's Day — big for riverside
    "2025-03-17": 1.25,   # St Patrick's Day
    "2025-04-18": 1.30,   # Good Friday
    "2025-04-20": 1.35,   # Easter Sunday
    "2025-04-21": 1.20,   # Easter Monday
    "2025-05-05": 1.25,   # Early May bank holiday
    "2025-05-26": 1.25,   # Spring bank holiday
    "2025-08-25": 1.30,   # Summer bank holiday
    "2025-10-31": 1.25,   # Halloween
    "2025-11-05": 1.20,   # Bonfire Night
    "2025-12-24": 1.45,   # Christmas Eve — big night out
    "2025-12-25": 0.40,   # Christmas Day — closed/quiet
    "2025-12-26": 1.30,   # Boxing Day
    "2025-12-31": 1.50,   # New Year's Eve — biggest night
}

# generate data
records = []

for date in dates:
    month   = date.month
    weekday = date.strftime("%A")
    date_str = str(date.date())

    # base sales from rules
    low, high = sales_rules[month][weekday]
    base_sales = np.random.randint(low, high)

    # weather for the day
    weather_pool = weather_by_month[month]
    weather = np.random.choice(weather_pool)

    # apply weather multiplier
    sales = base_sales * weather_multipliers[weather]

    # apply event multiplier if applicable
    if date_str in event_multipliers:
        sales *= event_multipliers[date_str]

    # add small random noise (±3%)
    noise = np.random.uniform(0.97, 1.03)
    sales = round(sales * noise, 2)

    # reservations — riverside pub books more (higher reservation rate)
    # roughly 1 reservation per £60 of sales, max 280 covers
    reservations = min(280, max(15, int(sales / 60) + np.random.randint(-5, 10)))

    records.append({
        "date":         date_str,
        "weekday":      weekday,
        "weather":      weather,
        "reservations": reservations,
        "sales_gbp":    sales,
    })

riverside_sales = pd.DataFrame(records)

# database saved
conn = sqlite3.connect("riverside_pub_sales.db")
riverside_sales.to_sql("daily_sales", conn, if_exists="replace", index=False)
conn.close()

# save excel sheet
riverside_sales.to_excel("riverside_pub_sales_2025.xlsx", index=False)

# summary
print(f"\nDataset generated successfully!")
print(f"Total rows      : {len(riverside_sales)}")
print(f"Date range      : {riverside_sales['date'].min()} to {riverside_sales['date'].max()}")
print(f"Mean daily sales: £{riverside_sales['sales_gbp'].mean():,.2f}")
print(f"Min daily sales : £{riverside_sales['sales_gbp'].min():,.2f}")
print(f"Max daily sales : £{riverside_sales['sales_gbp'].max():,.2f}")

print(f"\nSales by weekday (average):")
riverside_sales["date_dt"] = pd.to_datetime(riverside_sales["date"])
dow_avg = riverside_sales.groupby("weekday")["sales_gbp"].mean()
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
for day in day_order:
    print(f"  {day:<12}: £{dow_avg[day]:,.0f}")

print(f"\nWeather distribution:")
print(riverside_sales["weather"].value_counts().to_string())

print(f"\nTop 5 highest sales days:")
print(riverside_sales.nlargest(5, "sales_gbp")[["date","weekday","weather","sales_gbp"]].to_string(index=False))

print(f"\nSaved to: riverside_pub_sales.db")
print(f"Saved to: riverside_pub_sales_2025.xlsx")
print("\nDone.")