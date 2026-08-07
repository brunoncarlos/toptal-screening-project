
# 1. Import required libraries
import pycountry
from zoneinfo import ZoneInfo
import pandas as pd
import pickle
from unidecode import unidecode

# 3. Cache for resolved timezones (avoids repeated lookups)
timezone_cache = {}

# 4. Convert country names to ISO-2 codes
def country_to_iso(country):
    try:
        return pycountry.countries.lookup(country).alpha_2
    except LookupError:
        return None

# 5. Resolve timezone using country/city string (e.g., "Spain/Madrid")
def resolve_timezone(country_city):
    # 5.1 Check cache first
    if country_city in timezone_cache:
        return timezone_cache[country_city]

    # 5.2 Split "Country/City"
    country, city = country_city.split("/")

    # 5.3 Convert country name → ISO code
    country_code = country_to_iso(country)
    if country_code is None:
        timezone_cache[country_city] = "UTC"
        return "UTC"

    # 5.4 Exact match lookup
    key = (country_code, city)
    if key in timezone_lookup:
        tz = timezone_lookup[key]
        timezone_cache[country_city] = tz
        return tz

    # 5.5 ASCII fallback lookup
    key = (country_code, city)
    if key in timezone_lookup:
        tz = timezone_lookup[key]
        timezone_cache[country_city] = tz
        return tz

    # 5.6 Safe fallback
    timezone_cache[country_city] = "UTC"
    return "UTC"

# 6. Add temporal features based on local time
def add_local_time_features(df):
    df = df.copy()

    # 6.1 Build UTC timestamp from date + time
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        utc=True
    )

    # 6.2 Prepare lists for new features
    local_hours = []
    local_days = []
    is_weekend = []
    time_bins = []
    is_night = []

    # 6.3 Convert each timestamp to local time
    for ts, loc in zip(df["timestamp"], df["location"]):

        # 6.3.1 Resolve timezone
        tz_name = resolve_timezone(loc)

        # 6.3.2 Convert UTC → local timezone
        local_ts = ts.astimezone(ZoneInfo(tz_name))

        # 6.3.3 Extract hour and weekday
        hour = local_ts.hour
        day = local_ts.weekday()

        local_hours.append(hour)
        local_days.append(day)
        is_weekend.append(day >= 5)
        is_night.append(hour >= 23 or hour <= 4)

        # 6.3.4 Time-of-day binning
        if 5 <= hour <= 11:
            time_bins.append("morning")
        elif 12 <= hour <= 17:
            time_bins.append("afternoon")
        elif 18 <= hour <= 22:
            time_bins.append("evening")
        else:
            time_bins.append("night")

    # 6.4 Assign new temporal features
    df["local_hour"] = local_hours
    df["local_day_of_week"] = local_days
    df["is_local_weekend"] = is_weekend
    df["time_bin"] = time_bins
    df["is_night_session"] = is_night

    # 7. Manual one-hot encoding for time_bin
    for cat in ["morning", "afternoon", "evening", "night"]:
        df[f"time_bin_{cat}"] = (df["time_bin"] == cat).astype(int)

    # 8. Drop raw columns no longer needed
    df = df.drop(columns=['timestamp', 'timezone', 'location', 'time', 'date', 'time_bin'])

    return df

# # Example usage:
# from temporal_features import resolve_timezone, add_local_time_features

# df = train_df.sample(n=1000, random_state=42).copy()
# df["timezone"] = df["location"].apply(resolve_timezone)
# df = add_local_time_features(df)

# TEMPORAL_FEATURES = [
#     col for col in df.columns
#     if col not in train_df.columns
# ]

# print("TEMPORAL_FEATURES:\n", TEMPORAL_FEATURES)
# df.head()
