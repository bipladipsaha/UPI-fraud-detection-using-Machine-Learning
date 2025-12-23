import random
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# ----------------------------
# City & Coordinates
# ----------------------------
CITIES = ["Kolkata", "Delhi", "Mumbai", "Bangalore", "Chennai"]

CITY_COORDS = {
    "Kolkata": (22.5726, 88.3639),
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707)
}

# ----------------------------
# Distance function
# ----------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c


# ----------------------------
# MAIN FEATURE FUNCTION
# ----------------------------
def add_location_device_features(df, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Synthetic cities
    df["user_city"] = [random.choice(CITIES) for _ in range(len(df))]
    df["merchant_city"] = [random.choice(CITIES) for _ in range(len(df))]

    # Distance
    df["distance_km"] = df.apply(
        lambda x: haversine(
            CITY_COORDS[x["user_city"]][0],
            CITY_COORDS[x["user_city"]][1],
            CITY_COORDS[x["merchant_city"]][0],
            CITY_COORDS[x["merchant_city"]][1],
        ),
        axis=1
    )

    df["same_city"] = (df["user_city"] == df["merchant_city"]).astype(int)

    # Device type
    df["device_type"] = np.random.choice(
        ["Android", "iOS", "Web"],
        size=len(df),
        p=[0.6, 0.25, 0.15]
    )

    return df
