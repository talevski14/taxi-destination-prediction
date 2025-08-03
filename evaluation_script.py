import math

def haversine_distance(lat1, lon1, lat2, lon2):
    # Returns distance in kilometers
    R_earth = 6371  # Earth's radius in km

    # Convert degrees to radians
    lat_diff = math.radians(abs(lat1 - lat2))
    lon_diff = math.radians(abs(lon1 - lon2))
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    # Haversine formula
    a = math.sin(lat_diff / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(lon_diff / 2) ** 2
    d = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R_earth * d

    return distance