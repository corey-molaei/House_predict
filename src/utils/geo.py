import math
from typing import Iterable, Tuple


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points in km."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def distance_to_cbd_km(lat: Iterable[float], lon: Iterable[float]) -> list[float]:
    """Compute distance to Sydney CBD for each lat/lon pair."""
    cbd_lat, cbd_lon = -33.8688, 151.2093
    distances = []
    for la, lo in zip(lat, lon):
        if la is None or lo is None or math.isnan(la) or math.isnan(lo):
            distances.append(float("nan"))
        else:
            distances.append(haversine_km(la, lo, cbd_lat, cbd_lon))
    return distances
