import time
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests


# -----------------------------
# SETTINGS
# -----------------------------
INPUT_CSV = "tourist_attractions.csv"
OUTPUT_CSV = "tourist_attractions_FIXED.csv"
REPORT_CSV = "fix_report.csv"
CACHE_JSON = "geocode_cache.json"

# Nominatim public endpoint (OpenStreetMap). Respect usage policy: rate limit & User-Agent. :contentReference[oaicite:2]{index=2}
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

# IMPORTANT: Put a real contact email in User-Agent (usage policy expects identifiable UA). :contentReference[oaicite:3]{index=3}
USER_AGENT = "SmartTourismCSVFixer/1.0 (contact: umahinsab@gmail.com)"

# Public Nominatim policy: keep it gentle (<= 1 req/s). :contentReference[oaicite:4]{index=4}
REQUEST_DELAY_SEC = 1.1
TIMEOUT_SEC = 12

# Sri Lanka bounding box sanity check (WGS84)
SL_LAT_MIN, SL_LAT_MAX = 5.5, 10.2
SL_LON_MIN, SL_LON_MAX = 79.3, 82.2

# Confidence thresholds
MIN_IMPORTANCE = 0.30        # Nominatim 'importance' threshold (0..1). Lower means more ambiguity.
MAX_KM_IF_MINOR_ADJUST = 25  # If new coordinate is close to old, accept with lower importance.


# -----------------------------
# CATEGORY FIXES
# -----------------------------
# Deterministic overrides for known mislabels in your CSV dump. :contentReference[oaicite:5]{index=5}
CATEGORY_OVERRIDES = {
    "Trincomalee Beaches": "beach",
    "Batticaloa": "city",
    "Mannar Island": "beach",
    "Aukana Buddha Statue": "cultural",
    "Lipton's Seat": "mountain",
    "Bambarakanda Falls": "waterfall",
    "Ravana Falls": "waterfall",
    "Kalpitiya": "beach",
    "Nilaveli Beach": "beach",
    "Matara": "city",
    "Hambantota": "city",
    "Kataragama Temple": "temple",
    "Belilena Cave": "adventure",
    "Pidurangala Rock": "adventure",
    "Bundala National Park": "national_park",
    "Kumana National Park": "national_park",
    "Maduru Oya National Park": "national_park",
    "Gal Oya National Park": "national_park",
    "Tea Plantations Nuwara Eliya": "cultural",
    "Bandarawela": "city",
    "Mahiyangana": "cultural",
    "Ratnapura Gem Mines": "cultural",
    "Kitulgala Forest": "adventure",
    "Madu River Safari": "adventure",
    "Koggala Lake": "cultural",
    "Bentara River": "adventure",
    "Gregory Lake": "city",
    "Victoria Park Nuwara Eliya": "city",
    "Hakgala Botanical Garden": "cultural",
    "Handunugoda Tea Estate": "cultural",
    "Saman Villas": "city",
    "Geoffrey Bawa House": "cultural",
    "Independence Square": "city",
    "Viharamahadevi Park": "city",
    "Dehiwala Zoo": "wildlife",
    "Mount Lavinia Beach": "beach",
    "Dutch Hospital Shopping": "city",
    "Red Mosque": "temple",
    "Jami Ul Alfar Mosque": "temple",
    "St. Mary's Church": "temple",
    "Nallur Kovil": "temple",
    "Nagadeepa Temple": "temple",
    "Thiruketheeswaram Kovil": "temple",
    "Embekke Devalaya": "temple",
    "Lankatilaka Temple": "temple",
    "Gadaladeniya Temple": "temple",
    "Aluvihare Rock Temple": "temple",
    "Alu Vihara": "temple",
    "Kalutara Temple": "temple",
    "Kande Viharaya": "temple",
    "Seema Malaka": "temple",
    "Kandy Lake": "city",
    "Beira Lake": "city",
    "Dambulla Market": "city",
}

# Keyword-based fallback rules (applied only when no explicit override exists)
KEYWORD_CATEGORY_RULES = [
    (r"\b(temple|kovil|viharaya|vihara|mosque|church|dagoba|stupa|devalaya)\b", "temple"),
    (r"\b(beach|bay|island)\b", "beach"),
    (r"\b(national park|forest reserve|park)\b", "national_park"),
    (r"\b(falls|waterfall)\b", "waterfall"),
    (r"\b(rock|hike|hiking|rafting|safari|cave)\b", "adventure"),
    (r"\b(lake|river|gardens|botanical|estate|plantations|museum|fort)\b", "cultural"),
    (r"\b(city|market|square|shopping|tour)\b", "city"),
]


# -----------------------------
# GEO HELPERS
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def in_sri_lanka(lat: float, lon: float) -> bool:
    return (SL_LAT_MIN <= lat <= SL_LAT_MAX) and (SL_LON_MIN <= lon <= SL_LON_MAX)


# -----------------------------
# NOMINATIM (SEARCH + REVERSE)
# -----------------------------
class NominatimClient:
    def __init__(self, cache_path: str):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, Any] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def save_cache(self):
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sleep(self):
        time.sleep(REQUEST_DELAY_SEC)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Free-form search. Docs: Nominatim Search API. :contentReference[oaicite:6]{index=6}
        """
        cache_key = f"search::{query.lower().strip()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 5,
            "countrycodes": "lk",
            "accept-language": "en",
        }

        r = self.session.get(NOMINATIM_SEARCH_URL, params=params, timeout=TIMEOUT_SEC)
        self._sleep()
        r.raise_for_status()
        data = r.json()
        self.cache[cache_key] = data
        return data

    def reverse(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Reverse lookup. Docs: Nominatim Reverse API. :contentReference[oaicite:7]{index=7}
        """
        cache_key = f"reverse::{round(lat,6)},{round(lon,6)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
        }
        r = self.session.get(NOMINATIM_REVERSE_URL, params=params, timeout=TIMEOUT_SEC)
        self._sleep()
        if r.status_code != 200:
            return None
        data = r.json()
        self.cache[cache_key] = data
        return data


def choose_best_geocode_result(
    name: str,
    results: List[Dict[str, Any]],
    old_lat: float,
    old_lon: float,
    expected_category: str
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Returns (lat, lon) if confident, else None with reason.
    """
    if not results:
        return None, "no_results"

    # Prefer results whose display_name contains Sri Lanka
    filtered = [r for r in results if "Sri Lanka" in r.get("display_name", "")]
    if not filtered:
        filtered = results

    # Heuristic: prefer certain OSM types based on expected_category
    # (Nominatim returns class/type fields sometimes; not guaranteed.)
    cat_hint = expected_category.lower()
    prefer_keywords = []
    if cat_hint == "temple":
        prefer_keywords = ["temple", "vihara", "mosque", "church", "kovil", "stupa", "dagoba"]
    elif cat_hint == "beach":
        prefer_keywords = ["beach", "bay", "coast"]
    elif cat_hint == "national_park":
        prefer_keywords = ["national park", "park", "forest"]
    elif cat_hint == "waterfall":
        prefer_keywords = ["falls", "waterfall"]
    elif cat_hint == "city":
        prefer_keywords = ["city", "market", "square", "shopping"]
    else:
        prefer_keywords = []

    def score(r: Dict[str, Any]) -> float:
        imp = float(r.get("importance") or 0.0)
        disp = (r.get("display_name") or "").lower()
        bonus = 0.0
        for kw in prefer_keywords:
            if kw in disp:
                bonus += 0.05
        # Prefer closer to old point slightly (helps when old point is roughly right)
        try:
            lat = float(r["lat"]); lon = float(r["lon"])
            dist = haversine_km(old_lat, old_lon, lat, lon)
        except Exception:
            dist = 9999.0
        # Penalize very far moves unless the importance is strong
        far_penalty = min(dist / 400.0, 1.0) * 0.2
        return imp + bonus - far_penalty

    ranked = sorted(filtered, key=score, reverse=True)
    best = ranked[0]

    try:
        new_lat = float(best["lat"])
        new_lon = float(best["lon"])
    except Exception:
        return None, "parse_error"

    if not in_sri_lanka(new_lat, new_lon):
        return None, "outside_sri_lanka"

    importance = float(best.get("importance") or 0.0)
    dist_km = haversine_km(old_lat, old_lon, new_lat, new_lon)

    # Accept if:
    # - high importance OR
    # - small adjustment (close to old)
    if importance >= MIN_IMPORTANCE or dist_km <= MAX_KM_IF_MINOR_ADJUST:
        return (new_lat, new_lon), f"accepted(importance={importance:.2f},dist_km={dist_km:.1f})"

    return None, f"low_confidence(importance={importance:.2f},dist_km={dist_km:.1f})"


# -----------------------------
# CATEGORY NORMALIZATION
# -----------------------------
def normalize_category(name: str, current: str) -> str:
    if name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[name]

    n = name.lower()

    for pattern, cat in KEYWORD_CATEGORY_RULES:
        if re.search(pattern, n, flags=re.IGNORECASE):
            return cat

    # fallback: keep current
    return str(current).strip().lower()


# -----------------------------
# MAIN
# -----------------------------
def main():
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_CSV} in {Path.cwd()}")

    df = pd.read_csv(in_path)
    required_cols = {"attraction_id", "name", "category", "latitude", "longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    client = NominatimClient(CACHE_JSON)

    report_rows = []
    updated = df.copy()

    for idx, row in updated.iterrows():
        attraction_id = row["attraction_id"]
        name = str(row["name"]).strip()
        old_cat = str(row["category"]).strip().lower()

        try:
            old_lat = float(row["latitude"])
            old_lon = float(row["longitude"])
        except Exception:
            old_lat, old_lon = 0.0, 0.0

        # 1) Fix category
        new_cat = normalize_category(name, old_cat)

        cat_changed = (new_cat != old_cat)
        if cat_changed:
            updated.at[idx, "category"] = new_cat

        # 2) Fix coordinates (always validate, but only replace when confident)
        # Query format: "<name>, Sri Lanka"
        query = f"{name}, Sri Lanka"
        results = client.search(query)
        chosen, reason = choose_best_geocode_result(name, results, old_lat, old_lon, new_cat)

        coord_changed = False
        new_lat, new_lon = old_lat, old_lon

        if chosen is not None:
            new_lat, new_lon = chosen
            # Replace if the old coord is outside Sri Lanka OR far from the new coord
            if (not in_sri_lanka(old_lat, old_lon)) or (haversine_km(old_lat, old_lon, new_lat, new_lon) > 20):
                updated.at[idx, "latitude"] = new_lat
                updated.at[idx, "longitude"] = new_lon
                coord_changed = True

        # 3) Log
        if cat_changed or coord_changed:
            report_rows.append({
                "attraction_id": attraction_id,
                "name": name,
                "category_before": old_cat,
                "category_after": new_cat,
                "lat_before": old_lat,
                "lon_before": old_lon,
                "lat_after": new_lat if coord_changed else old_lat,
                "lon_after": new_lon if coord_changed else old_lon,
                "coord_change_reason": reason,
            })

    # Save outputs
    updated.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)

    client.save_cache()

    print("\nDONE")
    print(f"Saved fixed CSV  : {OUTPUT_CSV}")
    print(f"Saved fix report : {REPORT_CSV}")
    print(f"Saved cache      : {CACHE_JSON}")
    print("\nTip: Open fix_report.csv and sort by coord_change_reason to review any low-confidence cases.")


if __name__ == "__main__":
    main()
