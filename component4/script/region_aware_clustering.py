import numpy as np
import pandas as pd
from collections import Counter
import requests

# Sri Lanka District Definitions

SRI_LANKA_DISTRICTS = {
    'colombo': {'bounds': (6.85, 7.00, 79.80, 79.95), 'center': (6.9271, 79.8612)},
    'gampaha': {'bounds': (6.95, 7.15, 79.90, 80.10), 'center': (7.0873, 80.0142)},
    'kalutara': {'bounds': (6.50, 6.70, 79.90, 80.10), 'center': (6.5854, 80.0043)},
    'kandy': {'bounds': (7.20, 7.40, 80.55, 80.75), 'center': (7.2906, 80.6337)},
    'nuwara_eliya': {'bounds': (6.85, 7.05, 80.70, 80.90), 'center': (6.9497, 80.7891)},
    'galle': {'bounds': (6.00, 6.15, 80.15, 80.30), 'center': (6.0535, 80.2210)},
    'anuradhapura': {'bounds': (8.20, 8.45, 80.30, 80.50), 'center': (8.3114, 80.4037)},
    'polonnaruwa': {'bounds': (7.85, 8.05, 80.95, 81.15), 'center': (7.9403, 81.0188)},
    'badulla': {'bounds': (6.85, 7.10, 80.95, 81.20), 'center': (6.9934, 81.0550)},
}


# Clusterer

class RegionAwareClusterer:

    def __init__(self,
                 max_same_category=1,
                 target_hours_per_day=(6, 10),
                 max_daily_distance=75,
                 use_osrm=True):

        self.max_same_category = max_same_category
        self.target_hours_per_day = target_hours_per_day
        self.max_daily_distance = max_daily_distance
        self.use_osrm = use_osrm
        self.osrm_cache = {}

        self.max_attractions_per_day = 4
        self.min_attractions_per_day = 2

        self.trip_start_lat = None
        self.trip_start_lon = None

    # Public Entry

    def cluster_by_regions(self, attractions_df, num_days, start_lat, start_lon, max_distance):
        print(f"\n  [FINAL CLUSTERING] {len(attractions_df)} attractions → {num_days} days")
        print(f"  Daily distance cap: {max_distance} km")

        self.trip_start_lat = start_lat
        self.trip_start_lon = start_lon
        self.max_daily_distance = max_distance

        df = attractions_df.copy()
        df['district'] = df.apply(
            lambda r: self._assign_district(r['latitude'], r['longitude']), axis=1
        )

        candidates = df.to_dict('records')
        candidates.sort(key=lambda x: x.get('ml_score', 0), reverse=True)

        used_ids = set()
        itinerary = {}
        anchor_lat, anchor_lon = start_lat, start_lon

        for day in range(1, num_days + 1):
            day_key = f"Day {day}"
            day_plan = self._build_day(candidates, used_ids, anchor_lat, anchor_lon)
            itinerary[day_key] = day_plan

            if day_plan:
                last = day_plan[-1]
                anchor_lat, anchor_lon = last['latitude'], last['longitude']

        self._print_summary(itinerary)
        return itinerary

    # Day Builder

    def _build_day(self, candidates, used_ids, anchor_lat, anchor_lon):
        chosen = []
        category_count = Counter()
        target_min, target_max = self.target_hours_per_day

        def can_add(attr):
            if attr['attraction_id'] in used_ids:
                return False

            # Category uniqueness
            if category_count[attr['category']] >= self.max_same_category:
                return False

            # HARD DISTRICT LOCK
            if chosen and attr['district'] != chosen[0]['district']:
                return False

            # HARD ABSOLUTE DISTANCE FROM TRIP START
            if self._haversine(self.trip_start_lat, self.trip_start_lon,
                               attr['latitude'], attr['longitude']) > self.max_daily_distance:
                return False

            # Pairwise cohesion
            for a in chosen:
                if self._haversine(a['latitude'], a['longitude'],
                                   attr['latitude'], attr['longitude']) > self.max_daily_distance * 0.7:
                    return False

            # Route feasibility
            test_day = chosen + [attr]
            route_km, travel_h = self._estimate_route(test_day, anchor_lat, anchor_lon)
            if route_km > self.max_daily_distance:
                return False

            activity_h = sum(a['avg_duration_hours'] for a in test_day)
            if activity_h + travel_h > target_max:
                return False

            return True

        # Phase 1: diversity-first
        seen_categories = set()
        for attr in candidates:
            if attr['category'] in seen_categories:
                continue
            if can_add(attr):
                chosen.append(attr)
                used_ids.add(attr['attraction_id'])
                category_count[attr['category']] += 1
                seen_categories.add(attr['category'])
            if len(chosen) >= self.max_attractions_per_day:
                break

        # Phase 2: fill remaining slots
        for attr in candidates:
            if len(chosen) >= self.max_attractions_per_day:
                break
            if can_add(attr):
                chosen.append(attr)
                used_ids.add(attr['attraction_id'])
                category_count[attr['category']] += 1

        # Trim if over time
        while len(chosen) > 1:
            km, h = self._estimate_route(chosen, anchor_lat, anchor_lon)
            total_h = sum(a['avg_duration_hours'] for a in chosen) + h
            if total_h <= target_max:
                break
            weakest = min(chosen, key=lambda x: x.get('ml_score', 0))
            chosen.remove(weakest)

        return chosen

    # Utilities

    def _assign_district(self, lat, lon):
        for d, data in SRI_LANKA_DISTRICTS.items():
            lat_min, lat_max, lon_min, lon_max = data['bounds']
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return d
        # fallback to nearest center
        return min(
            SRI_LANKA_DISTRICTS.keys(),
            key=lambda d: self._haversine(
                lat, lon,
                SRI_LANKA_DISTRICTS[d]['center'][0],
                SRI_LANKA_DISTRICTS[d]['center'][1]
            )
        )

    def _estimate_route(self, attractions, start_lat, start_lon):
        total_km = 0.0
        total_h = 0.0
        cur_lat, cur_lon = start_lat, start_lon

        for a in attractions:
            km = self._haversine(cur_lat, cur_lon, a['latitude'], a['longitude'])
            total_km += km
            total_h += km / 45.0  # conservative Sri Lanka avg speed
            cur_lat, cur_lon = a['latitude'], a['longitude']

        return total_km, total_h

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def _print_summary(self, itinerary):
        print("\n    Clustering Summary:")
        for day, acts in itinerary.items():
            if not acts:
                continue
            cats = Counter(a['category'] for a in acts)
            print(f"      {day}: {len(acts)} attractions | "
                  f"{sum(a['avg_duration_hours'] for a in acts):.1f}h | "
                  f"District: {acts[0]['district']}")
            print(f"         Categories: {', '.join(cats.keys())}")
