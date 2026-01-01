import pandas as pd
import numpy as np
from datetime import datetime
import os


class BehaviorTracker:
    """Track user behavior and adapt preferences"""

    def __init__(self, attractions_file='../data/tourist_attractions.csv',
                 behavior_log_file='../data/behavior_log.csv'):

        self.attractions = pd.read_csv(attractions_file)
        self.behavior_log_file = behavior_log_file

        # Create behavior log if doesn't exist
        if not os.path.exists(behavior_log_file):
            self._initialize_behavior_log()

    def _initialize_behavior_log(self):
        """Create empty behavior log CSV"""
        columns = [
            'user_id', 'timestamp', 'latitude', 'longitude',
            'planned_attraction_id', 'planned_attraction_name',
            'actual_attraction_id', 'actual_attraction_name',
            'deviation_type', 'distance_from_plan_km', 'category',
            'duration_minutes', 'notes'
        ]

        df = pd.DataFrame(columns=columns)
        os.makedirs(os.path.dirname(self.behavior_log_file), exist_ok=True)
        df.to_csv(self.behavior_log_file, index=False)
        print(f" Initialized behavior log: {self.behavior_log_file}")

    def track_location(self, user_id, current_lat, current_lon,
                       planned_itinerary, current_day, timestamp=None):
        """
        Track user's current location and detect deviations

        Args:
            user_id: User identifier
            current_lat, current_lon: User's GPS coordinates
            planned_itinerary: The generated itinerary
            current_day: Which day of the trip (e.g., 'Day 1')
            timestamp: Optional timestamp (defaults to now)

        Returns:
            dict with deviation info
        """

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Get planned attractions for current day
        planned_today = planned_itinerary.get(current_day, [])

        if not planned_today:
            return {'status': 'no_plan', 'message': 'No planned attractions for today'}

        # Find nearest planned attraction
        nearest_planned = None
        min_distance = float('inf')

        for attr in planned_today:
            dist = self._haversine_distance(
                current_lat, current_lon,
                attr['latitude'], attr['longitude']
            )
            if dist < min_distance:
                min_distance = dist
                nearest_planned = attr

        # Check if user is at planned location (within 1km)
        if min_distance <= 1.0:
            return {
                'status': 'on_track',
                'planned_attraction': nearest_planned['name'],
                'distance_km': min_distance,
                'message': f" On track! At {nearest_planned['name']}"
            }

        # User deviated - find what they're actually visiting
        actual_attraction = self._identify_nearby_attraction(current_lat, current_lon)

        deviation_info = {
            'status': 'deviation',
            'planned_attraction': nearest_planned['name'] if nearest_planned else 'None',
            'actual_attraction': actual_attraction['name'] if actual_attraction else 'Unknown location',
            'distance_from_plan_km': min_distance,
            'message': f"⚠ Deviation detected"
        }

        # Log the deviation
        self._log_behavior(
            user_id=user_id,
            timestamp=timestamp,
            latitude=current_lat,
            longitude=current_lon,
            planned_attraction=nearest_planned,
            actual_attraction=actual_attraction,
            deviation_info=deviation_info
        )

        return deviation_info

    def _identify_nearby_attraction(self, lat, lon, radius_km=2.0):
        """Identify which attraction user is actually visiting"""

        distances = self.attractions.apply(
            lambda row: self._haversine_distance(
                lat, lon, row['latitude'], row['longitude']
            ), axis=1
        )

        # Find closest attraction within radius
        min_idx = distances.idxmin()
        min_dist = distances.min()

        if min_dist <= radius_km:
            attraction = self.attractions.iloc[min_idx].to_dict()
            attraction['distance_km'] = min_dist
            return attraction

        return None

    def _log_behavior(self, user_id, timestamp, latitude, longitude,
                      planned_attraction, actual_attraction, deviation_info):
        """Log behavior to CSV"""

        # Determine deviation type
        if actual_attraction:
            if planned_attraction and actual_attraction['attraction_id'] == planned_attraction['attraction_id']:
                deviation_type = 'on_track'
            else:
                deviation_type = 'unplanned_visit'
        else:
            deviation_type = 'off_route'

        log_entry = {
            'user_id': user_id,
            'timestamp': timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'planned_attraction_id': planned_attraction['attraction_id'] if planned_attraction else None,
            'planned_attraction_name': planned_attraction['name'] if planned_attraction else 'None',
            'actual_attraction_id': actual_attraction['attraction_id'] if actual_attraction else None,
            'actual_attraction_name': actual_attraction['name'] if actual_attraction else 'Unknown',
            'deviation_type': deviation_type,
            'distance_from_plan_km': deviation_info.get('distance_from_plan_km', 0),
            'category': actual_attraction['category'] if actual_attraction else 'unknown',
            'duration_minutes': None,  # Can be filled later
            'notes': ''
        }

        # Append to CSV
        df = pd.DataFrame([log_entry])
        df.to_csv(self.behavior_log_file, mode='a', header=False, index=False)

    def analyze_preferences(self, user_id):
        """
        Analyze user's behavior history to learn new preferences

        Returns:
            dict with discovered preferences
        """

        # Load behavior log
        if not os.path.exists(self.behavior_log_file):
            return {'message': 'No behavior data yet'}

        df = pd.read_csv(self.behavior_log_file)
        user_data = df[df['user_id'] == user_id]

        if len(user_data) == 0:
            return {'message': 'No data for this user'}

        # Analyze unplanned visits (user's true interests)
        unplanned = user_data[user_data['deviation_type'] == 'unplanned_visit']

        # Count categories visited
        category_counts = unplanned['category'].value_counts()

        # Discover new interests
        discovered_categories = category_counts.index.tolist()

        analysis = {
            'total_tracked_locations': len(user_data),
            'on_track_count': len(user_data[user_data['deviation_type'] == 'on_track']),
            'unplanned_visits': len(unplanned),
            'discovered_interests': discovered_categories,
            'category_distribution': category_counts.to_dict(),
            'adherence_rate': len(user_data[user_data['deviation_type'] == 'on_track']) / len(user_data)
        }

        return analysis

    def get_adapted_preferences(self, original_preferences, user_id):
        """
        Adapt user preferences based on behavior history

        Returns:
            Updated preferences dict
        """

        analysis = self.analyze_preferences(user_id)

        if 'discovered_interests' not in analysis:
            return original_preferences

        # Merge discovered interests with original
        adapted = original_preferences.copy()

        original_categories = set(adapted['activity_categories'])
        discovered = set(analysis['discovered_interests'])

        # Add newly discovered categories
        new_categories = discovered - original_categories

        if new_categories:
            adapted['activity_categories'] = list(original_categories | new_categories)
            adapted['adapted'] = True
            adapted['new_interests'] = list(new_categories)
        else:
            adapted['adapted'] = False

        return adapted

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def simulate_behavior(self, user_id, itinerary, num_deviations=2):
        """
        Simulate user behavior for demonstration
        (In production, this would come from actual GPS tracking)
        """

        print("\n" + "=" * 80)
        print("SIMULATING USER BEHAVIOR (GPS Tracking)")
        print("=" * 80)

        all_attractions = []
        for day, attractions in itinerary.items():
            all_attractions.extend(attractions)

        # Simulate following some planned attractions
        for i, attr in enumerate(all_attractions[:3]):
            print(f"\n[Tracking] Day {i + 1}: User visiting {attr['name']}")
            result = self.track_location(
                user_id=user_id,
                current_lat=attr['latitude'],
                current_lon=attr['longitude'],
                planned_itinerary=itinerary,
                current_day=f'Day {i + 1}'
            )
            print(f"  Status: {result['status']}")
            print(f"  {result['message']}")

        # Simulate some deviations (visiting unplanned places)
        print(f"\n[Simulation] Generating {num_deviations} unplanned visits...")

        for i in range(num_deviations):
            # Pick random attraction not in plan
            unplanned = self.attractions.sample(1).iloc[0]

            print(f"\n[Tracking] User visiting unplanned location: {unplanned['name']}")
            result = self.track_location(
                user_id=user_id,
                current_lat=unplanned['latitude'],
                current_lon=unplanned['longitude'],
                planned_itinerary=itinerary,
                current_day='Day 2'
            )
            print(f"  Status: {result['status']}")
            print(f"  Actual: {result.get('actual_attraction', 'Unknown')}")
            print(f"  Category: {unplanned['category']}")

        print("\n Behavior simulation complete")