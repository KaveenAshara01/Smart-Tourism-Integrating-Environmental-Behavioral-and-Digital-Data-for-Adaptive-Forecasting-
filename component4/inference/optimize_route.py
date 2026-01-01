import numpy as np
from itertools import permutations


class RouteOptimizer:
    """Optimize routes for multi-day itineraries"""

    def __init__(self):
        pass

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def nearest_neighbor_route(self, attractions, start_lat, start_lon):
        """Find optimal route using nearest neighbor algorithm"""

        if len(attractions) <= 1:
            return attractions

        unvisited = attractions.copy()
        route = []
        current_lat, current_lon = start_lat, start_lon

        while unvisited:
            # Find nearest unvisited attraction
            distances = [
                self.haversine_distance(current_lat, current_lon,
                                        attr['latitude'], attr['longitude'])
                for attr in unvisited
            ]

            nearest_idx = np.argmin(distances)
            nearest_attr = unvisited.pop(nearest_idx)
            route.append(nearest_attr)

            current_lat = nearest_attr['latitude']
            current_lon = nearest_attr['longitude']

        return route

    def optimize_itinerary(self, itinerary, start_lat, start_lon):
        """Optimize routes for entire itinerary"""

        optimized = {}

        for day, attractions in itinerary.items():
            if attractions:
                optimized[day] = self.nearest_neighbor_route(
                    attractions, start_lat, start_lon
                )

                # Update start for next day
                if optimized[day]:
                    start_lat = optimized[day][-1]['latitude']
                    start_lon = optimized[day][-1]['longitude']
            else:
                optimized[day] = []

        return optimized

    def calculate_total_distance(self, itinerary, start_lat, start_lon):
        """Calculate total travel distance"""

        total_distance = 0
        current_lat, current_lon = start_lat, start_lon

        for day, attractions in itinerary.items():
            for attr in attractions:
                dist = self.haversine_distance(
                    current_lat, current_lon,
                    attr['latitude'], attr['longitude']
                )
                total_distance += dist
                current_lat = attr['latitude']
                current_lon = attr['longitude']

        return total_distance

    def calculate_total_cost(self, itinerary, num_travelers):
        """Calculate estimated total cost"""

        total_cost = 0

        for day, attractions in itinerary.items():
            for attr in attractions:
                total_cost += attr['avg_cost'] * num_travelers

        return total_cost

    def calculate_total_time(self, itinerary):
        """Calculate total time needed"""

        total_hours = 0

        for day, attractions in itinerary.items():
            for attr in attractions:
                total_hours += attr['avg_duration_hours']

        return total_hours