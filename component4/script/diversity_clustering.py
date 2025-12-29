import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class DiversityAwareClusterer:
    """
    Cluster attractions into days ensuring:
    1. Category diversity (max 2 from same category per day)
    2. Balanced time per day (6-10 hours)
    3. Balanced cost per day
    4. Geographic clustering (minimize travel)
    """

    def __init__(self, target_hours_per_day=(6, 10), max_same_category=2):
        self.target_hours_per_day = target_hours_per_day
        self.max_same_category = max_same_category

    def cluster_attractions(self, attractions_df, num_days, start_lat, start_lon):
        """
        Main clustering method with diversity constraints

        Args:
            attractions_df: DataFrame with selected attractions
            num_days: Number of days available
            start_lat, start_lon: Starting coordinates

        Returns:
            dict: {day_num: [attractions]}
        """

        print(f"\n  [Diversity Clustering] Organizing {len(attractions_df)} attractions into {num_days} days")

        if len(attractions_df) == 0:
            return {}

        # Initial geographic clustering
        print("    Geographic clustering...")
        geo_clusters = self._geographic_clustering(attractions_df, num_days, start_lat, start_lon)

        # Apply diversity constraints
        print("    Applying diversity constraints...")
        diverse_clusters = self._apply_diversity_constraints(geo_clusters, num_days)

        # Balance time/cost
        print("    Balancing time and cost...")
        balanced_clusters = self._balance_time_cost(diverse_clusters, num_days)

        # Validate and adjust
        print("    Validating constraints...")
        final_clusters = self._validate_and_adjust(balanced_clusters)

        # Print summary
        self._print_summary(final_clusters)

        return final_clusters

    def _geographic_clustering(self, df, num_days, start_lat, start_lon):
        """Initial K-means clustering based on geography"""

        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Add distance from start to features
        coords = df[['latitude', 'longitude']].values

        if len(df) < num_days:
            # Fewer attractions than days - one per day
            return {i + 1: [df.iloc[i].to_dict()] for i in range(len(df))}

        # K-means clustering
        kmeans = KMeans(n_clusters=num_days, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(coords)

        # Convert to dict
        clusters = defaultdict(list)
        for cluster_id in range(num_days):
            cluster_data = df[df['cluster'] == cluster_id]
            clusters[cluster_id + 1] = cluster_data.to_dict('records')

        return clusters

    def _apply_diversity_constraints(self, clusters, num_days):
        """
        Ensure each day has diverse categories
        Max 2 attractions from same category per day
        """

        diverse_clusters = defaultdict(list)
        overflow = []  # Attractions that violate constraints

        for day, attractions in clusters.items():
            category_count = Counter()
            day_attractions = []

            for attr in attractions:
                category = attr['category']

                if category_count[category] < self.max_same_category:
                    day_attractions.append(attr)
                    category_count[category] += 1
                else:
                    # Too many of this category today
                    overflow.append(attr)

            diverse_clusters[day] = day_attractions

        # Redistribute overflow attractions
        if overflow:
            diverse_clusters = self._redistribute_overflow(diverse_clusters, overflow, num_days)

        return diverse_clusters

    def _redistribute_overflow(self, clusters, overflow, num_days):
        """Redistribute overflow attractions to days with capacity"""

        for attr in overflow:
            category = attr['category']

            # Find day with room for this category
            placed = False
            for day in range(1, num_days + 1):
                day_categories = [a['category'] for a in clusters[day]]
                category_count = day_categories.count(category)

                if category_count < self.max_same_category:
                    clusters[day].append(attr)
                    placed = True
                    break

            # If still not placed, force into day with least of this category
            if not placed:
                min_day = min(range(1, num_days + 1),
                              key=lambda d: [a['category'] for a in clusters[d]].count(category))
                clusters[min_day].append(attr)

        return clusters

    def _balance_time_cost(self, clusters, num_days):
        """
        Balance time and cost across days
        Target: 6-10 hours per day
        """

        min_hours, max_hours = self.target_hours_per_day

        # Calculate current time per day
        day_hours = {}
        for day, attractions in clusters.items():
            total_hours = sum(a['avg_duration_hours'] for a in attractions)
            day_hours[day] = total_hours

        # Rebalance if needed
        max_iterations = 10
        for iteration in range(max_iterations):
            # Find overloaded and underloaded days
            overloaded = [(day, hours) for day, hours in day_hours.items() if hours > max_hours]
            underloaded = [(day, hours) for day, hours in day_hours.items() if hours < min_hours]

            if not overloaded and not underloaded:
                break  # Balanced!

            # Move attractions from overloaded to underloaded days
            for over_day, over_hours in overloaded:
                if not underloaded:
                    break

                # Find smallest attraction to move
                over_attractions = clusters[over_day]
                if len(over_attractions) <= 1:
                    continue

                movable = sorted(over_attractions, key=lambda x: x['avg_duration_hours'])

                for attr in movable:
                    # Find best underloaded day
                    under_day = min(underloaded, key=lambda x: x[1])[0]

                    # Check diversity constraint
                    under_categories = [a['category'] for a in clusters[under_day]]
                    if under_categories.count(attr['category']) >= self.max_same_category:
                        continue

                    # Move attraction
                    clusters[over_day].remove(attr)
                    clusters[under_day].append(attr)

                    # Update hours
                    day_hours[over_day] -= attr['avg_duration_hours']
                    day_hours[under_day] += attr['avg_duration_hours']

                    # Recheck
                    underloaded = [(d, h) for d, h in day_hours.items() if h < min_hours]

                    if day_hours[over_day] <= max_hours:
                        break

        return clusters

    def _validate_and_adjust(self, clusters):
        """Final validation and adjustments"""

        # Remove empty days
        clusters = {day: attrs for day, attrs in clusters.items() if len(attrs) > 0}

        # Renumber days consecutively
        final = {}
        day_num = 1
        for day in sorted(clusters.keys()):
            final[f'Day {day_num}'] = clusters[day]
            day_num += 1

        return final

    def _print_summary(self, clusters):
        """Print clustering summary"""

        print("\n    Clustering Summary:")
        for day, attractions in clusters.items():
            total_hours = sum(a['avg_duration_hours'] for a in attractions)
            total_cost = sum(a['avg_cost'] for a in attractions)
            categories = [a['category'] for a in attractions]
            unique_categories = len(set(categories))

            print(f"      {day}: {len(attractions)} attractions, {total_hours:.1f}h, "
                  f"LKR {total_cost:,.0f}, {unique_categories} categories")

            # Show category distribution
            category_counts = Counter(categories)
            category_str = ', '.join([f"{cat}({cnt})" for cat, cnt in category_counts.most_common()])
            print(f"         Categories: {category_str}")


class DynamicCapacityAllocator:
    """
    Dynamically allocate attractions to days (NOVELTY #2)
    Days can have 1-5 attractions based on:
    - Activity duration
    - Travel time
    - Daily target hours (6-10h)
    """

    def __init__(self, target_hours=(6, 10)):
        self.target_hours = target_hours

    def allocate_attractions(self, attractions_df, num_days):
        """
        Allocate attractions to days with flexible capacity

        Returns:
            dict: {day: [attractions]}
        """

        print(f"\n  [Dynamic Allocation] Flexible capacity planning...")

        min_hours, max_hours = self.target_hours

        # Sort attractions by priority (from ML scores)
        sorted_attrs = attractions_df.sort_values('ml_score', ascending=False).to_dict('records')

        # Initialize days
        days = {f'Day {i + 1}': {'attractions': [], 'hours': 0, 'cost': 0}
                for i in range(num_days)}

        # Allocate attractions
        for attr in sorted_attrs:
            # Find day with capacity
            best_day = None

            for day_name, day_data in days.items():
                current_hours = day_data['hours']
                attr_hours = attr['avg_duration_hours']

                # Check if adding this attraction would keep day in target range
                if current_hours + attr_hours <= max_hours:
                    if best_day is None or day_data['hours'] < days[best_day]['hours']:
                        # Check category diversity
                        categories = [a['category'] for a in day_data['attractions']]
                        if categories.count(attr['category']) < 2:  # Max 2 per category
                            best_day = day_name

            # Add to best day
            if best_day:
                days[best_day]['attractions'].append(attr)
                days[best_day]['hours'] += attr['avg_duration_hours']
                days[best_day]['cost'] += attr['avg_cost']

        # Convert to simpler format
        result = {day: data['attractions'] for day, data in days.items()
                  if len(data['attractions']) > 0}

        # Print summary
        print(f"    Allocated to {len(result)} days:")
        for day, attrs in result.items():
            hours = sum(a['avg_duration_hours'] for a in attrs)
            print(f"      {day}: {len(attrs)} attractions, {hours:.1f}h")

        return result
