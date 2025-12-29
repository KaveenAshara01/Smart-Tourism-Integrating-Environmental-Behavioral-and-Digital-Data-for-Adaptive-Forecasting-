import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

class EnhancedItineraryGenerator:
    """Generate personalized itineraries with hybrid ML and diversity constraints"""

    def __init__(self, attractions_file='../data/tourist_attractions.csv',
                 xgboost_path='../models/xgboost_model.pkl',
                 fusion_path='../models/fusion_model.h5',
                 scaler_path='../models/scaler.pkl'):

        # Load attractions
        self.attractions = pd.read_csv(attractions_file)

        # Load models
        with open(xgboost_path, 'rb') as f:
            self.xgb_model = pickle.load(f)

        try:
            import tensorflow as tf
            self.fusion_model = tf.keras.models.load_model(fusion_path)
        except:
            self.fusion_model = None
            print("  Note: Fusion NN not available, using XGBoost only")

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load metadata
        metadata_path = xgboost_path.replace('xgboost_model.pkl', 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.user_feature_cols = self.metadata['user_feature_columns']
        self.attraction_feature_cols = self.metadata['attraction_feature_columns']
        self.activity_types = self.metadata['activity_types']

    def get_user_preferences(self):
        """Interactive questionnaire"""
        
        print("WELCOME TO SMART TOURISM ITINERARY PLANNER")
        print("Powered by XGBoost + Fusion Neural Network")
        
        print("\nPlease answer a few questions to personalize your itinerary:\n")

        # Budget
        print("1. What is your budget (in LKR)?")
        print("   a) Budget: < 50,000")
        print("   b) Moderate: 50,000 - 150,000")
        print("   c) Comfortable: 150,000 - 300,000")
        print("   d) Luxury: > 300,000")
        budget_choice = input("   Your choice (a/b/c/d): ").lower()
        budget_map = {'a': 35000, 'b': 100000, 'c': 225000, 'd': 400000}
        budget = budget_map.get(budget_choice, 100000)

        # Days
        available_days = int(input("\n2. How many days will you spend in Sri Lanka? "))

        # Distance
        print("\n3. How far are you willing to travel?")
        print("   a) Local: < 100 km")
        print("   b) Regional: 100-250 km")
        print("   c) Nationwide: > 250 km")
        distance_choice = input("   Your choice (a/b/c): ").lower()
        distance_map = {'a': 75, 'b': 175, 'c': 350}
        distance_preference = distance_map.get(distance_choice, 175)

        # Travelers
        num_travelers = int(input("\n4. How many people are traveling? "))

        # Activities (multiple)
        print("\n5. What types of activities interest you? (Select multiple, e.g., 1,3,5)")
        print("   1. Beach & Relaxation")
        print("   2. Historical Sites & Monuments")
        print("   3. Temples & Spiritual")
        print("   4. National Parks & Wildlife")
        print("   5. Waterfalls & Nature")
        print("   6. Mountains & Hiking")
        print("   7. Cultural Experiences")
        print("   8. City Life & Shopping")
        print("   9. Adventure & Sports")
        print("   10. Wildlife Safaris")

        activity_input = input("   Your choices: ")
        selected_activities = [int(x.strip()) for x in activity_input.split(',')]

        activity_map = {
            1: 'beach', 2: 'historical', 3: 'temple', 4: 'national_park',
            5: 'waterfall', 6: 'mountain', 7: 'cultural', 8: 'city',
            9: 'adventure', 10: 'wildlife'
        }
        selected_categories = [activity_map[x] for x in selected_activities if x in activity_map]

        # Season
        print("\n6. When are you planning to visit?")
        print("   1. Spring (March-May)")
        print("   2. Summer (June-August)")
        print("   3. Autumn (September-November)")
        print("   4. Winter (December-February)")
        season = int(input("   Your choice (1/2/3/4): "))

        # Start location
        print("\n7. Where will you start your trip?")
        print("   a) Colombo (Airport)")
        print("   b) Kandy (Cultural Triangle)")
        print("   c) Galle (Southern Coast)")
        start_choice = input("   Your choice (a/b/c): ").lower()

        start_locations = {
            'a': (6.9271, 79.8612),
            'b': (7.2906, 80.6337),
            'c': (6.0535, 80.2210)
        }
        start_lat, start_lon = start_locations.get(start_choice, (6.9271, 79.8612))

        preferences = {
            'budget': budget,
            'available_days': available_days,
            'distance_preference': distance_preference,
            'num_travelers': num_travelers,
            'activity_categories': selected_categories,
            'season': season,
            'start_latitude': start_lat,
            'start_longitude': start_lon
        }

        return preferences

    def predict_attraction_scores(self, preferences):
        """
        Predict preference scores using XGBoost + Fusion NN

        Returns:
            np.array of scores for each attraction
        """

        # Build user features
        user_features = {
            'budget': preferences['budget'],
            'available_days': preferences['available_days'],
            'distance_preference': preferences['distance_preference'],
            'num_travelers': preferences['num_travelers']
        }

        # One-hot encode activities
        for activity in self.activity_types:
            user_features[f'pref_{activity}'] = 1 if activity in preferences['activity_categories'] else 0

        # Season encoding
        user_features['season_spring'] = 1 if preferences['season'] == 1 else 0
        user_features['season_summer'] = 1 if preferences['season'] == 2 else 0
        user_features['season_autumn'] = 1 if preferences['season'] == 3 else 0
        user_features['season_winter'] = 1 if preferences['season'] == 4 else 0

        # Get user feature vector
        user_vec = np.array([user_features[col] for col in self.user_feature_cols])

        # Score each attraction
        scores = []
        X_batch = []

        for _, attr in self.attractions.iterrows():
            # Encode attraction categories
            attr_features = {}
            for activity in self.activity_types:
                attr_features[f'is_{activity}'] = 1 if attr['category'] == activity else 0

            # Get attraction feature vector
            attr_vec = np.array([
                attr['avg_duration_hours'],
                attr['avg_cost'],
                int(attr['outdoor']),
                attr['popularity_score'],
                attr['accessibility'],
                attr['tourist_density'],
                attr['safety_rating']
            ] + [attr_features[f'is_{activity}'] for activity in self.activity_types])

            # Combine user + attraction features
            combined = np.concatenate([user_vec, attr_vec])
            X_batch.append(combined)

        # Scale features
        X_scaled = self.scaler.transform(np.array(X_batch))

        # Predict with XGBoost
        scores_xgb = self.xgb_model.predict_proba(X_scaled)[:, 1]

        # If fusion model available, use it
        if self.fusion_model is not None:
            X_fusion = np.concatenate([X_scaled, scores_xgb.reshape(-1, 1)], axis=1)
            scores_fusion = self.fusion_model.predict(X_fusion, verbose=0).flatten()
            return scores_fusion

        return scores_xgb

    def apply_context_integration(self, attractions_scored, forecast_data=None,
                                  sentiment_data=None, risk_data=None, weather_data=None):
        """
        Apply contextual adjustments from Components 1, 2, 3
        """

        print("\n  [Context Integration] Applying external factors...")

        adjustments_applied = []

        # Component 1: Tourist Forecast (avoid overcrowded)
        if forecast_data:
            print("     Component 1: Tourist forecast adjustments")
            attractions_scored['crowd_penalty'] = attractions_scored.apply(
                lambda row: min(row.get('tourist_density', 0.5) * 0.2, 0.3), axis=1
            )
            attractions_scored['ml_score'] *= (1 - attractions_scored['crowd_penalty'])
            adjustments_applied.append('forecast')

        # Component 2: Sentiment (boost highly-rated)
        if sentiment_data:
            print("     Component 2: Sentiment score adjustments")
            attractions_scored['sentiment_boost'] = attractions_scored.apply(
                lambda row: sentiment_data.get(row['attraction_id'], 0) * 0.15, axis=1
            )
            attractions_scored['ml_score'] *= (1 + attractions_scored['sentiment_boost'])
            adjustments_applied.append('sentiment')

        # Component 3: Risk/Safety
        if risk_data:
            print("     Component 3: Safety adjustments")
            attractions_scored['ml_score'] *= attractions_scored['safety_rating']
            adjustments_applied.append('safety')

        # Weather data
        if weather_data:
            print("     Weather adjustments")
            rainfall = weather_data.get('rainfall_mm', 0)

            if rainfall > 30:
                # Heavy rain - penalize outdoor activities
                mask = attractions_scored['outdoor'] == True
                attractions_scored.loc[mask, 'ml_score'] *= 0.5
                adjustments_applied.append('weather')

        if not adjustments_applied:
            print("    No external context provided")

        return attractions_scored

    def filter_and_rank(self, attractions_scored, preferences):
        """Filter attractions by constraints and rank"""

        # Calculate distance from start
        start_lat, start_lon = preferences['start_latitude'], preferences['start_longitude']

        attractions_scored['distance_km'] = attractions_scored.apply(
            lambda row: self._haversine_distance(start_lat, start_lon,
                                                 row['latitude'], row['longitude']),
            axis=1
        )

        # Apply filters
        filtered = attractions_scored.copy()

        # Distance filter
        filtered = filtered[filtered['distance_km'] <= preferences['distance_preference']]

        # Budget filter (rough estimate)
        max_daily_cost = preferences['budget'] / preferences['available_days'] / preferences['num_travelers']
        filtered = filtered[filtered['avg_cost'] <= max_daily_cost * 3]

        # Category filter (if specified) - Apply STRONG boost
        if preferences['activity_categories']:
            # Create boolean mask for selected categories
            category_mask = filtered['category'].isin(preferences['activity_categories'])

            # Apply strong boost to selected categories (3x multiplier)
            filtered.loc[category_mask, 'ml_score'] *= 3.0

            print(f"     Applied 3x boost to selected categories: {', '.join(preferences['activity_categories'])}")

        # Season preference boost
        filtered['season_match'] = filtered['best_season'].apply(
            lambda x: 1.2 if x == preferences['season'] else 1.0
        )
        filtered['ml_score'] *= filtered['season_match']

        # Final ranking
        filtered = filtered.sort_values('ml_score', ascending=False)

        return filtered

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def generate_itinerary(self, preferences, forecast_data=None,
                          sentiment_data=None, risk_data=None, weather_data=None,
                          use_diversity_clustering=True):
        """
        Main method: Generate complete itinerary

        Args:
            preferences: User preferences dict
            forecast_data: Component 1 forecast data
            sentiment_data: Component 2 sentiment scores
            risk_data: Component 3 risk indexes
            weather_data: Weather conditions
            use_diversity_clustering: Use diversity-aware clustering (NOVELTY)

        Returns:
            tuple: (itinerary dict, selected attractions list)
        """

        
        print("GENERATING YOUR PERSONALIZED ITINERARY")
        

        # Step 1: ML Prediction
        print("\n Predicting attraction preferences (XGBoost + Fusion NN)...")
        scores = self.predict_attraction_scores(preferences)

        # Add scores to attractions
        attractions_scored = self.attractions.copy()
        attractions_scored['ml_score'] = scores
        print(f"   Scored {len(attractions_scored)} attractions")

        # Step 2: Context Integration
        print("\n Integrating context from Components 1, 2, 3...")
        attractions_scored = self.apply_context_integration(
            attractions_scored, forecast_data, sentiment_data, risk_data, weather_data
        )

        # Step 3: Filter and Rank
        print("\n Filtering and ranking...")
        filtered = self.filter_and_rank(attractions_scored, preferences)
        print(f"   Filtered to {len(filtered)} matching attractions")

        # Step 4: Select top attractions
        print("\n Selecting optimal attractions...")
        target_count = preferences['available_days'] * 4  # Target 4 per day (increased from 3)
        selected_df = filtered.head(min(target_count + 10, len(filtered)))  # Get more candidates
        print(f"   Selected {len(selected_df)} top-ranked attractions")

        # Step 5: Organize into days with diversity
        print("\n Organizing into day-by-day itinerary...")

        if use_diversity_clustering:
            from region_aware_clustering import RegionAwareClusterer
            clusterer = RegionAwareClusterer(
                max_same_category=1,  # no repeated category in a single day
                target_hours_per_day=(6, 10),
                use_osrm=True,
                max_daily_distance=preferences['distance_preference']
            )
            itinerary = clusterer.cluster_by_regions(
                selected_df,
                preferences['available_days'],
                preferences['start_latitude'],
                preferences['start_longitude'],
                max_distance=preferences['distance_preference']  # Pass user's distance preference
            )
        else:
            # Fallback: simple division
            itinerary = self._simple_day_division(selected_df, preferences['available_days'])

        print("   Itinerary generated successfully!")

        return itinerary, selected_df.to_dict('records')

    def _simple_day_division(self, attractions_df, num_days):
        """Simple fallback if clustering fails"""
        attractions_list = attractions_df.to_dict('records')
        per_day = len(attractions_list) // num_days

        itinerary = {}
        for day in range(1, num_days + 1):
            start_idx = (day - 1) * per_day
            end_idx = start_idx + per_day if day < num_days else len(attractions_list)
            itinerary[f'Day {day}'] = attractions_list[start_idx:end_idx]

        return itinerary