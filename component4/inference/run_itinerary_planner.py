import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add script directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.abspath(os.path.join(current_dir, '..', 'script'))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from enhanced_itinerary_generator import EnhancedItineraryGenerator
from optimize_route import RouteOptimizer
from track_behavior import BehaviorTracker




#  COLLECT USER PREFERENCES


print("\n  USER PREFERENCES")


generator = EnhancedItineraryGenerator()
preferences = generator.get_user_preferences()

# Save preferences
user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
preferences['user_id'] = user_id

os.makedirs('../data', exist_ok=True)
prefs_file = '../data/user_preferences.csv'

if os.path.exists(prefs_file):
    prefs_df = pd.read_csv(prefs_file)
    new_pref = pd.DataFrame([{
        'user_id': user_id,
        'budget': preferences['budget'],
        'available_days': preferences['available_days'],
        'distance_preference': preferences['distance_preference'],
        'num_travelers': preferences['num_travelers'],
        'activity_categories': ','.join(preferences['activity_categories']),
        'season': preferences['season'],
        'timestamp': datetime.now().isoformat()
    }])
    prefs_df = pd.concat([prefs_df, new_pref], ignore_index=True)
else:
    prefs_df = pd.DataFrame([{
        'user_id': user_id,
        'budget': preferences['budget'],
        'available_days': preferences['available_days'],
        'distance_preference': preferences['distance_preference'],
        'num_travelers': preferences['num_travelers'],
        'activity_categories': ','.join(preferences['activity_categories']),
        'season': preferences['season'],
        'timestamp': datetime.now().isoformat()
    }])

prefs_df.to_csv(prefs_file, index=False)
print(f"\n Preferences saved for {user_id}")


#  FETCH CONTEXT FROM OTHER COMPONENTS (Optional)


print("\n\n  CONTEXT INTEGRATION")


# In production, fetch from Components 1, 2, 3
# For demo, use simulated data

forecast_data = None  # Component 1: Tourist forecast
sentiment_data = None  # Component 2: Sentiment scores
risk_data = None  # Component 3: Risk indexes

# Example weather data
weather_data = {
    'rainfall_mm': 5.0,  # Light rain
    'temperature': 28.0,
    'humidity': 75.0
}

print("  Context sources:")
print(f" Component 1 (Forecast): {'' if forecast_data else '⊗ Not provided'}")
print(f" Component 2 (Sentiment): {'' if sentiment_data else '⊗ Not provided'}")
print(f" Component 3 (Safety): {'' if risk_data else '⊗ Not provided'}")
print(f"Weather Data:  Provided")


#  GENERATE ITINERARY WITH ML + DIVERSITY CLUSTERING


print("\n\n  GENERATE ITINERARY (ML + Diversity Clustering)")


itinerary, selected_attractions = generator.generate_itinerary(
    preferences=preferences,
    forecast_data=forecast_data,
    sentiment_data=sentiment_data,
    risk_data=risk_data,
    weather_data=weather_data,
    use_diversity_clustering=True  # Use novelty!
)

#  OPTIMIZE ROUTES
print("\n\n   ROUTE OPTIMIZATION")
optimizer = RouteOptimizer()

print(" Optimizing routes...")
optimized_itinerary = optimizer.optimize_itinerary(
    itinerary=itinerary,
    start_lat=preferences['start_latitude'],
    start_lon=preferences['start_longitude']
)

print(" Calculating statistics...")
total_distance = optimizer.calculate_total_distance(
    itinerary=optimized_itinerary,
    start_lat=preferences['start_latitude'],
    start_lon=preferences['start_longitude']
)

total_cost = optimizer.calculate_total_cost(
    itinerary=optimized_itinerary,
    num_travelers=preferences['num_travelers']
)

total_time = optimizer.calculate_total_time(optimized_itinerary)

print(f"\n   Total Distance: {total_distance:.1f} km")
print(f"   Estimated Cost: LKR {total_cost:,.0f}")
print(f"   Total Activity Time: {total_time:.1f} hours")


#  DISPLAY ITINERARY


print("\n\n YOUR PERSONALIZED ITINERARY")


for day, attractions in optimized_itinerary.items():
    print(f"\n{day}:")
    

    if not attractions:
        print("  No activities scheduled")
        continue

    day_cost = sum(attr['avg_cost'] * preferences['num_travelers'] for attr in attractions)
    day_time = sum(attr['avg_duration_hours'] for attr in attractions)
    categories = [attr['category'] for attr in attractions]
    unique_categories = len(set(categories))

    for i, attr in enumerate(attractions, 1):
        print(f"\n  {i}. {attr['name']}")
        print(f"     Category: {attr['category'].title()}")
        print(f"     Duration: {attr['avg_duration_hours']:.1f} hours")
        print(f"     Cost: LKR {attr['avg_cost'] * preferences['num_travelers']:,.0f}")
        print(f"     ML Score: {attr.get('ml_score', 0):.3f}")
        print(f"     Popularity: {attr['popularity_score']:.0%}")

    print(f"\n  Day Summary: {day_time:.1f}h | LKR {day_cost:,.0f} | {unique_categories} categories")




#  SAVE ITINERARY


print("\n\n  SAVING ITINERARY")


output_dir = '../metrics/itinerary_outputs'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'itinerary_{user_id}.json')

itinerary_data = {
    'user_id': user_id,
    'timestamp': datetime.now().isoformat(),
    'model': 'XGBoost + Fusion NN + Diversity Clustering',
    'preferences': {
        'budget': preferences['budget'],
        'days': preferences['available_days'],
        'travelers': preferences['num_travelers'],
        'categories': preferences['activity_categories'],
        'season': preferences['season']
    },
    'itinerary': {
        day: [
            {
                'name': attr['name'],
                'category': attr['category'],
                'latitude': attr['latitude'],
                'longitude': attr['longitude'],
                'duration_hours': attr['avg_duration_hours'],
                'cost': attr['avg_cost'] * preferences['num_travelers'],
                'ml_score': float(attr.get('ml_score', 0)),
                'popularity': float(attr['popularity_score'])
            }
            for attr in attractions
        ]
        for day, attractions in optimized_itinerary.items()
    },
    'statistics': {
        'total_distance_km': total_distance,
        'total_cost_lkr': total_cost,
        'total_activity_hours': total_time,
        'num_attractions': sum(len(attrs) for attrs in optimized_itinerary.values()),
        'avg_attractions_per_day': sum(len(attrs) for attrs in optimized_itinerary.values()) / preferences['available_days']
    },
    'novelty_features': {
        'ml_model': 'XGBoost + Fusion Neural Network',
        'negative_sampling': True,
        'diversity_clustering': True,
        'context_integration': bool(forecast_data or sentiment_data or risk_data or weather_data),
        'behavioral_adaptation': 'Available'
    }
}

with open(output_file, 'w') as f:
    json.dump(itinerary_data, f, indent=2)

print(f" Itinerary saved: {output_file}")


#  BEHAVIOR TRACKING DEMO


print("\n\n  BEHAVIOR TRACKING (DEMO)")

print("In production, this tracks user's actual GPS location")
print("For demo, we'll simulate user behavior\n")

tracker = BehaviorTracker()
tracker.simulate_behavior(
    user_id=user_id,
    itinerary=optimized_itinerary,
    num_deviations=2
)


#  ANALYZE BEHAVIOR & ADAPT


print("\n\n  ADAPTIVE LEARNING")


print("\n Analyzing behavior patterns...")
analysis = tracker.analyze_preferences(user_id)

if 'discovered_interests' in analysis:
    print(f"\n  Behavior Analysis:")
    print(f"    Tracked locations: {analysis['total_tracked_locations']}")
    print(f"    On-track visits: {analysis['on_track_count']}")
    print(f"    Unplanned visits: {analysis['unplanned_visits']}")
    print(f"    Adherence rate: {analysis['adherence_rate']:.1%}")

    if analysis['discovered_interests']:
        print(f"\n  Discovered New Interests:")
        for category in analysis['discovered_interests']:
            count = analysis['category_distribution'].get(category, 0)
            print(f"     {category.title()} ({count} visits)")

print("\n Adapting future recommendations...")
adapted_preferences = tracker.get_adapted_preferences(preferences, user_id)

if adapted_preferences.get('adapted', False):
    print(f"\n   Preferences updated!")
    print(f"    Original interests: {', '.join(preferences['activity_categories'])}")
    print(f"    New interests: {', '.join(adapted_preferences['new_interests'])}")
    print(f"    Updated interests: {', '.join(adapted_preferences['activity_categories'])}")
    print(f"\n  → Future itineraries will include these new categories!")
else:
    print("\n  → No new interests discovered yet")


# COMPLETION


print("\n\n" + "="*80)
print("✅ ITINERARY PLANNING COMPLETE!")


print(f"\nSummary for {user_id}:")
print(f"  • Trip Duration: {preferences['available_days']} days")
print(f"  • Total Attractions: {sum(len(attrs) for attrs in optimized_itinerary.values())}")
print(f"  • Avg per Day: {sum(len(attrs) for attrs in optimized_itinerary.values()) / preferences['available_days']:.1f}")
print(f"  • Total Distance: {total_distance:.1f} km")
print(f"  • Estimated Cost: LKR {total_cost:,.0f}")
print(f"  • Categories: {', '.join(preferences['activity_categories'])}")

print(f"\nTechnical Features:")
print(f"  • ML Model: XGBoost + Fusion Neural Network")
print(f"  • Negative Sampling: 1:3 ratio")
print(f"  • Diversity Clustering: Max 2 per category/day")
print(f"  • Dynamic Capacity: 1-5 attractions/day based on hours")
print(f"  • Context Integration: Components 1, 2, 3 + Weather")
print(f"  • Behavioral Adaptation: Reinforcement Learning")

print(f"\nFiles Created:")
print(f"  • Itinerary: {output_file}")
print(f"  • Preferences: ../data/user_preferences.csv")
print(f"  • Behavior Log: ../data/behavior_log.csv")


print("Thank you for using Smart Tourism Itinerary Planner!")
