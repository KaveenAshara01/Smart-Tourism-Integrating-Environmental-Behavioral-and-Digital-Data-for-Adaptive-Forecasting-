import sys

sys.path.append('.')

from performance_tracker import PerformanceTracker
import matplotlib.pyplot as plt
import pandas as pd
import os

# Initialize tracker
tracker = PerformanceTracker(db_path='../data/monitoring.db')

print("GENERATING MONITORING REPORT")

# Get performance trends
print("\n Fetching performance trends...")
perf_trend = tracker.get_performance_trend(limit=12)
print(perf_trend)

# Get drift history
print("\n Fetching drift detection history...")
drift_history = tracker.get_drift_history(limit=10)
print(drift_history)

# Generate visualizations
print("\n Generating visualizations...")

os.makedirs('../metrics/monitoring_plots', exist_ok=True)

# Plot 1: MAPE trend over time
if not perf_trend.empty:
    plt.figure(figsize=(12, 6))
    plt.plot(perf_trend['prediction_month'], perf_trend['mape'], marker='o', linewidth=2)
    plt.axhline(y=15, color='r', linestyle='--', label='Target (15%)')
    plt.axhline(y=25, color='orange', linestyle='--', label='Warning (25%)')
    plt.xlabel('Month')
    plt.ylabel('MAPE (%)')
    plt.title('Model Performance Trend - MAPE Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../metrics/monitoring_plots/mape_trend.png', dpi=300)
    print("  Saved: mape_trend.png")
    plt.close()

# Plot 2: R² trend
if not perf_trend.empty:
    plt.figure(figsize=(12, 6))
    plt.plot(perf_trend['prediction_month'], perf_trend['r2'], marker='o', linewidth=2, color='green')
    plt.axhline(y=0.7, color='r', linestyle='--', label='Target (0.7)')
    plt.xlabel('Month')
    plt.ylabel('R² Score')
    plt.title('Model Performance Trend - R² Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../metrics/monitoring_plots/r2_trend.png', dpi=300)
    print("  Saved: r2_trend.png")
    plt.close()

# Save summary report
print("\n Saving summary report...")

with open('../metrics/monitoring_summary.txt', 'w') as f:
    f.write("ADAPTIVE ML MONITORING SUMMARY\n")

    f.write("Performance Trends:\n")
    f.write(perf_trend.to_string(index=False))
    f.write("\n\n")

    f.write("Drift Detection History:\n")
    f.write(drift_history.to_string(index=False))
    f.write("\n\n")

print("  Saved: monitoring_summary.txt")

print("REPORT GENERATION COMPLETE")
