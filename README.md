# Smart Tourism: Integrating Environmental, Behavioral, and Digital Data for Adaptive Forecasting

## Project Overview

This project implements an advanced adaptive forecasting system for the tourism industry using a Hybrid Ensemble approach. By integrating diverse data sources—including environmental factors (weather), behavioral indicators (sentiment analysis), and digital footprints—the system predicts tourist arrivals with high accuracy across multiple districts.

The core technology utilizes a combination of **XGBoost** for capturing non-linear relationships and **LSTM (Long Short-Term Memory)** networks for modeling temporal sequences, enhanced by an optimized weighted ensemble strategy.

## System Architecture

The project is organized into four main components, each designed to handle specific aspects of the forecasting pipeline. While Component 1 is the primary reference implementation, the other components follow a similar structural pattern.

### Directory Structure

Each component adheres to the following standardized directory layout to ensure consistency and modularity:

```
componentX/
├── data/
│   ├── raw/             # Original raw datasets
│   └── processed/       # Cleaned and feature-engineered data
├── inference/           # Scripts for running model inference
├── metrics/             # Performance reports and visualization plots
├── models/              # Saved model artifacts (.pkl, .h5, .json)
├── script/              # Core training and evaluation scripts
└── utils/               # Helper functions and shared utilities
```

## Key Features

*   **Hybrid Ensemble Model**: Combines XGBoost and LSTM with MAPE-optimized weighting.
*   **Automated Feature Engineering**: Includes rolling averages, lag features, and peak season indicators.
*   **Auto-PCA**: Automatically applies Principal Component Analysis to weather features to reduce dimensionality while retaining 95% variance.
*   **Comprehensive Metrics**: Generates MAE, RMSE, R², and MAPE metrics for rigorous performance evaluation.
*   **Visual Reporting**: Automatically generates plots for Actual vs. Predicted values, Residuals, and Feature Importance.

## Prerequisites

Ensure you have **Python 3.8+** installed. The project relies on the following key libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xgboost`
*   `tensorflow` (for LSTM)
*   `matplotlib`
*   `seaborn`

### Installation

Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
```

## Workflow & Usage

### 1. Data Preparation
Ensure your processed dataset is placed in the `data/processed/` directory. The default expected filename is `final_training_dataset.csv`.

### 2. Training the Model
Navigate to the component's script directory and execute the training script.

**Example for Component 1:**

```bash
cd component1/script
python train_hybrid_ensemble_forecaster.py
```

### 3. Reviewing Results
Upon successful execution, the system will output artifacts in the following directories:

*   **`models/`**: Contains the trained XGBoost (`xgb_models.pkl`), LSTM (`lstm_model.h5`), and scaler objects.
*   **`metrics/`**: Contains the `performance_metrics.txt` report and visualization graphs (`actual_vs_predicted.png`, `feature_importance.png`, etc.).

## Output Artifacts

*   **`model_metadata.json`**: Stores configuration details, feature lists, and ensemble weights.
*   **`performance_metrics.txt`**: A detailed summary of model performance, including per-district error rates.
*   **Visualizations**: High-resolution PNGs comparing model predictions against ground truth.

---
*Note: This structure applies across all logic components (1-4) within the repository.*