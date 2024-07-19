# Supplier Reliability Prediction System

This repository contains a set of Python scripts for predicting and analyzing supplier reliability based on historical order data. The system uses various machine learning models to provide insights into supplier performance.

## Scripts

1. `ridge_regressor.py`: Implements a Ridge Regression model with hyperparameter tuning.
2. `bayesian_ridge.py`: Uses a Bayesian Ridge Regression model for robust predictions.
3. `model_comparison.py`: Compares multiple models (Bayesian Ridge, XGBoost, and LightGBM).

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- optuna (for `ridge_regressor.py`)

Install the required packages using:
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib optuna

## Data Preparation

Ensure your data is in CSV format and contains the following columns:
- SupplierCode
- SupplierName
- CountryOfOrigin
- Exfactory_EST
- Exfactory_actual
- GoodsReady_Actual
- goods_ready_date_est
- QtyShip
- CustomerPOQTY
- OrderNumber

Place your CSV file(s) in a known location on your system.

## Usage

### 1. Ridge Regressor (`ridge_regressor.py`)

This script uses Ridge Regression with hyperparameter tuning via Optuna.

Key functions:
- `downcast(df)`: Optimizes DataFrame memory usage.
- `calculate_date_score(days)`: Calculates a score based on delivery timeliness.
- `calculate_fulfillment_score(rate)`: Calculates a score based on order fulfillment rate.
- `objective(trial)`: Defines the optimization objective for Optuna.

To use:
1. Update the `file_paths` list in the script with your CSV file paths.
2. Run: `python ridge_regressor.py`

### 2. Bayesian Ridge Regressor (`bayesian_ridge.py`)

Implements a Bayesian Ridge Regression model for supplier reliability prediction.

Key functions:
- `load_data(file_path)`: Loads data from a CSV file.
- `clean_data(df)`: Removes rows with missing values in key columns.
- `preprocess_data(df)`: Converts date columns and calculates accuracy metrics.
- `calculate_scores(df)`: Computes various performance scores for suppliers.

To use:
1. Update the file path in the `load_data()` function call within `main()`.
2. Run: `python bayesian_ridge.py`

### 3. Model Comparison (`model_comparison.py`)

Compares multiple models for supplier reliability prediction.

Key functions:
- `cross_validate_model(model, X, y, kf)`: Performs cross-validation for a given model.
- `display_results(model_name, train_losses, val_losses, train_r2_scores, val_r2_scores)`: Displays cross-validation results.
- `evaluate_model(model, X_val, y_val)`: Evaluates the model on validation data.
- `plot_results(results)`: Visualizes actual vs predicted scores and error distribution.

To use:
1. Update the file path in the `load_data()` function call within `main()`.
2. Run: `python model_comparison.py`

## Output

Each script will generate:
- Model performance metrics (MSE, RMSE, R2, MAE, etc.)
- Visualizations of actual vs. predicted reliability scores
- Lists of top and bottom performing suppliers

## Customization

You can customize these scripts by:
- Modifying the feature selection in the data preparation functions
- Adjusting the hyperparameters of the models
- Adding new models to the comparison in `model_comparison.py`
- Changing the scoring methods in `calculate_date_score()` and `calculate_fulfillment_score()`

## Performance Considerations

These scripts are designed for medium-sized datasets. For very large datasets, consider:
- Implementing data chunking
- Using more memory-efficient data structures
- Utilizing distributed computing frameworks
