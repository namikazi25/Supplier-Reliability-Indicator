#!/usr/bin/env python
# coding: utf-8

"""
Supplier Reliability Predictor - Model Comparison

This script compares multiple machine learning models (Bayesian Ridge, XGBoost, and LightGBM)
for predicting supplier reliability scores based on historical order data. It includes
data preprocessing, feature engineering, model training, cross-validation, and evaluation.

Requirements:
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib

Usage:
1. Ensure all required libraries are installed.
2. Update the file path in the load_data() function to point to your CSV data file.
3. Run the script to train and compare models, and evaluate their performance.

Author: Mir Nafis Sharear Shopnil
Date: 16 July 2024
"""

"""
How to Use:
1. Environment Setup:
   - Ensure you have Python 3.x installed on your system.
   - Install required libraries using pip:
     pip install pandas numpy scikit-learn xgboost lightgbm matplotlib

2. Data Preparation:
   - Place your CSV data file in a known location on your system.
   - The CSV should contain columns for supplier information, order details, and dates.

3. Script Configuration:
   - Open this script in a text editor or IDE.
   - Locate the main() function at the bottom of the script.
   - Update the file path in the load_data() function call to point to your CSV file:
     df = load_data('/path/to/your/data/file.csv')

4. Running the Script:
   - Open a terminal or command prompt.
   - Navigate to the directory containing this script.
   - Run the script using Python:
     python supplier_reliability_predictor.py

5. Interpreting Results:
   - The script will output cross-validation results for each model.
   - It will display evaluation metrics for the final model.
   - Several plots will be generated to visualize the results:
     - Actual vs Predicted Scores
     - Predicted vs Actual Scores scatter plot
     - Distribution of Prediction Differences

6. Customization (Optional):
   - Modify the feature selection in the prepare_model_data() function if needed.
   - Adjust the models dictionary in main() to add or remove models for comparison.
   - Customize the plotting functions to change the visualizations.

Note: Ensure you have sufficient memory to handle your dataset size. For very large
datasets, you may need to implement data chunking or use a more memory-efficient approach.
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             median_absolute_error, explained_variance_score, max_error)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded DataFrame
    """
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Clean the data by dropping rows with missing values in specific columns.
    
    Args:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    columns_to_check = [
        'CountryOfOrigin', 'Exfactory_EST', 'Exfactory_actual', 'GoodsReady_Actual', 
        'goods_ready_date_est', 'QtyShip'
    ]
    return df.dropna(subset=columns_to_check)

def preprocess_data(df):
    """
    Preprocess the data by converting date columns and calculating accuracy metrics.
    
    Args:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: Preprocessed DataFrame
    """
    date_columns = ['Exfactory_actual', 'Exfactory_EST', 'GoodsReady_Actual', 'goods_ready_date_est']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    df['Production_Accuracy'] = (df['Exfactory_actual'] - df['Exfactory_EST']).dt.days
    df['Goods_Ready_Accuracy'] = (df['GoodsReady_Actual'] - df['goods_ready_date_est']).dt.days
    df['Order_Fulfillment_Accuracy'] = np.where(
        df['CustomerPOQTY'] > 0,
        (df['QtyShip'] / df['CustomerPOQTY']) * 100,
        0
    )
    return df

def calculate_scores(df):
    """
    Calculate scores based on accuracy metrics.
    
    Args:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with calculated scores
    """
    def calculate_date_score(days):
        if days == 0:
            return 100
        elif days > 0:
            return max(0, 100 - (days ** 2))
        else:
            return max(0, 100 - (abs(days) ** 1))
    
    def calculate_fulfillment_score(rate):
        if rate == 100:
            return 100
        elif rate > 100:
            return max(0, 100 - ((rate - 100) ** 1.8))
        else:
            return max(0, 100 - ((100 - rate) ** 1.8))
    
    calculate_date_score_vec = np.vectorize(calculate_date_score)
    calculate_fulfillment_score_vec = np.vectorize(calculate_fulfillment_score)
    
    df['ex_factory_score'] = calculate_date_score_vec(df['Production_Accuracy'])
    df['goods_ready_score'] = calculate_date_score_vec(df['Goods_Ready_Accuracy'])
    df['fulfillment_score'] = calculate_fulfillment_score_vec(df['Order_Fulfillment_Accuracy'])
    
    df['order_score'] = (
        0.4 * df['ex_factory_score'] +
        0.2 * df['goods_ready_score'] +
        0.4 * df['fulfillment_score']
    )
    return df

def aggregate_supplier_data(df):
    """
    Aggregate data at the supplier level.
    
    Args:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: Aggregated DataFrame
    """
    agg_data = df.groupby('SupplierCode').agg({
        'Production_Accuracy': ['mean', 'std'],
        'Goods_Ready_Accuracy': ['mean', 'std'],
        'Order_Fulfillment_Accuracy': ['mean', 'std'],
        'order_score': 'mean',
        'OrderNumber': 'count',
        'SupplierName': 'first'
    }).reset_index()

    agg_data.columns = ['SupplierCode', 
                        'ProductionAccuracyMean', 'ProductionAccuracyStd', 
                        'GoodsReadyAccuracyMean', 'GoodsReadyAccuracyStd',
                        'OrderFulfillmentAccuracyMean', 'OrderFulfillmentAccuracyStd',
                        'ReliabilityScore',
                        'OrderCount', 'SupplierName']
    return agg_data

def prepare_model_data(agg_data):
    """
    Prepare data for model training.
    
    Args:
    agg_data (pandas.DataFrame): Aggregated supplier data
    
    Returns:
    tuple: X (features), y (target), X_scaled (scaled features)
    """
    X = agg_data[['ProductionAccuracyMean', 'ProductionAccuracyStd', 
                  'GoodsReadyAccuracyMean', 'GoodsReadyAccuracyStd',
                  'OrderFulfillmentAccuracyMean', 'OrderFulfillmentAccuracyStd',
                  'OrderCount']]
    y = agg_data['ReliabilityScore']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, y, X_scaled

def cross_validate_model(model, X, y, kf):
    """
    Perform cross-validation for a given model.
    
    Args:
    model: Machine learning model
    X (numpy.ndarray): Feature matrix
    y (numpy.ndarray): Target vector
    kf (sklearn.model_selection.KFold): K-Fold cross-validator
    
    Returns:
    tuple: Lists of train/validation losses and R2 scores
    """
    train_losses, val_losses, train_r2_scores, val_r2_scores = [], [], [], []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_losses.append(mean_squared_error(y_train, train_pred))
        val_losses.append(mean_squared_error(y_val, val_pred))
        train_r2_scores.append(r2_score(y_train, train_pred))
        val_r2_scores.append(r2_score(y_val, val_pred))

    return train_losses, val_losses, train_r2_scores, val_r2_scores

def display_results(model_name, train_losses, val_losses, train_r2_scores, val_r2_scores):
    """
    Display cross-validation results for a model.
    
    Args:
    model_name (str): Name of the model
    train_losses, val_losses, train_r2_scores, val_r2_scores: Lists of scores from cross-validation
    """
    print(f"{model_name} Results:")
    print(f"Train MSE: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
    print(f"Validation MSE: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Train R2: {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
    print(f"Validation R2: {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on the validation set.
    
    Args:
    model: Trained machine learning model
    X_val (numpy.ndarray): Validation feature matrix
    y_val (numpy.ndarray): Validation target vector
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    median_ae = median_absolute_error(y_val, y_pred)
    evs = explained_variance_score(y_val, y_pred)
    max_err = max_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MedianAE': median_ae,
        'ExplainedVarianceScore': evs,
        'MaxError': max_err,
        'MAPE': mape
    }

def plot_results(results):
    """
    Plot actual vs predicted scores and the distribution of differences.
    
    Args:
    results (pandas.DataFrame): DataFrame containing actual and predicted scores
    """
    # Plot actual vs predicted scores
    plt.figure(figsize=(10, 6))
    plt.scatter(results.index, results['Actual Score'], label='Actual Score', alpha=0.7)
    plt.scatter(results.index, results['Predicted Score'], label='Predicted Score', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('Actual vs Predicted Scores')
    plt.legend()
    plt.show()

    # Plot predicted vs actual scores
    plt.figure(figsize=(10, 10))
    plt.scatter(results['Actual Score'], results['Predicted Score'], alpha=0.5)
    plt.plot([results['Actual Score'].min(), results['Actual Score'].max()], 
             [results['Actual Score'].min(), results['Actual Score'].max()], 
             'r--', lw=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Predicted vs Actual Scores')
    plt.show()

    # Plot the distribution of differences
    plt.figure(figsize=(10, 6))
    results['Difference'].hist(bins=30)
    plt.xlabel('Prediction Difference (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Differences')
    plt.show()

def main():
    # Load and preprocess data
    df = load_data('/project/Dataset/new data/CL17799_2024.csv')
    df_cleaned = clean_data(df)
    df_preprocessed = preprocess_data(df_cleaned)
    df_scored = calculate_scores(df_preprocessed)
    agg_data = aggregate_supplier_data(df_scored)

    # Prepare data for modeling
    X, y, X_scaled = prepare_model_data(agg_data)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Train and evaluate models
    models = {
        'Bayesian Ridge': BayesianRidge(),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }

    for name, model in models.items():
        train_losses, val_losses, train_r2_scores, val_r2_scores = cross_validate_model(model, X_scaled, y, kf)
        display_results(name, train_losses, val_losses, train_r2_scores, val_r2_scores)

    # Train final model (using Bayesian Ridge as an example)
    final_model = BayesianRidge()
    final_model.fit(X_scaled, y)
    predicted_scores = final_model.predict(X_scaled)

    # Evaluate final model
    metrics = evaluate_model(final_model, X_val, y_val)
    print("\nFinal Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create results DataFrame
    results = pd.DataFrame({
        'Actual Score': y,
        'Predicted Score': predicted_scores
    })
    results['Difference'] = results['Predicted Score'] - results['Actual Score']

    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()
