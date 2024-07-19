#!/usr/bin/env python
# coding: utf-8

"""
Supplier Reliability Predictor

This script trains a Bayesian Ridge Regression model to predict supplier reliability scores
based on historical order data. It includes data preprocessing, feature engineering,
model training, hyperparameter tuning, and evaluation.

Requirements:
- Python 3.x
- Libraries: dask, pandas, numpy, scikit-learn, matplotlib, optuna

Usage:
1. Ensure all required libraries are installed.
2. Update the file_paths list with the correct paths to your CSV data files.
3. Run the script to train the model and evaluate its performance.
4. Use the trained model to predict reliability scores for new suppliers.

Author: [Your Name]
Date: [Current Date]
"""

# Import required libraries
import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             median_absolute_error, explained_variance_score, max_error)
import optuna

def downcast(df):
    """
    Downcast integer and float columns in a DataFrame to reduce memory usage.
    
    Args:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with downcasted dtypes
    """
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def load_and_preprocess_data(file_paths):
    """
    Load and preprocess data from multiple CSV files.
    
    Args:
    file_paths (list): List of file paths to CSV files
    
    Returns:
    pandas.DataFrame: Preprocessed and combined DataFrame
    """
    # Define the dtype for specific columns
    dtypes = {
        'PObooked_Org': 'object',
        'POapproved_Org': 'object',
        'GoodsReady_Actual': 'object',
        'DCdeliverycompleted_Org': 'object',
    }

    # Create a list to hold Dask DataFrames
    dask_dfs = []

    # Read and downcast each file in the list
    for file_path in file_paths:
        ddf = dd.read_csv(file_path, assume_missing=True, low_memory=False, dtype=dtypes)
        ddf = ddf.map_partitions(downcast)
        dask_dfs.append(ddf)

    # Concatenate all Dask DataFrames
    df = dd.concat(dask_dfs, axis=0, interleave_partitions=True)

    # Compute the result into a single DataFrame
    return df.compute()

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

def engineer_features(df):
    """
    Engineer features for the model.
    
    Args:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with engineered features
    """
    # Convert date columns to datetime
    date_columns = ['Exfactory_actual', 'Exfactory_EST', 'GoodsReady_Actual', 'goods_ready_date_est']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Calculate accuracy metrics
    df['Production_Accuracy'] = (df['Exfactory_actual'] - df['Exfactory_EST']).dt.days
    df['Goods_Ready_Accuracy'] = (df['GoodsReady_Actual'] - df['goods_ready_date_est']).dt.days
    df['Order_Fulfillment_Accuracy'] = np.where(
        df['CustomerPOQTY'] > 0,
        (df['QtyShip'] / df['CustomerPOQTY']) * 100,
        0
    )

    # Calculate scores
    df['ex_factory_score'] = calculate_date_score_vec(df['Production_Accuracy'])
    df['goods_ready_score'] = calculate_date_score_vec(df['Goods_Ready_Accuracy'])
    df['fulfillment_score'] = calculate_fulfillment_score_vec(df['Order_Fulfillment_Accuracy'])

    # Calculate overall order score
    df['order_score'] = (
        0.33 * df['ex_factory_score'] +
        0.33 * df['goods_ready_score'] +
        0.33 * df['fulfillment_score']
    )

    return df

def calculate_date_score(days):
    """
    Calculate a score based on the number of days early or late.
    
    Args:
    days (int): Number of days early (negative) or late (positive)
    
    Returns:
    float: Calculated score
    """
    if days == 0:
        return 100
    elif days > 0:  # Late delivery
        return max(0, 100 - (days ** 2))
    else:  # Early delivery
        return max(0, 100 - (abs(days) ** 1))

calculate_date_score_vec = np.vectorize(calculate_date_score)

def calculate_fulfillment_score(rate):
    """
    Calculate a score based on the order fulfillment rate.
    
    Args:
    rate (float): Order fulfillment rate
    
    Returns:
    float: Calculated score
    """
    if rate == 100:
        return 100
    elif rate > 100:  # Over-delivery
        return max(0, 100 - ((rate - 100) ** 1.8))
    else:  # Under-delivery
        return max(0, 100 - ((100 - rate) ** 1.8))

calculate_fulfillment_score_vec = np.vectorize(calculate_fulfillment_score)

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
                  'OrderFulfillmentAccuracyMean', 'OrderFulfillmentAccuracyStd']]
    y = agg_data['ReliabilityScore']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, X_scaled

def optimize_model(X_scaled, y):
    """
    Optimize the Bayesian Ridge model using Optuna.
    
    Args:
    X_scaled (numpy.ndarray): Scaled feature matrix
    y (numpy.ndarray): Target vector
    
    Returns:
    dict: Best hyperparameters
    """
    def objective(trial):
        y_std = np.std(y)
        params = {
            'alpha_1': trial.suggest_loguniform('alpha_1', 1e-10, 1e-1),
            'alpha_2': trial.suggest_loguniform('alpha_2', 1e-10, 1e-1),
            'lambda_1': trial.suggest_loguniform('lambda_1', 1e-10, 1e-1),
            'lambda_2': trial.suggest_loguniform('lambda_2', 1e-10, 1e-1),
            'alpha_init': 1.0 / (y_std ** 2),
            'lambda_init': 1.0 / (y_std ** 2),
            'compute_score': trial.suggest_categorical('compute_score', [True, False]),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'n_iter': trial.suggest_int('n_iter', 100, 3000)
        }
        
        model = BayesianRidge(**params)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1500)
    return study.best_params

def train_and_evaluate_model(X_train, X_val, y_train, y_val, best_params):
    """
    Train the model with best parameters and evaluate its performance.
    
    Args:
    X_train, X_val, y_train, y_val: Training and validation data
    best_params (dict): Best hyperparameters from optimization
    
    Returns:
    tuple: Trained model, prediction results, evaluation metrics
    """
    best_model = BayesianRidge(**best_params)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_val)
    
    metrics = {
        'MSE': mean_squared_error(y_val, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
        'R2': r2_score(y_val, y_pred),
        'MAE': mean_absolute_error(y_val, y_pred),
        'MedianAE': median_absolute_error(y_val, y_pred),
        'ExplainedVariance': explained_variance_score(y_val, y_pred),
        'MaxError': max_error(y_val, y_pred),
        'MAPE': np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    }
    
    return best_model, y_pred, metrics

def main():
    # Define file paths
    file_paths = [
        '/project/Dataset/new data/CL17799_2024.csv',
        '/project/Dataset/new data/CL17799_2023.csv',
        '/project/Dataset/new data/CL17799_2022.csv',
        '/project/Dataset/new data/CL17799_2021.csv',
        '/project/Dataset/new data/CL17799_2020.csv',
        '/project/Dataset/new data/CL17799_2019.csv',
        '/project/Dataset/new data/CL17799_2018.csv'
    ]

    # Load and preprocess data
    df = load_and_preprocess_data(file_paths)
    df_cleaned = clean_data(df)
    df_engineered = engineer_features(df_cleaned)
    agg_data = aggregate_supplier_data(df_engineered)

    # Prepare data for modeling
    X, y, X_scaled = prepare_model_data(agg_data)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Optimize model
    best_params = optimize_model(X_scaled, y)

    # Train and evaluate model
    best_model, y_pred, metrics = train_and_evaluate_model(X_train, X_val, y_train, y_val, best_params)

    # Print evaluation metrics
    print("Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save the model
    import pickle
    with open('supplier_reliability_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print("Model saved as 'supplier_reliability_model.pkl'")

if __name__ == "__main__":
    main()