# src/metrics.py

import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).
    
    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values
    
    Returns:
    - MSE value
    """
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    """
    Compute the coefficient of determination (R² score).
    
    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values
    
    Returns:
    - R² score (float): 1 means perfect prediction, 0 means model predicts the mean.
    """
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - numerator / denominator