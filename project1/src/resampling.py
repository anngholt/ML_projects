# src/resampling.py

import numpy as np
from src.metrics import mean_squared_error

def bootstrap(X, y, model_class, n_bootstrap=100, **model_kwargs):
    """
    Perform bootstrap resampling for a given regression model.

    Parameters:
    - X: Feature matrix
    - y: Target values
    - model_class: Class implementing .fit() and .predict()
    - n_bootstrap: Number of bootstrap iterations
    - model_kwargs: Parameters to pass to the model (e.g., lmbda for Ridge)

    Returns:
    - mse: Mean squared error
    - bias: Bias squared
    - variance: Variance
    """
    n = X.shape[0]
    y_pred = np.zeros((n, n_bootstrap))

    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        X_resampled = X[indices]
        y_resampled = y[indices]

        model = model_class(**model_kwargs)
        model.fit(X_resampled, y_resampled)
        y_pred[:, i] = model.predict(X)

    y_pred_mean = np.mean(y_pred, axis=1)

    bias = np.mean((y - y_pred_mean) ** 2)
    variance = np.mean(np.var(y_pred, axis=1))
    mse = np.mean(np.mean((y_pred - y[:, np.newaxis]) ** 2, axis=1))

    return mse, bias, variance
