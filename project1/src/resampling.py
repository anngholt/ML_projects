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


def k_fold_cross_validation(X, y, model_class, k=5, **model_kwargs):
    """
    Perform k-Fold Cross-Validation.

    Parameters:
    - X: Design matrix (n_samples x n_features)
    - y: Target values (n_samples,)
    - model_class: Regression model class (e.g., LinearRegression)
    - k: Number of folds (default 5)
    - model_kwargs: Additional keyword args to pass to model_class constructor

    Returns:
    - avg_mse: Average Mean Squared Error across folds
    - avg_bias: Average bias^2 across folds
    - avg_var: Average variance across folds
    """
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1  # Distribute remainder

    current = 0
    mses = []
    biases = []
    variances = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = np.mean((y_test - y_pred) ** 2)
        bias = np.mean((y_test - np.mean(y_pred)) ** 2)
        var = np.var(y_pred)

        mses.append(mse)
        biases.append(bias)
        variances.append(var)

        current = stop

    avg_mse = np.mean(mses)
    avg_bias = np.mean(biases)
    avg_var = np.mean(variances)

    return avg_mse, avg_bias, avg_var
