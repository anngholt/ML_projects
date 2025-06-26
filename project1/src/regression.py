# src/regression.py

import numpy as np

class LinearRegression:
    """
    Ordinary Least Squares Regression (OLS).
    """
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        """
        Fit OLS model to training data using normal equation.

        Parameters:
        - X: Design matrix (n_samples x n_features)
        - y: Target values (n_samples,)
        """
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predict target values using the trained model.

        Parameters:
        - X: Design matrix (n_samples x n_features)

        Returns:
        - y_pred: Predicted target values
        """
        return X @ self.beta


class RidgeRegression:
    """
    Ridge Regression with L2 regularization.
    """
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.beta = None

    def fit(self, X, y):
        """
        Fit Ridge model to training data using the closed-form solution.

        Parameters:
        - X: Design matrix (n_samples x n_features)
        - y: Target values (n_samples,)
        """
        n_features = X.shape[1]
        I = np.eye(n_features)
        self.beta = np.linalg.pinv(X.T @ X + self.lmbda * I) @ X.T @ y

    def predict(self, X):
        """
        Predict target values using the trained Ridge model.

        Parameters:
        - X: Design matrix (n_samples x n_features)

        Returns:
        - y_pred: Predicted target values
        """
        return X @ self.beta