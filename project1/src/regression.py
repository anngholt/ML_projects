# src/regression.py

import numpy as np

# --------------------
# Ordinary Least Squares Regression
# --------------------

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

# --------------------
# Ridge Regression
# --------------------

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

# --------------------
# Lasso Regression
# --------------------

class LassoRegression:
    """
    Lasso Regression using coordinate descent.
    Solves: minimize ||y - Xβ||² + λ * ||β||₁
    """
    def __init__(self, lmbda=1.0, max_iter=1000, tol=1e-4):
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None

    def soft_threshold(self, rho, lmbda):
        """
        Soft thresholding operator.
        """
        if rho < -lmbda:
            return rho + lmbda
        elif rho > lmbda:
            return rho - lmbda
        else:
            return 0.0

    def fit(self, X, y):
        """
        Fit the Lasso model to the training data using coordinate descent.
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)

        for iteration in range(self.max_iter):
            beta_old = self.beta.copy()

            for j in range(n_features):
                tmp_beta = self.beta.copy()
                tmp_beta[j] = 0.0
                residual = y - X @ tmp_beta
                rho = X[:, j] @ residual

                self.beta[j] = self.soft_threshold(rho, self.lmbda)

            # Check convergence
            if np.sum(np.abs(self.beta - beta_old)) < self.tol:
                break

    def predict(self, X):
        """
        Predict using the Lasso regression model.
        """
        return X @ self.beta

