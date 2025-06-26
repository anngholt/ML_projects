# src/franke_function.py

import numpy as np


def franke_function(x, y, noise_scale=0.0, seed=None):
    """
    Compute the Franke function value at the points (x, y) with optional Gaussian noise.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates (1D or 2D numpy array).
    y : np.ndarray
        y-coordinates (same shape as x).
    noise_scale : float, optional
        Standard deviation of Gaussian noise to add (default: 0.0).
    seed : int or None
        Seed for reproducibility (default: None).

    Returns
    -------
    z : np.ndarray
        Franke function values with noise.
    """
    if seed is not None:
        np.random.seed(seed)

    term1 = 0.75 * np.exp(-((9 * x - 2)**2 + (9 * y - 2)**2) / 4)
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - (9 * y + 1) / 10.0)
    term3 = 0.5 * np.exp(-((9 * x - 7)**2 + (9 * y - 3)**2) / 4)
    term4 = -0.2 * np.exp(-((9 * x - 4)**2 + (9 * y - 7)**2))

    z = term1 + term2 + term3 + term4
    noise = noise_scale * np.random.randn(*x.shape)
    return z + noise


def create_design_matrix(x, y, degree):
    """
    Create a design matrix with polynomial terms up to the given degree for 2D data.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates (flattened).
    y : np.ndarray
        y-coordinates (flattened).
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    X : np.ndarray
        Design matrix of shape (len(x), num_polynomial_terms)
    """
    if len(x.shape) > 1:
        x = x.ravel()
    if len(y.shape) > 1:
        y = y.ravel()

    N = len(x)
    num_terms = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, num_terms))

    idx = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, idx] = (x**(i - j)) * (y**j)
            idx += 1

    return X