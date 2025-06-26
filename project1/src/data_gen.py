# src/data_gen.py

import numpy as np
from .franke_function import franke_function  # Relative import

def generate_data(n=100, noise_std=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x_mesh, y_mesh = np.meshgrid(x, y)

    z = franke_function(x_mesh, y_mesh) + noise_std * np.random.randn(n, n)

    return x_mesh.ravel(), y_mesh.ravel(), z.ravel()


def create_design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    n_terms = int((degree + 1)*(degree + 2)/2)
    X = np.ones((N, n_terms))

    idx = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, idx] = (x**(i - j)) * (y**j)
            idx += 1

    return X