import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.franke_function import franke_function
from src.data_gen import create_design_matrix
from src.regression import LinearRegression
from src.resampling import bootstrap


def run_bias_variance_analysis(
    max_degree=15,
    n_points=100,
    noise_std=0.1,
    n_bootstrap=100,
    seed=42,
    save_path="figures/bias_variance_tradeoff.png"
):
    """
    Compute and plot bias², variance, and MSE as functions of polynomial degree.

    Parameters:
    - max_degree: maximum polynomial degree to evaluate
    - n_points: number of points in each meshgrid dimension
    - noise_std: standard deviation of Gaussian noise
    - n_bootstrap: number of bootstrap resamplings
    - seed: random seed for reproducibility
    - save_path: path to save the resulting plot
    """
    np.random.seed(seed)

    # Generate data
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    z = franke_function(x_mesh, y_mesh) + noise_std * np.random.randn(*x_mesh.shape)

    x_flat, y_flat, z_flat = x_mesh.ravel(), y_mesh.ravel(), z.ravel()

    degrees = range(1, max_degree + 1)
    mses, biases, variances = [], [], []

    for deg in degrees:
        X = create_design_matrix(x_flat, y_flat, degree=deg)
        X_train, _, z_train, _ = train_test_split(X, z_flat, test_size=0.2, random_state=seed)

        mse, bias, var = bootstrap(X_train, z_train, LinearRegression, n_bootstrap=n_bootstrap)

        mses.append(mse)
        biases.append(bias)
        variances.append(var)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, mses, label="MSE")
    plt.plot(degrees, biases, label="Bias²")
    plt.plot(degrees, variances, label="Variance")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff vs Model Complexity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    run_bias_variance_analysis()
