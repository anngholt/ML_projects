# src/analysis/kfold_cross_validation_analysis.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt

from src.franke_function import franke_function
from src.data_gen import create_design_matrix
from src.resampling import k_fold_cross_validation
from src.regression import LinearRegression

def run_kfold_analysis(
    degree=5,
    n_points=100,
    noise_std=0.1,
    k=5,
    seed=42,
    save_path="figures/kfold_cv_results.png"
):
    """
    Run k-Fold Cross-Validation analysis on Franke function regression.
    
    Parameters:
    - degree: polynomial degree for design matrix
    - n_points: number of points per dimension
    - noise_std: Gaussian noise std deviation
    - k: number of folds for cross-validation
    - seed: random seed for reproducibility
    - save_path: file path to save MSE plot
    """

    np.random.seed(seed)

    # Generate mesh grid data
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # Generate noisy Franke function values
    z = franke_function(x_mesh, y_mesh) + noise_std * np.random.randn(*x_mesh.shape)
    
    x_flat, y_flat, z_flat = x_mesh.ravel(), y_mesh.ravel(), z.ravel()
    
    # Create design matrix
    X = create_design_matrix(x_flat, y_flat, degree=degree)

    # Run k-Fold Cross-Validation
    mse, bias, var = k_fold_cross_validation(X, z_flat, LinearRegression, k=k)

    # Print results
    print(f"k-Fold Cross-Validation Results (k={k}, degree={degree}):")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Bias^2: {bias:.6f}")
    print(f"Variance: {var:.6f}")

    # Plot and save MSE for visual reference
    plt.figure(figsize=(8, 5))
    plt.bar(['MSE', 'Bias²', 'Variance'], [mse, bias, var], color=['blue', 'orange', 'green'])
    plt.title(f"k-Fold CV Results (degree={degree})")
    plt.ylabel("Error")
    plt.tight_layout()

    # Ensure directory exists and save plot
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_path = os.path.join(project_root, "figures/kfold_cv_results.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Current working directory:", os.getcwd())
    print("Saving plot to:", os.path.abspath(save_path))
    
    plt.savefig(save_path)
    plt.show()

    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    run_kfold_analysis()