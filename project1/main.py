"""
main.py - Regression analysis on Franke's function.

This script generates 2D surface data from the Franke function and fits regression models
(OLS, Ridge, or Lasso) to analyze model behavior with different polynomial degrees and noise levels.

"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.franke_function import franke_function
from src.data_gen import create_design_matrix
from src.metrics import mean_squared_error, r2_score
from src.regression import LinearRegression, RidgeRegression, LassoRegression
from src.resampling import bootstrap

# -----------------------
# Configuration Flags
# -----------------------
MODEL_TYPE = "ridge"       # Options: "ols", "ridge", "lasso"
LAMBDA = 1e-4              # Regularization strength for Ridge/Lasso
POLY_DEGREE = 5            # Polynomial degree for design matrix
N_POINTS = 100             # Number of points along each axis
NOISE_STD = 0.1            # Standard deviation of Gaussian noise
TEST_SIZE = 0.2            # Proportion of test data
SEED = 42                  # Random seed for reproducibility
SAVE_PLOT = True           # Save plot to figures/
SHOW_PLOT = True           # Display 3D surface plot

# -----------------------
# Data Generation
# -----------------------
np.random.seed(SEED)
x = np.linspace(0, 1, N_POINTS)
y = np.linspace(0, 1, N_POINTS)
x_mesh, y_mesh = np.meshgrid(x, y)

# Franke function + Gaussian noise
z = franke_function(x_mesh, y_mesh) + NOISE_STD * np.random.randn(*x_mesh.shape)

# Flatten for regression
x_flat = x_mesh.ravel()
y_flat = y_mesh.ravel()
z_flat = z.ravel()

# -----------------------
# Design Matrix
# -----------------------
X = create_design_matrix(x_flat, y_flat, degree=POLY_DEGREE)

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=TEST_SIZE, random_state=SEED)

# -----------------------
# Model Selection
# -----------------------
if MODEL_TYPE == "ols":
    model = LinearRegression()
elif MODEL_TYPE == "ridge":
    model = RidgeRegression(lmbda=LAMBDA)
elif MODEL_TYPE == "lasso":
    model = LassoRegression(lmbda=LAMBDA)
else:
    raise ValueError("MODEL_TYPE must be one of: 'ols', 'ridge', 'lasso'")

# -----------------------
# Training and Prediction
# -----------------------
start_time = time.time()
model.fit(X_train, z_train)
elapsed = time.time() - start_time

z_train_pred = model.predict(X_train)
z_test_pred = model.predict(X_test)

# -----------------------
# Evaluation
# -----------------------
print(f"\nModel: {MODEL_TYPE.upper()}")
print(f"Polynomial degree: {POLY_DEGREE}")
if MODEL_TYPE in ["ridge", "lasso"]:
    print(f"Lambda: {LAMBDA}")
print(f"Training time: {elapsed:.4f} sec\n")

print(f"Train MSE: {mean_squared_error(z_train, z_train_pred):.4f}")
print(f"Test  MSE: {mean_squared_error(z_test, z_test_pred):.4f}")
print(f"Train R²:  {r2_score(z_train, z_train_pred):.4f}")
print(f"Test  R²:  {r2_score(z_test, z_test_pred):.4f}")

# -----------------------
# 3D Plot
# -----------------------
if SHOW_PLOT or SAVE_PLOT:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_flat, y_flat, model.predict(X), cmap='viridis', linewidth=0.2)
    ax.set_title(f"{MODEL_TYPE.upper()} Prediction Surface (deg = {POLY_DEGREE})")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()

    if SAVE_PLOT:
        os.makedirs("figures", exist_ok=True)
        filename = f"figures/{MODEL_TYPE}_deg{POLY_DEGREE}_lambda{LAMBDA:.0e}.png" if MODEL_TYPE != "ols" else f"figures/ols_deg{POLY_DEGREE}.png"
        plt.savefig(filename, dpi=300)
        print(f"\nPlot saved to {filename}")

    if SHOW_PLOT:
        plt.show()


# --------------------
# Bootstrap Resampling (Bias-Variance)
# --------------------
mse, bias, var = bootstrap(X_train, z_train, LinearRegression, n_bootstrap=100)
print("\nBootstrap Results (Train Set, OLS):")
print(f"MSE:   {mse:.4f}")
print(f"Bias²: {bias:.4f}")
print(f"Var:   {var:.4f}")