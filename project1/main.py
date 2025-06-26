# main.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from src.franke_function import franke_function
from src.data_gen import create_design_matrix
from src.metrics import mean_squared_error, r2_score

# --------------------
# Configuration
# --------------------
n_points = 100        # Number of points in each spatial dimension
noise_std = 0.1       # Standard deviation of added Gaussian noise
poly_degree = 5       # Degree of polynomial for design matrix
seed = 42             # Random seed for reproducibility

# --------------------
# Generate data
# --------------------
np.random.seed(seed)
x = np.linspace(0, 1, n_points)
y = np.linspace(0, 1, n_points)
x_mesh, y_mesh = np.meshgrid(x, y)

z = franke_function(x_mesh, y_mesh) + noise_std * np.random.randn(*x_mesh.shape)

# Flatten for regression
x_flat = x_mesh.ravel()
y_flat = y_mesh.ravel()
z_flat = z.ravel()

# --------------------
# Create design matrix
# --------------------
X = create_design_matrix(x_flat, y_flat, degree=poly_degree)

# --------------------
# Train-test split
# --------------------
X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2, random_state=seed)

# --------------------
# OLS Regression
# --------------------
ols = LinearRegression(fit_intercept=False)
ols.fit(X_train, z_train)

z_train_pred = ols.predict(X_train)
z_test_pred = ols.predict(X_test)

# --------------------
# Evaluation
# --------------------
print(f"Train MSE: {mean_squared_error(z_train, z_train_pred):.4f}")
print(f"Test MSE:  {mean_squared_error(z_test, z_test_pred):.4f}")
print(f"Train R²:  {r2_score(z_train, z_train_pred):.4f}")
print(f"Test R²:   {r2_score(z_test, z_test_pred):.4f}")

# --------------------
# Visualization
# --------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x_flat, y_flat, ols.predict(X), cmap='viridis', linewidth=0.2)
ax.set_title("OLS Prediction Surface (Degree = 5)")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.show()
