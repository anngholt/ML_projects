import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# ----------------------
# Metrics: R^2 and MSE
# ----------------------
def R2(y_true, y_pred):
    """Coefficient of determination."""
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def MSE(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

# ---------------------------
# Generate synthetic dataset
# ---------------------------
np.random.seed(3155)  # For reproducibility
n = 100               # Number of data points
x = np.random.rand(n)
y = 2.0 + 5 * x**2 + 0.1 * np.random.randn(n)  # Quadratic function with noise

# -------------------------------
# Construct polynomial features
# -------------------------------
def create_design_matrix(x, degree):
    """Builds the design matrix with polynomial features up to a given degree."""
    X = np.vstack([x**i for i in range(degree + 1)]).T
    return X

# ---------------------------------------------
# Settings for the model and regularization loop
# ---------------------------------------------
degrees = range(1, 15)  # Polynomial degrees to test
lambdas = np.logspace(-4, 1, 20)  # Ridge regularization strengths

# Arrays to store error metrics
mse_train = np.zeros((len(degrees), len(lambdas)))
mse_test = np.zeros((len(degrees), len(lambdas)))
r2_train = np.zeros((len(degrees), len(lambdas)))
r2_test = np.zeros((len(degrees), len(lambdas)))

# -----------------------------
# Ridge Regression Evaluation
# -----------------------------
for d_idx, degree in enumerate(degrees):
    X = create_design_matrix(x, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for l_idx, lmb in enumerate(lambdas):
        # Manual Ridge Regression
        I = np.eye(X_train.shape[1])
        beta_ridge = np.linalg.inv(X_train.T @ X_train + lmb * I) @ X_train.T @ y_train

        # Predictions
        y_train_pred = X_train @ beta_ridge
        y_test_pred = X_test @ beta_ridge

        # Error metrics
        mse_train[d_idx, l_idx] = MSE(y_train, y_train_pred)
        mse_test[d_idx, l_idx] = MSE(y_test, y_test_pred)
        r2_train[d_idx, l_idx] = R2(y_train, y_train_pred)
        r2_test[d_idx, l_idx] = R2(y_test, y_test_pred)

# -----------------------------
# Plot MSE vs Degree of Polynomial
# -----------------------------
best_lambda_idx = 10  # Example: Choose a moderate lambda for visualization

plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_train[:, best_lambda_idx], label='Training MSE')
plt.plot(degrees, mse_test[:, best_lambda_idx], label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title(f'MSE vs Polynomial Degree (lambda = {lambdas[best_lambda_idx]:.4f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot MSE vs Lambda for Fixed Degree
# -----------------------------
fixed_degree_idx = 4  # Example: Degree = 5 (index 4 since degrees starts at 1)

plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, mse_train[fixed_degree_idx], label='Training MSE')
plt.semilogx(lambdas, mse_test[fixed_degree_idx], label='Test MSE')
plt.xlabel('Lambda (Regularization strength)')
plt.ylabel('Mean Squared Error')
plt.title(f'MSE vs Lambda (Polynomial Degree = {degrees[fixed_degree_idx]})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Identify Optimal Model
# -----------------------------
opt_idx = np.unravel_index(np.argmin(mse_test), mse_test.shape)
opt_degree = degrees[opt_idx[0]]
opt_lambda = lambdas[opt_idx[1]]
opt_mse = mse_test[opt_idx]

print(f"\nOptimal model: Degree = {opt_degree}, Lambda = {opt_lambda:.4f}, Test MSE = {opt_mse:.5f}")

# --- RESULT ---
# Optimal model: Degree = 5, Lambda = 0.0001, Test MSE = 0.00793