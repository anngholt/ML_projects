import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
x = np.random.rand(100,1)
y = 2.0 + 5*x**2 + 0.1*np.random.randn(100,1)

# --- Manual Fit using Normal Equation ---
X = np.hstack((np.ones_like(x), x, x**2))
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Manual fit coefficients (beta):")
print(beta.ravel())

# --- Scikit-learn Polynomial Regression ---
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

print("Scikit-learn fit coefficients:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# --- Evaluation Metrics ---
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R² score:", r2)

# --- RESULTS ---
# Manual fit coefficients (beta): [1.99050631 0.1870752  4.78175794]
# Scikit-learn fit coefficients: 
# Intercept: [1.99050631]
# Coefficients: [[0.         0.1870752  4.78175794]]
# Mean Squared Error: 0.008862148439478542
# R² score: 0.9966267034372802