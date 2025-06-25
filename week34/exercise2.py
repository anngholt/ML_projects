import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate random input data
x = np.random.rand(100,1)
y = 2.0 + 5*x**2 + 0.1*np.random.randn(100,1)  # true model + noise

# Build design matrix manually for 2nd-order polynomial: [1, x, x^2]
X = np.hstack((np.ones_like(x), x, x**2))

# Solve for beta using the Normal Equation: β = (XᵀX)⁻¹Xᵀy
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Manual fit coefficients (beta):")
print(beta.ravel())



# Create polynomial features up to degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)  # Automatically includes 1, x, x^2

# Fit using scikit-learn
model = LinearRegression()
model.fit(X_poly, y)

# Predict on training data
y_pred = model.predict(X_poly)

# Output coefficients
print("Scikit-learn fit coefficients:")
print(model.intercept_, model.coef_)

# Compute Mean Squared Error
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
