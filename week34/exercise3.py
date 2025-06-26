import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Part A ---

# Ensure reproducibility
np.random.seed(42)

# Parameters
n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

# Polynomial design matrix for degree 5
poly5 = PolynomialFeatures(degree=5)
X5 = poly5.fit_transform(x)

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X5, y, test_size=0.3, random_state=42)

# --- Part B ---

# Fit Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate MSE
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("Degree 5 polynomial fit:")
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# --- Part C ---

max_degree = 20
mse_train_list = []
mse_test_list = []

for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_degree + 1), mse_train_list, label="Training MSE", marker='o')
plt.plot(range(1, max_degree + 1), mse_test_list, label="Test MSE", marker='s')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Test MSE for Polynomial Fits")
plt.legend()
plt.grid(True)
plt.show()
