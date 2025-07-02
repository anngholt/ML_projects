# Project 1 – Regression Analysis with the Franke Function


In this project, I explore how to model a 2D function (Franke's function) using different regression techniques and study model behavior under various conditions such as noise, model complexity, and regularization.


## Goals of the Project

- Fit the Franke function with:
  - Ordinary Least Squares (OLS)
  - Ridge Regression
  - Lasso Regression
- Evaluate model performance using:
  - Mean Squared Error (MSE)
  - R² Score
- Explore:
  - Impact of polynomial degree
  - Effect of noise on fit quality
  - Regularization and model complexity
  - Bias-variance tradeoff
- Apply resampling methods:
  - Bootstrapping
  - k-Fold Cross-Validation

## How to use main.py

Change this section to switch experiments:

MODEL_TYPE = "ridge"  # Choose: "ols", "ridge", or "lasso"
LAMBDA = 1e-3
POLY_DEGREE = 5
NOISE_STD = 0.1




## Structure

Project1/
├── src/
│   ├── franke_function.py
│   ├── metrics.py
├── main.py
├── results/
├── figures/
├── requirements.txt
└── README.md
