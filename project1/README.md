# Project 1 – Regression Analysis with the Franke Function


In this project, I explore how to model a 2D function (Franke's function) using different regression techniques and study model behavior under various conditions such as noise, model complexity, and regularization.

---

## Goals of the Project

- Fit the Franke function with:
  - **Ordinary Least Squares (OLS)**
  - **Ridge Regression**
  - **Lasso Regression**
- Evaluate model performance using:
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Explore:
  - Impact of polynomial degree
  - Effect of noise on fit quality
  - Regularization and model complexity
  - **Bias-variance tradeoff**
- Apply resampling methods:
  - **Bootstrapping**
  - **k-Fold Cross-Validation**

---

## The Franke Function

A commonly used test function in two dimensions:

z = FrankeFunction(x, y) + noise

## Project Structure

Project1/
├── src/
│   ├── data_gen.py         # Data generation using Franke function
│   ├── regression.py       # OLS, Ridge, and Lasso implementations
│   ├── metrics.py          # MSE, R², bias, variance, etc.
│   └── plots.py            # Visualization utilities
├── notebooks/
│   └── project1_analysis.ipynb  # Main exploratory notebook
├── results/                # Output files, model scores, tables
├── figures/                # Plots and surfaces
├── main.py                 # Script for running everything
├── requirements.txt        # Python dependencies
└── README.md               # This file