# OLS Linear Regression from Scratch

This repository contains a manual implementation of Ordinary Least Squares (OLS) simple linear regression in Python using core mathematical formulas (without high-level machine learning libraries like `scikit-learn` or `statsmodels`).

## Overview

* **Dataset**: `pollution_cleaneddata.csv` ($n = 60$)
* **Target Variables**: 
  * Independent variable ($X$): `POOR` (% of families with income < $3000)
  * Dependent variable ($Y$): `MORT` (Total age-adjusted mortality rate)
* **Objective**: Compute OLS regression parameters, residual standard error, and make expected value predictions with standard errors ($SE_{mean}$) for target values (10%, 18%, 25%).

## Implementation Details

* **Slope ($\beta_1$) & Intercept ($\beta_0$)**: Derived using $SS_{xy} / SS_x$ and $\bar{y} - \beta_1 \bar{x}$.
* **Residual Standard Error ($s_e$)**:
  $$s_e = \sqrt{\frac{1}{n-2} \left(SS_y - \frac{SS_{xy}^2}{SS_x}\right)}$$
* **Standard Error of Mean Response**:
  $$SE(\hat{\mu}) = s_e \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{SS_x}}$$
* **Visualization**: Generated using `matplotlib` to plot actual data points, the fitted regression line, and targeted predictions.

## How to Run

1. Ensure Python and required libraries (`numpy`, `pandas`, `matplotlib`) are installed.
2. Place `pollution_cleaneddata.csv` in the same directory as the script.
3. Run the script:

```bash
python Exercise-01.py
