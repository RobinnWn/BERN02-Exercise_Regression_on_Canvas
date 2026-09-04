# OLS Linear Regression and local regression from Scratch

This repository contains a manual implementation of Ordinary Least Squares (OLS) simple linear regression and local regression in Python using core mathematical formulas.

## Overview

* **Dataset**: `pollution_cleaneddata.csv` ($n = 60$)
* **Target Variables**: 
  * Independent variable ($X$): `POOR` (% of families with income < $3000)
  * Dependent variable ($Y$): `MORT` (Total age-adjusted mortality rate)
* **Objective**: Compute OLS regression parameters, residual standard error, and make expected value predictions with standard errors ($SE_{mean}$) for target values (10%, 18%, 25%).

## Implementation Details

### Linear Regression
* **Slope ($\beta_1$) & Intercept ($\beta_0$)**: Derived using $SS_{xy} / SS_x$ and $\bar{y} - \beta_1 \bar{x}$.
* **Residual Standard Error ($s_e$)**:
  $$s_e = \sqrt{\frac{1}{n-2} \left(SS_y - \frac{SS_{xy}^2}{SS_x}\right)}$$
* **Standard Error of Mean Response**:
  $$SE(\hat{\mu}) = s_e \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{SS_x}}$$
* **Visualization**: Generated using `matplotlib` to plot actual data points, the fitted regression line, and targeted predictions.
### Local Regression
* **Neighbor Selection ($k$-NN)**: For each target evaluation point $x_0$, the model selects the $k$ nearest neighbor observations $(x_k, y_k)$ based on Euclidean distance $|x - x_0|$.
* **Local Slope ($\beta_1$) & Intercept ($\beta_0$)**: Derived using OLS over the $k$ local neighbors: $SS_{xy} / SS_x$ and $\bar{y}_k - \beta_1 \bar{x}_k$.
* **Local Residual Standard Error ($s_e$)**: Evaluated over the $k$ local neighbors with $k - 2$ degrees of freedom:
  $$s_e = \sqrt{\frac{1}{k-2} \left(SS_{y,k} - \frac{SS_{xy,k}^2}{SS_{x,k}}\right)}$$
* **Standard Error of Expected Value ($SE_{\hat{y}}$)**: Incorporates local sample size $k$ and leverage effect of $x_0$ relative to the local neighborhood mean $\bar{x}_k$:
  $$SE(\hat{y}) = s_e \sqrt{\frac{1}{k} + \frac{(x_0 - \bar{x}_k)^2}{SS_{x,k}}}$$
* **Visualization**: Generated using `matplotlib` to plot actual observed data points, the fitted global local-regression curve, targeted predictions at $x_0$, and individual local linear fit segments.

## Results and visualization
### 1. Manual OLS Prediction Results
* POOR = 10% | Expected values (MORT): 913.51 | standard error (SE): 10.76
* POOR = 18% | Expected values (MORT): 962.62 | standard error (SE): 9.83
* POOR = 25% | Expected values (MORT): 1005.59 | standard error (SE): 20.41
### 2. Manual Local Regression Prediction Results
* POOR = 10'| pred: 909.73| standard_error: 18.90
* POOR = 18'| pred: 942.63| standard_error: 16.61
* POOR = 25'| pred: 999.61| standard_error: 25.33
### 3. Regression Line & Prediction Plot
<img width="2362" height="1556" alt="image" src="https://github.com/user-attachments/assets/37e877aa-a330-4fda-a64e-9af1dd7c7be2" />
<img width="2368" height="1490" alt="image" src="https://github.com/user-attachments/assets/49e10d85-b3c6-475d-8075-fe7f6a0a5e37" />


## How to Run

1. Ensure Python and required libraries (`numpy`, `pandas`, `matplotlib`) are installed.
2. Place `pollution_cleaneddata.csv` in the same directory as the script.
3. Run the script:

```bash
python Luobin_0903_BERN02_Exercise-01.py
python Luobin_0903_BERN02_local_Regression-01.py

