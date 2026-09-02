import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


D_C = pd.read_csv('pollution_cleaneddata.csv')
print ("Test the data type:", D_C.info()) #make sure all data are consistent

#Ordinary Least Squares (OLS) regression is used to expect the value of Total age-adjusted mortality rate per 100,000 
x = D_C['POOR'].values
y = D_C['MORT'].values
n = len(x)

x_m = np.mean(x)
y_m = np.mean(y)

SSxy = np.sum(x*y)-n*x_m*y_m
SSx = np.sum(np.square(x-x_m))
SSy = np.sum(np.square(y-y_m))

Slope = SSxy/SSx
Inter = y_m - Slope*x_m
y_e = Inter + Slope * x

#print(Slope)
#print(Inter)

# Calculate standard error
se =np.sqrt((1/(n-2))*(SSy-(np.square(SSxy)/SSx)))

target_x = np.array([10, 18, 25])
target_y = x_e = Inter + Slope * target_x

for x_new in target_x:
    x_e = Inter + Slope * x_new
    se_error = se * np.sqrt((1 / n) + ((x_new - x_m) ** 2) / SSx) # which is mean standard error
    print(print(f"POOR = {x_new}% | Expected values (MORT): {x_e:.2f} | standard error (SE): {se_error:.2f}"))

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', alpha=0.6, label='Actual Data')
plt.plot(x, y_e, color='blue', linewidth=2, label='Expected Value Line')
plt.scatter(target_x, target_y, color='red', s=70, zorder=5, label='Predictions (10%, 18%, 25%)')

plt.title('POOR vs Expected Mortality Rate (MORT)', fontsize=12)
plt.xlabel('% of families with income < $3000 (POOR)', fontsize=10)
plt.ylabel('Total age-adjusted mortality rate (MORT)', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()