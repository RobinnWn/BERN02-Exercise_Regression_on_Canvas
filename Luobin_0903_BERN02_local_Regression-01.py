import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


D_C = pd.read_csv('pollution_cleaneddata.csv')
x_vex = D_C['POOR'].values
y_vex = D_C['MORT'].values


def local_regression(y, x, k, x0):
    pred = []
    standard_error = []
    
    for x0_i in x0:
        dis = np.abs(x-x0_i)
        k_near = np.argsort(dis)[:k]
        
        x_k = x[k_near]
        y_k = y[k_near]

        x_k_m = np.mean(x_k)
        y_k_m = np.mean(y_k)
        
        SSxy = np.sum(x_k*y_k)- k * x_k_m * y_k_m
        
        SSx = np.sum(np.square(x_k-x_k_m))
        SSy = np.sum(np.square(y_k-y_k_m))
        Slope = SSxy/SSx
        
        Inter = y_k_m - Slope * x_k_m
        
        y_e = Inter + Slope * x0_i
        se =np.sqrt((1/(k-2))*(SSy-(np.square(SSxy)/SSx)))
        se_y = se * np.sqrt((1 / k) + ((x0_i - x_k_m) ** 2) / SSx)

        pred.append(y_e)
        standard_error.append(se_y)
    return np.array(pred), np.array(standard_error)
    
k_valu = 20
x0_vax = np.array([10, 18, 25])


prediction, s_e  =  local_regression(y_vex, x_vex, k_valu, x0_vax) 
for x_val, p, s in zip (x0_vax, prediction, s_e):
    print(f"POOR = {x_val}%'| pred:{p: .2f}| standard_error:{s: .2f}")

plt.figure(figsize=(10, 6), dpi=120)
plt.scatter(x_vex, y_vex, color='black', alpha=0.6, label='Actual Data')
x_grid = np.linspace(x_vex.min(), x_vex.max(), 200)
pred_grid, _ = local_regression(y_vex, x_vex, k_valu, x_grid)
plt.plot(
    x_grid,
    pred_grid,
    color="#1f77b4",
    linewidth=1.5,
    label=f"Local Linear Regression (k={k_valu})",
)
colors = ["#d95f02", "#7570b3", "#e7298a"]
for i, x0_i in enumerate(x0_vax):
    dis = np.abs(x_vex - x0_i)
    k_near = np.argsort(dis)[:k_valu]
    x_k = x_vex[k_near]
    y_k = y_vex[k_near]

    x_k_m = np.mean(x_k)
    y_k_m = np.mean(y_k)

    SSxy = np.sum(x_k * y_k) - k_valu * x_k_m * y_k_m
    SSx = np.sum(np.square(x_k - x_k_m))

    Slope = SSxy / SSx
    Inter = y_k_m - Slope * x_k_m
    x_line = np.linspace(x_k.min(), x_k.max(), 50)
    y_line = Inter + Slope * x_line
    plt.plot(
        x_line,
        y_line,
        color=colors[i],
        linewidth=2,
        linestyle="--",
        label=f"Local Fit at POOR={x0_i}%",
    )
    p_i = prediction[i]
    plt.scatter(
        x0_i, p_i, color=colors[i], s=80, zorder=5, edgecolors="black"
        )
    plt.annotate(
        f"x0={x0_i}\npred={p_i:.1f}",
        (x0_i, p_i),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        fontweight="bold",
        bbox=dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec=colors[i],
        lw=1.5,
        alpha=0.9,
            ),
)
plt.title(
    "Local Linear Regression with 3 Local Fits (k=20)", fontsize=13, pad=12
)
plt.xlabel("POOR (% of poor families)", fontsize=11)
plt.ylabel("MORT (Mortality rate)", fontsize=11)
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()


