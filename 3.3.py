import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data = {
    'глюкоза': [80, 90, 100, 110, 120, 130, 140, 150],
    'гемоглобін': [13.5, 14.0, 14.6, 15.0, 15.2, 15.5, 15.7, 16.0]
}
df = pd.DataFrame(data)
x = df['глюкоза'].values
y = df['гемоглобін'].values


# Поліном 2-го порядку
def poly_model(x, a, b, c):
    return a*x**2 + b*x + c


params, covariance = curve_fit(poly_model, x, y)
a, b, c = params
print(f"Оцінені параметри: a={a:.6f}, b={b:.6f}, c={c:.6f}")


# Візуалізація
x_fit = np.linspace(min(x), max(x), 100)
y_fit = poly_model(x_fit, a, b, c)


plt.scatter(x, y, label='Дані')
plt.plot(x_fit, y_fit, 'r-', label='Поліноміальна модель')
plt.xlabel('Стабілізована глюкоза')
plt.ylabel('Гемоглобін')
plt.title('Нелінійна залежність гемоглобін – глюкоза')
plt.legend()
plt.show()


# Оцінка якості
residuals = y - poly_model(x, a, b, c)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"Коефіцієнт детермінації R^2 = {r_squared:.3f}")
