import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# === 1. Тестові дані ===
data = {
    'глюкоза': [80, 90, 100, 110, 120, 130, 140, 150],
    'ліпопротеїни': [150, 200, 180, 220, 170, 190, 210, 160],
    'гемоглобін': [13.5, 14.0, 14.6, 15.0, 15.2, 15.5, 15.7, 16.0]
}
df = pd.DataFrame(data)


# === 2. Вибір ознак та цільової змінної ===
X = df[['глюкоза', 'ліпопротеїни']]
y = df['гемоглобін']


# === 3. Побудова регресійної моделі ===
model = LinearRegression()
model.fit(X, y)


# === 4. Коефіцієнти регресії ===
beta0 = model.intercept_
beta1, beta2 = model.coef_
print(f"Математичне рівняння моделі:")
print(f"Гемоглобін = {beta0:.3f} + {beta1:.3f}*Глюкоза + {beta2:.3f}*Ліпопротеїни")


# === 5. Оцінка моделі ===
r_squared = model.score(X, y)
print(f"Коефіцієнт детермінації R^2 = {r_squared:.3f}")


# === 6. Візуалізація впливу змінних ===
sns.pairplot(df, x_vars=['глюкоза','ліпопротеїни'], y_vars='гемоглобін', height=4, kind='reg')
plt.suptitle('Вплив змінних на гемоглобін', y=1.02)
plt.show()
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# === 1. Тестові дані ===
data = {
    'глюкоза': [80, 90, 100, 110, 120, 130, 140, 150],
    'ліпопротеїни': [150, 200, 180, 220, 170, 190, 210, 160],
    'гемоглобін': [13.5, 14.0, 14.6, 15.0, 15.2, 15.5, 15.7, 16.0]
}
df = pd.DataFrame(data)


# === 2. Вибір ознак та цільової змінної ===
X = df[['глюкоза', 'ліпопротеїни']]
y = df['гемоглобін']


# === 3. Побудова регресійної моделі ===
model = LinearRegression()
model.fit(X, y)


# === 4. Коефіцієнти регресії ===
beta0 = model.intercept_
beta1, beta2 = model.coef_
print(f"Математичне рівняння моделі:")
print(f"Гемоглобін = {beta0:.3f} + {beta1:.3f}*Глюкоза + {beta2:.3f}*Ліпопротеїни")


# === 5. Оцінка моделі ===
r_squared = model.score(X, y)
print(f"Коефіцієнт детермінації R^2 = {r_squared:.3f}")


# === 6. Візуалізація впливу змінних ===
sns.pairplot(df, x_vars=['глюкоза','ліпопротеїни'], y_vars='гемоглобін', height=4, kind='reg')
plt.suptitle('Вплив змінних на гемоглобін', y=1.02)
plt.show()
