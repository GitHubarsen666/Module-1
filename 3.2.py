import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# === 1. Тестові дані ===
data = {
    'ліпопротеїни': [150, 200, 180, 220, 170, 190, 210, 160],
    'гемоглобін': [13.5, 15.2, 14.8, 15.5, 14.0, 14.6, 15.3, 13.8]
}


df = pd.DataFrame(data)


# === 2. Обчислення коефіцієнта кореляції Пірсона ===
correlation_matrix = df.corr()
correlation_value = correlation_matrix.loc['ліпопротеїни', 'гемоглобін']
print(f"Коефіцієнт кореляції Пірсона: {correlation_value:.2f}")


# === 3. Візуалізація кореляції ===
sns.scatterplot(data=df, x='ліпопротеїни', y='гемоглобін')
plt.title('Кореляція ліпопротеїни vs гемоглобін')
plt.xlabel('Ліпопротеїни')
plt.ylabel('Гемоглобін')
plt.show()
