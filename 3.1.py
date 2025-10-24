# Побудова дерева рішень для моніторингу активності користувачів бібліотеки
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# === 1. Тестові дані ===
# Приклад: користувачі, їх активність, тип запитів, час активності
data = {
    'кількість_запитів': [5, 2, 15, 7, 1, 10, 8, 3],
    'тип_документів': ['статті', 'книги', 'книги', 'статті', 'статті', 'книги', 'статті', 'книги'],
    'час_активності': ['ранок', 'вечір', 'ранок', 'день', 'вечір', 'день', 'вечір', 'ранок'],
    'ціль': ['низька', 'низька', 'висока', 'середня', 'низька', 'висока', 'середня', 'низька']
}


df = pd.DataFrame(data)


# === 2. Перетворення категоріальних ознак ===
df_encoded = pd.get_dummies(df, columns=['тип_документів', 'час_активності'])


# === 3. Вибір ознак та цільової змінної ===
X = df_encoded.drop('ціль', axis=1)
y = df_encoded['ціль']


# === 4. Побудова дерева рішень ===
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X, y)


# === 5. Візуалізація дерева ===
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
plt.show()


# === 6. Важливість ознак ===
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Важливість ознак:")
print(feature_importance)
