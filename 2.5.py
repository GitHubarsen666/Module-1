# Аналіз запитів користувачів електронної бібліотеки
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter


# Приклад текстів (запити користувачів)
texts = [
    "штучний інтелект у навчанні",
    "оптимізація процесу навчання",
    "інтелектуальний аналіз даних",
    "системи підтримки прийняття рішень",
    "ефективність навчального процесу"
]


# 1. Розрахунок TF-IDF (ваг слів)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()


# Середнє значення ваги кожного слова
avg_tfidf = tfidf_matrix.mean(axis=0).A1
word_weights = dict(zip(feature_names, avg_tfidf))


print("Найінформативніші слова:")
for word, weight in sorted(word_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{word:30} -> {weight:.4f}")


# 2. Закон Ципфа (частота слів)
all_words = " ".join(texts).split()
counts = Counter(all_words)
ranks = range(1, len(counts) + 1)
frequencies = [freq for _, freq in counts.most_common()]


# 3. Візуалізація
plt.figure(figsize=(6, 4))
plt.loglog(ranks, frequencies, marker="o")
plt.title("Закон Ципфа для запитів користувачів бібліотеки")
plt.xlabel("Ранг слова")
plt.ylabel("Частота слова")
plt.grid(True)
plt.show()
