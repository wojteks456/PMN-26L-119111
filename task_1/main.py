import pandas
import matplotlib.pyplot as plt  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.manifold import TSNE

#Wczytanie zbioru danych Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

#Tworzenie DataFrame dla lepszej analizy
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. Krótka analiza danych (wypisanie w konsoli)
print("--- Krótka analiza danych zbioru Iris ---")
print("Kształt zbioru danych:", df.shape)
print("\nPodstawowe statystyki:")
print(df.describe())
print("\nLiczebność poszczególnych klas:")
print(df['target'].value_counts().rename(index=dict(enumerate(target_names))))
print("-" * 40)

# 3. Podział na zbiór treningowy (70%) i testowy (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Klasyfikacja przy użyciu algorytmu KNN (Grupa 1)
# K = 5 to standardowa, często optymalna wartość startowa
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
y_pred = knn.predict(X_test)

# 5. Obliczenie i wyświetlenie metryk
accuracy = accuracy_score(y_test, y_pred)
# Używamy average='weighted' ponieważ mamy więcej niż 2 klasy
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n--- Metryki Klasyfikacji KNN ---")
print(f"Accuracy (Dokładność): {accuracy:.4f}")
print(f"Precision (Precyzja):  {precision:.4f}")
print(f"Recall (Czułość):      {recall:.4f}")
print(f"F1-score:              {f1:.4f}")

# 6. Wizualizacja T-SNE po klasyfikacji (na zbiorze testowym)
# T-SNE redukuje 4 wymiary do 2 wymiarów, aby można było je narysować na płaskim wykresie
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
X_test_tsne = tsne.fit_transform(X_test)

plt.figure(figsize=(8, 6))
# Rysujemy punkty, kolorując je według przewidzianych klas (y_pred)
scatter = plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=100)
plt.title('Wizualizacja t-SNE po klasyfikacji KNN (Zbiór Testowy)')
plt.xlabel('Wymiar t-SNE 1')
plt.ylabel('Wymiar t-SNE 2')

# Dodanie legendy odpowiadającej nazwom kwiatów
handles, _ = scatter.legend_elements()
plt.legend(handles, target_names, title="Przewidziana klasa")

# Zapis do pliku i wyświetlenie
plt.savefig('tsne_plot.png')
plt.show()