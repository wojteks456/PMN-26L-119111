import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode

# 1. Wczytanie danych
iris = load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names

# Krótka analiza danych (wyświetlana w konsoli)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y_true
print("--- Podstawowe statystyki zbioru Iris ---")
print(df.describe())
print("\nBrakujące wartości:\n", df.isnull().sum())

# 2. Grupowanie K-Means
# Znamy liczbę gatunków (3), więc n_clusters=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# 3. Mapowanie klastrów na prawdziwe etykiety
# K-Means nadaje losowe numery klastrom. Aby policzyć metryki (accuracy, f1), 
# musimy przypisać każdemu klastrowi najczęściej występującą w nim prawdziwą etykietę.
mapped_labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    # Przypisanie najczęstszej prawdziwej klasy w danym klastrze
    mapped_labels[mask] = mode(y_true[mask], keepdims=True)[0][0]

# 4. Obliczenie metryk
accuracy = accuracy_score(y_true, mapped_labels)
precision = precision_score(y_true, mapped_labels, average='weighted')
recall = recall_score(y_true, mapped_labels, average='weighted')
f1 = f1_score(y_true, mapped_labels, average='weighted')

print("\n--- Wyniki metryk po zmapowaniu klastrów ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 5. Redukcja wymiarów T-SNE i Wizualizacja
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
# Rysujemy wykres na podstawie wyznaczonych przez nas klastrów K-Means
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=mapped_labels, cmap='viridis', edgecolors='k', s=60)
plt.title('Wizualizacja grupowania K-Means (zbiór Iris) przy użyciu T-SNE')
plt.xlabel('T-SNE Cecha 1')
plt.ylabel('T-SNE Cecha 2')

# Dodanie legendy
handles, _ = scatter.legend_elements()
plt.legend(handles, iris.target_names, title="Zmapowane Klastry")

plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('tsne_plot.png')
print("\nWykres został zapisany jako 'tsne_plot.png'.")
plt.show()
