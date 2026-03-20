# Sprawozdanie: Zadanie 1 - Uczenie Maszynowe
**Numer indeksu:** 119111
**Grupa:** 1

## Krótka analiza danych
Zbiór danych **Iris** składa się ze 150 próbek kwiatów kosaćca, podzielonych na 3 równe klasy (Setosa, Versicolor, Virginica) – po 50 sztuk dla każdej klasy. 
Każda próbka opisana jest za pomocą 4 cech liczbowych wyrażonych w centymetrach:
1. Długość działki kielicha (sepal length)
2. Szerokość działki kielicha (sepal width)
3. Długość płatka (petal length)
4. Szerokość płatka (petal width)

Zbiór jest idealnie zbalansowany i nie posiada brakujących wartości, co czyni go świetnym zbiorem referencyjnym do zadań klasyfikacji. Cechy płatków (petal) zazwyczaj wykazują najwyższą korelację z klasą docelową.

## Algorytm Klasyfikacji
Do klasyfikacji wykorzystano algorytm **KNN (K-Nearest Neighbors)** z liczbą sąsiadów ustawioną na `k=5`. Zbiór danych został wcześniej podzielony na zbiór treningowy (70%) oraz zbiór testowy (30%) przy użyciu stratyfikacji (aby zachować proporcje klas).

## Metryki ewaluacyjne
Wyniki klasyfikacji na zbiorze testowym prezentują się następująco:
* **Accuracy (Dokładność):** 1.0000 (100%)
* **Precision (Precyzja):** 1.0000 (100%)
* **Recall (Czułość):** 1.0000 (100%)
* **F1-score:** 1.0000 (100%)

Model poradził sobie perfekcyjnie, co jest typowe dla zbioru Iris przy użyciu algorytmu KNN, ponieważ klasy w tym zbiorze są od siebie bardzo dobrze odseparowane (szczególnie klasa Setosa względem dwóch pozostałych).

## Wizualizacja t-SNE
Algorytm t-SNE został wykorzystany do zredukowania 4-wymiarowej przestrzeni cech do 2 wymiarów, co pozwoliło na narysowanie punktów na wykresie 2D. Wykres (plik `tsne_plot.png`) przedstawia próbki ze zbioru testowego, pokolorowane zgodnie z przewidzianymi przez model klasami. Klastry dla poszczególnych gatunków są wyraźnie odseparowane.