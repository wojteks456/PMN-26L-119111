import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

def load_cleveland_data(url):
    """
    Wczytywanie zbiorów danych Heart Disease (Cleveland), nadaje nazwy kolumnom
    i zamienia braki danych oznaczane jako '?' na np.nan.
    """
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(url, names=columns)
    
    # Zamiana '?' na NaN i konwersja na typy numeryczne
    df.replace('?', np.nan, inplace=True)
    df['ca'] = pd.to_numeric(df['ca'])
    df['thal'] = pd.to_numeric(df['thal'])
    
    # Zmiana problemu na klasyfikację binarną (0 - zdrowy, 1 - chory)
    # W oryginalnym zbiorze wartości 1, 2, 3, 4 oznaczają różne stadia choroby
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def plot_target_distribution(df, target_col='target'):
    """
    Rysuje wykres słupkowy rozkładu klas.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, palette='Set2')
    plt.title("Rozkład klas decyzyjnych (0 = Zdrowy, 1 = Chory)")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba pacjentów")
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Rysuje macierz pomyłek oraz krzywą ROC dla podanego modelu.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Macierz pomyłek
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Zdrowy", "Chory"]).plot(ax=ax[0], cmap='Blues')
    ax[0].set_title("Macierz pomyłek")
    
    # Krzywa ROC
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1])
    ax[1].set_title("Krzywa ROC")
    ax[1].plot([0, 1], [0, 1], color='red', linestyle='--') # linia losowego klasyfikatora
    
    plt.tight_layout()
    plt.show()

