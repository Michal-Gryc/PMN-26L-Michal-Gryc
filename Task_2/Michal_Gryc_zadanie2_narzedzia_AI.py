import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

# -----------------------------------------------1. WCZYTANIE PLIKU I DATA FRAME----------------------------------------------
# Definiowanie nazw kolumn 
column_names = [
    "age", "sex", "cp", "trestbps", "chol", 
    "fbs", "restecg", "thalach", "exang", 
    "oldpeak", "slope", "ca", "thal", "target"
]

# Wczytanie pliku ,parametr na_values='?'- zamiana na puste wartości NaN, z którymi sklearn umie pracować.
df = pd.read_csv('processed.cleveland.data', header=None, names=column_names, na_values='?')


# ---------------------------------------------2. ANALIZA SUROWYCH DANYCH Z PLIKU---------------------------------------------
def analyze_data(df):
    # 1. Liczebność i typy danych
    print("\n=== STATYSTYKI OPISOWE ===")
    print(df.describe())

    # 2. Analiza brakujących danych
    missing = df.isnull().sum()
    print("\n=== BRAKUJĄCE WARTOŚCI ===")
    print(missing[missing > 0])

# Ustawienie stylu wykresów
sns.set_theme(style="whitegrid")

def plot_categorical_distributions(df):
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    plt.figure(figsize=(18, 12)) 
    
    # kopia do wykresów
    df_plot = df.copy()
    
    for i, col in enumerate(categorical_features, 1):
        ax = plt.subplot(3, 3, i)
        
        # Zastępstwo pustej wartości napisem "Brak"
        df_plot[col] = df_plot[col].fillna("No data")
        
        sns.countplot(x=col, data=df_plot, palette='Blues_r') # Personalizacja wykresów, paleta kolorów
        plt.title(f'Feature distribution: {col}')   # Tytuły wykresów
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresów na w jednym oknie

plot_categorical_distributions(df)

def plot_numerical_distributions(df):
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        # kde=True linia trendu (gęstości)
        sns.histplot(df[col], kde=True, color='skyblue', bins=20)
        plt.title(f'Histogram: {col}')
    
    plt.tight_layout()
    plt.show()

plot_numerical_distributions(df)

# -------------------------------------------3. PRZYGOTOWANIE DANYCH (PREPROCESSING)------------------------------------------
# Zastępienie NaN najczęstszą wartością 
df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

# Zamiana target z (0-4) na wartość binarną (0 - zdrowy, 1 - chory)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Dzielenie danych na cechy (X) i zmienną przewidywaną (y)
X = df.drop('target', axis=1)
y = df['target']

# Podział na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --------------------------------------------4. DRZEWO DECYZYJNE I WIZUALIZACJE---------------------------------------------
def run_experiments():
    # --- EKSPERYMENT: Wpływ głębokości (max_depth) na jakość ---
    depths = range(1, 15)
    train_acc = []
    test_acc = []

    for d in depths:
        # Tworzenie drzewa o zadanej głębokości
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train) # Uczymy model
        
        # Zapisanie dokładność (accuracy) dla treningu i testu
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

    # Wykres z eksperymentu
    plt.figure(figsize=(10, 5))
    plt.plot(depths, train_acc, label='Zbiór treningowy', marker='o')
    plt.plot(depths, test_acc, label='Zbiór testowy', marker='o')
    plt.xlabel('Maksymalna głębokość drzewa (max_depth)')
    plt.ylabel('Dokładność (Accuracy)')
    plt.title('Eksperyment: Wpływ głębokości drzewa na jakość modelu')
    plt.legend()
    plt.show()

    # --- KRZYWA ROC DLA OPTYMALNEGO MODELU ---
    # Dobierany max_depth=3 oraz random_state=42 - niestety te paremetry w głównej mierze mają wpływ na końcowy wynik
    best_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    best_clf.fit(X_train, y_train)

    # Prawdopodobieństwo, że dany pacjent jest chory (1)
    y_probs = best_clf.predict_proba(X_test)[:, 1]
    
    # Punkty do krzywej ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Wyświetlanie krzywej ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Odsetek fałszywych alarmów (False Positive Rate)')
    plt.ylabel('Czułość (True Positive Rate)')
    plt.title('Krzywa ROC dla Drzewa Decyzyjnego')
    plt.legend(loc="lower right")
    plt.show()

run_experiments()