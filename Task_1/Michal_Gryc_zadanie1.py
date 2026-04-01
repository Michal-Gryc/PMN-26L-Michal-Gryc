import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE

# Tworzenie dataframe
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('iris.data', names=columns)


X = data.drop('class', axis=1)
y = data['class']


# Dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=79)


# Szkolenie modelu
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Wyświetlanie raportu
print("--- METRYKI KLASYFIKACJI ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


# Tworzenie wykresu 
tsne = TSNE(n_components=2, random_state=79)
X_tsne = tsne.fit_transform(X)


plt.figure(figsize=(10, 7))
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}

for species in colors:
    indices = data['class'] == species
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=colors[species], label=species)

plt.title('Wizualizacja zbioru Iris przy użyciu t-SNE (2D)')
plt.legend()
plt.show()