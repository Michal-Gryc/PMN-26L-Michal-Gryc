import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Zmiana pikseli obrazow z zakresu 0-255 do 0-1 formatu HWC na CHW (Channels, Height, Width)
transform = transforms.ToTensor()  

# Wczytanie danych z biblioteki Pytorcha - listy z pikselami i numerami klass
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
classes = train_set.classes

# Analiza bazy danych
def analyze_dataset(dataset):
    print("--- Statystyki zbioru ---")
    # 1. Wymiary i zakres
    img, label = dataset[0]
    print(f"Wymiary pojedynczego obrazu: {img.shape}")
    print(f"Minimalna wartość piksela: {img.min():.4f}")
    print(f"Maksymalna wartość piksela: {img.max():.4f}")
    
    # 2. Rozkład klas
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    for i, count in enumerate(class_counts):
        print(f"Klasa {classes[i]}: {count} obrazów")

analyze_dataset(train_set)

# Przykładowy obraz
image, label = train_set[0]
plt.imshow(image.permute(1, 2, 0))
plt.show()


# Definicja klasy MLP - Multilayer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size = 32*32*3, hidden1=128, output_size=10, dropout_p=0.00):
        super(MLP, self).__init__()    
        self.fc1 = nn.Linear(input_size, hidden1) # pierwsza warstwa ukryta
        self.fc2 = nn.Linear(hidden1, hidden1//2)         # druga wartsta ukryta
        self.fc3 = nn.Linear(hidden1//2, output_size) # wyjście
        self.dropout = nn.Dropout(p=dropout_p) # Dodany dropout

    def forward(self, X):
        X = X.view(X.size(0), -1) # Spłaszczanie obrazu do 1D
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        X = self.fc3(X)

        return F.log_softmax(X, dim=1) # dim=1 oznacza, że prawdopodobieństwa mają sumować się do 1


# Testowanie napisanego modelu na losowo wygranych danych
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    # Przełączenie modelu w tryb uczenia
    model.train() 

    # Słownik do przechowywania historii wyników dla wykresów
    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Pętla po paczkach danych (batches)
        for images, labels in train_loader:
            
            optimizer.zero_grad() # 1. Zerowanie gradientów optymalizatora
            
            outputs = model(images) # 2. Forward pass: obliczenie przewidywań modelu
            
            loss = criterion(outputs, labels) # 3. Obliczenie funkcji straty (jak bardzo model się pomylił)
            
            loss.backward()  # 4. Backward pass: obliczenie gradientów (wsteczna propagacja błędu)
            
            optimizer.step() # 5. Aktualizacja modelu
            
            running_loss += loss.item() # 6. Liczenie błędów
            
            # Klasa z najwyższym prawdopodobieństwem
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Średnia strata i dokładność dla całej epoki
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Zapis do historii
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return history


# ----------------------------------------------Testowanie modelu - Seria 1---------------------------------------------
def series_1():
    results_lr = {} # Wyniki z każdego testu
    learning_rates = [0.01, 0.001, 0.0001]  # Lista wartości "Learning Rate" do przetestowania
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Funkcja straty
    criterion = nn.CrossEntropyLoss()
    # Testowanie kolejnych wartość learning_rates
    for i in learning_rates:
        print(f"\n--- Eksperyment dla LR = {i} ---")
        new_model = MLP() 
        optimizer = torch.optim.Adam(new_model.parameters(), lr=i)
        
        history = train_model(new_model, train_loader, criterion, optimizer, epochs=5)
        
        results_lr[i] = history

    # Tworzenie okna dla wykresu
    plt.figure(figsize=(10, 5))

    # Pętla z wyników
    for lr, hist in results_lr.items():
        # Rysowanie linii dla konkretnych wyników.
        plt.plot(hist['acc'], label=f'LR = {lr}')

    plt.title('Wpływ Learning Rate na Accuracy w kolejnych epokach')
    plt.xlabel('Epoka (od 0 do 4)')
    plt.ylabel('Dokładność (Accuracy %)')
    plt.legend()
    plt.show()

# ----------------------------------------------Testowanie modelu - Seria 2---------------------------------------------
def series_2():
    results_batch = {}
    batch_sizes = [16, 32, 64, 128]  # Lista wartości "Batch Sizes" do przetestowania   

    # Funkcja straty
    criterion = nn.CrossEntropyLoss()

    # Testowanie kolejnych wartość Batch Sizes
    for i in batch_sizes:
        print(f"\n--- Eksperyment dla Batch Size = {i} ---")
        current_train_loader = DataLoader(train_set, batch_size=i, shuffle=True)
        new_model = MLP() 
        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)
        
        history = train_model(new_model, current_train_loader, criterion, optimizer, epochs=5)
        
        results_batch[i] = history

    # Tworzenie okna dla wykresu
    plt.figure(figsize=(12, 6))

    # Pętla z wyników
    for bs, hist in results_batch.items():
        # Rysowanie linii dla konkretnych wyników.
        plt.plot(hist['acc'], label=f'Batch Size = {bs}')

    plt.title('Wpływ Batch Size na Accuracy (CIFAR-10)')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------------------------Testowanie modelu - Seria 3---------------------------------------------
def series_3():
    results_dropout = {}
    dropout_values = [0.0, 0.3, 0.5]  #  Lista wartości "Dropout" do przetestowania
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Funkcja straty
    criterion = nn.CrossEntropyLoss()

    # Testowanie kolejnych wartość Dropoutu
    for i in dropout_values:
        print(f"\n--- Ekseryment dla Dropout = {i} ---")
        new_model = MLP(dropout_p=i)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)
        
        history = train_model(new_model, train_loader, criterion, optimizer, epochs=10)
        
        results_dropout[i] = history

    # Tworzenie okna dla wykresu
    plt.figure(figsize=(12, 6))

    # Pętla z wyników
    for drp, hist in results_dropout.items():
        # Rysowanie linii dokładności dla każdej wartości dropoutu
        plt.plot(hist['acc'], label=f'Dropout = {drp}')

    plt.title('Wpływ Dropout na Accuracy (CIFAR-10)')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------------------------Testowanie modelu - Seria 4---------------------------------------------
def series_4():
    results_neurons = {}
    neuron_configs = [64, 128, 256]  # Lista wartości "liczby neuronów " do przetestowania
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    for n in neuron_configs:
        print(f"\n--- Eksperyment dla liczby neuronów = {n} ---")
        new_model = MLP(hidden1=n)
        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)
        
        history = train_model(new_model, train_loader, criterion, optimizer, epochs=10)
        
        results_neurons[n] = history

     # Tworzenie okna dla wykresu
    plt.figure(figsize=(12, 6))

    # Pętla z wyników
    for n, hist in results_neurons.items():
        # Rysowanie linii dokładności dla każdej wartości liczby neuronów
        plt.plot(hist['acc'], label=f'{n} neuronów')

    plt.title('Wpływ liczby neuronów na Accuracy (CIFAR-10)')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Wywołanie eksperymentów
series_1()
series_2()
series_3()
series_4()

def final_model():
    final_model = MLP(dropout_p=0.0)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.0001)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    print("Trenowanie finałowego modelu do metryk...")
    train_model(final_model, train_loader, criterion, optimizer, epochs=15)
    return final_model

def run_final_metrics(model, test_loader, classes):
    # Ustawienie modelu w tryb testowy 
    model.eval()
    
    y_true = []
    y_pred = []
    
    # 2. Przewidywania modelu dla całego zbioru testowego
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    
    # Raport klasyfikacji - tabela z precyzją, czułością i F1
    print("SZCZEGÓŁOWY RAPORT KLASYFIKACJI")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Macierz pomyłek - mapa ciepła
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Macierz pomyłek (Confusion Matrix)')
    plt.ylabel('Klasa prawdziwa')
    plt.xlabel('Klasa przewidziana przez model')
    plt.show()

run_final_metrics(final_model(), test_loader, classes)