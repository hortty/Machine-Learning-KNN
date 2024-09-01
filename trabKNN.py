from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

class KNN:

    def __init__(self, neighbours = 1):

        # Verificar neighbours
        self.k = neighbours
        self.data = []
        self.labels = []

    def fit(self, train_data, train_labels):
        self.data = train_data
        self.labels = train_labels

    def predict(self, test_data):

        # Validação se treino tem mesma dimensão que teste
        if self.data.shape[1] == test_data.shape[1]:
          predict_list = []
          ids = []
          dimensao = self.data.shape[1]
        else:
          print("test_data não possui mesma dimensão que train_data!!")

        for new_points in test_data:
            distance_list = []
            for id, old_points in enumerate(self.data):

              # Calculo da distancia
              soma = 0
              for dim in range(dimensao):

                soma += (old_points[dim] - new_points[dim])**2
                # distance = ((old_points[0] - new_points[0])**2 + (old_points[1] - new_points[1])**2)**(1/2)

              distance = soma**(1/2)
              # ---------------------

              distance_list.append({
                    "id": id,
                    "distance": distance
                })

            distance_list.sort(key=lambda x: x["distance"])

            # faz slice com k elementos, depois pega o conjunto de elementos e extrai somente o campo id
            ids = [item['id'] for item in distance_list[:self.k]]

            labels_subset = [self.labels[i] for i in ids]

            predict_list.append(statistics.mode(labels_subset))
            distance_list = []

        return predict_list

# CARREGANDO DATASET IRIS
print("STARTING IRIS TEST")
dataset = load_iris()
data = dataset.data
labels = dataset.target
label_names = dataset.target_names

# PEGANDO NUM CLASSES
n_classes = len(np.unique(labels))

# VIEW DATASET
colours = plt.cm.rainbow(np.linspace(0, 1, n_classes))
fig, ax = plt.subplots()
for n_class in range(n_classes):
    ax.scatter(data[labels == n_class, 0], data[labels == n_class, 1],
               c=[colours[n_class]], s=10, label=str(n_class))

ax.legend(loc='upper right')
plt.show()

# Holdout (Train-Val | Test)
random_seed = 1525
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.7, test_size=0.3, random_state=random_seed)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, train_size=0.7, test_size=0.3, random_state=random_seed)

# KNN implementado
qtde_neighbours = 13
print(f"usando {qtde_neighbours} k-vizinhos")
my_knn = KNN(qtde_neighbours)
inicio = time.time()
my_knn.fit(train_data=train_data, train_labels=train_labels)
my_predicts = my_knn.predict(test_data)
fim = time.time()
print(f"Tempo de execução do KNN implementado: {(fim - inicio) * 1000:.3f} ms")

# KNN SKLEAN
knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=qtde_neighbours)
inicio = time.time()
knn.fit(train_data, train_labels)
predicts = knn.predict(test_data)
fim = time.time()
print(f"Tempo de execução do KNN scikit: {(fim - inicio) * 1000:.3f} ms")

# ACCURACY
acc = accuracy_score(test_labels, my_predicts)
print(f'Accuracy (KNN implementado): {acc}')
acc2 = accuracy_score(test_labels, predicts)
print(f'Accuracy (KNN scikit): {acc2}')

# Relatório de classificação
print("\nRelatório de classificação (sklearn):")
print(classification_report(test_labels, predicts, target_names=label_names))
print("\nRelatório de classificação (personalizado):")
print(classification_report(test_labels, my_predicts, target_names=label_names))

# Matriz de Confusão
cm = confusion_matrix(test_labels, my_predicts, labels=np.unique(labels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax)
ax.set_title('KNN implementado')
plt.show()

cm2 = confusion_matrix(test_labels, predicts, labels=np.unique(labels))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=label_names)
fig2, ax2 = plt.subplots(figsize=(8, 6))
disp2.plot(ax=ax2)
ax2.set_title('KNN Scikit-Learn')
plt.show()

# CARREGANDO DATASET WINE
print("STARTING WINE TEST")
dataset = load_wine()
data = dataset.data
labels = dataset.target
label_names = dataset.target_names

# PEGANDO NUM CLASSES
n_classes = len(np.unique(labels))

# VIEW DATASET
colours = plt.cm.rainbow(np.linspace(0, 1, n_classes))
fig, ax = plt.subplots()
for n_class in range(n_classes):
    ax.scatter(data[labels == n_class, 0], data[labels == n_class, 1],
               c=[colours[n_class]], s=10, label=str(n_class))

ax.legend(loc='upper right')
plt.show()

# Holdout (Train-Val | Test)
random_seed = 6786
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.7, test_size=0.3, random_state=random_seed)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, train_size=0.7, test_size=0.3, random_state=random_seed)

# KNN implementado
qtde_neighbours = 21
print(f"usando {qtde_neighbours} k-vizinhos")
my_knn = KNN(qtde_neighbours)
inicio = time.time()
my_knn.fit(train_data=train_data, train_labels=train_labels)
my_predicts = my_knn.predict(test_data)
fim = time.time()
print(f"Tempo de execução do KNN implementado: {(fim - inicio) * 1000:.3f} ms")

# KNN SKLEAN
knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=qtde_neighbours)
inicio = time.time()
knn.fit(train_data, train_labels)
predicts = knn.predict(test_data)
fim = time.time()
print(f"Tempo de execução do KNN scikit: {(fim - inicio) * 1000:.3f} ms")

# ACCURACY
acc = accuracy_score(test_labels, my_predicts)
print(f'Accuracy (KNN implementado): {acc}')
acc2 = accuracy_score(test_labels, predicts)
print(f'Accuracy (KNN scikit): {acc2}')

# Relatório de classificação
print("\nRelatório de classificação (sklearn):")
print(classification_report(test_labels, predicts, target_names=label_names))
print("\nRelatório de classificação (personalizado):")
print(classification_report(test_labels, my_predicts, target_names=label_names))

# Matriz de Confusão
cm = confusion_matrix(test_labels, my_predicts, labels=np.unique(labels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax)
ax.set_title('KNN implementado')
plt.show()

cm2 = confusion_matrix(test_labels, predicts, labels=np.unique(labels))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=label_names)
fig2, ax2 = plt.subplots(figsize=(8, 6))
disp2.plot(ax=ax2)
ax2.set_title('KNN scikit')
plt.show()

# WINE COM KVIZINHOS DIFERENTE
# KNN implementado
qtde_neighbours = 8
print(f"STARTING WINE COM {qtde_neighbours} k-vizinhos")
my_knn = KNN(qtde_neighbours)
inicio = time.time()
my_knn.fit(train_data=train_data, train_labels=train_labels)
my_predicts = my_knn.predict(test_data)
fim = time.time()
print(f"Tempo de execução do KNN implementado: {(fim - inicio) * 1000:.3f} ms")

# KNN SKLEAN
knn = KNeighborsClassifier(metric='minkowski', p=2, n_neighbors=qtde_neighbours)
inicio = time.time()
knn.fit(train_data, train_labels)
predicts = knn.predict(test_data)
fim = time.time()
print(f"Tempo de execução do KNN scikit: {(fim - inicio) * 1000:.3f} ms")

# ACCURACY
acc = accuracy_score(test_labels, my_predicts)
print(f'Accuracy (KNN implementado): {acc}')
acc2 = accuracy_score(test_labels, predicts)
print(f'Accuracy (KNN scikit): {acc2}')

# Relatório de classificação
print("\nRelatório de classificação (sklearn):")
print(classification_report(test_labels, predicts, target_names=label_names))
print("\nRelatório de classificação (personalizado):")
print(classification_report(test_labels, my_predicts, target_names=label_names))

# Matriz de Confusão
cm = confusion_matrix(test_labels, my_predicts, labels=np.unique(labels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax)
ax.set_title('KNN implementado')
plt.show()

cm2 = confusion_matrix(test_labels, predicts, labels=np.unique(labels))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=label_names)
fig2, ax2 = plt.subplots(figsize=(8, 6))
disp2.plot(ax=ax2)
ax2.set_title('KNN scikit')
plt.show()
