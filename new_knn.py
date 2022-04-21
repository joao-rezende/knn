import os, operator
import pandas as pd
import numpy as np

def euclideanDistance(data, data_, data_len):
    dist = 0
    for i in range(data_len):
        dist = dist + np.square(data[i] - data_[i + 1])
    return np.sqrt(dist)

def knn(dataset, testset, k):
    distances = {}
    # Quantidade de dados para calcular a distância
    length = testset.shape[1]

    # Repetição com os dados do dataset (base)
    for i in range(len(dataset)):
        # Calculando a distância do dado de teste com a base
        distance = euclideanDistance(testset, dataset.iloc[i], length)
        distances[i] = distance[0]

    # Ordenando o array de distâncias
    sort_distances = sorted(distances.items(), key=operator.itemgetter(1))

    # Buscando K vizinhos mais próximos
    neighbors = []
    for i in range(k):
        neighbors.append(sort_distances[i][0])

    # Lista para contagem das espécies dos vizinhos mais próximos
    counts = {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}

    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1]
        counts[response] += 1

    #Ordenando a contagem de espécies 
    sort_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return sort_counts[0][0]

data = pd.read_csv(os.path.join("./", "iris.csv"), names=['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie'])

#Embaralhando os índices para dividir entre a base e os testes
indexes = np.random.permutation(data.shape[0])

#Definido um corte de 60%, ou seja, 60% do dados fornecidos serão de base e o resto de teste
cut = int(0.6 * len(indexes))
base_id, test_id = indexes[:cut], indexes[cut:]
baseset, test_data = data.loc[base_id,:], data.loc[test_id,:]

test_specie = list(test_data.iloc[:,-1])

# Criando uma lista apenas com os dados de teste
testset = []
for index, rows in test_data.iterrows():
    testset.append([[rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]])

k_values = [1, 3, 5, 7]
for k in k_values:
    counts = {
        "Iris-setosa" : {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}, 
        "Iris-versicolor" : {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}, 
        "Iris-virginica" : {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}
    }
    correct = 0
    for i in range(len(testset)):
        specie_result = knn(baseset, pd.DataFrame(testset[i]), k)
        counts[test_specie[i]][specie_result] += 1
        if (specie_result == test_specie[i]):
            correct += 1

    print("K = " + str(k) + " / Taxa de reconhecimento = " + str(correct * 100 / len(testset)))
    print("Matriz de confusão")
    print("                  Iris setosa | Iris versicolor | Iris virginica")
    print("Iris setosa     |          " + str(counts['Iris-setosa']['Iris-setosa']) + " |               " + str(counts['Iris-setosa']['Iris-versicolor']) + " |              " + str(counts['Iris-setosa']['Iris-virginica']) + "")
    print("Iris versicolor |          " + str(counts['Iris-versicolor']['Iris-setosa']) + " |               " + str(counts['Iris-versicolor']['Iris-versicolor']) + " |              " + str(counts['Iris-versicolor']['Iris-virginica']) + "")
    print("Iris virginica  |          " + str(counts['Iris-virginica']['Iris-setosa']) + " |               " + str(counts['Iris-virginica']['Iris-versicolor']) + " |              " + str(counts['Iris-virginica']['Iris-virginica']) + "")
    print("\n")
