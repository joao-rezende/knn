import os
import pandas as pd
import numpy as np
import operator
# import matplotlib.pyplot as plt

def euclideanDistance(data_1, data_2, data_len):
    dist = 0
    for i in range(data_len):
        dist = dist + np.square(data_1[i] - data_2[i])
    return np.sqrt(dist)

def knn(dataset, testInstance, k): 
    distances = {}
    length = testInstance.shape[1]

    for x in range(len(dataset)):
        dist_up = euclideanDistance(testInstance, dataset.iloc[x], length)
        distances[x] = dist_up[0]

    sort_distances = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    # Extracting nearest k neighbors
    for x in range(k):
        neighbors.append(sort_distances[x][0])
    # Initializing counts for 'class' labels counts as 0
    counts = {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}
    # Computing the most frequent class
    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1] 
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
    # Sorting the class in reverse order to get the most frequest class
    sort_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return(sort_counts[0][0])

data = pd.read_csv(os.path.join("C:\\Users", "Joao", "Documents", "Python Codes", "knn", "iris.csv"), names=['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie'])

indexes = np.random.permutation(data.shape[0])
div = int(0.75 * len(indexes))
development_id, test_id = indexes[:div], indexes[div:]

development_set, test_set = data.loc[development_id,:], data.loc[test_id,:]
# print("Development Set:\n", development_set, "\n\nTest Set:\n", test_set)

test_class = list(test_set.iloc[:,-1])
dev_class = list(development_set.iloc[:,-1])

# Criando uma lista com todas as colunas da lista original com exceção da espécie
row_list = []
for index, rows in development_set.iterrows():
    my_list =[rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]       
    row_list.append([my_list])
    
# Diferentes K para testar
# k_n = [1, 3, 5, 7]
k_n = [1, 3]
development_set_obs_k = {}
for k in k_n:
    development_set_obs = []
    for i in range(len(row_list)):
        development_set_obs.append(knn(development_set, pd.DataFrame(row_list[i]), k))
    development_set_obs_k[k] = development_set_obs
# Dicionário contendo a classe para cada k
obs_k = development_set_obs_k
# print(obs_k)

# Calculando a taxa de reconhecimento 
rate = {}
for k_value in obs_k.keys():
    #print('k = ', key)
    count = 0
    for i,j in zip(dev_class, obs_k[k_value]):
        if i == j:
            count = count + 1
        else:
            pass
    rate[k_value] = count/(len(dev_class))

df_res = pd.DataFrame({'k': k_n})
value = list(rate.values())
df_res = value
# print(df_res)

# Creating a list of list of all columns except 'class' by iterating through the development set
row_list_test = []
for index, rows in test_set.iterrows(): 
    my_list =[rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]       
    row_list_test.append([my_list])
test_set_obs = []
for i in range(len(row_list_test)):
    test_set_obs.append(knn(test_set, pd.DataFrame(row_list_test[i]), 3)) # No lugar de 3, deve ser o melhor K
#print(test_set_obs)

count = 0
for i,j in zip(test_class, test_set_obs):
    if i == j:
        count = count + 1
    else:
        pass
accuracy_test = count/(len(test_class))
print('Final Accuracy of the Test dataset is ', accuracy_test)