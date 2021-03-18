import numpy as np
import math as m
import pandas as p

dset = p.read_csv('copy_diabetes.csv', header = None)
clase = p.read_csv('clase_copy_diabetes.csv', header = None)
X = dset.iloc[:,:].values
y = clase.iloc[:,:].values
clusters = int(input("Ingrese un valor para k: "))
K = clusters
iteraciones = 100
n_ejemplos = len(X)
n_columnas = 8

# lista de índices de muestra para cada cluster
t_clusters = [[] for _ in range(K)]
# Total de centroides (vector de característica media) para cada cluster
t_centroides = []
#centroides convergentes
b_centroides = []


#Media = np.average(b_centroides)
#Desviacion = np.std(b_centroides)

# Calcula la distancia entre 2 puntos
def distancia_euclidiana(a, b):
    distancia = 0
    for x in range(1,len(a)):
    	distancia += pow(a[x] - b[x], 2)
    return m.sqrt(distancia)

# Se asignan las muestras a los centroides más cercanos
# Se calculan nuevos centroides a partir de los clústeres
# Se comprueba si los grupos han cambiado
def predictor(X):
    indices_aleatorios = np.random.choice(n_ejemplos, K, replace = False)
    centroides = [X[indice] for indice in indices_aleatorios]
    for _ in range(iteraciones):
        clusters = crear_clusters(centroides)   
        t_clusters.append(clusters)
        centroides_v = centroides
        centroides = centroide(clusters)
        t_centroides.append(centroides)
        #print(centroides)    
        if convergencia(centroides_v, centroides):
            b_centroides.append(centroides)
            print("Convergencia Encontrada")
            break
    return etiquetas_cluster(clusters)

# Se asigna a cada muestra la etiqueta del grupo al que se le asignó    
def etiquetas_cluster(clusters):
    etiquetas = np.empty(n_ejemplos)
    for indice_cluster, cluster in enumerate(clusters):
        for indice_registro in cluster:
            etiquetas[indice_registro] = indice_cluster
    return etiquetas

# Asigna las muestras a los centroides más cercanos para crear clusters
def crear_clusters(centroides):
    clusters = [[] for _ in range(K)]
    for indice, registro in enumerate(X):
        indice_centroide = centroide_cercano(registro, centroides)
        clusters[indice_centroide].append(indice)
    return clusters
    
# Obtiene la distancia de la muestra actual a uno de los centroides 
def centroide_cercano(registro, centroides):        
    distancias = [distancia_euclidiana(registro, puntero) for puntero in centroides]
    indice_cercano = np.argmin(distancias)
    return indice_cercano
    
# Asigna el promedio de los grupos a los centroides
def centroide(clusters):  
    centroides = np.zeros((K, n_columnas))
    for indice_cluster, cluster in enumerate(clusters):
        media_cluster = np.mean(X[cluster], axis=0)
        centroides[indice_cluster] = media_cluster
    return centroides
    
# Obtiene la variacion entre cada centroide antiguo y nuevo de todos los centroides
def convergencia(centroides_v, centroides):
    distancias = [distancia_euclidiana(centroides_v[i], centroides[i]) for i in range(K)]
    variacion = sum(distancias)
    print(variacion)
    return variacion == 0


y_pred = predictor(X)
print("-----------------PATRONES----------------- ")
print(t_centroides)
print("-----------------PATRONES CONVERGENTES----------------- ")
print(b_centroides)
    
