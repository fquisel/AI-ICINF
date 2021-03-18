i = 1
resultado = []
import operator as o
import math as m
import numpy as n
from index import Folds,dset,clase

k = int(input("Ingrese el valor para k: " ))
for train, test in Folds(dset):
    dset_entrenamiento = dset[train]
    dset_prueba = dset[test]
    clase_entrenamiento = clase[train]
    clase_prueba = clase[test]
    i += 1     
    
    
    '''-------------------KNN---------------------------------'''
    
    #Distancia Euclidiana entre Prueba y Entrenamiento
    def distancia_euclidiana(a, b):
    	distancia= 0
    	for x in range(1,len(a)):
    		distancia += pow(a[x] - b[x], 2)
    	return m.sqrt(distancia)
    
    
    #K vecinos más cercanos calculando la distancia entre el punto de prueba y el conjunto de prueba
    #Ordena las distancias
    def vecinos_cercanos(entrenamiento, p_prueba, k):
    	distancias = []
    	for x in range(len(entrenamiento)):
    		dist = distancia_euclidiana((p_prueba), entrenamiento[x])
    		distancias.append((entrenamiento[x], dist))
    	distancias.sort(key = o.itemgetter(1))
    	vecinos = []
    	for x in range(k):
    		vecinos.append(distancias[x][0])
    	return vecinos
    
    #Clase más común entre los k vecinos
    def clase_comun(vecinos):
    	listas = {}
    	for x in range(len(vecinos)):
    		lista = vecinos[x][0]
    		if lista in listas:
    			listas[lista] += 1;
    		else:
    			listas[lista] = 1;
    	lista_ordenada = sorted(listas.items(), key = lambda kv: kv[1], reverse = True)
    	return lista_ordenada[0][0]
    
    #Porcentaje de precisión
    def precision(prueba, prediccion):
    	acierto = 0
    	for x in range(len(prueba)):
    		if prueba[x][0] == prediccion[x]:
    			acierto += 1
    	return (acierto / float(len(test))) * 100
     
    
    #Recorre funciones definidas, entrega de Porcentaje de Acierto
    prediccion = []
    for x in range(len(dset_prueba)):
     	vecinos = vecinos_cercanos(dset_entrenamiento, dset_prueba[x], k)
     	aux = clase_comun(vecinos)
     	prediccion.append(aux) 
    print("FOLD",i-1,"- Porcentaje de Acierto: ", precision(dset_prueba, prediccion))   
    resultado.append(precision(dset_prueba,prediccion)) 

         
print("Media: ",n.mean(resultado))
print("Desviación Estándar: ",n.std(resultado))


