import pandas as p
import numpy as n

dset = p.read_csv('copy_diabetes.csv')
clase = p.read_csv('clase_copy_diabetes.csv')
dset = dset.iloc[:,:].values
clase = clase.iloc[:,:].values

#Creación de Folds para Entrenamiento y Prueba
def Folds(dset, shuffle = True ):
    registros = dset.shape[0]
    entrada_i = n.arange(registros)
    if shuffle:
        random = n.random.RandomState(1)
        random.shuffle(entrada_i)
    for i in Particion(entrada_i, registros):
        entrenamiento = entrada_i[n.logical_not(i)]
        prueba = entrada_i[i]
        yield entrenamiento, prueba 

#Se definen los folds a utilizar por 5 iteraciones
def Particion(entrada_i, registros):
    particiones = (registros // 5) * n.ones(5, dtype = n.int)
    aux = 0
    for particion in particiones:
        i, f = aux, aux + particion
        entrada_i_lista = entrada_i[i:f]
        lista = n.zeros(registros, dtype = n.bool)
        lista[entrada_i_lista] = True
        yield lista
        aux = f 
        
        