import pandas as p
#from Index import (dset_copy,clase)

#Se importa csv Normalizado (con enzabezado)
dset_copy = p.read_csv('copy_diabetes.csv')
dset_copy.head()

#Desordenar Indices con sus respectivos registros
ds = dset_copy.sample(916, random_state = 1)

#Creacion de Folds
fold = [ds.iloc[0:183], ds.iloc[183:366], ds.iloc[366:549], ds.iloc[549:732],ds.iloc[732:916]]

#Creación de registros de prueba y entrenamiento por iteraciones
entrenamiento = []
prueba = []
kfold = {'entrenamiento': entrenamiento, 'prueba': prueba}
for i, pruebai in enumerate(fold):
    entrenamiento.append(fold[:i] + fold[i+1:])
    prueba.append(pruebai)


#Imprimir Índices para verificación por iteración
for k in range(len(entrenamiento)):
    new = entrenamiento[k]
    print("")
    print("--------------------- ITERACION ", k+1," ----------------------")
    test = prueba[k]
    print("PRUEBA: ",test.index)
    for i in range(4):
        train = new[i]
        print("ENTRENAMIENTO: ",train.index)