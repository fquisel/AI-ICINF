import pandas as p
#import numpy

dset = p.read_csv('diabetes.csv')
dset.head()

pi_dset = dset.Outcome.value_counts()

#Conjunto de instrucciones para eliminar valores perdidos
dset = dset.drop(dset.query('Glucose == 0').index)
dset = dset.drop(dset.query('BloodPressure == 0').index)
dset = dset.drop(dset.query('SkinThickness == 0').index)
dset = dset.drop(dset.query('BMI == 0').index)

#Conjunto de instrucciones para limpieza y verificacion 
dset = dset.reset_index(drop = True)
info_dset = dset.describe().T
p_dset = dset.Outcome.value_counts()
#print(p_dset)

#Conjunto de instrucciones para eliminar registros sobrantes (balancear), limpieza y verificacion
dset_copy = dset.drop(dset.query('Outcome == 0').sample(n = 484, random_state = 1).index)
dset_copy = dset_copy.reset_index(drop = True)
info_dset_copy = dset_copy.describe().T
p_dset_copy = dset_copy.Outcome.value_counts()
#print(p_dset_copy)


#Conjunto de instrucciones para separar la clase
clase = dset_copy.Outcome
dset_copy = dset_copy.drop(['Outcome'], axis = 1)

#Conjunto de intrucciones para calcular la media y desviacion
mean_dset_copy = dset_copy.mean(axis = 0)
desv_dset_copy = dset_copy.std(axis = 0)
#print(mean_dset_copy)
#print(desv_dset_copy)


#Conjunto de instrucciones para normalización y generar nuevo archivo
dset_copy = (dset_copy - mean_dset_copy) / desv_dset_copy
dset_copy.to_csv(r'C:\Users\SystemOutp\IA_PROYECTO\CODE_PY\copy_diabetes.csv', index = False, header = False)

