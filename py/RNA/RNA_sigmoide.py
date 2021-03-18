i = 1
resultado = []
import numpy as np
import random as r
import sys
from index import Folds,dset,clase


sys.stdout = open("salida.txt", "w") 
 
for train, test in Folds(dset):
    dset_entrenamiento = dset[train]
    dset_prueba = dset[test]
    clase_entrenamiento = clase[train]
    clase_prueba = clase[test]
    i += 1     
    clase_t = 50
    
    '''------------------ RED NEURONAL----------------------- '''
    # Función de activación sigmoide
    def sigmoide(x):
      return 1 / (1 + np.exp(-x))
    
    # Retorna y_real y y_pred, que son matrices de la misma longitud en la salida.
    def mse_error(y_real, y_pred):
      return ((y_real - y_pred) ** 2).mean()
    
    class Red_Neuronal:
      '''
      La red neuronal contiene:
        - 8 entradas
        - capa oculta con 2 neuronas (h1, h2)
        - salida con 1 neurona (o1)
      '''
      def __init__(self):
          
        numeros = [r.randint(1,99) for x in range(18)]
             
        # Pesos
        self.w1 = numeros[0]/100
        self.w2 = numeros[1]/100
        self.w3 = numeros[2]/100
        self.w4 = numeros[3]/100
        self.w5 = numeros[4]/100
        self.w6 = numeros[5]/100
        self.w7 = numeros[6]/100
        self.w8 = numeros[7]/100
        self.w9 = numeros[8]/100
        self.w10 = numeros[9]/100
        self.w11 = numeros[10]/100
        self.w12 = numeros[11]/100
        self.w13 = numeros[12]/100
        self.w14 = numeros[13]/100
        self.w15 = numeros[14]/100
        self.w16 = numeros[15]/100
        self.w17 = numeros[16]/100
        self.w18 = numeros[17]/100
    
    
      def hacia_adelante(self, x):
        # x es la matriz de 8 atributos.
        h1 = sigmoide(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.w4 * x[3] + self.w5 * x[4] + self.w6 * x[5] + self.w7 * x[6] + self.w8 * x[7])
        h2 = sigmoide(self.w9 * x[0] + self.w10 * x[1] + self.w11 * x[2] + self.w12 * x[3] + self.w13 * x[4] + self.w14 * x[5] + self.w15 * x[6] + self.w16 * x[7])
        o1 = sigmoide(self.w17 * h1 + self.w18 * h2)
        return o1
    
      def entrenamiento(self, datos, clase):
        '''
        - los datos son una matriz numérica (n x 8), n = # de muestras en el conjunto de datos.
        - clase es una matriz numerosa con n elementos.
          Los elementos de clase corresponden a los datos.
        '''
        
        aprendizaje = 0.1
        #número de veces para recorrer todo el conjunto de datos
    
        for i in range(clase_t):
          for x, y_real in zip(datos, clase):
            # --- Seguir adelante
            sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.w4 * x[3] + self.w5 * x[4] + self.w6 * x[5] + self.w7 * x[6] + self.w8 * x[7]
            h1 = sigmoide(sum_h1)
    
            sum_h2 = self.w9 * x[0] + self.w10 * x[1] + self.w11 * x[2] + self.w12 * x[3] + self.w13 * x[4] + self.w14 * x[5] + self.w15 * x[6] + self.w16 * x[7]
            h2 = sigmoide(sum_h2)
    
            sum_o1 = self.w17 * h1 + self.w18 * h2
            o1 = sigmoide(sum_o1)
            y_pred = o1
    
            # --- Calcula derivadas parciales.
            # --- Nombre: d_L_d_w1 representa "L parcial / w1 parcial"
            d_L_d_ypred = -2 * (y_real - y_pred)
    
            # Neurona o1
            d_ypred_d_w17 = h1 * sigmoide(sum_o1)
            d_ypred_d_w18 = h2 * sigmoide(sum_o1)
            
    
            d_ypred_d_h1 = self.w17 * sigmoide(sum_o1)
            d_ypred_d_h2 = self.w18 * sigmoide(sum_o1)
    
            # Neurona h1
            d_h1_d_w1 = x[0] * sigmoide(sum_h1)
            d_h1_d_w2 = x[1] * sigmoide(sum_h1)
            d_h1_d_w3 = x[2] * sigmoide(sum_h1)
            d_h1_d_w4 = x[3] * sigmoide(sum_h1)
            d_h1_d_w5 = x[4] * sigmoide(sum_h1)
            d_h1_d_w6 = x[5] * sigmoide(sum_h1)
            d_h1_d_w7 = x[6] * sigmoide(sum_h1)
            d_h1_d_w8 = x[7] * sigmoide(sum_h1)
        
    
            # Neurona h2
            d_h2_d_w9 = x[0] * sigmoide(sum_h2)
            d_h2_d_w10 = x[1] * sigmoide(sum_h2)
            d_h2_d_w11 = x[2] * sigmoide(sum_h2)
            d_h2_d_w12 = x[3] * sigmoide(sum_h2)
            d_h2_d_w13 = x[4] * sigmoide(sum_h2)
            d_h2_d_w14 = x[5] * sigmoide(sum_h2)
            d_h2_d_w15 = x[6] * sigmoide(sum_h2)
            d_h2_d_w16 = x[7] * sigmoide(sum_h2)
         
    
            # --- Actualizar ponderaciones
            # Neurona h1
            self.w1 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
            self.w2 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
            self.w3 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
            self.w4 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w4
            self.w5 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w5
            self.w6 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w6
            self.w7 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w7
            self.w8 -= aprendizaje * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w8

    
            # Neurona h2
            self.w9 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w9
            self.w10 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w10
            self.w11 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w11
            self.w12 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w12
            self.w13 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w13
            self.w14 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w14
            self.w15 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w15
            self.w16 -= aprendizaje * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w16
    
            # Neurona o1
            self.w17 -= aprendizaje * d_L_d_ypred * d_ypred_d_w17
            self.w18 -= aprendizaje * d_L_d_ypred * d_ypred_d_w18

    
          # --- Calcula el error total
          if i % 1 == 0:  
            y_preds = np.apply_along_axis(self.hacia_adelante, 1, datos)
            error = mse_error(clase, y_preds)
            resultado.append(error)
            print("    clase: {}  Salida: {} ".format(clase_entrenamiento[i],error))


    print("FOLD",i-1,": ")
    # Entrenamiento
    network = Red_Neuronal()
    network.entrenamiento(dset_entrenamiento, clase_entrenamiento)
    print("                   ECM:", np.mean(resultado))


