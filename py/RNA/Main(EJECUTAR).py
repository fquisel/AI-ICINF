import os

op = int(input("(Tiempo ejecución 7min) Digite '1' Sigmoide o Digite '2' Rampa: "))
if op == 1:
    os.system ("python RNA_sigmoide.py")   
elif op == 2:
    os.system ("python RNA_rampa.py") 
    
    