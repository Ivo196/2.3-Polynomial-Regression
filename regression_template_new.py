# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:06:06 2023

@author: ivoto
"""

#Regresion template

#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values 
y = dataset.iloc[:, 2:3].values 


#Training & Test 
'''
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )
'''

#Escalado de variables(Datos)
'''
#Escalamos el conjunto de entrenamiento 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #Ahora quedan escalados entre -1 y 1 pero es una STANDARIZACION (Normal) por lo que tendremos valores mayores a 1 y menores a -1
X_test = sc_X.transform(X_test) #Solo detecta la transformacion y la aplica
'''

#Ajustamos la regresion con el dataset
'''
Crear nuestro modelo de regresion 
'''


#Prediccion de nuestro modelos 
y_pred = regression.predict(6.5) #tengo que hacer la tranformacion polinomica primero
 
#Visualizacion de los resultados del Modelo Polinomico
X_grid = np.arange(min(X), max(X), 0.1 )
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red') 
plt.plot(X_grid, regression.predict(X_grid),color = 'blue')
plt.title('Modelo de Regresion')
plt.xlabel('Posicion del empleado')
plt.ylabel=('Sueldo (en $)')
plt.show()


