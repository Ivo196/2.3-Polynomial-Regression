# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:37:50 2023

@author: ivoto
"""

#Regresion polinomica 

#Plantilla de Pre-Procesado 
#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values 
y = dataset.iloc[:, 2:3].values 


#Training & Test 
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )

#Escalado de variables(Datos)
'''
#Escalamos el conjunto de entrenamiento 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #Ahora quedan escalados entre -1 y 1 pero es una STANDARIZACION (Normal) por lo que tendremos valores mayores a 1 y menores a -1
X_test = sc_X.transform(X_test) #Solo detecta la transformacion y la aplica
'''
#Ajustamos la regresion lineal con el dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Ajustamos la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4 ) #Lo que hace es elevar al cuadrado o a la n  cada uno de los terminos que le pasemos
X_poly = poly_reg.fit_transform(X) #Me tranforma la X en una polinomica de grado 2 (Termino independiente = 1, Valores de X, El cuadrado de cada uno de los valores de X)
#Para hacer una regresion Polinomica se usa el mismo modelo que la regresion lineal 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#Visualizacion de los resultados del Modelo Lineal
plt.scatter(X, y, color='red') 
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Modelo de Regresion Lineal')
plt.xlabel('Posicion del empleado')
plt.ylabel=('Sueldo (en $)')
plt.show()

#Visualizacion de los resultados del Modelo Polinomico
plt.scatter(X, y, color='red') 
plt.plot(X, lin_reg_2.predict(X_poly),color = 'blue')
plt.title('Modelo de Regresion Polinomica')
plt.xlabel('Posicion del empleado')
plt.ylabel=('Sueldo (en $)')
plt.show()

#Prediccion de nuestro modelos 
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) #tengo que hacer la tranformacion polinomica primero





























