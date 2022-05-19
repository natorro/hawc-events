#!/usr/bin/env python

file = "/Users/natorro/Desktop/programming/yardiel_projects/hawc/datasets/hawc_data/hawc_crudos200k.csv"
import csv
archivo = open(file)
csvreader = csv.reader(archivo)

vec=[]
next(csvreader)
for i in range(200000):
    vec.append(next(csvreader))

print(vec[0])
print(len(vec[8]))

clase = []
for i in vec:
    clase.append(i[len(i)-1])
    
print(clase[:3])

salida = []
for i in range(200000):
    salida.append([])
    salida[i].append(clase[i])

print(salida[:100])

print(clase)

elementos = []

count = 0
for i in vec:
    i.pop(0)
    i.pop(300)
    count += 1
    elementos.append(i)

print(elementos[0])

len(elementos[0])

numeros = []
for i in range(200000):
    numeros.append(i)

print(numeros)

from random import shuffle
shuffle(numeros)

print(numeros)

entrada=[]
salidas=[]
for i in range(160000):
    entrada.append(elementos[numeros[i]])
    salidas.append(salida[numeros[i]])
len(entrada[9])


len(salidas[9])

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from tensorflow import keras


from keras import models
from keras import layers
from keras import losses
from keras import metrics

import numpy as np

entrada = np.array(entrada, "float32")
salidas = np.array(salidas, "float32")

model = models.Sequential()
model.add(layers.Dense(300, activation = 'sigmoid', input_shape = (300, )))
model.add(layers.Dense(300, activation = 'sigmoid'))
model.add(layers.Dense(300, activation = 'sigmoid'))
model.add(layers.Dense(100, activation = 'sigmoid'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(learning_rate = 0.001), 
              loss = losses.binary_crossentropy, 
              metrics = [metrics.binary_accuracy])

model.fit(entrada, salidas, epochs = 1000)


model.save('modelodia1.h5')


from keras.models import load_model
model1=load_model('modelodia1.h5')

entradaRed
pred1=model1.predict_on_batch(entradaRed)






