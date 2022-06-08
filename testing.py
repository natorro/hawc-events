
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.models import load_model

model1 = load_model('modelodia1.h5')

file = "/Users/natorro/Desktop/programming/yardiel_projects/hawc-events/datasets/hawc_data/hawc_crudos200k.csv"
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

for i in range(200000):
    pred1=model1.predict(numeros[i])
    print(pred1)

