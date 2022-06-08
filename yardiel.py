import csv
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(41)
rn.seed(41)
tf.random.set_seed(41)

from tensorflow import keras
from keras import optimizers
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# se carga la libreria de tensorflow
import tensorflow.compat.v1 as tf

# se carga el aditamento para seleccionar el gpu
import os

os.environ ['CUDA_VISIBLE_DEVICES'] = '0'

#escoger porcentage de memorya en gpu para ser usada
#config.gpu_options.per_process_gpu_memory_fraction=0.9
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# despues de la configuracion validar, se inicia la sesion
sess = tf.Session (config = config)
import time
start_time = time.time()




archivo = "/Users/natorro/Desktop/programming/yardiel_projects/hawc-events/datasets/hawc_data/hawc_crudos200k.csv"

archivo = open(archivo)
csvreader = csv.reader(archivo)

vec=[]

next(csvreader)
for i in range(160000):
    vec.append(next(csvreader))


print(len(vec[8]))


clase = []
for i in vec:
    clase.append(i[len(i)-1])
    

clase[0]

salida = []
for i in range(160000):
    salida.append([])
    salida[i].append(clase[i])

print(salida)
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




entrada = np.array(elementos, "float32")
salidas = np.array(salida, "float32")

model = models.Sequential()
model.add(layers.Dense(300, activation = 'sigmoid', input_shape = (300, )))
model.add(layers.Dense(300, activation = 'relu'))
model.add(layers.Dense(300, activation = 'sigmoid'))
model.add(layers.Dense(100, activation = 'sigmoid'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(learning_rate = 0.01), 
              loss = losses.binary_crossentropy, 
              metrics = [metrics.binary_accuracy])
model.fit(entrada, salidas, epochs = 200)



model.save('modelodia1.h5')
model1=load_model('modelodia1.h5')

print("\n \n \n Time to finish: --- %s seconds ---" % (time.time() - start_time))
