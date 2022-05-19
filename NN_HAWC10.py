# NN_HAWC10
# Programa con alteracion de archivo de validacion o de entada 
import os
import random
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf
from tensorflow              import keras
from sklearn.metrics         import roc_curve, roc_auc_score, recall_score, f1_score
from keras.utils.vis_utils   import plot_model
np.random.seed(41)
random.seed(41)

def plot_roc_curve(fpr, tpr, auc_ROC, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0,1],[0,1], 'k--')
	plt.xlabel("Falso Positivo", size = 14)
	plt.ylabel("Verdadero Positivo (recall)", size = 14)
	plt.title(f"ROC auRoc = {auc_ROC:0.2f}")

print("\n________________________  NN API HAWC  _________________________\n")
current_path = os.getcwd()      #regresa directorio de trabjo
file = os.path.sep.join(['', 'datasets', 'hawc_data', 'hawc_crudos10k.csv'])
data_size = 1000
data = pd.read_csv(current_path + file)  #lee archivo .csv

dataX = data.copy().drop(['evento','class'],axis=1)
dataY = data['class'].copy()

tr_size  = int(data_size*0.8)
tst_size = int(data_size*0.9)
val_size = data_size - tst_size

X_train, X_test, X_valid = dataX[:tr_size], dataX[tr_size:tst_size], dataX[tst_size:]
y_train, y_test, y_valid = dataY[:tr_size], dataY[tr_size:tst_size], dataY[tst_size:]
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X_valid = X_valid.to_numpy()
y_valid = y_valid.to_numpy()
#------
#--------0---------0---------0---------0---------0---------0---------0--
#--- Tensor Board -----

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("RUN%m:%d%H%M")
    return os.path.join(root_logdir, run_id),run_id

run_logdir, run_id = get_run_logdir()

#---- model ---------
np.random.seed(42)
tf.random.set_seed(42)

#-------  A C T I V A T I O N -----------
activation_hidd = "elu"
activation_out  = "softmax"
#---
#activation = "softmax"    #mas usada en ultima capa y multiclase
#activation = "sigmoid"    #(logistic) mas usada en binaria
#activation  = "relu"       #bueno en hidd, muy usada
#activation = "softplus"
#sctivation = "softsign"
#activation = "tanh"  #muy comun
#activation = "selu"
#activation = "elu"
#activation = "exponential"
#activation = "LeakyReLU(alpha=0.01)"

#------ L O S S  ---------
loss = "sparse_categorical_crossentropy"

# Probabilistic
# "BinaryCrossentropy"
# "binary_crossentropy"
# "CategoricalCrossentropy"
# "categorical_crossentropy" -
# "sparse_categorical_crossentropy" ok
# "SparseCategoricalCrossentropy"
# "Poisson"
# "poisson" ok
# "KLDivergence" ok
# "kl_divergence" ok

# Regession
# "mean_squared_error" ok
# "MeanSquaredError" ok
# "MeanAbsolutePercentageError"  ok
# "MeanSquaredLogarithmicError" ok
# "mean_absolute_error" ok
# "cosine_similarity" ok
# "Huber" ok
# "LogCosh" ok

# hinge
# "hinge"
# "SquaredHinge"
# "CategoricalHinge"

#------ O P T I M I Z E R -------------
learning_rate = 2.0e-4     # default = 1.0e-2
#---
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
#optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#optimizer = keras.RMSprop(learning_rate=learning_rate)
#optimizer = keras.Adadelta(learning_rate=learning_rate)

#-------- M E T R I C S ---------
metrics = ["accuracy"]
#metrics = [tf.keras.metrics.Accuracy()]
#metrics = ["binary_accuracy"]
#metrics = [tf.keras.metrics.BinaryAccuracy()]

#-------- E P O C H S --------
epochs = 200 

#--------- L A Y E R S ---------
hidd_lay_1 = 300
#hidd_lay_1 = float(input("hid_lay_1 = "))
#hidd_lay_2 = float(input("hid_lay_2 = "))
#hidd_lay_3 = float(input("hid_lay_3 = "))
n_hidd_lay = 1

print()
print("Input file      = ", file)
print("Data Size       = ", data_size)
print("Run id          = ", run_id)
print("Epochs          = ", epochs)
print("Hidd lay 1 siz  = ", hidd_lay_1)
#print("Hidd lay 2 siz    = ", hidd_lay_2)
#print("Hidd lay 3 siz    = ", hidd_lay_3)
print("N hidd_lay      = ", n_hidd_lay)
print("Learning_rate   = ", learning_rate)
print("Activation_hidd = ", activation_hidd)
print("Activation_out  = ", activation_out)
print("Optimizer       = ", optimizer)
print("Loss            = ", loss)
print("Metrics         = ", metrics)
print()

#-------- M O D E L O -------
model = keras.models.Sequential()
model.add(keras.layers.Dense(hidd_lay_1, input_shape=[300,], activation=activation_hidd))
#model.add(keras.layers.Dense(hidd_lay_2, activation=activation_hidd))
#model.add(keras.layers.Dense(hidd_lay_3, activation=activation_hidd))
model.add(keras.layers.Dense(2, activation=activation_out))

print()
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#---------  C O M P I L E ---------
model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)

#--------- F I T  --------
history = model.fit(X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb],
                    verbose=0)
#------------
train_loss,train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"Train loss      = {train_loss:5.2f}, train accuracy      = {train_acc:5.2f}")
#---
test_loss,test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss       = {test_loss:5.2f}, test accuracy       = {test_acc:5.2f}")
#---
val_loss,val_acc = model.evaluate(X_valid, y_valid, verbose=0)
print(f"Validation loss = {val_loss:5.2f}, validation accuracy = {val_acc:5.2f}")

#------------

#---------P R E D I C T --------------
TP = TN = FP = FN = 0
g_calculados = []
h_calculados = []
curva_g_calculados = []
curva_h_calculados = []
g_target = []
h_target = []
size = int(data_size*0.1)
y_scores = []
#-------   interin 

size = int(data_size*0.1)
X_new = X_valid   #[:i]
y_new_target = y_valid

prediction = model.predict(X_new)
print("\nlen(X_new) =", len(X_new))
print("np.shape(X_new) = ",np.shape(X_new))
print("X_new.size",X_new[0][:].size)
print("\nlen(prediction) =", len(prediction))
print("np.shape(prediction) = ",np.shape(prediction))
print("prediction.size",prediction[0][:].size)

for j in range(len(X_new)):
	g_predicho = prediction[j][1]
	g_calculados.append(g_predicho)
	h_predicho = prediction[j][0]
	h_calculados.append(h_predicho)
	if bool(y_new_target[j]):
		curva_g_predicho = prediction[j][1]
		curva_g_calculados.append(curva_g_predicho)
		if g_predicho > 0.5: 
			TP += 1
		else: 
			FN += 1
	else:
		curva_h_predicho = 1.0 - prediction[j][0]
		curva_h_calculados.append(curva_h_predicho)
		if h_predicho > 0.5: 
			TN += 1
		else: 
			FP += 1
#----------------

print(f"\nConfusion Matrix:\nTP={TP:g} FP={FP:g}")
print(                     f"FN={FN:g} TN={TN:g}")
prec = TP/(TP+FP+0.1)	# mejor para HAWC
ACC = 100.0*(TP + TN)/(TP+TN+FP+FN)
rec = TP/(TP+FN+0.1)
f1 = 2.0 * prec*rec/(prec + rec+0.1)
print(f"\nACC = {ACC:.0f} %")
print(f"Precision = {prec:.2f}")
print(f"Recall    = {rec:.2f}")
print(f"f1        = {f1:.2f}")

y_scores = np.array(g_calculados)
fpr, tpr, thresholds = roc_curve(y_valid, y_scores)
auc_ROC = roc_auc_score(y_valid, y_scores)
plot_roc_curve(fpr, tpr, auc_ROC)
plt.show()
print(f"\nauc_ROC = {auc_ROC:.2f}")
#-------
plt.figure(figsize=(8,6))
plt.hist(curva_g_calculados, bins=50, range=(0.0,1.0), alpha=0.5, label="Gamma")
plt.hist(curva_h_calculados, bins=50, range=(0.0,1.0), alpha=0.5, label="Protones")
plt.xlabel("Probabilidad", size = 14)
plt.ylabel("Eventos"    , size = 14)
plt.title("Distribucion predicha")
plt.legend(loc = 'upper right')
plt.savefig("overlapping_histograms_with_matplotlib_Python_2.png")
plt.show()
#-----

#--- Tensor Board
root_logdir = os.path.join(os.curdir, "my_logs")
run_logdir = get_run_logdir()
run_logdir
print("para ver tensorboard")
print("\n > tensorboard --logdir=./my_logs --port=6006")
print("http://localhost:6006/#scalars")

