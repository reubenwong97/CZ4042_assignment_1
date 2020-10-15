import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from util.scaler import norm_scale
from util.plots import plot_loss, plot_predictions

NUM_CLASSES = 7

# determined from early stopping
epochs = 40
batch_size = 8
num_neurons = 10
weight_decay = 10e-3
lr = 10e-3
seed = 10

histories={}

np.random.seed(seed)
tf.random.set_seed(seed)

#read and divide data into test and train sets 
admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)

# perform split and shuffling
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, shuffle=True, random_state=seed)

# scale both, no CV here
X_train = norm_scale(X_train)
X_test = norm_scale(X_test)

# create a network
starter_model = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1)
])

starter_model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
              loss=keras.losses.MeanSquaredError(),
              metrics=['mse'])

# learn the network
history =starter_model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose = 2,
                        validation_data=(X_test, y_test))

# plot learning curves
plot_loss(history.history, 'training_val_losses', 'mse', 'epochs vs mse losses', path='./figures/2a_1/')
plot_predictions(starter_model, X_test, y_test, 50, 'predictions_targets_scatter', 'predictions and targets', path='./figures/2a_1/')
plot_predictions(starter_model, X_test, y_test, 50, 'predictions_targets_line', 'predictions and targets', scatter=False,path='./figures/2a_1/')
