import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from util.scaler import norm_scale
from util.plots import plot_loss, plot_predictions, compare_subset_lengths, compare_more_models

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

############################################# handling dataset #############################################
X_train_all = X_train
X_test_all = X_test
X_train_6 = np.delete(X_train, [0], axis=1)
X_test_6 = np.delete(X_test, [0], axis=1)
X_train_5 = np.delete(X_train_6, [1], axis=1)
X_test_5 = np.delete(X_test_6, [1], axis=1)

#* final handling
X_train = X_train_5
X_test = X_test_5
############################################# handling dataset #############################################

model_3 = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1)
])

model_3.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

model_4_drop = keras.Sequential([
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dropout(rate=0.2), 
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(1),
])

model_4_drop.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

model_4_nodrop = keras.Sequential([
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1),
])

model_4_nodrop.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

model_5_drop = keras.Sequential([
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dropout(rate=0.2), 
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(1),
])

model_5_drop.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

model_5_nodrop = keras.Sequential([
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1),
])

model_5_nodrop.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

hist_3 = model_3.fit(X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            validation_data=(X_test, y_test))

hist_4_drop = model_4_drop.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=2,
                validation_data=(X_test, y_test))

hist_4_nodrop = model_4_nodrop.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(X_test, y_test))         
        
hist_5_drop = model_5_drop.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=2,
                validation_data=(X_test, y_test))

hist_5_nodrop = model_5_nodrop.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(X_test, y_test))

histories = [hist_3, hist_4_drop, hist_4_nodrop, hist_5_drop, hist_5_nodrop]
mse_array = [history.history['val_mse'] for history in histories]
names = ['3_layer', '4_layer_dropout', '4_layer_nodropout', '5_layer_dropout', '5_layer_nodropout']

compare_more_models(mse_array, names, 'dropout_analysis_75_epochs', 'comparison: models with / without dropout',
                        path='./figures/2a_3/')