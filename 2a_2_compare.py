import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from util.scaler import norm_scale
from util.plots import plot_loss, plot_predictions, compare_subset_lengths

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
X_train_6 = np.delete(X_train, [4], axis=1)
X_test_6 = np.delete(X_test, [4], axis=1)
X_train_5 = np.delete(X_train_6, [2], axis=1)
X_test_5 = np.delete(X_test_6, [2], axis=1)
X_train_4 = np.delete(X_train_5, [4], axis=1)
X_test_4 = np.delete(X_test_5, [4], axis=1)
X_train_3 = np.delete(X_train_4, [0], axis=1)
X_test_3 = np.delete(X_test_4, [0], axis=1)
############################################# handling dataset #############################################

# baseline to compare to
baseline = keras.Sequential([
        keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
        keras.layers.Dense(1)
    ])

baseline.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

    # learn the network
baseline_history = baseline.fit(X_train_all, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_test_all, y_test))

# model on subset length 6
model_6 = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1)
])

model_6.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

    # learn the network
model_6_history = model_6.fit(X_train_6, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_test_6, y_test))

# model on subset length 5
model_5 = keras.Sequential([
        keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
        keras.layers.Dense(1)
    ])

model_5.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

    # learn the network
model_5_history = model_5.fit(X_train_5, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_test_5, y_test))

# model on subset length 5
# model_4 = keras.Sequential([
#         keras.layers.Dense(num_neurons, activation='relu'),
#         keras.layers.Dense(1)
#     ])

# model_4.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
#                 loss=keras.losses.MeanSquaredError(),
#                 metrics=['mse'])

#     # learn the network
# model_4_history = model_4.fit(X_train_4, y_train,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         verbose=2,
#                         validation_data=(X_test_4, y_test))

# model_3 = keras.Sequential([
#         keras.layers.Dense(num_neurons, activation='relu'),
#         keras.layers.Dense(1)
#     ])

# model_3.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
#                 loss=keras.losses.MeanSquaredError(),
#                 metrics=['mse'])

#     # learn the network
# model_3_history = model_3.fit(X_train_3, y_train,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         verbose=2,
#                         validation_data=(X_test_3, y_test))


# mse_array = [baseline_history.history['val_mse'], model_6_history.history['val_mse'], 
#     model_5_history.history['val_mse'], model_4_history.history['val_mse'],
#     model_3_history.history['val_mse']]

mse_array = [baseline_history.history['val_mse'], model_6_history.history['val_mse'], 
    model_5_history.history['val_mse']]

last_mse_array = [baseline_history.history['val_mse'][-1], model_6_history.history['val_mse'][-1], 
    model_5_history.history['val_mse'][-1]]

first_rows = [X_train_all[0], X_train_6[0], X_train_5[0]]

compare_subset_lengths(mse_array, 7, 'comparing_subset_567', 'perfomance of models trained on subset of features', path='./figures/2a_2/')

print("...LAST MSE...\n", last_mse_array)
print("...FIRST ROWS...\n", first_rows)