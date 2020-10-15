# for random debugging
import numpy as np
from util.plots import compare_feature_losses, plot_loss, compare_models
import tensorflow as tf
from sklearn.model_selection import train_test_split
from util.scaler import norm_scale
from tensorflow import keras

# best_mse_history = np.load('./data/2a_2/auto_full/best_mse_history.npy')
# compare_feature_losses(best_mse_history, [4, 2, 6, 0, 3, 1], 'full_rse_sweep', 'full_rse_sweep best features', path='./figures/2a_2/auto_full/')

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

X_train_ = np.delete(X_train, [5], axis=1)
X_test_ = np.delete(X_test, [5], axis=1)

alternate = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1)
])

alternate.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
            loss=keras.losses.MeanSquaredError(),
            metrics=['mse'])

# learn the network
alt_history = alternate.fit(X_train_, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_test_, y_test))

# train only on 5th feature
X_train = X_train[:, 5]
X_test = X_test[:, 5]

model = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay)),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
            loss=keras.losses.MeanSquaredError(),
            metrics=['mse'])

# learn the network
history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_test, y_test))

plot_loss(history.history, 'debug_loss', 'mse', 'loss on 1 feature', path='./figures/2a_2/auto_full/')
plot_loss(alt_history.history, 'alt_debug_loss', 'mse', 'loss on other features', path='./figures/2a_2/auto_full/')

compare_models(history.history, alt_history.history, 'mse', '1_feature', 'other_features', 'comparing_1_others', 'comparing_1_others', path='./figures/2a_2/auto_full/')