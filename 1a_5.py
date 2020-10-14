import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from util.plots import plot_acc, plot_loss, plot_accs, plot_time, compare_models
from util.scaler import scale
from util.callbacks import TimeHistory
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

wandb.init(project="nndl_assignment_1")

NUM_CLASSES = 3

# validation loss stabalises at around 250 epochs -> 500 epochs for plotting
epochs = 500
num_neurons = 10
batch_size = 32
weight_decay = 10e-6
seed = 10

np.random.seed(seed)
tf.random.set_seed(seed)

histories = {}

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
data_X, data_y = train_input[1:, :21], train_input[1:,-1].astype(int)
data_y = data_y-1

# split and scale data to prevent leakage of distribution
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=seed)
#* only scale test here
X_test = scale(X_test, np.min(X_test, axis=0), np.max(X_test, axis=0)) 

################################# TEST #######################################
X_train = scale(X_train, np.min(X_train, axis=0), np.max(X_train, axis=0))
model_4 = keras.Sequential([
            keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay),
                            bias_regularizer=keras.regularizers.l2(weight_decay)),
            keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay),
                            bias_regularizer=keras.regularizers.l2(weight_decay)),
            keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
        ])

model_4.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

# fit to full training set now
history_4 = model_4.fit(X_train, y_train,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[WandbCallback()])

plot_acc(history_4.history, 'full_test_train_acc', 'epochs vs train_acc', path='./figures/1a_5/')
plot_loss(history_4.history, 'full_test_train_loss', 'sparse_categorical_crossentropy','epochs vs loss', path='./figures/1a_5/')

# 3 layer optimum model
model_3 = keras.Sequential([
            keras.layers.Dense(20, activation='relu', kernel_regularizer=keras.regularizers.l2(10e-6),
                            bias_regularizer=keras.regularizers.l2(10e-6)),
            keras.layers.Dense(NUM_CLASSES)
])
model_3.compile(optimizer='sgd',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

history_3 = model_3.fit(X_train, y_train,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[WandbCallback()])

compare_models(history_3.history, history_4.history, 'accuracy', '3_layer', '4_layer', 'compare_3_4_layer_acc', 'comparison of 3 layer and 4 layer accuracy', path='./figures/1a_5/')
compare_models(history_3.history, history_4.history, 'sparse_categorical_crossentropy', '3_layer', '4_layer', 'compare_3_4_layer_loss', 'comparison of 3 layer and 4 layer loss', path='./figures/1a_5/')
################################ END TEST #####################################
