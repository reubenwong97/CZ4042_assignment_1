#
# Project 1, starter code part a
#

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from util.plots import plot_acc, plot_loss
from util.scaler import scale
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 3

epochs = 1000
batch_size = 32
num_neurons = 10
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
X_train = scale(X_train, np.min(X_train, axis=0), np.max(X_train, axis=0))
X_test = scale(X_test, np.min(X_test, axis=0), np.max(X_test, axis=0)) 

# create the model
starter_model = keras.Sequential([
    keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(10e-6),
                    bias_regularizer=keras.regularizers.l2(10e-6)),
    keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
])

starter_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

# train the model
histories['starter'] = starter_model.fit(X_train, y_train,
                                        epochs=epochs,
                                        verbose = 2,
                                        batch_size=batch_size,
                                        validation_data=(X_test, y_test))

# plot learning curves
plot_acc(histories['starter'].history, 'starter_acc', title='epochs vs starter model accuracy')
plot_loss(histories['starter'].history, 'starter_loss', loss='sparse_categorical_crossentropy', 
    title='epochs vs starter model crossentropy')
plot_loss(histories['starter'].history, 'starter_loss_with_reg', loss='loss', 
    title='epochs vs starter model crossentropy loss with regularization penalty')
