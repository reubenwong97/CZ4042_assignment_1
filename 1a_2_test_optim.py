import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from util.plots import plot_acc, plot_loss, plot_accs, plot_time
from util.scaler import scale
from util.callbacks import TimeHistory
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

wandb.init(project="nndl_assignment_1")

NUM_CLASSES = 3

# validation loss stabalises at around 250 epochs -> 300 epochs for plotting
epochs = 500
OPTIM_BATCH_SIZE = 8
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
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
################################# TEST #######################################
model = keras.Sequential([
            keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(10e-6),
                            bias_regularizer=keras.regularizers.l2(10e-6)),
            keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
        ])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

# fit to full training set now
history = model.fit(X_train, y_train,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=OPTIM_BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[WandbCallback()])

plot_acc(history.history, 'full_test_train_acc', 'epochs vs acc', path='./figures/1a_2/')
plot_loss(history.history, 'full_test_train_loss', 'epochs vs loss', path='./figures/1a_2/')
################################ END TEST #####################################
