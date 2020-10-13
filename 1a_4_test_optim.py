import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from util.plots import plot_acc, plot_loss, plot_accs, plot_time
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
OPTIM_BATCH_SIZE = 32
OPTIM_NUM_NEURONS = 20
OPTIM_WEIGHT_DECAY = 10e-6
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
model = keras.Sequential([
            keras.layers.Dense(OPTIM_NUM_NEURONS, activation='relu', kernel_regularizer=keras.regularizers.l2(OPTIM_WEIGHT_DECAY),
                            bias_regularizer=keras.regularizers.l2(OPTIM_WEIGHT_DECAY)),
            keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
        ])

model.compile(optimizer='sgd',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

# fit to full training set now
history = model.fit(X_train, y_train,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=OPTIM_BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[WandbCallback()])

plot_acc(history.history, 'full_test_train_acc', 'epochs vs train_acc', path='./figures/1a_4/')
plot_loss(history.history, 'full_test_train_loss', 'epochs vs loss', path='./figures/1a_4/')
################################ END TEST #####################################
