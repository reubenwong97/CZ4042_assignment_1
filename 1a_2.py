import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from util import scale, plot_acc, plot_loss, plot_accs, TimeHistory, plot_time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_CLASSES = 3

# validation loss stabalises at around 250 epochs -> 300 epochs for plotting
epochs = 300
BATCH_SIZES = [4, 8, 16, 32, 64]
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
#* only scale test here
X_test = scale(X_test, np.min(X_test, axis=0), np.max(X_test, axis=0)) 

# perform splitting
kf = KFold(n_splits=5, random_state=seed, shuffle=False)
kf.get_n_splits(X_train)

AVG_TRAIN_ACCS = []
AVG_VAL_ACCS = []
AVG_TIMES = []

for batch_size in tqdm(BATCH_SIZES):
    # get average scoring for each hyperparameter to optimize towards
    val_accs = []
    train_accs = []
    times = []
    for train_idx, val_idx in kf.split(X_train):
        time_callback = TimeHistory()
        # X_train_ for model to be trained on and X_val to be evaluated on
        X_train_, X_val = X_train[train_idx], X_train[val_idx]
        y_train_, y_val = y_train[train_idx], y_train[val_idx]

        #* normalise train and val seperately here
        X_train_ = scale(X_train_, np.min(X_train_, axis=0), np.max(X_train_, axis=0))
        X_val = scale(X_val, np.min(X_val, axis=0), np.max(X_val, axis=0))

        # create the model: assume regularization is kept
        model = keras.Sequential([
            keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(10e-6),
                            bias_regularizer=keras.regularizers.l2(10e-6)),
            keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
        ])

        model.compile(optimizer='sgd',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

        # train the model
        #! FIT TO CV TRAIN SET
        history = model.fit(X_train_, y_train_,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[time_callback])

        train_accs.append(history.history['accuracy'])
        val_accs.append(history.history['val_accuracy'])
        times.append(time_callback.times[0])

        train_accs.clear()
        val_accs.clear()
        times.clear()

    AVG_TRAIN_ACCS.append(np.mean(np.stack(train_accs, axis=0), axis=0))
    AVG_VAL_ACCS.append(np.mean(np.stack(val_accs, axis=0), axis=0))
    AVG_TIMES.append(np.mean(times))

    #! IDEA: collect the val data from fit -> stack over CV runs -> take mean -> plot

np.save('./data/1a_2/train_accs.npy', AVG_TRAIN_ACCS)
np.save('./data/1a_2/val_accs.npy', AVG_VAL_ACCS)
np.save('./data/1a_2/avg_time_1_epoch.npy', AVG_TIMES)

plot_accs(AVG_TRAIN_ACCS, 'train_accs', 'epochs vs train_accs', 'batch_size', BATCH_SIZES, train=True)
plot_accs(AVG_VAL_ACCS, 'val_accs', 'epochs vs val_accs', 'batch_size', BATCH_SIZES, train=False)
plot_time(AVG_TIMES, 'avg_time', 'batch_size vs avg_time per epoch', 'batch_size', BATCH_SIZES)

################################## TEST #######################################
OPTIM_BATCH_SIZE = 32
X_train = scale(X_train, np.min(X_train, axis=0), np.max(X_train, axis=0))
model = keras.Sequential([
            keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(10e-6),
                            bias_regularizer=keras.regularizers.l2(10e-6)),
            keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
        ])

model.compile(optimizer='sgd',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

# fit to full training set now
history = model.fit(X_train, y_train,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[time_callback])

plot_acc(history.history, 'full_test_train_acc', 'epochs vs train_acc')
################################# END TEST #####################################