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
WEIGHT_DECAYS = [0, 10e-3, 10e-6, 10e-9, 10e-12]
# param to optimize

# validation loss stabalises at around 250 epochs -> 300 epochs for plotting
epochs = 300
batch_size = 32
num_neurons = 20
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

# perform splitting
kf = KFold(n_splits=5, random_state=seed, shuffle=False)
kf.get_n_splits(X_train)

AVG_TRAIN_ACCS = []
AVG_VAL_ACCS = []
AVG_TIMES = []

for hp_i, weight_decay in enumerate(WEIGHT_DECAYS):
    print('hp_i:\n', hp_i)
    # get average scoring for each hyperparameter to optimize towards
    val_accs = []
    train_accs = []
    times = []
    for train_idx, val_idx in kf.split(X_train):
        time_callback = TimeHistory()
        # X_train_ for model to be trained on and X_val to be evaluated on
        X_train_, X_val = X_train[train_idx], X_train[val_idx]
        y_train_, y_val = y_train[train_idx], y_train[val_idx]

        X_val = scaler.transform(X_val)

        # create the model: assume regularization is kept
        model = keras.Sequential([
            keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay),
                            bias_regularizer=keras.regularizers.l2(weight_decay)),
            keras.layers.Dense(NUM_CLASSES) # softmax not needed as loss specifies from_logits
        ])

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.SparseCategoricalCrossentropy(from_logits=True)])

        # train the model
        #! FIT TO CV TRAIN SET
        history = model.fit(X_train_, y_train_,
                            epochs=epochs,
                            verbose = 2,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[time_callback, WandbCallback(log_weights=True)])

        train_accs.append(history.history['accuracy'])
        val_accs.append(history.history['val_accuracy'])
        times.append(time_callback.times[0])

    AVG_TRAIN_ACCS.append(np.mean(np.stack(train_accs, axis=0), axis=0))
    AVG_VAL_ACCS.append(np.mean(np.stack(val_accs, axis=0), axis=0))
    AVG_TIMES.append(np.mean(times))

    #! IDEA: collect the val data from fit -> stack over CV runs -> take mean -> plot

np.save('./data/1a_4/train_accs_350.npy', AVG_TRAIN_ACCS)
np.save('./data/1a_4/val_accs_350.npy', AVG_VAL_ACCS)
np.save('./data/1a_4/avg_time_1_epoch_350.npy', AVG_TIMES)

plot_accs(AVG_TRAIN_ACCS, 'train_accs_350', 'epochs vs train_accs', 'weight_decay', WEIGHT_DECAYS, train=True, path='./figures/1a_4/')
plot_accs(AVG_VAL_ACCS, 'val_accs_350', 'epochs vs val_accs', 'weight_decay', WEIGHT_DECAYS, train=False, path='./figures/1a_4/')
plot_time(AVG_TIMES, 'avg_time_350', 'num_neurons vs avg_time per epoch', 'weight_decay', WEIGHT_DECAYS, path='./figures/1a_4/')
