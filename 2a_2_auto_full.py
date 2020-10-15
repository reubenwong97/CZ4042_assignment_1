import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from util.scaler import norm_scale
from util.plots import plot_loss, plot_predictions, compare_feature_losses

# pseudocode

# require array of length L: store best mse for that subset
# require array of length L: best feature to drop

# for each subset of length L:
# 	scan through all columns and drop one by one
# 	record mse of this subset

# 	find the index corresponding to the best mse
# 	append best mse
# 	append feature index to drop

# 	stop when no improvement or performance worsens

NUM_CLASSES = 7

# determined from early stopping
epochs = 75
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

# baseline to compare to
baseline = keras.Sequential([
        keras.layers.Dense(num_neurons, activation='relu'),
        keras.layers.Dense(1)
    ])

baseline.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=['mse'])

    # learn the network
baseline_history = baseline.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose = 2,
                        validation_data=(X_test, y_test))

final_mse = np.mean(baseline_history.history['val_mse'][-5:])
baseline_mse = baseline_history.history['val_mse']
np.save('./data/2a_2/baseline_test_mse.npy', baseline_history.history['val_mse'])
np.save('./data/2a_2/baseline_train_mse.npy', baseline_history.history['mse'])

# informed that while cross-validation is correct, not done for 
# this assignment due to time required

BEST_MSES = []
BEST_MSE_HIST = []
BEST_IDXS = []
ALL_MSES = []
ALL_MSE_HIST = []
FIRST_ROW_X = []

original_feature_len = X_train.shape[1]
best_mse = final_mse

for j in range(original_feature_len-1):
# maximum number of times to loop -> loop until 1 feature left
    print(f'...subset length {original_feature_len-j}...')
    best_feature_idx = None
    subset_mses = [] 
    first_xes = []
    subset_mse_histories = []

    # loop for searching through inputs
    for i in range(X_train.shape[1]):
        print(f'...analysing {i}th feature...')
        # drop ith feature from train and test sets
        X_train_ = np.delete(X_train, [i], axis=1)
        X_test_ = np.delete(X_test, [i], axis=1)

        model = keras.Sequential([
            keras.layers.Dense(num_neurons, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                    loss=keras.losses.MeanSquaredError(),
                    metrics=['mse'])

        # learn the network
        history = model.fit(X_train_, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=2,
                                validation_data=(X_test_, y_test))

        last_mse = np.mean(history.history['val_mse'][-5:])
        subset_mse_histories.append(history.history['val_mse'])
        subset_mses.append(last_mse)
        first_xes.append((f'subset length: {len(X_train_[0])}', X_train_[0]))

    best_mse = np.min(subset_mses)
    best_feature_idx = np.argmin(subset_mses)
    best_mse_history = subset_mse_histories[best_feature_idx]

    ALL_MSES.append(subset_mses)	
    FIRST_ROW_X.append(first_xes)
    ALL_MSE_HIST.append(subset_mse_histories)

    # record data
    BEST_MSES.append(best_mse)
    BEST_IDXS.append(best_feature_idx)
    BEST_MSE_HIST.append(best_mse_history)

    # alter subset
    X_train = np.delete(X_train, [best_feature_idx], axis=1)
    X_test = np.delete(X_test, [best_feature_idx], axis=1)

# save arrays
np.save('./data/2a_2/auto_full/best_mse_scores.npy', BEST_MSES)
np.save('./data/2a_2/auto_full/best_mse_history.npy', BEST_MSE_HIST)
np.save('./data/2a_2/auto_full/best_idxs.npy', BEST_IDXS)
np.save('./data/2a_2/auto_full/all_mses.npy', ALL_MSES)
np.save('./data/2a_2/auto_full/all_mse_hist.npy', ALL_MSE_HIST)

print("...BEST MSES...\n", BEST_MSES)
print("...BEST IDXS...\n", BEST_IDXS)
print("...ALL MSES...\n", ALL_MSES)
print("...FIRST ROWS...\n", FIRST_ROW_X)

# plots
compare_feature_losses(BEST_MSE_HIST, [4, 2, 6, 0, 3, 1], 'full_rse_sweep', 'full_rse_sweep best features', baseline=baseline_mse, path='./figures/2a_2/auto_full/')