import numpy as np

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def norm_scale(X, X_train):
    return (X - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)