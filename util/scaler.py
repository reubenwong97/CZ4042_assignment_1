import numpy as np

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def norm_scale(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)