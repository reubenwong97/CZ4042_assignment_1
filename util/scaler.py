# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)