# for random debugging
import numpy as np
from util.plots import compare_feature_losses

best_mse_history = np.load('./data/2a_2/auto_full/best_mse_history.npy')
compare_feature_losses(best_mse_history, [4, 2, 6, 0, 3, 1], 'full_rse_sweep', 'full_rse_sweep best features', path='./figures/2a_2/auto_full/')