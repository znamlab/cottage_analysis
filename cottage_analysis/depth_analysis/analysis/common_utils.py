import numpy as np


def calculate_R_squared(actual_data, predicted_data):
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    residual_var = np.sum((predicted_data - actual_data) ** 2)
    total_var = np.sum((actual_data - np.mean(actual_data)) ** 2)
    R_squared = 1 - residual_var / total_var
    return R_squared
