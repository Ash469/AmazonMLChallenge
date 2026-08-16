import numpy as np

def calculate_mape(y_true, y_pred):
    """
    Computes Mean Absolute Percentage Error (MAPE).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    y_true_clipped = np.clip(y_true, 1e-5, None) # for close to zero target values.
    
    mape = np.mean(np.abs(y_true - y_pred) / y_true_clipped)
    return float(mape)
