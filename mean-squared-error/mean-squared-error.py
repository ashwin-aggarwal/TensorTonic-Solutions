import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    pred, labels = np.asarray(y_pred, dtype= float), np.asarray(y_true, dtype=float)

    return np.mean( (pred - labels) ** 2) 