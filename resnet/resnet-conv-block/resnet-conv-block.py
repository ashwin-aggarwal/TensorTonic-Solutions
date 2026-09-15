import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype = float)
    Ws = np.array(Ws, dtype = float)
    def relu(v):
        return np.maximum(0,v)

    skip = x @ Ws
    h = relu( x @ W1)
    F = h @ W2
    y = relu(F + skip)
    return np.round(y, 4).tolist()