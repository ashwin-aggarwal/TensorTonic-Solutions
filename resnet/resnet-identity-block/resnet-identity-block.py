import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype = float)
    def relu(x):
        return np.maximum(0,x)
        
    h = relu(x @ W1.T)
    F = h @ W2.T
    y = relu(F + x)
    for row in y:
        for col in row:
            round(float(col), 4)
    return y
    
    