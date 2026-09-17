import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws=None):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.atleast_2d(np.array(x, dtype=float))
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    W3 = np.array(W3, dtype=float)

    def relu(v):
        return np.maximum(0, v)

    # Projection shortcut if Ws is given, identity otherwise
    I = x @ np.array(Ws, dtype=float) if Ws is not None else x

    h = relu(x @ W1)   # reduce
    h = relu(h @ W2)   # transform
    F = h @ W3         # expand, no ReLU before the add
    y = relu(F + I)
    return [[round(float(v), 4) for v in row] for row in y]