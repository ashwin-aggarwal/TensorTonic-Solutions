import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = X.shape          # n = samples, m = features

    w = np.zeros(m)         # one weight per feature, 1-D
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w + b)   # (n,) predicted probabilities
        diff = (p - y) / n        # (n,) dL/dz, already averaged

       
        dw =  X.T @ diff #X.T is mxn and diff is nx1
        w -= lr*dw

        db = diff.sum()
        b -= lr*db
        
    return w, float(b)