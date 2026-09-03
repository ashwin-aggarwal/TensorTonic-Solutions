import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here

    labels = np.asarray(y_true)
    predictions = np.asarray(y_score)

    losses = np.maximum(0.0, margin - (predictions * labels))
    if reduction == "mean":
        return float(np.mean(losses)) 

    return float(sum(losses))