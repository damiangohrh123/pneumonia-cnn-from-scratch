import math

def huber_loss(y_true: int, y_pred: float, delta: float = 1.0) -> float:
    """
    Computes the Huber loss, which is less sensitive to outliers.

    Args:
        y_true: Ground truth label (0 or 1).
        y_pred: Model output probability (0.0 to 1.0).
        delta: The threshold at which to switch from quadratic to linear.

    Returns:
        The calculated scalar loss value.
    """
    error = y_true - y_pred
    if abs(error) <= delta:
        return 0.5 * error ** 2
    else:
        return delta * (abs(error) - 0.5 * delta)

def huber_loss_derivative(y_true: int, y_pred: float, delta: float = 1.0) -> float:
    """
    Computes the derivative of the Huber loss w.r.t. y_pred.

    Args:
        y_true: Ground truth label (0 or 1).
        y_pred: Predicted probability.
        delta: The threshold at which to switch from quadratic to linear.

    Returns:
        The gradient of the Huber loss.
    """
    error = y_true - y_pred
    if abs(error) <= delta:
        return -error
    else:
        return -delta if error > 0 else delta