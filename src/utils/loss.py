import math

def weighted_binary_cross_entropy(y_true: int, y_pred: float, w_pos: float = 5.0) -> float:
    """
    Computes BCE loss with a penalty weight for positive cases.
    
    Args:
        y_true: Ground truth label (0 or 1).
        y_pred: Model output probability (0.0 to 1.0).
        w_pos: Multiplier for the loss when y_true is 1.
        
    Returns:
        The calculated scalar loss value.
    """
    # Subtract epsilon from 1 to prevent log(0) errors
    epsilon = 1e-15 
    y_pred = max(epsilon, min(1 - epsilon, y_pred))
    
    if y_true == 1:
        # Penalize missing a positive case more heavily
        return -w_pos * math.log(y_pred)
    else:
        return -math.log(1 - y_pred)

def loss_derivative(y_true: int, y_pred: float, w_pos: float = 5.0) -> float:
    """
    Computes the derivative of the weighted BCE loss w.r.t. y_pred.
    
    Args:
        y_true: Ground truth label (0 or 1).
        y_pred: Predicted probability.
        w_pos: The same weight used in the loss function.
    """

    # Subtract epsilon from 1 to prevent division by 0
    epsilon = 1e-15
    y_pred = max(epsilon, min(1 - epsilon, y_pred))
    
    if y_true == 1:
        grad = -w_pos / y_pred
    else:
        grad = 1 / (1 - y_pred)
    
    # GRADIENT CLIPPING: Limit the magnitude to 100.0 or 10.0
    return max(-100.0, min(100.0, grad))