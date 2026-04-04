import math

def sigmoid(z: float) -> float:
    """
    Computes the sigmoid activation function.
    
    Args:
        z: The raw input (logit) from a linear layer.
        
    Returns:
        A value between 0 and 1.
    """
    # Clamping z to avoid overflow in math.exp
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(output_val: float) -> float:
    """
    Computes the gradient of the sigmoid function.
    
    Args:
        output_val: The result of sigmoid(z).
        
    Returns:
        The local gradient for backpropagation.
    """
    return output_val * (1 - output_val)

def relu(z: float) -> float:
    """
    Leaky ReLU: Returns z if positive, else a tiny fraction of z.
    Prevents 'dead' neurons in pure Python implementations.
    """
    return z if z > 0 else 0.01 * z

def relu_derivative(z: float) -> float:
    """
    Computes the gradient of Leaky ReLU.

    Args:
        z: The raw input to the ReLU function.
        
    Returns:
        The local gradient for backpropagation.
    """
    return 1.0 if z > 0 else 0.01