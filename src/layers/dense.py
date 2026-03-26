import math
import random
from typing import List

class DenseLayer:
    """
    The final classification head of the network.
    Connects every input from the flattened feature map to the output neuron.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initializes weights and biases for the fully connected layer.
        
        Args:
            input_size: Number of elements in the flattened input vector.
            output_size: Number of output neurons (1 for binary classification).
        """
        self.input_size: int = input_size
        self.output_size: int = output_size
        
        # Initialize weights: List[output_neuron][input_connection]
        # Calculate the 'He' scale factor
        scale = math.sqrt(2.0 / input_size)

        self.weights: List[List[float]] = [
            [random.gauss(0, scale) for _ in range(input_size)] 
            for _ in range(output_size)
        ]
        self.biases: List[float] = [0.01 for _ in range(output_size)]
        
        # Cache for backpropagation
        self.last_input: List[float] = []

    def forward(self, input_vector: List[float]) -> List[float]:
        """
        Calculates the weighted sum for each output neuron.
        
        Args:
            input_vector: The 1D flattened list of features.
            
        Returns:
            The raw logits (pre-activation values).
        """
        self.last_input = input_vector
        outputs: List[float] = [0.0] * self.output_size
        
        for i in range(self.output_size):
            # z = sum(w * x) + b
            summ: float = 0.0
            for j in range(self.input_size):
                summ += input_vector[j] * self.weights[i][j]
            outputs[i] = summ + self.biases[i]
            
        return outputs

    def backward(self, d_L_d_out: List[float], learning_rate: float) -> List[float]:
        """
        Updates weights/biases and passes the error back to the flattened vector.
        
        Args:
            d_L_d_out: Gradient of loss with respect to the output neurons.
            learning_rate: Factor to scale weight updates.
            
        Returns:
            The gradient with respect to the input vector (to pass back to Pooling).
        """
        # Initialize the input gradient vector (the error signal to pass back)
        d_L_d_input: List[float] = [0.0] * self.input_size
        
        for i in range(self.output_size):
            # Error signal for the current output neuron (i)
            grad: float = d_L_d_out[i]
            
            for j in range(self.input_size):
                # Calculate gradient w.r.t. input (Error * Weight)
                # This 'blame' is accumulated for feature j to be passed to the Pooling layer.
                d_L_d_input[j] += grad * self.weights[i][j]
                
                # Calculate gradient w.r.t. weight (Error * Input)
                # This identifies how much weight [i][j] contributed to the specific error.
                weight_gradient: float = grad * self.last_input[j]

                # Update weight using Stochastic Gradient Descent (SGD)
                self.weights[i][j] -= learning_rate * weight_gradient
            
            # Update bias
            self.biases[i] -= learning_rate * grad
            
        return d_L_d_input

def flatten(input_3d: List[List[List[float]]]) -> List[float]:
    """
    Converts a 3D feature map into a 1D vector for the Dense layer.
    
    Args:
        input_3d: The 3D output from the last pooling layer.
        
    Returns:
        A 1D list containing all feature map values.
    """
    flattened: List[float] = []
    # Loop order must match the backward pass logic (Row -> Col -> Filter)
    for row in input_3d:
        for col in row:
            for val in col:
                flattened.append(val)
    return flattened

def unflatten(d_L_d_flat: List[float], shape: List[int]) -> List[List[List[float]]]:
    """
    Reshapes a 1D gradient back into 3D for convolutional/pooling backpropagation.
    
    Args:
        d_L_d_flat: The 1D gradient vector from the Dense layer.
        shape: The target [height, width, filters] shape.
        
    Returns:
        The reconstructed 3D gradient map.
    """
    h, w, f = shape
    output: List[List[List[float]]] = [
        [[0.0 for _ in range(f)] for _ in range(w)] for _ in range(h)
    ]
    
    idx: int = 0
    for i in range(h):
        for j in range(w):
            for k in range(f):
                output[i][j][k] = d_L_d_flat[idx]
                idx += 1
    return output