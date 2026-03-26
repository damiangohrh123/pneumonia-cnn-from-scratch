import random
from typing import List

class ConvolutionLayer:
    """
    Performs a 2D convolution operation on a single-channel (grayscale) image.
    Uses a set of learnable kernels to extract features.
    """

    def __init__(self, num_filters: int, kernel_size: int):
        """
        Initializes filters and biases with small random values.
        
        Args:
            num_filters: How many different feature maps to produce.
            kernel_size: The width/height of the square filter (e.g., 3 for 3x3).
        """
        self.num_filters: int = num_filters
        self.k: int = kernel_size
        
        # Initialize filters: List[filter_index][row][col]
        self.filters: List[List[List[float]]] = [
            [[random.uniform(-0.1, 0.1) for _ in range(kernel_size)] 
             for _ in range(kernel_size)] for _ in range(num_filters)
        ]
        self.biases: List[float] = [0.0] * num_filters
        
        # Cache for backpropagation
        self.last_input: List[List[float]] = []

    def forward(self, input_2d: List[List[float]]) -> List[List[List[float]]]:
        """
        Scans the input image with kernels to produce feature maps.
        
        Args:
            input_2d: The grayscale image or previous feature map.
            
        Returns:
            A 3D list where the 3rd dimension represents different filters.
        """
        self.last_input = input_2d
        in_h: int = len(input_2d)
        in_w: int = len(input_2d[0])
        
        # Output dimensions: (N - K + 1)
        out_h: int = in_h - self.k + 1
        out_w: int = in_w - self.k + 1
        
        # Initialize 3D output: [row][col][filter]
        output: List[List[List[float]]] = [
            [[0.0 for _ in range(self.num_filters)] for _ in range(out_w)] 
            for _ in range(out_h)
        ]

        # Iterate through each filter in the layer
        for f in range(self.num_filters):
            # Slide the filter vertically across the image (rows)
            for i in range(out_h):
                # Slide the filter horizontally across the image (columns)
                for j in range(out_w):
                    
                    # Iterate through the kernel's area and sum the weights into a scalar
                    summ: float = 0.0
                    for m in range(self.k):
                        for n in range(self.k):
                            summ += input_2d[i + m][j + n] * self.filters[f][m][n]
                    
                    # Store the result in the feature map and add the learned bias
                    # Resulting shape is [row][col][filter] (Channel-Last format)
                    output[i][j][f] = summ + self.biases[f]
        return output

    def backward(self, d_L_d_out: List[List[List[float]]], learning_rate: float) -> List[List[float]]:
        """
        Calculates gradients for kernels and passes error back to input with L2 regularization.
        
        Args:
            d_L_d_out: The gradient of the loss with respect to this layer's output.
            learning_rate: Factor to scale the weight updates.
            
        Returns:
            The gradient with respect to the input image (to pass to previous layers).
        """
        in_h: int = len(self.last_input)
        in_w: int = len(self.last_input[0])
        out_h: int = len(d_L_d_out)
        out_w: int = len(d_L_d_out[0])

        # Strength of the weight penalty
        l2_lambda: float = 0.001
        
        # Initialize gradient storage for filters [filter][row][col]
        d_L_d_filters: List[List[List[float]]] = [
            [[0.0 for _ in range(self.k)] for _ in range(self.k)] 
            for _ in range(self.num_filters)
        ]

        # Initialize gradient storage for the input image [row][col]
        d_L_d_input: List[List[float]] = [[0.0 for _ in range(in_w)] for _ in range(in_h)]

        # 1. Gradient Accumulation Phase
        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    # Local gradient from the next layer (Pooling/ReLU)
                    grad: float = d_L_d_out[i][j][f]
                    
                    # Backpropagate error through the convolution operation
                    for m in range(self.k):
                        for n in range(self.k):
                            # dL/dW = dL/dOut * Input
                            d_L_d_filters[f][m][n] += grad * self.last_input[i + m][j + n]

                            # dL/dIn = dL/dOut * Weight
                            d_L_d_input[i + m][j + n] += grad * self.filters[f][m][n]

        # 2. Update Phase (SGD + L2 Regularization)
        for f in range(self.num_filters):
            # Calculate Bias Gradient: The sum of all output errors for this filter
            bias_gradient: float = 0.0
            for i in range(out_h):
                for j in range(out_w):
                    bias_gradient += d_L_d_out[i][j][f]

            # Adjust bias
            self.biases[f] -= learning_rate * bias_gradient

            # Adjust kernel weights using L2 Weight Decay
            for m in range(self.k):
                for n in range(self.k):
                    # Calculate the penalty based on the current weight value
                    l2_penalty: float = l2_lambda * self.filters[f][m][n]
                    
                    # Update: New_W = Old_W - (LR * (Gradient + Penalty))
                    self.filters[f][m][n] -= learning_rate * (d_L_d_filters[f][m][n] + l2_penalty)

        # Return input gradients to update previous layers (or the original image)
        return d_L_d_input