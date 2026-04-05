import random
from typing import List, Union, Any

class ConvolutionLayer:
    """
    Performs a 3D convolution operation. 
    Can handle single-channel images (2D) or multi-channel feature maps (3D).
    """

    def __init__(self, num_filters: int, kernel_size: int):
        """
        Sets the filter configuration. Actual weight initialization is 
        deferred to the first forward pass once input depth is known.
        
        Args:
            num_filters: How many different feature maps to produce.
            kernel_size: The width/height of the square filter (e.g., 3 for 3x3).
        """
        self.num_filters: int = num_filters
        self.k: int = kernel_size
        
        # Structure: [filter_index][channel_index][row][col]
        self.filters: List[List[List[List[float]]]] = []
        self.biases: List[float] = [0.0] * num_filters
        
        # Cache to store the input for backpropagation
        self.last_input: Union[List[List[float]], List[List[List[float]]]] = []
    
    def _init_filters(self, depth: int):
        """
        Initializes weights using He (Kaiming) Normal Initialization.
        
        This prevents the 'Vanishing Gradient' problem by scaling weights 
        based on the number of input connections (fan-in).
        """
        # fan-in = (kernel_width * kernel_height * input_channels)
        # He initialization scale: sqrt(2 / fan-in)
        he_scale = (2.0 / (self.k * self.k * depth)) ** 0.5

        # Apply an additional "Safety Factor" (0.1 or 0.01) to prevent immediate Sigmoid saturation
        tuned_scale = he_scale * 0.1
    
        self.filters = [
            [[[random.gauss(0, tuned_scale) for _ in range(self.k)]  # Dimension 4: kernel row
                for _ in range(self.k)]                        # Dimension 3: kernel column                                               
                for _ in range(depth)]                         # Dimension 2: input channels        
                for _ in range(self.num_filters)               # Dimension 1: number of filters
        ]

    def forward(self, input_data: Any) -> List[List[List[float]]]:
        """
        Scans the input (image or feature map) with 3D kernels.
        
        Args:
            input_data: A 2D grayscale image or a 3D feature map [row][col][channel].
            
        Returns:
            A 3D list where the 3rd dimension represents the new filter set.
        """
        self.last_input = input_data
        
        # 1. Determine if input is 2D (Image) or 3D (Feature Map)
        # This allows the layer to be used at the start (1 channel) or deep in the stack (e.g., 16 channels).
        if isinstance(input_data[0][0], list):
            # If it's 3D [row][col][channel] (Channel-Last format)
            in_h, in_w, in_d = len(input_data), len(input_data[0]), len(input_data[0][0])
        else:
            # If it's 2D [row][col]. We reshape to 3D for consistent math across layers.
            in_h, in_w = len(input_data), len(input_data[0])
            in_d = 1
            input_data = [[ [pixel] for pixel in row] for row in input_data]

        # 2. Delayed Weight Initialization. We wait until the first pass to know the 'in_d' (depth) of our filters.
        if not self.filters:
            self._init_filters(in_d)

        # 3. Calculate Output Dimensions. Standard convolution output formula: (Input_Dim - Kernel_Size + 1)
        out_h, out_w = in_h - self.k + 1, in_w - self.k + 1
        
        # Initialize 3D output: [height (out_h)][width (out_w)][channels (num_filters)]
        output = [[[0.0 for _ in range(self.num_filters)] for _ in range(out_w)] for _ in range(out_h)]

        # 4. The 3D Convolution Operation
        for f in range(self.num_filters):            # Loop over each filter
            for i in range(out_h):                   # Loop over output height
                for j in range(out_w):               # Loop over output width
                    summ = 0.0
                    for c in range(in_d):            # Loop over input channels
                        for m in range(self.k):      # Loop over kernel height
                            for n in range(self.k):  # Loop over kernel width
                                # weight format: [filter][channel][row][col]
                                summ += input_data[i + m][j + n][c] * self.filters[f][c][m][n]
                    
                    # Store result and add the unique bias for this filter
                    output[i][j][f] = summ + self.biases[f]
        return output

    def backward(self, d_L_d_out: List[List[List[float]]], learning_rate: float) -> Any: 
        """
        Calculates gradients for kernels and passes error back to input with L2 regularization.
        Handles both 2D (initial image) and 3D (feature map) inputs.
        
        Args:
            d_L_d_out: The gradient of the loss with respect to this layer's output.
            learning_rate: Factor to scale the weight updates.
            
        Returns:
            The gradient with respect to the input (to pass to previous layers).
        """
        # Check if the pixel is a single value (2D image) or a list of features (3D map)
        is_2d_input = not isinstance(self.last_input[0][0], list)

        # Standardize input to 3D: If 2D [row][col], wrap pixels to create [row][col][1 channel]
        curr_input = [[ [p] for p in row] for row in self.last_input] if is_2d_input else self.last_input
        
        in_h, in_w, in_d = len(curr_input), len(curr_input[0]), len(curr_input[0][0])
        out_h, out_w = len(d_L_d_out), len(d_L_d_out[0])
        
        # L2 Regularization Hyperparameter (prevents overfitting by penalizing large weights)
        l2_lambda = 0.0001 

        # Initialize Gradients
        # Empty 4D list to store weight gradient updates: [filter][channel][row][col]
        d_L_d_filters = [[[[0.0 for _ in range(self.k)] for _ in range(self.k)] for _ in range(in_d)] for _ in range(self.num_filters)]

        # Empty 3D list to store error for the previous layer: [height][width][depth]
        d_L_d_input = [[[0.0 for _ in range(in_d)] for _ in range(in_w)] for _ in range(in_h)]

        # Backpropagation through Convolution
        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    # The incoming gradient from the next layer for this specific filter/pixel
                    grad = d_L_d_out[i][j][f]
                    
                    # Clipping to prevent explosion (Optional but can help with stability)
                    grad = max(-5.0, min(5.0, grad))

                    for c in range(in_d):
                        for m in range(self.k):
                            for n in range(self.k):
                                # dL/dW: How much this specific weight contributed to the error
                                d_L_d_filters[f][c][m][n] += grad * curr_input[i + m][j + n][c]
                                
                                # dL/dX: Pass error to the previous layer
                                d_L_d_input[i + m][j + n][c] += grad * self.filters[f][c][m][n]

        # Update Weights and Biases
        for f in range(self.num_filters):
            # Sum up total gradient for this filter then update bias
            bias_grad = sum(d_L_d_out[i][j][f] for i in range(out_h) for j in range(out_w))
            self.biases[f] -= learning_rate * max(-1.0, min(1.0, bias_grad))
            
            # Update Filters
            for c in range(in_d):
                for m in range(self.k):
                    for n in range(self.k):
                        # Regularization: Apply L2 "Weight Decay" to prevent overfitting
                        total_grad = d_L_d_filters[f][c][m][n] + (l2_lambda * self.filters[f][c][m][n])
                        
                        # Move weights in opposite direction of error (with a safety cap)
                        self.filters[f][c][m][n] -= learning_rate * max(-2.0, min(2.0, total_grad))

        # Return error map matched to the original input shape
        if is_2d_input:
            # Flatten 3D [H][W][1] back to 2D [H][W] for the previous layer
            return [[channel[0] for channel in row] for row in d_L_d_input]
        return d_L_d_input  # Keep as 3D if the input was originally a feature map