from typing import List, Any
from src.layers.convolution import ConvolutionLayer
from src.layers.pooling import MaxPoolingLayer
from src.layers.dense import DenseLayer, flatten, unflatten
from src.utils.activations import relu, relu_derivative, sigmoid

class Model:
    """
    A modular CNN container that chains layers together.
    Manages the sequential flow of data and gradients.
    """

    def __init__(self):
        """
        Initializes the specific architecture for Pneumonia detection.
        """
        # Feature Extraction Stage
        self.conv: ConvolutionLayer = ConvolutionLayer(num_filters=16, kernel_size=3)
        self.pool: MaxPoolingLayer = MaxPoolingLayer(size=2)
        
        # Classification Stage (Dense input size depends on pool output)
        # For a 64x64 input: 
        # Conv (3x3) -> 62x62
        # Pool (2x2) -> 31x31
        # 31 * 31 * 16 filters = 15376 total inputs
        self.dense: DenseLayer = DenseLayer(input_size=15376, output_size=1)
        
        # Cache for backpropagation
        self.last_conv_raw: List[List[List[float]]] = []

    def forward(self, image_2d: List[List[float]]) -> float:
        """
        Passes an image through the entire network to get a prediction.
        
        Args:
            image_2d: A 64x64 normalized grayscale image.
            
        Returns:
            The final sigmoid probability.
        """
        # 1. Convolution
        conv_out = self.conv.forward(image_2d)
        
        # 2. ReLU Activation
        # We must store the pre-activation values for backpropagation
        self.last_conv_raw = conv_out
        for i in range(len(conv_out)):
            for j in range(len(conv_out[0])):
                for f in range(len(conv_out[0][0])):
                    conv_out[i][j][f] = relu(conv_out[i][j][f])
        
        # 3. Max Pooling
        pool_out = self.pool.forward(conv_out)
        
        # 4. Flattening
        flat_out = flatten(pool_out)
        
        # 5. Dense Layer & Sigmoid
        dense_out = self.dense.forward(flat_out)
        prediction = sigmoid(dense_out[0])
        
        self.last_prediction = prediction

        return prediction

    def backward(self, d_L_d_pred: float, learning_rate: float) -> None:
        """
        Propagates the error signal backward through all layers.
        
        Args:
            d_L_d_pred: The initial gradient from the loss function.
            learning_rate: The speed of optimization.
        """
        # 1. Output Activation Backpropagation
        # The gradient must pass through the Sigmoid derivative: s(z) * (1 - s(z))
        # This converts the loss gradient into the error signal for the Dense layer.
        sig_grad = (self.last_prediction * (1.0 - self.last_prediction)) + 0.01  # Add small value (0.01) to prevent zero gradient
        d_L_d_dense_out = d_L_d_pred * sig_grad

        # 2. Dense Layer Backpropagation
        # Updates weights/biases in the Dense layer and returns gradient w.r.t inputs.
        d_L_d_flat = self.dense.backward([d_L_d_dense_out], learning_rate)
        
        # 3. Unflatten back to 3D (31x31x16)
        d_L_d_pool = unflatten(d_L_d_flat, [31, 31, 16])
        
        # 4. Pooling Backward
        d_L_d_relu = self.pool.backward(d_L_d_pool)
        
        # 5. Leaky ReLU Backward
        # Instead of setting gradients to 0 for negative values, we multiply by the derivative (0.01) to keep the 'dead' neurons learning.
        for i in range(len(d_L_d_relu)):
            for j in range(len(d_L_d_relu[0])):
                for f in range(len(d_L_d_relu[0][0])):
                    # Use the stored raw convolution values to find the gradient
                    raw_val = self.last_conv_raw[i][j][f]
                    d_L_d_relu[i][j][f] *= relu_derivative(raw_val)
        
        # 6. Convolution Backward
        self.conv.backward(d_L_d_relu, learning_rate)