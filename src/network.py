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
        # Convolutional Layer 1: 16 filters of size 3x3
        self.conv1 = ConvolutionLayer(num_filters=16, kernel_size=3)
        self.pool1 = MaxPoolingLayer(size=2)

        # Convolutional Layer 2: 32 filters of size 3x3
        # Processes the 16 channels from the previous layer
        self.conv2 = ConvolutionLayer(num_filters=32, kernel_size=3)
        self.pool2 = MaxPoolingLayer(size=2)
        
        # Classification Stage (Dense input size depends on pool output)
        # For a 64x64 input: 
        # Conv1 (3x3) -> 62x62
        # Pool1 (2x2) -> 31x31
        # Conv2 (3x3) -> 29x29
        # Pool2 (2x2) -> 14x14
        # 14 * 14 * 32 filters = 6272 total inputs
        self.dense = DenseLayer(input_size=6272, output_size=1)
        
        # Cache for backpropagation (storing raw values for both layers)
        self.last_conv1_raw: List[List[List[float]]] = []
        self.last_conv2_raw: List[List[List[float]]] = []
        self.last_prediction: float = 0.0

    def forward(self, image_2d: List[List[float]]) -> float:
        """
        Passes an image through the entire network to get a prediction.
        
        Args:
            image_2d: A 64x64 normalized grayscale image.
            
        Returns:
            The final sigmoid probability.
        """
        # --------- BLOCK 1: Feature Extraction (Low-level features) ---------
        # Convolution: output shape is (16, 62, 62) -> 16 feature maps, each 62x62 pixels
        conv1_out = self.conv1.forward(image_2d)
        self.last_conv1_raw = [[ [f for f in row] for row in channel] for channel in conv1_out]  # Save to cache for backpropagation

        # Apply the ReLU function to every single data point
        for i in range(len(conv1_out)):
            for j in range(len(conv1_out[0])):
                for f in range(len(conv1_out[0][0])):
                    conv1_out[i][j][f] = relu(conv1_out[i][j][f])
        
        # Max Pooling
        pool1_out = self.pool1.forward(conv1_out)

        # --------- BLOCK 2: Feature Extraction (Mid-level features) ---------
        # Convolution: output shape is (32, 29, 29) -> 32 feature maps, each 29x29 pixels
        conv2_out = self.conv2.forward(pool1_out)
        self.last_conv2_raw = [[ [f for f in row] for row in channel] for channel in conv2_out]  # Save to cache for backpropagation

        # Apply the ReLU function to every single data point
        for i in range(len(conv2_out)):
            for j in range(len(conv2_out[0])):
                for f in range(len(conv2_out[0][0])):
                    conv2_out[i][j][f] = relu(conv2_out[i][j][f])

        # Max Pooling
        pool2_out = self.pool2.forward(conv2_out)
        
        # Flattening (Now 14x14x32)
        flat_out = flatten(pool2_out)
        
        # Dense Layer & Sigmoid
        dense_out = self.dense.forward(flat_out)
        dense_out = [max(-10, min(10, x)) for x in dense_out]  # Clamp inputs to sigmoid
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
        # The gradient must pass through the Sigmoid derivative: s(z) * (1 - s(z))
        sig_grad = (self.last_prediction * (1.0 - self.last_prediction)) + 1e-6  # Small epsilon to prevent zero gradient

        # Take the initial error and multiply by the sigmoid gradient
        d_L_d_dense_out = d_L_d_pred * sig_grad

        # Dense Layer Backpropagation
        d_L_d_flat = self.dense.backward([d_L_d_dense_out], learning_rate)
        
        # Unflatten back to 3D (14x14x32)
        d_L_d_pool2 = unflatten(d_L_d_flat, [14, 14, 32])
        
        # Second Pooling & Leaky ReLU Backward
        d_L_d_relu2 = self.pool2.backward(d_L_d_pool2)
        for i in range(len(d_L_d_relu2)):
            for j in range(len(d_L_d_relu2[0])):
                for f in range(len(d_L_d_relu2[0][0])):
                    raw_val = self.last_conv2_raw[i][j][f]
                    d_L_d_relu2[i][j][f] *= relu_derivative(raw_val)
        
        # Second Convolution Backward. Returns the gradient for the output of Pool1 (31x31x16)
        d_L_d_pool1 = self.conv2.backward(d_L_d_relu2, learning_rate)

        # First Pooling & Leaky ReLU Backward
        d_L_d_relu1 = self.pool1.backward(d_L_d_pool1)
        for i in range(len(d_L_d_relu1)):
            for j in range(len(d_L_d_relu1[0])):
                for f in range(len(d_L_d_relu1[0][0])):
                    raw_val = self.last_conv1_raw[i][j][f]
                    d_L_d_relu1[i][j][f] *= relu_derivative(raw_val)
        
        # First Convolution Backward
        self.conv1.backward(d_L_d_relu1, learning_rate)