from typing import List, Tuple

class MaxPoolingLayer:
    """
    Reduces spatial dimensions by taking the maximum value in 2x2 windows.
    Helps the model become invariant to small shifts in the X-ray image.
    """

    def __init__(self, size: int = 2):
        """
        Initializes the pooling window size.
        
        Args:
            size: The dimensions of the square pooling window (default is 2x2).
        """
        self.size: int = size

        # Cache for backpropagation (storing the forward pass input)
        self.last_input: List[List[List[float]]] = []

    def forward(self, input_3d: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Downsamples the feature maps by selecting the maximum value in each window.
        
        Args:
            input_3d: The 3D feature maps from the previous activation layer.
            
        Returns:
            A 3D list with reduced height and width.
        """
        # Cache input for the backward pass to identify 'winning' pixel coordinates
        self.last_input = input_3d

        in_h: int = len(input_3d)
        in_w: int = len(input_3d[0])
        f_count: int = len(input_3d[0][0]) # Filter count

        # Output dimensions: (H // size) and (W // size)
        out_h: int = in_h // self.size
        out_w: int = in_w // self.size
        
        # Initialize 3D output: [row][col][filter]
        output: List[List[List[float]]] = [
            [[0.0 for _ in range(f_count)] for _ in range(out_w)] 
            for _ in range(out_h)
        ]

        # Iterate through each filter in the layer
        for f in range(f_count):
            # Slide the pooling window vertically across the image (rows)
            for i in range(out_h):
                # Slide the pooling window horizontally across the image (columns)
                for j in range(out_w):

                    # Identify the maximum value within the pooling window
                    max_val: float = -float('inf')
                    for m in range(self.size):
                        for n in range(self.size):

                            # Calculate the input coordinate based on stride (i * size)
                            val: float = input_3d[i * self.size + m][j * self.size + n][f]
                            if val > max_val:
                                max_val = val
                    
                    # Store the highest signal found.
                    # Resulting shape is [row][col][filter] (Channel-Last format)
                    output[i][j][f] = max_val
        return output

    def backward(self, d_L_d_out: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Routes the gradient back to the exact location of the maximum element.
        
        Args:
            d_L_d_out: The gradient of the loss from the subsequent layer.
            
        Returns:
            A 3D list of gradients with the same shape as the original input.
        """
        in_h: int = len(self.last_input)
        in_w: int = len(self.last_input[0])
        f_count: int = len(self.last_input[0][0]) # Filter count

        # Dimensions of the incoming gradient (the shrunk version)
        out_h: int = len(d_L_d_out)
        out_w: int = len(d_L_d_out[0])
        
        # Initialize gradient map with zeros. Shape matches the original input.
        d_L_d_input: List[List[List[float]]] = [
            [[0.0 for _ in range(f_count)] for _ in range(in_w)] for _ in range(in_h)
        ]
        
        # Iterate through each filter in the layer
        for f in range(f_count):
            # Slide the pooling window vertically across the shrunk gradient map
            for i in range(out_h):
                # Slide the pooling window horizontally across the shrunk gradient map
                for j in range(out_w):

                    # Re-identify the maximum position (Argmax) to route the error
                    max_val: float = -float('inf')
                    max_pos: Tuple[int, int] = (0, 0)
                    
                    # Scan the local size x size neighborhood in the cached forward input
                    for m in range(self.size):
                        for n in range(self.size):

                            # Calculate the absolute input coordinate using the stride (i * size)
                            curr_val: float = self.last_input[i * self.size + m][j * self.size + n][f]
                            if curr_val > max_val:
                                max_val = curr_val

                                # Store the specific row and column of the winner
                                max_pos = (i * self.size + m, j * self.size + n)
                    
                    # Route the gradient: Only the winner receives the incoming error signal
                    # All other pixels in this 2x2 neighborhood remain zero
                    d_L_d_input[max_pos[0]][max_pos[1]][f] = d_L_d_out[i][j][f]
        
        return d_L_d_input