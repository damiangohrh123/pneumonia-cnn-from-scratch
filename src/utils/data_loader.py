import os
import json
from typing import List, Tuple

class ImageProcessor:
    """
    Handles the transformation of raw pixel data into a format suitable for the CNN.
    Implements resizing and normalization without external libraries.
    """

    @staticmethod
    def resize(image: List[List[int]], target_h: int, target_w: int) -> List[List[float]]:
        """
        Resizes a 2D image using Nearest Neighbor interpolation.
        
        Args:
            image: The original 2D list of pixel intensities (0-255).
            target_h: The desired output height (e.g., 64).
            target_w: The desired output width (e.g., 64).
            
        Returns:
            A resized 2D list of pixels, normalized between 0.0 and 1.0.
        """
        in_h: int = len(image)
        in_w: int = len(image[0])
        
        # Initialize the output grid
        resized: List[List[float]] = [[0.0 for _ in range(target_w)] for _ in range(target_h)]
        
        # Calculate scaling factors
        row_ratio: float = in_h / target_h
        col_ratio: float = in_w / target_w
        
        for i in range(target_h):
            for j in range(target_w):
                # Map the target coordinate back to the source coordinate
                source_i: int = int(i * row_ratio)
                source_j: int = int(j * col_ratio)
                
                # Pixel Normalization (0-255 -> 0.0-1.0)
                resized[i][j] = image[source_i][source_j] / 255.0
                
        return resized

    @staticmethod
    def grayscale_convert(r: int, g: int, b: int) -> int:
        """
        Converts an RGB pixel to Grayscale using the luminance formula.
        
        Args:
            r: Red channel (0-255).
            g: Green channel (0-255).
            b: Blue channel (0-255).
            
        Returns:
            A single grayscale intensity value.
        """
        # Luminance formula
        return int(0.299 * r + 0.587 * g + 0.114 * b)

def load_processed_data(data_type: str) -> List[Tuple[List[List[float]], int]]:
    dataset: List[Tuple[List[List[float]], int]] = []
    base_path = f"data/processed/{data_type}"
    categories = {"normal": 0, "pneumonia": 1} # Ensure case matches your folders!
    
    for category, label in categories.items():
        folder_path = os.path.join(base_path, category)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r') as f:
                img_data = json.load(f)
                dataset.append((img_data, label))
    return dataset