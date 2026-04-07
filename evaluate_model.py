import json
import os
import sys
from typing import List, Tuple
from src.network import Model
from src.utils.loss import huber_loss

"""
NOTE: This evaluation script is designed specifically for Model Architecture v5.0+ (Dual Convolutional Layers). 

To evaluate v1-v4 (Single Conv), you must manually revert the Model() class 
in src/network.py to the older architecture to avoid shape mismatch errors.
"""

def load_processed_data(data_type: str) -> List[Tuple[List[List[float]], int]]:
    """Loads JSON-formatted images and pairs them with numeric labels."""
    dataset = []
    base_path = f"data/processed/{data_type}"
    categories = {"normal": 0, "pneumonia": 1}

    for category, label in categories.items():
        folder_path = os.path.join(base_path, category)

        # Skip if the specific category folder is missing
        if not os.path.exists(folder_path): continue

        # Iterate through every processed image in the category folder
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r') as f:
                # Load the pixel array (2D or 3D list) from JSON
                img_data = json.load(f)

                # Store as a (Feature, Label)
                dataset.append((img_data, label))
    return dataset

def run_final_eval(model_path: str):
    # 1. Initialize a blank model
    nn = Model()
    
    # 2. Load the weights from the model.JSON file
    print(f"Loading weights from {model_path}...")
    try:
        with open(model_path, 'r') as f:
            weights = json.load(f)
            
        # Manually inject the saved weights back into the layers
        nn.conv1.filters = weights["conv1_filters"]
        nn.conv1.biases = weights["conv1_biases"]
        nn.conv2.filters = weights["conv2_filters"]
        nn.conv2.biases = weights["conv2_biases"]
        nn.dense.weights = weights["dense_weights"]
        nn.dense.biases = weights["dense_biases"]
        print("Weights successfully loaded!")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load the Test Data
    print("Loading test data...")
    test_data = load_processed_data("test")
    
    # 4. Run the Evaluation
    correct = 0
    total_loss = 0.0
    total = len(test_data)
    
    print(f"Evaluating on {total} images...")
    for i, (image, label) in enumerate(test_data):
        prediction = nn.forward(image)

        # Calculate loss
        total_loss += huber_loss(label, prediction)

        # Calculate accuracy
        pred_label = 1 if prediction > 0.5 else 0
        if pred_label == label:
            correct += 1
        
        # Simple progress indicator
        if i % 50 == 0:
            print(f"Processed {i}/{total}...")

    final_acc = (correct / total) * 100
    avg_loss = total_loss / total

    print("\n" + "="*30)
    print(f"FINAL TEST ACCURACY: {final_acc:.2f}%")
    print(f"AVERAGE LOSS: {avg_loss:.4f}")
    print("="*30)

if __name__ == "__main__":
    # Check if an argument was even provided
    if len(sys.argv) < 2:
        print("\n[ERROR] No model file provided.")
        print("Usage: python main.py <path_to_model.json>")
        sys.exit(1)

    target_path = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(target_path):
        print(f"\n[ERROR] The file '{target_path}' could not be found.")
        print("Please check the filename and try again.")
        sys.exit(1)

    # Everything is correct, run the evaluation
    run_final_eval(target_path)