import os
import json
import random
from typing import List, Tuple
from src.network import Model
from src.utils.loss import weighted_binary_cross_entropy, loss_derivative

def load_processed_data(data_type: str) -> List[Tuple[List[List[float]], int]]:
    """
    Loads the pre-processed JSON images and their labels.
    
    Args:
        data_type: 'train' or 'test'.
        
    Returns:
        A list of tuples containing (image_data, label).
    """
    dataset: List[Tuple[List[List[float]], int]] = []
    base_path = f"data/processed/{data_type}"
    categories = {"normal": 0, "pneumonia": 1}
    
    for category, label in categories.items():
        folder_path = os.path.join(base_path, category)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r') as f:
                img_data = json.load(f)
                dataset.append((img_data, label))
                
    return dataset

def save_model(model: Model, filename: str = "best_pneumonia_model.json") -> None:
    """Saves weights and biases to a JSON file."""
    data = {
        "conv_filters": model.conv.filters,
        "conv_biases": model.conv.biases,
        "dense_weights": model.dense.weights,
        "dense_biases": model.dense.biases
    }
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"\n[SAVED] New best model saved to {filename}")

def progress_bar(current: int, total: int, length: int = 30) -> str:
    """ASCII Progress Bar."""
    percent = float(current) / total
    arrow = '-' * int(round(percent * length) - 1) + '>'
    spaces = ' ' * (length - len(arrow))
    return f"[{arrow + spaces}] {int(percent * 100)}%"

def evaluate(model: Model, test_data: List[Tuple[List[List[float]], int]]) -> float:
    """Evaluates the model on the test set."""
    correct = 0
    for image, label in test_data:
        prediction = model.forward(image)
        pred_label = 1 if prediction > 0.5 else 0
        if pred_label == label:
            correct += 1
    return (correct / len(test_data)) * 100

def train() -> None:
    nn = Model()
    train_data = load_processed_data("train")
    test_data = load_processed_data("test")
    
    epochs: int = 20
    learning_rate: float = 0.00001
    w_pos: float = 5.0 
    best_accuracy: float = 0.0

    # Early Stopping Setup
    patience: int = 3 
    epochs_without_improvement: int = 0

    print(f"Starting training on {len(train_data)} images...")

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss: float = 0.0
        correct_train: int = 0

        for i, (image, label) in enumerate(train_data):
            # Capture weights BEFORE update to check for change
            old_weight = nn.dense.weights[0][0]

            # Forward & Backward Pass
            prediction = nn.forward(image)
            loss = weighted_binary_cross_entropy(label, prediction, w_pos)
            total_loss += loss
            
            if (1 if prediction > 0.5 else 0) == label:
                correct_train += 1
            
            d_loss = loss_derivative(label, prediction, w_pos)
            nn.backward(d_loss, learning_rate)

            # Debug Prints every 100 images
            if i % 100 == 0:
                new_weight = nn.dense.weights[0][0]
                weight_delta = new_weight - old_weight
                
                print(f"\n--- DEBUG [Image {i}] ---")
                print(f"Target Label: {label}")
                print(f"Model Prediction: {prediction:.6f}")
                print(f"Loss Gradient (d_loss): {d_loss:.6f}")
                print(f"Weight Delta (Sample): {weight_delta:.10f}")
                
                # Check for "Dead" Model
                if abs(weight_delta) < 1e-15:
                    print("!!! WARNING: Weights are not updating. Gradients might be vanishing.")
                
                bar = progress_bar(i, len(train_data))
                print(f"Epoch {epoch+1} {bar} Loss: {loss:.4f}")

        # Results per Epoch
        train_acc = (correct_train / len(train_data)) * 100
        test_acc = evaluate(nn, test_data)
        avg_loss = total_loss / len(train_data)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Avg Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Early Stopping and Saving Logic
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_without_improvement = 0  # Reset counter on success
            save_model(nn)
            print(f"New best Test Accuracy! Saving model...")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in Test Acc. Patience: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOP] Stopping at Epoch {epoch+1} to prevent overfitting.")
            break
            
        print("-" * 35)

if __name__ == "__main__":
    train()