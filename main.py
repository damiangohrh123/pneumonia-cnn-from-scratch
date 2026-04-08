import os
import json
import random
from typing import List, Tuple
from src.utils.data_loader import load_processed_data
from src.network import Model
from src.utils.loss import huber_loss, huber_loss_derivative

def augment_image(image_data: List[List[float]]) -> List[List[float]]:
    """Applies random transformations to the image."""

    # Create a deep copy so we don't ruin the original training set
    current_image = [row[:] for row in image_data] 
    
    rows = len(current_image)
    cols = len(current_image[0])
    
    # 50% chance to flip horizontally
    if random.random() > 0.5:
            current_image = [row[::-1] for row in current_image]
        
    # Small random pixel shift (Translation)
    # This prevents the model from relying on exact pixel coordinates
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    
    # Simple shift implementation
    new_image = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            old_r, old_c = r + shift_y, c + shift_x
            if 0 <= old_r < rows and 0 <= old_c < cols:
                new_image[r][c] = current_image[old_r][old_c]
                
    return new_image

def save_model(model: Model, filename: str = "best_pneumonia_model.json") -> None:
    data = {
        "conv1_filters": model.conv1.filters,
        "conv1_biases": model.conv1.biases,
        "conv2_filters": model.conv2.filters,
        "conv2_biases": model.conv2.biases,
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

    # Count the classes
    zeros = sum(1 for _, label in train_data if label == 0)
    ones = sum(1 for _, label in train_data if label == 1)

    print(f"Dataset Stats: Normal (0): {zeros} | Pneumonia (1): {ones}")
    
    epochs: int = 20
    learning_rate: float = 0.0025
    w_pos: float = 2.0 
    best_accuracy: float = 0.0

    # Early Stopping Setup
    patience: int = 5 
    epochs_without_improvement: int = 0

    print(f"Starting training on {len(train_data)} images...")

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss: float = 0.0
        correct_train: int = 0

        for i, (image, label) in enumerate(train_data):
            # Do augmentation on training images
            augmented_image = augment_image(image)

            # Capture weights BEFORE update to check for change
            old_weight = nn.dense.weights[0][0]

            # Forward & Backward Pass
            prediction = nn.forward(augmented_image)
            loss = huber_loss(label, prediction)
            total_loss += loss
            
            if (1 if prediction > 0.5 else 0) == label:
                correct_train += 1
            
            d_loss = huber_loss_derivative(label, prediction)

            # If label is 0, multiply the gradient by w_pos to make the penalty heavier.
            if label == 0: 
                d_loss *= w_pos
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