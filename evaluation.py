import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import load_processed_data
from src.network import Model
import json

def run_full_evaluation(model, test_data):
    """
    Runs the model on the test set, calculates clinical metrics, 
    and displays a professional Confusion Matrix.
    """
    # Initialize counts for the confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    all_probs = []
    all_labels = []

    print(f"Starting Evaluation on {len(test_data)} Test Samples")

    for image, label in test_data:
        # Pass image through model and extract the predicted probability
        output = model.forward(image)
        prediction_prob = output[0] if isinstance(output, list) else output
        
        # Binary Classification (Threshold = 0.5)
        prediction = 1 if prediction_prob >= 0.5 else 0

        # Categorize for Confusion Matrix
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1
            
        all_probs.append(prediction_prob)
        all_labels.append(label)

    # Calculate Metrics
    accuracy = (tp + tn) / len(test_data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nEvaluation Results:")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1_score:.4f}")

    # Plotting the Confusion Matrix with Matplotlib
    conf_matrix = np.array([[tn, fp], 
                            [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
    plt.colorbar()

    classes = ['Normal', 'Pneumonia']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Annotate the matrix squares with counts
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label (Ground Truth)')
    plt.xlabel('Predicted Label (Model)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the test data
    test_data = load_processed_data("test")
    
    # Reconstruct the Model Architecture
    nn_v6 = Model() 
    
    # Load the weights model_v6_augmented.json file
    filename = "models/model_v6_augmented.json"
    
    try:
        with open(filename, "r") as f:
            weights = json.load(f)
            
        # Inject the saved weights into the model layers
        nn_v6.conv1.filters = weights["conv1_filters"]
        nn_v6.conv1.biases = weights["conv1_biases"]
        nn_v6.conv2.filters = weights["conv2_filters"]
        nn_v6.conv2.biases = weights["conv2_biases"]
        nn_v6.dense.weights = weights["dense_weights"]
        nn_v6.dense.biases = weights["dense_biases"]
        
        # Run the evaluation
        run_full_evaluation(nn_v6, test_data)

    except FileNotFoundError:
        print(f"Error: '{filename}' not found in the current directory.")