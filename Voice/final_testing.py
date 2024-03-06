import numpy as np

# Function to evaluate the model
def evaluate_model(predictions, true_labels):
    # Compute accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Compute confusion matrix
    confusion_matrix = np.zeros((len(np.unique(true_labels)), len(np.unique(true_labels))))
    for i in range(len(predictions)):
        confusion_matrix[true_labels[i]][predictions[i]] += 1
    
    # Compute precision, recall, and F1-score for each class
    precision = np.zeros(len(np.unique(true_labels)))
    recall = np.zeros(len(np.unique(true_labels)))
    f1_score = np.zeros(len(np.unique(true_labels)))
    for i in range(len(np.unique(true_labels))):
        if np.sum(confusion_matrix[i]) == 0:
            precision[i] = 0
            recall[i] = 0
            f1_score[i] = 0
        else:
            precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[:, i])
            recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[i])
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    # Compute average precision, recall, and F1-score
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1_score = np.mean(f1_score)
    
    return accuracy, avg_precision, avg_recall, avg_f1_score

# Load predictions and true labels from file
def load_predictions(file_path):
    data = np.load(file_path)
    predictions = data['predictions']
    true_labels = data['true_labels']
    return predictions, true_labels

if __name__ == "__main__":
    # Load predictions and true labels
    predictions, true_labels = load_predictions("predictions.npz")
    
    # Evaluate the model
    accuracy, avg_precision, avg_recall, avg_f1_score = evaluate_model(predictions, true_labels)
    
    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1-score:", avg_f1_score)
