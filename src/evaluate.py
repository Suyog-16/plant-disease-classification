import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module,
             device: torch.device):
    """
    Evaluates the performance of a trained model on a given dataset.

    Args:
        model (torch.nn.Module): The trained neural network model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test/validation dataset.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device on which the model and data are loaded (e.g., "cuda" or "cpu").

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - accuracy (float): Overall classification accuracy.
            - precision (float): Weighted average precision across all classes.
            - recall (float): Weighted average recall across all classes.
            - f1 (float): Weighted F1-score, which balances precision and recall.
            - cm (numpy.ndarray): Confusion matrix representing class-wise performance.

    Example:
        accuracy, precision, recall, f1, cm = evaluate(model, test_loader, loss_fn, device)
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
    """
    # To store predictions and true labels for calculating the metrics
    all_preds = []
    all_labels = []
    
    # Variable to accumulate total correct predictions for accuracy
    correct_preds = 0
    total_preds = 0

    # Evaluate the model using the existing test_step function and collect predictions
    model.eval()  # Switch model to evaluation mode
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Move data to the device
            X, y = X.to(device), y.to(device)

            # Get model predictions using test_step
            test_pred_logits = model(X)

            # Get predicted class labels
            test_pred_labels = test_pred_logits.argmax(dim=1)
            
            # Store predictions and true labels
            all_preds.extend(test_pred_labels.cpu().numpy())  # move to CPU for metrics calculation
            all_labels.extend(y.cpu().numpy())
            
            # Calculate correct predictions for accuracy
            correct_preds += (test_pred_labels == y).sum().item()
            total_preds += len(y)

    # Calculate accuracy
    accuracy = correct_preds / total_preds

    # Calculate the precision, recall, and f1-score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm
