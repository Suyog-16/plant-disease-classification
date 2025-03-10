import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    """
    Performs a single training step for one epoch.

    Args:
        model (torch.nn.Module): The CNN model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        loss_fn (torch.nn.Module): Loss function to compute model loss (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters (e.g., Adam, SGD).

    Returns:
        tuple: A tuple containing:
            - train_loss (float): The average training loss over all batches.
            - train_acc (float): The average training accuracy over all batches.

    """
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    """
       Performs a single evaluation step on a given dataset.

    Args:
        model (torch.nn.Module): The trained CNN model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test/validation dataset.
        loss_fn (torch.nn.Module): Loss function to compute the model’s loss (e.g., CrossEntropyLoss).

    Returns:
        tuple: A tuple containing:
            - test_loss (float): The average loss over the test dataset.
            - test_acc (float): The average accuracy over the test dataset.
    """
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

from tqdm.auto import tqdm


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    """
    Trains and evaluates a PyTorch model over a specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing/validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        loss_fn (torch.nn.Module, optional): Loss function for training and evaluation (default: CrossEntropyLoss).
        epochs (int, optional): Number of training epochs (default: 5).

    Returns:
        dict: A dictionary containing training and testing results with the following keys:
            - "train_loss" (list): Training loss values per epoch.
            - "train_acc" (list): Training accuracy values per epoch.
            - "test_loss" (list): Testing loss values per epoch.
            - "test_acc" (list): Testing accuracy values per epoch.
    """

    # Initialize results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Training loop
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        # Print training progress
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Store results
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results

