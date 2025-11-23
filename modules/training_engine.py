from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple
import torch
import time

from modules.utils import save_model
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> Tuple[float, float]:
    """
    Perform a single training step.

    Turns train mode and runs training loop for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for the training dataset.
        loss_fn: Loss function to compute the loss.
        optimizer: Optimizer for updating model parameters.
        device: Device to perform computations on (CPU or GPU).
    
    Returns:
        A tuple with training loss and training accuracy eg. (0.5, 0.8).
    """

    model.train()

    train_loss, train_acc = 0, 0

    for _, (X, y) in enumerate(tqdm(dataloader, desc="Training", miniters=20, mininterval=5)):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #Convert logits to prediction probabilities and choose the class with highest probability
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device) -> Tuple[float, float]:
    
    """
    Tests NN model for 1 epoch.

    Runs model in eval mode and performs forward pass on testing dataset.

    Args: 
        model: model to test
        dataloader: dataloader with testing data
        loss_fn: loss fn
        device: target device

    Returns:
        A tuple of test loss and test accuracy eg. (0.21, 0.23)
    """

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for _, (X, y) in enumerate(tqdm(dataloader, desc="Testing", miniters=20, mininterval=5)):

            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)

            loss = loss_fn(y_pred_logits, y)

            test_loss += loss.item()

            # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

            test_acc += (pred_labels == y).sum().item() / len(pred_labels)
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_best_model: bool = False,
          best_model_dir: Optional[str] = None,
          best_model_name: Optional[str] = None
          #writer: SummaryWriter
          ) -> Dict[str, List]:
    """
    Trains a PyTorch model.

    Args:
        model: model,
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_fn,
        epochs,
        device,
        save_best_model: whether to persist the best test accuracy model during training.
        best_model_dir: directory where the best model should be saved when requested.
        best_model_name: filename to use when saving the best model.
        writer: TensorBoard SummaryWriter for logging metrics.
    
    Returns:
        A dictionary of training and testing loss and accuracy. Each metric has a value for an epoch.
        For example epochs = 2:
            {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973],
                  train_time: 123.45} 
    """

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "train_time": 0
    }

    best_test_acc = float("-inf")
    should_save_best = save_best_model and best_model_dir and best_model_name

    start_time = time.time()

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}\n"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if should_save_best and test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model(model=model, target_dir=best_model_dir, model_name=best_model_name)
    
    end_time = time.time()
    results["train_time"] = end_time - start_time

    return results




