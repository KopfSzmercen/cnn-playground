
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import List, Optional

def get_predicions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[List[int], List[int]]:
    """
    Get predictions and true labels from the model.
    
    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader containing the dataset.
        device: Device to run the model on (CPU or GPU).
        
    Returns:
        A tuple of lists containing predicted labels and true labels. Eg. [predictions, true_labels].
    """
    model.eval()
    true_labels = []
    predictions = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get the predicted class with the highest score
            _, preds = torch.max(outputs, 1)
            
            # Collect predictions and true labels -- convert to CPU and numpy for easier handling
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    return predictions, true_labels

def create_confusion_matrix(
    true_labels: List[int],
    predictions: List[int],
    lasss_names: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create and display a confusion matrix.
    
    Args:
        true_labels: List of true labels.
        predictions: List of predicted labels.
        class_names: Optional list of class names for labeling the matrix.
        figsize: Size of the figure for the plot.
        save_path: Optional path to save the confusion matrix image.
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=lasss_names, yticklabels=lasss_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix_from_model(
        model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None)-> None:
    """
    Plot confusion matrix from a model's predictions.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader containing the dataset.
        device: Device to run the model on (CPU or GPU).
        class_names: Optional list of class names for labeling the matrix.
        figsize: Size of the figure for the plot.
        save_path: Optional path to save the confusion matrix image.
    """

    predictions, true_labels = get_predicions(model, dataloader, device)
    
    create_confusion_matrix(
        true_labels=true_labels,
        predictions=predictions,
        lasss_names=class_names,
        figsize=figsize,
        save_path=save_path
    )