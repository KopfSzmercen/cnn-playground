
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def view_random_N_dataloader_images(
        dataloader: torch.utils.data.DataLoader,
        n: int = 4):

    """
    View a random sample of images from a dataloader using matplotlib.
    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing images.
        n (int): The number of images to display.
    """ 
    if n > 15:
        raise ValueError("n must be less than or equal to 15 to fit in a 4x4 grid.")
    
    random_indices = random.sample(range(len(dataloader.dataset)), n)
    images = []
    labels = []

    for idx in random_indices:
        image, label = dataloader.dataset[idx]
        images.append(image)
        labels.append(label)

    for i in range(n):
        plt.subplot(4, 4, i + 1)

        #reshape the image to 3 channels for visualization
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f"Label: {labels[i]}")
        plt.axis(False)

    plt.show()


def plot_train_val_progress(
    train_results: Dict,
    fig_name: str = "train_val_progress.png"
):
    """
    Plots results of train and val.

    Args:
        train_results: dictionary of train_loss, train_acc, test_loss, test_acc
        fig_name: name of the file to save the figure
    """
    plt.figure(figsize=(10, 8))

    num_epochs = len(train_results["train_loss"])
    
    # Adjust ticks if too many epochs
    if num_epochs > 30:
        step = max(1, num_epochs // 15)  # Aim for roughly 15 ticks
        ticks = range(0, num_epochs, step)
    else:
        ticks = range(num_epochs)

    plt.subplot(2, 1, 1)
    plt.plot(train_results["train_loss"], label="train_loss")
    plt.plot(train_results["test_loss"], label="test_loss")
    plt.title("Train and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.xticks(ticks)

    plt.subplot(2, 1, 2)
    plt.plot(train_results["train_acc"], label="train_acc")
    plt.plot(train_results["test_acc"], label="test_acc")
    plt.title("Train and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.xticks(ticks)

    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


def plot_classification_report(
    class_report: Dict,
    fig_name: str = "classification_report.png"
):
    """
    Minimalist visualization of classification report metrics.
    Shows precision, recall, and f1-score for each class in a grouped bar chart.
    
    Args:
        class_report: dictionary from sklearn's classification_report
        fig_name: name of the file to save the figure
    """
    # Extract class names and metrics (exclude accuracy and averages)
    classes = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    precisions = [class_report[c]['precision'] for c in classes]
    recalls = [class_report[c]['recall'] for c in classes]
    f1_scores = [class_report[c]['f1-score'] for c in classes]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Create bars
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Classification Report (Accuracy: {class_report["accuracy"]:.1%})', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    plt.show()


def plot_classification_heatmap(
    class_report: Dict,
    fig_name: str = "classification_heatmap.png"
):
    """
    Heatmap visualization of precision, recall, and f1-score.
    Minimalist alternative showing all metrics in one view.
    
    Args:
        class_report: dictionary from sklearn's classification_report
        fig_name: name of the file to save the figure
    """
    # Extract class names and metrics
    classes = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Create matrix
    metrics_matrix = np.array([
        [class_report[c]['precision'] for c in classes],
        [class_report[c]['recall'] for c in classes],
        [class_report[c]['f1-score'] for c in classes]
    ])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(['Precision', 'Recall', 'F1-Score'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(3):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Classification Metrics Heatmap (Accuracy: {class_report["accuracy"]:.1%})',
                fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    plt.show()
