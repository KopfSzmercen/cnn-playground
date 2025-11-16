
import random
import torch
import matplotlib.pyplot as plt
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

    plt.subplot(2, 1, 1)
    plt.plot(train_results["train_loss"], label="train_loss")
    plt.plot(train_results["test_loss"], label="test_loss")
    plt.title("Train and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    epochs = range(len(train_results["train_loss"]))
    plt.xticks(epochs)

    plt.subplot(2, 1, 2)
    plt.plot(train_results["train_acc"], label="train_acc")
    plt.plot(train_results["test_acc"], label="test_acc")
    plt.title("Train and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.xticks(epochs)

    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
