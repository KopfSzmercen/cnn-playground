import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    """
    Create DataLoaders for training and testing datasets.
    
    Args:
        train_dir: Directory containing training images.
        test_dir: Directory containing testing images.
        transform: Transformations to apply to the images.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
        
    Returns:
        train_loader: DataLoader for training dataset.
        test_loader: DataLoader for testing dataset.
        class_names: List of class names in the dataset.
    """
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = train_dataset.classes

    return train_loader, test_loader, class_names