import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a PyTorch model to the specified directory.
    Args:
        model: The PyTorch model to save.
        target_dir: The directory where the model should be saved.
        model_name: The name of the model file (e.g., "model.pth").
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth"), "Model name must end with .pth"

    print(f"[INFORMATION] Saving model to {target_dir}/{model_name}...")

    torch.save(model.state_dict(), f=target_dir / model_name)


# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_truncated_dataset(dataset: torch.utils.data.Dataset,
                             proportion: float=0.1) -> torch.utils.data.Dataset:
    """
    Creates a truncated version of a dataset by selecting a proportion of its samples.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        proportion (float): The proportion of the dataset to keep (between 0 and 1).

    Returns:
        torch.utils.data.Dataset: A truncated dataset containing the specified proportion of samples.
    """
    if not (0 < proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1.")

    total_samples = len(dataset)
    truncated_size = int(total_samples * proportion)

    indices = torch.randperm(total_samples)[:truncated_size]

    truncated_dataset = torch.utils.data.Subset(dataset, indices)

    return truncated_dataset