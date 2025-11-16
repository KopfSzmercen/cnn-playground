from sklearn.metrics import classification_report
from typing import List, Optional
import torch

def calculate_metrics(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: Optional[List[str]] = None
    ) -> str:
    """
    Calculate and return a classification report.
    
    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader containing the dataset.
        device: Device to run the model on (CPU or GPU).
        class_names: Optional list of class names for labeling the report. 
    """
    with torch.inference_mode():
        model.eval()
        all_preds = []
        all_labels = []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get the predicted class with the highest score
            _, preds = torch.max(outputs, 1)

            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    print("Classification Report Generated.")

    return report