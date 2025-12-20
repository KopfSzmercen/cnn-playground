from modules.training_engine import train
from modules.utils import save_model, create_truncated_dataset
from torchvision import transforms, datasets, models
import torch
from modules.visualizations import view_random_N_dataloader_images, plot_train_val_progress, plot_classification_report, plot_classification_heatmap
from modules.confusion_matrix import plot_confusion_matrix_from_model
from torchinfo import summary
import os
import argparse

from modules.metrics import calculate_metrics

BATCH_SIZE = 32

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10 with options")
parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of training epochs (default: 3)")
parser.add_argument("--percent", "-p", type=float, default=0.1, help="Fraction of dataset to use (0-1) or percent (1-100). Default=0.1 (10%%)")
parser.add_argument(
    "--save-best-model",
    "-b",
    nargs="?",
    const=True,
    default=False,
    type=str_to_bool,
    help="Save the best test-accuracy checkpoint during training instead of the final model",
)
args = parser.parse_args()

cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

percent = float(args.percent)
if percent > 1:
    percent = percent / 100.0
if percent <= 0 or percent > 1:
    raise ValueError("`--percent` must be in range (0, 1] when given as fraction or (0,100] when given as percentage")

cifar_train_dataset = create_truncated_dataset(cifar_train_dataset, percent)
cifar_test_dataset = create_truncated_dataset(cifar_test_dataset, percent)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device)

model.to(device)

train_loader = torch.utils.data.DataLoader(
    cifar_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    cifar_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

view_random_N_dataloader_images(
    dataloader=train_loader,
    n=4
)


for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last convolutional block (Layer 4) for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

model.to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 0.01}
], momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    patience=5, 
    factor=0.1
)

EPOCHS = int(args.epochs)

print(f"Using epochs={EPOCHS}, dataset fraction={percent}")

summary(model, 
        input_size=(32, 3, 224, 224),
        verbose=1,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


train_results = train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    device=device,
    scheduler=scheduler,
    save_best_model=args.save_best_model,
    best_model_dir="models",
    best_model_name="resnet.pth"
)

print(f"Total training time: {train_results['train_time']:.2f} seconds")
print(f"Average time per epoch: {train_results['avg_epoch_time']:.2f} seconds")

if args.save_best_model:
    print("[INFORMATION] Loading best model for evaluation...")
    best_model_path = os.path.join("models", "resnet.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

classification_report = calculate_metrics(
    model=model,
    dataloader=test_loader,
    device=device,
    class_names=class_names
)

if not os.path.exists("results"):
    os.makedirs("results")

plot_train_val_progress(
    train_results=train_results,
    fig_name="results/train_val_progress.png"
)

plot_confusion_matrix_from_model(
    model=model,
    dataloader=test_loader,
    class_names=class_names,    
    device=device,
    figsize=(10, 8),
    save_path="results/confusion_matrix.png"
)

plot_classification_report(
    class_report=classification_report,
    fig_name="results/classification_report.png"
)

plot_classification_heatmap(
    class_report=classification_report,
    fig_name="results/classification_heatmap.png"
)


if not args.save_best_model:
    save_model(
        model=model,
        target_dir="models",
        model_name="resnet.pth"
    )