
from modules.data_setup import create_dataloaders
from modules.training_engine import train
from modules.utils import save_model, create_truncated_dataset
from torchvision import transforms, datasets
import torch
from modules.visualizations import view_random_N_dataloader_images, plot_train_val_progress
from modules.confusion_matrix import plot_confusion_matrix_from_model
from torchinfo import summary
import os

from modules.metrics import calculate_metrics

BATCH_SIZE = 32

model = torch.hub.load("pytorch/vision", "mobilenet_v3_small", weights="DEFAULT", skip_validation=True)

data_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor()
])


cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)

cifar_train_dataset = create_truncated_dataset(cifar_train_dataset, 0.1)
cifar_test_dataset = create_truncated_dataset(cifar_test_dataset, 0.1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cpu")
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

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=576, out_features=len(class_names),bias=True)
).to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

EPOCHS = 3

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
    device=device
)

calculate_metrics(
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

save_model(
    model=model,
    target_dir="models",
    model_name="mobilenet_v3_small_sports.pth"
)