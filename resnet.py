import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt


# ----------------------------
# Custom Dataset to Apply Transform per Subset
# ----------------------------
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.loader = dataset.loader

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, target = self.dataset.samples[self.indices[idx]]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target


# ----------------------------
# Simple CNN model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, image_size=224):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat_size = 64 * (image_size // 8) * (image_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------
# Train or Eval for one epoch
# ----------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if training:
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ----------------------------
# Plot Training Curves
# ----------------------------
def plot_curves(train_values, test_values, ylabel, title, save_path):
    plt.figure()
    plt.plot(train_values, label='Train')
    plt.plot(test_values, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# ----------------------------
# Main training function
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load full dataset (no transform yet)
    full_dataset = datasets.ImageFolder(args.data_dir)
    class_names = full_dataset.classes
    print("Classes:", class_names)

    # Stratified split
    targets = [sample[1] for sample in full_dataset.samples]
    train_idx, test_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=(1 - args.train_split),
        stratify=targets,
        random_state=42
    )

    # Wrap datasets with transforms
    train_dataset = TransformedSubset(full_dataset, train_idx, train_transform)
    test_dataset = TransformedSubset(full_dataset, test_idx, test_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    if args.model == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(class_names))
        )
    elif args.model == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(class_names))
        )
    else:
        model = SimpleCNN(num_classes=len(class_names), image_size=args.image_size)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Early stopping check
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > args.patience:
                print("⏹️ Early stopping triggered.")
                break

    # Final report
    print("\n✅ Final Train Accuracy: {:.2f}%".format(train_accuracies[-1] * 100))
    print("✅ Final Test Accuracy: {:.2f}%".format(test_accuracies[-1] * 100))

    # Save plots
    if args.plot:
        os.makedirs("plots", exist_ok=True)
        plot_curves(train_losses, test_losses, "Loss", "Train vs Test Loss", "plots/loss.png")
        plot_curves(train_accuracies, test_accuracies, "Accuracy", "Train vs Test Accuracy", "plots/accuracy.png")
        print("📊 Plots saved to `plots/` folder.")


# ----------------------------
# CLI Args
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retina/Choroid CNN Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset root")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet50", "simple"], default="resnet18", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/test split")
    parser.add_argument("--plot", action="store_true", help="Save training plots")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    main(args)
