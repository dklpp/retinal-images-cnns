import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import timm
import matplotlib.pyplot as plt
from datetime import datetime


# Apply transform only to a subset
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
        image = self.transform(image)
        return image, target


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if train_mode:
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def plot_curve(train_values, test_values, ylabel, title, path):
    plt.figure()
    plt.plot(train_values, label="Train")
    plt.plot(test_values, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Using device:", device)

    # Data transforms
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

    # Load dataset
    full_dataset = datasets.ImageFolder(args.data_dir)
    class_names = full_dataset.classes
    print("Classes:", class_names)

    targets = [sample[1] for sample in full_dataset.samples]
    train_idx, test_idx = train_test_split(
        list(range(len(full_dataset))),
        stratify=targets,
        test_size=(1 - args.train_split),
        random_state=42
    )

    train_dataset = TransformedSubset(full_dataset, train_idx, train_transform)
    test_dataset = TransformedSubset(full_dataset, test_idx, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # EfficientNet via timm
    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name)) if args.tensorboard else None

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_loss = float("inf")
    patience_counter = 0

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

        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if writer:
        writer.close()

    print(f"\n✅ Final Train Accuracy: {train_accuracies[-1]*100:.2f}%")
    print(f"✅ Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")

    if args.plot:
        os.makedirs("plots", exist_ok=True)
        plot_curve(train_losses, test_losses, "Loss", "Train vs Test Loss", "plots/loss.png")
        plot_curve(train_accuracies, test_accuracies, "Accuracy", "Train vs Test Accuracy", "plots/accuracy.png")
        print("📊 Plots saved to 'plots/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNet Classifier for Retina/Choroid")

    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="EfficientNet or other timm model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--plot", action="store_true", help="Save training plots")

    args = parser.parse_args()
    main(args)
