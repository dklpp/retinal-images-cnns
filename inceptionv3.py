import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt

# ------------------------------------
# Custom Dataset with Transform Override
# ------------------------------------
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


# ------------------------------------
# Training or Evaluation for One Epoch
# ------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs, _ = model(inputs) if training else (model(inputs), None)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if training:
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# ------------------------------------
# Plot Helper
# ------------------------------------
def plot_curve(train_vals, test_vals, ylabel, title, path):
    plt.figure()
    plt.plot(train_vals, label="Train")
    plt.plot(test_vals, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path)
    plt.close()


# ------------------------------------
# Main Training Function
# ------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms — InceptionV3 expects 299x299
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Dataset
    dataset = datasets.ImageFolder(args.data_dir)
    class_names = dataset.classes
    print("Classes:", class_names)

    targets = [sample[1] for sample in dataset.samples]
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        stratify=targets,
        test_size=(1 - args.train_split),
        random_state=42
    )

    train_dataset = TransformedSubset(dataset, train_idx, train_transform)
    test_dataset = TransformedSubset(dataset, test_idx, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ------------------------------------
    # InceptionV3 model (special case)
    # ------------------------------------
    model = models.inception_v3(pretrained=True, aux_logits=True)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, len(class_names))
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    run_name = f"inceptionv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                print("⏹️ Early stopping triggered.")
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


# ------------------------------------
# CLI
# ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InceptionV3 Retina/Choroid Classifier")

    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    main(args)
