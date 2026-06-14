
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os


def get_data_transforms():
   
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Standard ImageNet normalization (since we use a pre-trained ResNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'   -> EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'best_eurosat_model.pth')
        self.val_loss_min = val_loss


def build_model(num_classes=10, device='cpu'):
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.to(device)



def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Generates a heatmap to show exactly which crops the model is confusing.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('EuroSAT Crop & Land Cover Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def unnormalize(img):
    """Reverts normalization so we can display the satellite image properly."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    return np.clip(img, 0, 1)

def visualize_predictions(dataloader, model, classes, device):

    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Real Sentinel-2 Predictions (EuroSAT)', fontsize=16)

    for idx in range(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        img = unnormalize(images[idx].cpu())
        ax.imshow(img)
        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx]]

        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color, fontsize=10)

    plt.tight_layout()
    plt.show()

def visualize_saliency_maps(dataloader, model, classes, device):
    
    print("\n Generating Explainable AI Saliency Maps...")
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    images.requires_grad_()

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    score = outputs.gather(1, preds.view(-1, 1)).squeeze()

    model.zero_grad()
    score.backward(torch.ones_like(score))

    saliency, _ = torch.max(images.grad.data.abs(), dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Explainable AI: Saliency Maps (Model Focus Areas)', fontsize=16)

    for idx in range(5):
        img = unnormalize(images[idx].detach().cpu())
        axes[0, idx].imshow(img)
        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx]]
        color = 'green' if true_label == pred_label else 'red'
        axes[0, idx].set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color, fontsize=10)
        axes[0, idx].axis('off')

        sal = saliency[idx].cpu().numpy()
        axes[1, idx].imshow(sal, cmap='hot')
        axes[1, idx].set_title('Attention Heatmap', fontsize=10)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.show()

def train_eurosat_pipeline():
    print("[INFO] Initializing Real Earth Observation Pipeline (EuroSAT)...")

    BATCH_SIZE = 32
    EPOCHS = 15 
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {DEVICE}")

    
    train_transform, val_transform = get_data_transforms()

    print(" Downloading/Loading EuroSAT Dataset (Real Sentinel-2 images)...")
    # This automatically downloads the 27,000 real satellite images if not present
    full_dataset = torchvision.datasets.EuroSAT(
        root='./data',
        download=True,
        transform=train_transform
    )

    class_names = full_dataset.classes
    print(f"[INFO] Classes being monitored: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names), device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=4)

    print("\n Starting Training...")
    best_val_loss = np.inf

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(" Early stopping triggered. Halting training to prevent overfitting.")
            break

    print("\n Training Complete. Generating final metrics on unseen validation data...")

    model.load_state_dict(torch.load('best_eurosat_model.pth'))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(all_labels, all_preds, class_names)
    visualize_predictions(val_loader, model, class_names, DEVICE)

    visualize_saliency_maps(val_loader, model, class_names, DEVICE)

if __name__ == "__main__":
    train_eurosat_pipeline()