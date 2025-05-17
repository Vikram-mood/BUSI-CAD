import os
import random
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import joblib
import numpy as np
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data Splitting (Stratified)
def split_and_save_txt_stratified(dataset_root, output_dir, seed=42, train_ratio=0.7, val_ratio=0.15):
    image_paths = glob(os.path.join(dataset_root, '*', '*.png'))
    labels = [os.path.dirname(path).split('/')[-1] for path in image_paths]

    # Stratified split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, train_size=train_ratio, stratify=labels, random_state=seed
    )
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths, temp_labels, train_size=val_ratio_adjusted, stratify=temp_labels, random_state=seed
    )

    def write_txt(filename, paths):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for path in paths:
                rel_path = os.path.relpath(path, dataset_root)
                f.write(f"{rel_path}\n")

    os.makedirs(output_dir, exist_ok=True)
    write_txt("train.txt", train_paths)
    write_txt("val.txt", val_paths)
    write_txt("test.txt", test_paths)

# 2. Custom Dataset
class GrayscaleImageDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        with open(txt_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        self.root_dir = root_dir
        self.transform = transform
        self.labels = [path.split('/')[0] for path in self.image_paths]
        self.class_to_idx = {'benign': 0, 'malignant': 1, 'normal': 2}
        self.targets = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('L').convert('RGB')  # Grayscale to RGB
        if self.transform:
            image = self.transform(image)
        label = self.targets[idx]
        return image, label

# 3. CNN+Transformer Model
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=3, num_heads=8, num_layers=2, dim_feedforward=2048):
        super(CNNTransformer, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*(list(resnet.children())[:-1]))  # Remove FC layer
        self.feature_dim = 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.cnn(x).view(batch_size, -1)  # Shape: (batch_size, 512)
        features = features.unsqueeze(1)  # Shape: (batch_size, 1, 512)
        transformer_out = self.transformer(features).squeeze(1)  # Shape: (batch_size, 512)
        logits = self.fc(transformer_out)
        return logits

# 4. Training Function for DL Models
def train_dl_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies

# 5. Plotting Function for DL Models
def plot_dl_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss vs Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(f'{model_name} - Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'metrics_{model_name.lower()}.png')
    plt.close()

# 6. Evaluation Function
def evaluate_model(model, test_loader, device, model_name, class_names, is_dl=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            if is_dl:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            else:
                outputs = model.predict(inputs.cpu().numpy())
                preds = outputs
            all_preds.extend(preds.cpu().numpy() if is_dl else preds)
            all_labels.extend(labels.cpu().numpy() if is_dl else labels.numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🎯 {model_name} Test Accuracy: {accuracy:.4f}")
    print(f"\n🔍 {model_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f'cm_{model_name.lower()}.png')
    plt.close()

    return accuracy

# 7. Feature Extraction for ML Models
def extract_features(txt_file, root_dir, resnet, transform):
    dataset = GrayscaleImageDataset(txt_file, root_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features, labels = [], []

    resnet.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            out = resnet(imgs).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 512)
            features.append(out.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.vstack(features), np.array(labels)

# Main Execution
if __name__ == "__main__":
    # Data splitting
    dataset_root = "Dataset_BUSI_with_GT1_pre"
    output_dir = "split_dataset"
    split_and_save_txt_stratified(dataset_root, output_dir)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets and Dataloaders
    train_dataset = GrayscaleImageDataset("split_dataset/train.txt", dataset_root, train_transform)
    val_dataset = GrayscaleImageDataset("split_dataset/val.txt", dataset_root, val_test_transform)
    test_dataset = GrayscaleImageDataset("split_dataset/test.txt", dataset_root, val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    class_names = ["Benign", "Malignant", "Normal"]
    os.makedirs("models", exist_ok=True)

    # DL Models
    dl_models = [
        ("ResNet18", models.resnet18(pretrained=True)),
        ("GoogleNet", models.googlenet(pretrained=True)),
        ("CNNTransformer", CNNTransformer(num_classes=3))
    ]

    dl_accuracies = []
    for model_name, model in dl_models:
        print(f"\nTraining {model_name}...")
        if model_name == "ResNet18":
            model.fc = nn.Linear(model.fc.in_features, 3)
        elif model_name == "GoogleNet":
            model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        save_path = f"models/{model_name.lower()}_best.pth"

        train_losses, val_losses, train_accuracies, val_accuracies = train_dl_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, device=device, save_path=save_path
        )

        plot_dl_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name)

        model.load_state_dict(torch.load(save_path))
        accuracy = evaluate_model(model, test_loader, device, model_name, class_names, is_dl=True)
        dl_accuracies.append((model_name, accuracy))

    # Feature Extraction for ML Models
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()  # Remove FC layer
    resnet = resnet.to(device)
    X_train, y_train = extract_features("split_dataset/train.txt", dataset_root, resnet, val_test_transform)
    X_test, y_test = extract_features("split_dataset/test.txt", dataset_root, resnet, val_test_transform)

    # ML Models
    ml_models = [
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(kernel='rbf', probability=True, random_state=42))
    ]

    ml_accuracies = []
    for model_name, model in ml_models:
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{model_name.lower()}_model.pkl")
        accuracy = evaluate_model(model, test_loader, device, model_name, class_names, is_dl=False)
        ml_accuracies.append((model_name, accuracy))

    # Summary
    print("\n📊 Model Performance Summary:")
    print("Deep Learning Models:")
    for name, acc in dl_accuracies:
        print(f"{name}: Test Accuracy = {acc:.4f}")
    print("Machine Learning Models:")
    for name, acc in ml_accuracies:
        print(f"{name}: Test Accuracy = {acc:.4f}")