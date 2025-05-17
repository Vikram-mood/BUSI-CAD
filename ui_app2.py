import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter

class MedianFilterTransform:
    def __init__(self, size=3):
        self.size = size

    def __call__(self, img):
        img_np = np.array(img)
        filtered = median_filter(img_np, size=self.size)
        return Image.fromarray(filtered)

class HighPassAndEdgeEnhanceTransform:
    def __init__(self, gaussian_std=20):
        self.gaussian_std = gaussian_std

    def __call__(self, img):
        img_np = np.array(img, dtype=np.float64)
        blurred = gaussian_filter(img_np, sigma=self.gaussian_std)
        high_pass = img_np - blurred
        enhanced = img_np + high_pass
        enhanced = np.clip(enhanced, 0, 255)
        return Image.fromarray(enhanced.astype(np.uint8))

# Preprocessing pipeline for saving (excludes normalization)
preprocessing_pipeline = transforms.Compose([
    MedianFilterTransform(size=3),
    HighPassAndEdgeEnhanceTransform(gaussian_std=20),
    transforms.Resize((224, 224)),
])
# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale if it's not

    # Convert grayscale to 3 channels (for ResNet)
    img = img.convert('RGB')

    img_processed = preprocessing_pipeline(img)

    transform_to_tensor = transforms.ToTensor()
    img_tensor = transform_to_tensor(img_processed)

    return img_tensor.unsqueeze(0) 


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
label_map = {'benign': 0, 'malignant': 1, 'normal': 2}
reverse_label_map = {v: k for k, v in label_map.items()}

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# CNN+Transformer Model Definition
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

# Load Models
def load_models():
    # Define map_location based on device
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None

    # DL Models
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 3)
    resnet_model.load_state_dict(torch.load("ver2/models/resnet18_best.pth", map_location=map_location))
    resnet_model = resnet_model.to(device).eval()

    googlenet_model = models.googlenet(pretrained=True)
    googlenet_model.fc = nn.Linear(googlenet_model.fc.in_features, 3)
    googlenet_model.load_state_dict(torch.load("ver2/models/googlenet_best.pth", map_location=map_location))
    googlenet_model = googlenet_model.to(device).eval()

    cnntransformer_model = CNNTransformer(num_classes=3)
    cnntransformer_model.load_state_dict(torch.load("ver2/models/cnntransformer_best.pth", map_location=map_location))
    cnntransformer_model = cnntransformer_model.to(device).eval()

    # ML Models
    rf_model = joblib.load("ver2/models/randomforest_model.pkl")
    adaboost_model = joblib.load("ver2/models/adaboost_model.pkl")
    svm_model = joblib.load("ver2/models/svm_model.pkl")

    # Feature extractor for ML
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor.fc = nn.Identity()
    feature_extractor = feature_extractor.to(device).eval()

    return resnet_model, googlenet_model, cnntransformer_model, rf_model, adaboost_model, svm_model, feature_extractor
def show_preprocessed_image(image_path):
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.convert('RGB')  # Convert to RGB for preprocessing
        img_processed = preprocessing_pipeline(img)  # Apply preprocessing pipeline

        # Convert to numpy array for display
        img_np = np.array(img_processed)

        # Display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.title("Preprocessed Image")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying preprocessed image: {str(e)}")
        return None



# Extract Features for ML
def extract_features_for_ml(img_tensor, feature_extractor):
    with torch.no_grad():
        features = feature_extractor(img_tensor).squeeze(-1).squeeze(-1)  # Shape: (1, 512)
    return features.cpu().numpy()

# Classify Image
def classify_image(image_path, use_deep=True):
    img_tensor = preprocess_image(image_path)
    result_text = ""
    prob_text = ""

    if use_deep:
        # Deep Learning Models
        with torch.no_grad():
            resnet_out = resnet_model(img_tensor)
            googlenet_out = googlenet_model(img_tensor)
            cnntransformer_out = cnntransformer_model(img_tensor)

            # Apply softmax to get probabilities
            resnet_probs = F.softmax(resnet_out, dim=1)[0].cpu().numpy()
            googlenet_probs = F.softmax(googlenet_out, dim=1)[0].cpu().numpy()
            cnntransformer_probs = F.softmax(cnntransformer_out, dim=1)[0].cpu().numpy()

            # Format probabilities for terminal
            prob_text += "ResNet18 Probabilities:\n"
            for i, prob in enumerate(resnet_probs):
                prob_text += f"  {reverse_label_map[i]}: {prob:.4f}\n"
            prob_text += "GoogleNet Probabilities:\n"
            for i, prob in enumerate(googlenet_probs):
                prob_text += f"  {reverse_label_map[i]}: {prob:.4f}\n"
            prob_text += "CNN+Transformer Probabilities:\n"
            for i, prob in enumerate(cnntransformer_probs):
                prob_text += f"  {reverse_label_map[i]}: {prob:.4f}\n"

            # Ensemble: Max probability across models
            all_probs = np.stack([resnet_probs, googlenet_probs, cnntransformer_probs])
            max_probs = np.max(all_probs, axis=0)
            predicted_class = np.argmax(max_probs)
            predicted_label = reverse_label_map[predicted_class]
            result_text = f"Predicted Class (DL Ensemble): {predicted_label}"
    else:
        # Machine Learning Models
        features = extract_features_for_ml(img_tensor, feature_extractor)

        rf_probs = rf_model.predict_proba(features)[0]
        adaboost_probs = adaboost_model.predict_proba(features)[0]
        svm_probs = svm_model.predict_proba(features)[0]

        # Format probabilities for terminal
        prob_text += "Random Forest Probabilities:\n"
        for i, prob in enumerate(rf_probs):
            prob_text += f"  {reverse_label_map[i]}: {prob:.4f}\n"
        prob_text += "AdaBoost Probabilities:\n"
        for i, prob in enumerate(adaboost_probs):
            prob_text += f"  {reverse_label_map[i]}: {prob:.4f}\n"
        prob_text += "SVM Probabilities:\n"
        for i, prob in enumerate(svm_probs):
            prob_text += f"  {reverse_label_map[i]}: {prob:.4f}\n"

        # Ensemble: Max probability across models
        all_probs = np.stack([rf_probs, adaboost_probs, svm_probs])
        max_probs = np.max(all_probs, axis=0)
        predicted_class = np.argmax(max_probs)
        predicted_label = reverse_label_map[predicted_class]
        result_text = f"Predicted Class (ML Ensemble): {predicted_label}"

    return result_text, predicted_label, prob_text



# GUI
class CADInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("BUSI Dataset CAD Interface")
        self.root.geometry("600x400")

        # Load models
        global resnet_model, googlenet_model, cnntransformer_model, rf_model, adaboost_model, svm_model, feature_extractor
        try:
            resnet_model, googlenet_model, cnntransformer_model, rf_model, adaboost_model, svm_model, feature_extractor = load_models()
        except Exception as e:
            self.root.destroy()
            raise RuntimeError(f"Failed to load models: {str(e)}")

        # Widgets
        self.label = tk.Label(root, text="Select an Image for Classification", font=("Arial", 14))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=5)

        self.image_path_label = tk.Label(root, text="No image selected", wraplength=500)
        self.image_path_label.pack(pady=5)
        # New button for displaying preprocessed image
        self.show_preprocessed_button = tk.Button(root, text="Show Preprocessed Image", command=self.show_preprocessed, state=tk.DISABLED)
        self.show_preprocessed_button.pack(pady=5)

        self.model_type_label = tk.Label(root, text="Select Model Type:", font=("Arial", 12))
        self.model_type_label.pack(pady=10)

        self.model_type = tk.StringVar(value="Deep Learning")
        self.dl_radio = tk.Radiobutton(root, text="Deep Learning", variable=self.model_type, value="Deep Learning")
        self.ml_radio = tk.Radiobutton(root, text="Machine Learning", variable=self.model_type, value="Machine Learning")
        self.dl_radio.pack()
        self.ml_radio.pack()

        self.classify_button = tk.Button(root, text="Classify", command=self.classify, state=tk.DISABLED)
        self.classify_button.pack(pady=10)

        self.result_text = tk.Text(root, height=5, width=60, font=("Arial", 10))
        self.result_text.pack(pady=10)


    def select_image(self):
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
            if file_path:
                self.image_path_label.config(text=file_path)
                self.image_path = file_path
                self.classify_button.config(state=tk.NORMAL)
                self.show_preprocessed_button.config(state=tk.NORMAL)  # Enable the preprocess button
            else:
                self.image_path_label.config(text="No image selected")
                self.classify_button.config(state=tk.DISABLED)
                self.show_preprocessed_button.config(state=tk.DISABLED)  # Disable the preprocess button

    def show_preprocessed(self):
        if not hasattr(self, 'image_path'):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please select an image first.")
            return
        try:
            show_preprocessed_image(self.image_path)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Displaying preprocessed image for: {os.path.basename(self.image_path)}")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error displaying preprocessed image: {str(e)}")
    
    def classify(self):
        if not hasattr(self, 'image_path'):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please select an image first.")
            return

        use_deep = self.model_type.get() == "Deep Learning"
        try:
            result_text, predicted_label, prob_text = classify_image(self.image_path, use_deep=use_deep)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Image: {os.path.basename(self.image_path)}\n\n")
            self.result_text.insert(tk.END, result_text)
            print(f"\nProbabilities for {os.path.basename(self.image_path)}:")
            print(prob_text)
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error during classification: {str(e)}")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = CADInterface(root)
    root.mainloop()