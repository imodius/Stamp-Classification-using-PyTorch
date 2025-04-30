import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import os
import time
import copy
import numpy as np
import cv2
import torch.nn.functional as F

# --- Configuration (should match your training script) ---
DATA_DIR = 'stamps_dataset'
IMAGE_SIZE = 224
NUM_CLASSES = 31 # Make sure this matches the number of classes you trained with
MODEL_SAVE_PATH = 'best_model_weights.pth'
# Note: Saving the heatmap directly in Streamlit might not be ideal in a multi-user environment.
# For this example, we'll generate and display it directly.

# --- Model Definition (should match your training script) ---
def build_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- Grad-CAM Global Variables and Hook Functions ---
# These need to be outside the functions to be accessible by the hooks
gradients = None
feature_maps = None

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def save_feature_maps(module, input, output):
    global feature_maps
    feature_maps = output

# --- Grad-CAM Function (Manual Implementation Attempt) ---
def generate_grad_cam(model, input_image_tensor, target_class_index, target_layer):
    global gradients, feature_maps

    model.eval()
    if input_image_tensor.grad is not None:
        input_image_tensor.grad.zero_()

    # Register hooks
    hook_feature_maps = target_layer.register_forward_hook(save_feature_maps)
    hook_gradients = target_layer.register_full_backward_hook(save_gradients)

    # Enable gradients for the input tensor
    input_image_tensor.requires_grad_(True)

    # Forward pass
    output = model(input_image_tensor)

    # Zero gradients for the output
    model.zero_grad()

    # Calculate gradient of the target class score
    target_class_score = output[0, target_class_index]
    target_class_score.backward()

    # Remove hooks
    hook_feature_maps.remove()
    hook_gradients.remove()

    # --- Grad-CAM Calculation ---
    if feature_maps is None or gradients is None:
        st.error("Error: Could not retrieve feature maps or gradients for Grad-CAM.")
        gradients = None
        feature_maps = None
        return None

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    weighted_feature_maps = feature_maps * pooled_gradients.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    heatmap = torch.mean(weighted_feature_maps, dim=1).squeeze()
    heatmap = F.relu(heatmap)

    heatmap_max = torch.max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max
    else:
        heatmap = torch.zeros_like(heatmap)

    # Resize heatmap to the original image size
    # Note: input_image_tensor is already IMAGE_SIZE x IMAGE_SIZE after preprocessing
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze().cpu().detach().numpy()

    # Reset global variables after use
    gradients = None
    feature_maps = None

    return heatmap

# --- Load the Trained Model ---
@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            st.success(f"Model loaded successfully from {model_path}")
            return model, device
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {e}")
            return None, None
    else:
        st.error(f"Model file not found at {model_path}. Please train the model first using the other script.")
        return None, None

# --- Load Class Names ---
@st.cache_resource
def load_class_names(data_dir):
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                          for x in ['train']}
        class_names = image_datasets['train'].classes
        st.success("Class names loaded successfully.")
        return class_names
    except FileNotFoundError:
        st.error(f"Data directory not found at {data_dir}. Cannot load class names.")
        return None
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return None

# --- Streamlit App Layout ---
st.title("Stamp Identification Program with Grad-CAM")

st.write("Upload an image of a stamp to get it identified and see where the model focused.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a stamp image...", type=["jpg", "jpeg", "png"])

# Load model and class names
model, device = load_model(MODEL_SAVE_PATH, NUM_CLASSES)
class_names = load_class_names(DATA_DIR)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add a button to trigger prediction and Grad-CAM
    if st.button("Identify Stamp and Show Heatmap"):
        if model and class_names:
            # Define transformations for the single image (should match validation transforms)
            preprocess = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            try:
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                original_image_np = np.array(image) # For heatmap overlay

                # Perform prediction
                with torch.no_grad(): # Standard prediction without gradient tracking initially
                    outputs = model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    confidence = torch.softmax(outputs, dim=1)[0] * 100
                    predicted_class_index = predicted.item()

                    if predicted_class_index < len(class_names):
                        predicted_class_name = class_names[predicted_class_index]
                    else:
                        predicted_class_name = f"Unknown Class Index: {predicted_class_index}"
                        st.warning(f"Predicted class index {predicted_class_index} is out of bounds for class names list of size {len(class_names)}.")

                    confidence_score = confidence[predicted_class_index].item()

                # --- Generate and Display Grad-CAM ---
                st.subheader("Identification Results:")
                st.write(f"**Predicted Stamp:** {predicted_class_name}")
                st.write(f"**Confidence:** {confidence_score:.2f}%")

                st.subheader("Model Focus (Grad-CAM Heatmap):")
                # Generate heatmap (gradients are handled inside generate_grad_cam)
                heatmap = generate_grad_cam(model, image_tensor.clone().detach(), predicted_class_index, model.layer4[-1]) # Pass target layer

                if heatmap is not None:
                    # Convert heatmap to a displayable format and overlay
                    original_image_resized = cv2.resize(original_image_np, (IMAGE_SIZE, IMAGE_SIZE))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    overlayed_image = cv2.addWeighted(original_image_resized.astype(np.uint8), 0.6, heatmap_colored.astype(np.uint8), 0.4, 0)

                    # Display the heatmap without forcing it to use the full column width
                    st.image(overlayed_image, caption="Grad-CAM Heatmap")
                    st.write("The heatmap shows areas the model focused on (red = high focus, blue = low focus).")

                else:
                    st.warning("Could not generate Grad-CAM heatmap.")

                # Option to print results (using browser's print functionality)
                st.write("---")
                st.write("You can print this page using your browser's print function (Ctrl+P or Cmd+P).")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

        else:
            st.warning("Model or class names not loaded. Please ensure the model is trained and the data directory is correct.")

else:
    st.info("Please upload an image to begin.")

st.write("---")
st.write("Note: This program requires a trained model file ('best_model_weights.pth') and the 'stamps_dataset' directory to be accessible.")
