import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import os
import time # Import time to track training duration
import copy # Import copy for deep copying the model
import multiprocessing # Import multiprocessing
import numpy as np # Import numpy for image processing
import cv2 # Import OpenCV for image processing (requires pip install opencv-python)
import torch.nn.functional as F # Import functional for ReLU and resize
from torch.utils.data import DataLoader, Dataset # Import DataLoader and Dataset

# You might need to install pytorch-gradcam for a more robust implementation,
# but this attempts a manual integration of the core logic.
# from pytorch_grad_cam import GradCAM # Example import if using a library
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # Example import
# from pytorch_grad_cam.utils.image import show_cam_on_image # Example import

# --- Configuration ---
DATA_DIR = 'stamps_dataset' # Directory containing stamp images, organized by class (e.g., stamps_dataset/countryA/stamp1.jpg)
IMAGE_SIZE = 224 # Image size for model input
BATCH_SIZE = 64
NUM_EPOCHS = 100 # Reduced for demonstration; a real model needs more
LEARNING_RATE = 0.00001
NUM_CLASSES = 31 # Example: replace with the actual number of unique stamps you want to identify
MODEL_SAVE_PATH = 'best_model_weights.pth' # Path to save the best model weights
GRAD_CAM_OUTPUT_PATH = 'grad_cam_heatmap.jpg' # Path to save the Grad-CAM heatmap image

# --- Data Loading and Preprocessing ---

# Define transformations for the images
# These transformations prepare the images for the model
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE), # Randomly crop and resize for data augmentation
        transforms.RandomHorizontalFlip(), # Randomly flip horizontally for data augmentation
        transforms.ToTensor(), # Convert image to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize pixel values
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE), # Resize for validation
        transforms.CenterCrop(IMAGE_SIZE), # Center crop for validation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Model Definition ---

# Load a pre-trained ResNet model
# ResNet is a powerful CNN architecture pre-trained on a large dataset (ImageNet)
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# Replace the last fully connected layer to match the number of stamp classes
# This is fine-tuning: we keep the pre-trained layers and train a new classification head
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Grad-CAM Function (Manual Implementation Attempt) ---
# This function attempts to generate a Grad-CAM heatmap.
# It requires the model, the input image tensor, the target class index, and the device.
# It also needs access to the gradients and feature maps of a target convolutional layer.

# Store gradients and feature maps
gradients = None
feature_maps = None

# Corrected save_gradients function to accept all arguments from the hook
def save_gradients(module, grad_input, grad_output):
    global gradients
    # In this case, grad_output contains the gradients with respect to the output of the module
    gradients = grad_output[0] # Assuming a single output tensor

def save_feature_maps(module, input, output):
    global feature_maps
    feature_maps = output

def generate_grad_cam(model, input_image_tensor, target_class_index, device):
    # Declare global variables at the beginning
    global gradients, feature_maps

    # Ensure model is in evaluation mode
    model.eval()

    # Clear previous gradients
    if input_image_tensor.grad is not None:
        input_image_tensor.grad.zero_()

    # Register hooks to save feature maps and gradients
    # We'll target the last convolutional layer in ResNet18 (layer4[-1])
    target_layer = model.layer4[-1]
    hook_feature_maps = target_layer.register_forward_hook(save_feature_maps)
    # Using register_full_backward_hook which provides grad_input and grad_output
    hook_gradients = target_layer.register_full_backward_hook(save_gradients)

    # Enable gradients for the input tensor (needed for backpropagation)
    input_image_tensor.requires_grad_(True)

    # Forward pass
    output = model(input_image_tensor)

    # Zero gradients for the output
    model.zero_grad()

    # Calculate gradient of the target class score with respect to the output
    # Need to ensure the output tensor is the one we calculate gradient against
    # If output is a tuple or list, select the appropriate tensor
    target_class_score = output[0, target_class_index]
    target_class_score.backward()

    # Remove hooks
    hook_feature_maps.remove()
    hook_gradients.remove()

    # --- Grad-CAM Calculation ---
    if feature_maps is None or gradients is None:
        print("Error: Could not retrieve feature maps or gradients for Grad-CAM.")
        # Reset global variables in case of error
        gradients = None
        feature_maps = None
        return None

    # Pool the gradients over spatial dimensions to get weights
    # gradients shape: [batch_size, channels, height, width]
    # Assuming batch_size is 1 for a single image prediction
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the feature maps by the pooled gradients
    # feature_maps shape: [batch_size, channels, height, width]
    # Assuming batch_size is 1
    weighted_feature_maps = feature_maps * pooled_gradients.unsqueeze(0).unsqueeze(2).unsqueeze(3)


    # Average the weighted feature maps and apply ReLU
    heatmap = torch.mean(weighted_feature_maps, dim=1).squeeze()
    heatmap = F.relu(heatmap)

    # Normalize the heatmap
    # Avoid division by zero if heatmap is all zeros
    heatmap_max = torch.max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max
    else:
        heatmap = torch.zeros_like(heatmap) # Set to zeros if max is zero


    # Resize heatmap to the original image size
    # Note: input_image_tensor is already IMAGE_SIZE x IMAGE_SIZE after preprocessing
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze().cpu().detach().numpy()

    # Reset global variables after use
    gradients = None
    feature_maps = None

    return heatmap

# --- Prediction (Inference) ---

# To make a prediction on a new image:

# 1. Load the trained model weights (if not already loaded)
# This part is now handled in the main execution block to load the model once
def predict_stamp(image_path, model, class_names, device, generate_cam=False):
    model.eval() # Set model to evaluation mode

    # Define transformations for a single image
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB') # Open image and ensure it's RGB
        original_image = np.array(image) # Keep original for heatmap overlay

        image_tensor = preprocess(image).unsqueeze(0) # Apply transformations and add batch dimension
        image_tensor = image_tensor.to(device) # Move image to the same device as the model

        # Perform prediction
        # Need to enable gradients for the input tensor if Grad-CAM is requested
        if generate_cam:
             image_tensor.requires_grad_(True)

        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0] * 100
        predicted_class_index = predicted.item()

        if predicted_class_index < len(class_names):
            predicted_class_name = class_names[predicted_class_index]
        else:
            predicted_class_name = f"Unknown Class Index: {predicted_class_index}"
            print(f"Warning: Predicted class index {predicted_class_index} is out of bounds for class names list of size {len(class_names)}.")


        confidence_score = confidence[predicted_class_index].item()

        # --- Grad-CAM Integration ---
        heatmap_path = None
        if generate_cam and class_names:
             print(f"Attempting to generate Grad-CAM for predicted class: {predicted_class_name}")
             # Generate the heatmap using the implemented function
             heatmap = generate_grad_cam(model, image_tensor, predicted_class_index, device)

             # Save or display the heatmap (requires OpenCV)
             if heatmap is not None:
                 # Convert heatmap to a displayable format (e.g., apply a colormap)
                 # Resize original image to match heatmap size for overlay
                 original_image_resized = cv2.resize(original_image, (IMAGE_SIZE, IMAGE_SIZE))
                 heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                 # Overlay heatmap on the original image
                 # Use a transparency factor (e.g., 0.4 for heatmap, 0.6 for original)
                 # Ensure both images are the same data type for cv2.addWeighted
                 overlayed_image = cv2.addWeighted(original_image_resized.astype(np.uint8), 0.6, heatmap_colored.astype(np.uint8), 0.4, 0)

                 cv2.imwrite(GRAD_CAM_OUTPUT_PATH, overlayed_image)
                 heatmap_path = GRAD_CAM_OUTPUT_PATH
                 print(f"Grad-CAM heatmap saved to {heatmap_path}")


        return predicted_class_name, confidence_score, heatmap_path

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Reset global variables in case of error during prediction
        global gradients, feature_maps
        gradients = None
        feature_maps = None
        return None, None, None


# --- Main Execution Block ---
if __name__ == '__main__':
    # This block ensures that multiprocessing works correctly on Windows

    # Check if a trained model already exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading trained model from {MODEL_SAVE_PATH}")
        # Load the model architecture
        loaded_model = models.resnet18(weights=None) # Start with a fresh model structure
        num_ftrs_loaded = loaded_model.fc.in_features
        loaded_model.fc = nn.Linear(num_ftrs_loaded, NUM_CLASSES)
        # Load the saved weights
        loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        loaded_model = loaded_model.to(device)
        model = loaded_model # Use the loaded model for prediction
        # Since we loaded a model, we might not need to load the dataset or train
        # depending on whether the goal is just prediction or training + prediction.
        # For this script, we'll assume if a model is loaded, we can proceed to prediction examples.
        # If you want to train again even if a model exists, you can move the training block
        # outside this if condition.

        # Load data just to get class_names for prediction
        try:
            image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                                      data_transforms[x])
                              for x in ['train', 'val']}
            class_names = image_datasets['train'].classes
            print("Class names loaded from dataset structure.")
        except FileNotFoundError:
             print(f"Error: Data directory not found at {DATA_DIR}. Cannot load class names for prediction.")
             class_names = [] # Set empty if data directory is missing
             # Depending on your use case, you might need to save/load class names differently


    else:
        # Load the dataset using ImageFolder, which assumes data is organized into folders by class
        # Example: stamps_dataset/class1/img1.jpg, stamps_dataset/class2/img2.jpg
        # Ensure your dataset is in the DATA_DIR with 'train' and 'val' subdirectories
        try:
            image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                                      data_transforms[x])
                              for x in ['train', 'val']}
            # Set num_workers to 0 for potential debugging on Windows if issues persist
            dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                         shuffle=True, num_workers=4)
                          for x in ['train', 'val']}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
            class_names = image_datasets['train'].classes
            print("Data loaded successfully.")

        except FileNotFoundError:
            print(f"Error: Data directory not found at {DATA_DIR}")
            print("Please create the 'stamps_dataset' directory with 'train' and 'val' subdirectories,")
            print("each containing folders for your stamp classes, and place your images there.")
            # Exit or handle the error appropriately if data is not found
            exit()

        # --- Loss Function and Optimizer ---
        # Define the loss function (Cross-Entropy Loss is common for classification)
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer (Stochastic Gradient Descent is a common choice)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Training the Model ---

        # This is a simplified training loop. A real-world loop would include:
        # - More sophisticated learning rate scheduling
        # - Model checkpointing (saving the best model)
        # - Early stopping
        # - More detailed logging and evaluation metrics

        print("Starting training...")

        since = time.time() # Record the start time

        best_model_wts = copy.deepcopy(model.state_dict()) # Keep track of the best model weights
        best_acc = 0.0 # Keep track of the best accuracy

        for epoch in range(NUM_EPOCHS):
            print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model if it's the best validation accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print() # Print a newline after each epoch

        time_elapsed = time.time() - since # Calculate training time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model weights into the model
        model.load_state_dict(best_model_wts)

        # --- Saving the Trained Model ---
        # You can save the trained model weights to a file
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model weights saved to {MODEL_SAVE_PATH}")


    # --- Example usage for Prediction ---
    # After training (or if a trained model was loaded), you can use this to predict
    test_image_path = 'test_stamp.jpg' # <--- REPLACE WITH YOUR IMAGE PATH
    # Check if class_names were loaded successfully before predicting
    if class_names:
        # Call predict_stamp and request Grad-CAM generation
        # Note: Pass generate_cam=True to attempt Grad-CAM
        predicted_stamp, score, heatmap_path = predict_stamp(test_image_path, model, class_names, device, generate_cam=True)
        if predicted_stamp:
            print(f'Predicted stamp: {predicted_stamp} with confidence {score:.2f}%')
            if heatmap_path:
                print(f'Grad-CAM heatmap saved to {heatmap_path}')
                print("Inspect the saved image to see where the model focused.")
    else:
        print("Cannot perform prediction because class names could not be loaded.")


    print("\n--- Code Structure Explained ---")
    print("1. Configuration: Sets up parameters like data directory, image size, and training settings.")
    print("2. Data Loading and Preprocessing: Defines how to load images and apply transformations for training and validation.")
    print("3. Model Definition: Loads a pre-trained CNN (ResNet18) and modifies it for stamp classification.")
    print("4. Loss Function and Optimizer: Sets up how the model's performance is measured and how its weights is updated.")
    print("5. Training Loop: Active to train the model on your dataset (skipped if a saved model is found).")
    print("6. Prediction (Inference): Provides a function to take a new image and get a stamp prediction after the model is trained.")
    print("7. Grad-CAM (Manual Implementation Attempt): Attempts to generate a Grad-CAM heatmap to visualize model focus.")
    print("\nTo make this code functional, you would need:")
    print("- A dataset of stamp images organized into 'train' and 'val' subdirectories within the 'stamps_dataset' folder.")
    print("- To run the script. It will train if no saved model is found, or load the saved model.")
    print(f"- To provide a path to a new image by replacing 'path/to/your/test_stamp.jpg' in the script.")
    print("- To install OpenCV (`pip install opencv-python`) for image handling in the Grad-CAM part.")
    print("\nNote: The manual Grad-CAM implementation is an attempt and might require debugging or refinement.")