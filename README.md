# Stamp-Classification-using-PyTorch
Stamp Classification using PyTorch
This repository contains Python code for training an image classification model to identify different types of stamps. The model is based on a pre-trained ResNet18 architecture, fine-tuned on a custom dataset of stamp images.
Description
The goal of this project is to build a convolutional neural network (CNN) model capable of recognizing and classifying various postage stamps. The code handles data loading, preprocessing, model training, evaluation, and prediction. It also includes an attempt at implementing Grad-CAM to visualize which parts of the image the model focuses on during prediction.
Features
* Fine-tuning of a pre-trained ResNet18 model.
* Data augmentation during training.
* Training and validation loops with loss and accuracy tracking.
* Saving of the best model weights based on validation accuracy.
* Prediction on new, unseen stamp images.
* Attempt at generating Grad-CAM heatmaps to visualize model attention.
Setup and Installation
1. Clone the repository:
git clone https://github.com/your_username/your_repository_name.git
cd your_repository_name

(Replace your_username and your_repository_name with your actual GitHub details)
2. Install dependencies:
It's recommended to use a virtual environment.
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# Install required libraries
pip install torch torchvision pillow numpy opencv-python

(You might need a specific version of torch and torchvision depending on your CUDA setup if you plan to use a GPU. Refer to the official PyTorch website for installation instructions.)
Dataset Structure
The script expects the dataset to be organized in a specific directory structure. Create a main data directory (default is stamps_dataset) with train and val subdirectories. Inside train and val, create separate folders for each stamp class.
stamps_dataset/
├── train/
│   ├── class_name_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_name_2/
│   │   ├── imageA.png
│   │   └── ...
│   └── ...
└── val/
   ├── class_name_1/
   │   ├── imageX.jpg
   │   └── ...
   └── ...

Replace class_name_1, class_name_2, etc., with the actual names of your stamp categories.
How to Run
   1. Place your dataset: Organize your stamp images according to the dataset structure described above and place the stamps_dataset folder in the same directory as the Python script.
   2. Update configuration: Open the Python script (your_script_name.py) and update the following variables in the Configuration section:
   * DATA_DIR: Path to your dataset directory (if different from stamps_dataset).
   * NUM_CLASSES: The total number of unique stamp classes in your dataset.
   * Adjust BATCH_SIZE, NUM_EPOCHS, and LEARNING_RATE as needed (refer to comments in the code and consider monitoring training metrics).
   3. Run the script:
python your_script_name.py

The script will automatically start training if no saved model weights are found at MODEL_SAVE_PATH. If weights are found, it will load the existing model.
Prediction
After training (or if a trained model is loaded), the script will attempt to make a prediction on a test image.
      * Update the test_image_path variable in the if __name__ == '__main__': block with the path to the image you want to classify.
      * The script will print the predicted class and confidence score.
Grad-CAM Visualization
If generate_cam=True in the predict_stamp function call (which is the default in the example usage), the script will attempt to generate a Grad-CAM heatmap. This heatmap is overlaid on the input image to show the areas the model focused on for its prediction.
      * The heatmap image will be saved to the path specified by GRAD_CAM_OUTPUT_PATH.
      * Inspect this image to understand the model's decision-making process.
Note: The Grad-CAM implementation in the script is a manual attempt and might require debugging or refinement depending on your specific PyTorch version and model architecture. Using a dedicated library like pytorch-grad-cam is often more robust.
Dependencies
      * torch
      * torchvision
      * Pillow
      * numpy
      * opencv-python
License
(Choose a license and add its details here. For example, the MIT License or the Apache 2.0 License are common choices.)
Acknowledgements
(Optional: Thank any individuals, resources, or datasets that helped you with this project.)
