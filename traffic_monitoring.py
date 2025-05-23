import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import svm
import random
from torchvision import transforms

# Set a random seed for reproducibility
random.seed(42)

# Define the path to the KITTI dataset images
image_data_dir = r"KITTI/2011_09_26_drive_0117/image_02/data"

# Path to the Haar Cascade XML file for car detection
car_cascade_path = os.path.join(os.path.dirname(__file__), "cars.xml")

# Load the Haar Cascade for vehicle detection
if not os.path.exists(car_cascade_path):
    raise FileNotFoundError(f"Haar Cascade file not found at: {car_cascade_path}")

car_cascade = cv2.CascadeClassifier(car_cascade_path)

# Define an enhanced CNN for feature extraction
class EnhancedTrafficCNN(nn.Module):
    def __init__(self):
        super(EnhancedTrafficCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Instantiate the CNN model
cnn_model = EnhancedTrafficCNN()

# Preprocessing function for images
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

# Haar Cascade detection only for initial testing
image_files = sorted(os.listdir(image_data_dir))
selected_files = random.sample(image_files, 10)

for img_name in selected_files:
    img_path = os.path.join(image_data_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is not None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Test Haar Cascade Detection Only
        haar_detected_regions = car_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.05, minNeighbors=2, minSize=(50, 50))

        # Draw Haar Cascade detections for debugging
        for (x, y, w, h) in haar_detected_regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Traffic Detection by Shubh Rakesh Nahar")
        plt.axis('off')
        plt.show()

# To fully utilize the sliding window and SVM, continue with the full detection pipeline after verifying Haar Cascade.
