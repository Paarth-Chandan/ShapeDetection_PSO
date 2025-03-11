import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Define dataset path
DATASET_PATH = "dataset/train"
CATEGORIES = ["circle", "kite", "parallelogram", "rectangle", "rhombus", "square", "trapezoid", "triangle"]


# Function to extract HOG features
def extract_hog_features(image):
    if image is None:
        return None  # Return None if the image is invalid
    image = cv2.resize(image, (64, 64))  # Resize for uniformity
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)  # Disable visualization for speed
    return features


# Load dataset
X = []  # Feature vectors
y = []  # Labels

for label, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found, skipping...")
        continue
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)  # Read image
        if img is None:
            print(f"Warning: Could not read image '{img_path}', skipping...")
            continue

        features = extract_hog_features(img)
        if features is not None:
            X.append(features)
            y.append(label)  # Assign numerical label

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Check if dataset is empty
if len(X) == 0:
    raise ValueError("Dataset is empty. Please add images before training.")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate model
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred) * 100  # Convert to percentage
print(f"Validation Accuracy: {accuracy:.2f}%")  # Display as percentage

# Save the trained model
joblib.dump(svm_model, "shape_classifier_svm.pkl")
print("Model saved as 'shape_classifier_svm.pkl'")
