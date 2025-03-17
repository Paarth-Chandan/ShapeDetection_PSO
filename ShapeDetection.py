import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define dataset path and categories
DATASET_PATH = "dataset"
CATEGORIES = ["circle", "kite", "parallelogram", "rectangle", "rhombus", "square", "trapezoid", "triangle"]

# Function to extract HOG features
def extract_hog_features(image):
    if image is None:
        return None
    image = cv2.resize(image, (64, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features


# Function to load data from a given folder (train/val)
def load_dataset(dataset_path):
    X = []  # Feature vectors
    y = []  # Labels

    for label, category in enumerate(CATEGORIES):
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found, skipping...")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image '{img_path}', skipping...")
                continue

            features = extract_hog_features(img)
            if features is not None:
                X.append(features)
                y.append(label)

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_dataset(os.path.join(DATASET_PATH, "train"))

    # Load validation data
    print("Loading validation data...")
    X_val, y_val = load_dataset(os.path.join(DATASET_PATH, "val"))

    # Check if datasets are empty
    if len(X_train) == 0:
        raise ValueError("Training dataset is empty. Please add images before training.")
    if len(X_val) == 0:
        print("Warning: Validation dataset is empty. The model will be trained without validation.")

    # Train SVM model
    print("Training SVM model...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    print("Model training complete!")

    # Validate the model if validation data is available
    if len(X_val) > 0:
        y_val_pred = svm_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred) * 100
        print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the trained model
    joblib.dump(svm_model, "shape_classifier_svm.pkl")
    print("Model saved as 'shape_classifier_svm.pkl'")