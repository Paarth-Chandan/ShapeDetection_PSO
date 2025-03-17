import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define dataset path and categories
DATASET_PATH = "dataset"
CATEGORIES = ["circle", "kite", "parallelogram", "rectangle", "rhombus", "square", "trapezoid", "triangle"]

# Function to extract HOG features (same as training code)
def extract_hog_features(image):
    if image is None:
        return None
    image = cv2.resize(image, (64, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features


# Function to load dataset (test only)
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


# Function to evaluate the model
def evaluate_model(model, X, y):
    # Make predictions
    y_pred = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=CATEGORIES))

    # Show confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the trained model
    model_path = "shape_classifier_svm.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Train the model first.")

    print("Loading trained model...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")

    # Load test data
    print("Loading test data...")
    X_test, y_test = load_dataset(os.path.join(DATASET_PATH, "test"))

    if len(X_test) == 0:
        raise ValueError("Test dataset is empty. Please add test images to evaluate the model.")

    # Evaluate the model on test data
    print("Evaluating model on test data...")
    evaluate_model(model, X_test, y_test)
