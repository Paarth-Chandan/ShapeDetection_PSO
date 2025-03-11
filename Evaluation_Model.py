import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define dataset path
CATEGORIES = ["circle", "kite", "parallelogram", "rectangle", "rhombus", "square", "trapezoid", "triangle"]


# Function to extract HOG features (same as in your training code)
def extract_hog_features(image):
    if image is None:
        return None
    image = cv2.resize(image, (64, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features


def evaluate_model(model, dataset_path, categories):
    # Lists to store features and labels
    X_test = []
    y_test = []

    # Load test images
    for label, category in enumerate(categories):
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
                X_test.append(features)
                y_test.append(label)

    # Convert to NumPy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    return X_test, y_test, y_pred


def detailed_evaluation(y_true, y_pred, categories):
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=categories))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the trained model
    model = joblib.load("shape_classifier_svm.pkl")

    # Evaluate on test dataset
    print("\nEvaluating on test dataset:")
    X_test, y_test, y_test_pred = evaluate_model(model, "dataset/test", CATEGORIES)

    # Evaluate on validation dataset
    print("\nEvaluating on validation dataset:")
    X_val, y_val, y_val_pred = evaluate_model(model, "dataset/val", CATEGORIES)

    # Detailed evaluation on test dataset
    print("\nDetailed evaluation on test dataset:")
    detailed_evaluation(y_test, y_test_pred, CATEGORIES)