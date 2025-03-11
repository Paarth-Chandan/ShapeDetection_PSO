import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load your trained model
model = joblib.load("shape_classifier_svm.pkl")

# Define your categories in the same order used during training
CATEGORIES = ["circle", "kite", "parallelogram", "rectangle", "rhombus", "square", "trapezoid", "triangle"]


# Function to extract HOG features (same as in your training code)
def extract_hog_features(image):
    if image is None:
        return None
    image = cv2.resize(image, (64, 64))  # Resize for uniformity
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features


def predict_shape(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Display the image
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()

    # Extract features
    features = extract_hog_features(img)
    if features is None:
        print("Error: Could not extract features from the image")
        return None

    # Reshape features for prediction (model expects 2D array)
    features = features.reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]

    # Get predicted class
    predicted_shape = CATEGORIES[prediction]

    return predicted_shape


# Test an image
image_path = "image6.jpeg"  # Replace with your image path
shape = predict_shape(image_path)

if shape:
    print(f"The predicted shape is: {shape}")