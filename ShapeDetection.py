import cv2  # OpenCV Library

# Correct file path for the image
image_path = r"C:\Users\KIIT\PycharmProjects\ShapeDetection_PSO\images.png"

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Error: Unable to load image at {image_path}. Check the file path and file integrity.")
    exit(1)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

# Find contours in the threshold image
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour to identify and label shapes
for i, contour in enumerate(contours):
    if i == 0:  # Skip the first contour as it might be the outer border
        continue

    # Approximate the shape
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Draw the contours on the original image
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + (w / 2))  # Estimate x midpoint
    y_mid = int(y + (h / 2))  # Estimate y midpoint
    coords = (x_mid, y_mid)
    colour = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Identify the shape based on the number of vertices
    if len(approx) == 3:
        cv2.putText(image, "Triangle", coords, font, 0.5, colour, 1)
    elif len(approx) == 4:
        cv2.putText(image, "Quadrilateral", coords, font, 0.5, colour, 1)
    elif len(approx) == 5:
        cv2.putText(image, "Pentagon", coords, font, 0.5, colour, 1)
    elif len(approx) == 6:
        cv2.putText(image, "Hexagon", coords, font, 0.5, colour, 1)
    else:
        cv2.putText(image, "Circle", coords, font, 0.5, colour, 1)

# Display the image with detected shapes
cv2.imshow("Shapes Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()