import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\KIIT\PycharmProjects\ShapeDetection_PSO\image5.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}. Check the file path and file integrity.")
    exit(1)

# Convert to grayscale and apply thresholding
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) < 100:  # Ignore very small contours (noise)
        continue

    # Approximate contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Draw contours
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    # Bounding box for text placement
    x, y, w, h = cv2.boundingRect(approx)
    text_x = x + w // 4
    text_y = y + h // 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    colour = (255, 0, 0)

    # Detect polygons based on vertices
    vertex_count = len(approx)

    if vertex_count == 3:
        shape_name = "Triangle"
    elif vertex_count == 4:
        aspect_ratio = float(w) / h
        shape_name = "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle"
    elif vertex_count == 5:
        shape_name = "Pentagon"
    elif vertex_count == 6:
        shape_name = "Hexagon"
    elif vertex_count > 6:
        # Check for Circle or Oval using ellipse fitting
        if len(contour) > 5:  # Ensure valid ellipse fitting
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            aspect_ratio = major_axis / minor_axis

            if 0.9 < aspect_ratio < 1.1:
                shape_name = "Circle"
            else:
                shape_name = "Oval"
        else:
            shape_name = "Polygon"

    else:
        shape_name = "Polygon"

    # Put the detected shape name on the image
    cv2.putText(image, shape_name, (text_x, text_y), font, 0.6, colour, 2)

# Display the processed image
cv2.imshow("Fixed Shape Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()