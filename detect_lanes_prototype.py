import cv2
import numpy as np

def detect_lanes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    try: 
        # Define ROI (Region of Interest) mask
        mask = np.zeros_like(edges)
        height, width = mask.shape
        roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
        cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

        # Draw detected lines on original image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw ROI on the image
        cv2.polylines(image, [np.int32([roi_vertices])], True, (0, 0, 255), 2)
    except Exception:
        return image

    return image


camera = cv2.VideoCapture(0)

cv2.namedWindow("capture")

while True:
    ret, frame = camera.read()

    frame = detect_lanes(frame)

    if not ret:
        print("failed to grab frame")
        break  
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    if k%256 == 32:
        # space bar hit, break
        break

camera.release()

cv2.destroyAllWindows()