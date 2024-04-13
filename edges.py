
import cv2
import numpy as np
import colors

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # create mask
    mask = np.zeros_like(edges)
    height, width = mask.shape
    # {x, y}
    robot_trajectory = [0, 1]

    # draw robot trajectory which is assumed to be normal to the camera sensor
    cv2.line(image, (int(width/2), height), (int(width/2), 0), colors.red, 3)

    # Define ROI (Region of Interest) mask
    roi_vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Use Hough Line Transform to detect lines
    edges = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    return edges

# draws edges on the image and returns the image
def draw_edges(image, edges):
    '''Draws edges on the image and returns the image and edges.
       image, edges (x1, y1, x2, y2)'''
    for n, edge in enumerate(edges):
        x1, y1, x2, y2 = edge[0]
        line_vector = [x2-x1, y2-y1]
        line_unit_vector = line_vector/np.linalg.norm(line_vector)

        cv2.line(image, (x1,y1),(x2, y2), colors.green, 1)
        cv2.circle(image, (x1,y1), radius=3, color=colors.red, thickness=3)
        cv2.circle(image, (x2,y2), radius=3, color=colors.red, thickness=3)

    return image, edges

def filter_edges_by_midpoint(edges):

    return None