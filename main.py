
import cv2
import numpy as np
import os
video = './VID_20240405_120134.mp4'



def process_image(image):


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # create mask
    mask = np.zeros_like(edges)
    height, width = mask.shape

    # draw robot trajectory which is assumed to be normal to the camera sensor
    cv2.line(image, (int(width/2), height), (int(width/2), 0), (5, 5, 255), 3)

    # Define ROI (Region of Interest) mask
    roi_vertices = [(0, height), (0, height*1/4), (width, height*1/4), (width, height)]
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

    return image


def extract_images_from_video(path_in, subsample_rate, debug=False):
    vidcap = cv2.VideoCapture(path_in)
    if not vidcap.isOpened():
        raise IOError

    if debug:
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = vidcap.get(cv2.CAP_PROP_FPS)
        print ("Length: %.2f | Width: %.2f | Height: %.2f | Fps: %.2f" % (length, width, height, fps))


    success, image = vidcap.read() #extract first frame.
    frame_count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (frame_count*subsample_rate))
        success, image = vidcap.read()
        image = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
        processed_image = process_image(image)
        cv2.imshow("Frame", processed_image)
        cv2.waitKey(250)
        frame_count = frame_count + 1
    
    vidcap.release()
    return frame_count


extract_images_from_video(video, 1000, True)
