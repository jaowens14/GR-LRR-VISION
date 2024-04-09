
import cv2
import numpy as np
import serial



#video = './VID_20240405_120134.mp4'
video = 0

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

h7 = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1) 

def process_image(image):
    vectors = {}
    edge_stats = []

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
    cv2.line(image, (int(width/2), height), (int(width/2), 0), red, 3)

    # Define ROI (Region of Interest) mask
    roi_vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)


    # this is a really complicated line that sorts the edges found by x1
    # the purpose is to get the edges are that closest to the middle of the image
    lines = np.array(sorted(lines, key=lambda x: abs(x[0][0] - int(width/2))))[0:10]
    print(lines)
    # Draw detected lines on original image
    for n, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        line_vector = [x2-x1, y2-y1]
        line_unit_vector = line_vector/np.linalg.norm(line_vector)

        

        try: 
            arr = round(abs(np.dot(robot_trajectory, line_unit_vector)), 4)
            if arr > 0.9:
                cv2.line(image, (x1,y1),(x2, y2), green, 1)
                cv2.circle(image, (x1,y1), radius=3, color=red, thickness=3)
                cv2.circle(image, (x2,y2), radius=3, color=red, thickness=3)
                cv2.putText(image, str(arr), (x1,y1), 1, 1, red)
                cv2.putText(image, str(arr), (x2,y2), 1, 1, red)
                edge_stats.append(arr)
        except:
            print("error with dot product")



        edge_stats.sort()

    print(edge_stats)
    print(" ")
    # Draw ROI on the image
    cv2.polylines(image, [np.int32([roi_vertices])], True, blue, 2)

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
