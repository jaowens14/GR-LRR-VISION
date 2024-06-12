
import cv2
import numpy as np
import serial
import traceback


video = './VID_20240405_120134.mp4'
#video = 0

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

#7 = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1) 

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
    #vidcap = cv2.VideoCapture(path_in, cv2.CAP_DSHOW)
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
        
            

        try:
            processed_image = process_image(image)
            cv2.imshow("Frame", processed_image)
            cv2.waitKey(1000)
            frame_count = frame_count + 1
            
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    
    vidcap.release()
    return frame_count



def fft(I):
    rows, cols = I.shape
    m = cv2.getOptimalDFTSize( rows )
    n = cv2.getOptimalDFTSize( cols )
    padded = cv2.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes) # Add to the expanded another plane with zeros

    cv2.dft(complexI, complexI) # this way the result may fit in the source matrix

    cv2.split(complexI, planes) # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv2.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    magI = planes[0]

    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI) # switch to logarithmic scale
    cv2.log(magI, magI)

    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)

    q0 = magI[0:cx, 0:cy] # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy] # Top-Right
    q2 = magI[0:cx, cy:cy+cy] # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right

    tmp = np.copy(q0) # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp

    tmp = np.copy(q1) # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp

    cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX) # Transform the matrix with float values into a

    cv2.imshow("Input Image" , I ) # Show the result
    cv2.imshow("spectrum magnitude", magI)

extract_images_from_video(video, 1000, True)
