
import cv2
import numpy as np
import serial
import traceback


video = './VID_20240405_120134.mp4'
#video = 0


red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)



font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
  
# Line thickness of 2 px 
thickness = 1

#7 = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1) 


def process_image(image, frame_count):
    image = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
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

    limit1_point1 = [0, int(height/5)]
    limit1_point2 = [width, int(height/5)]

    limit2_point1 = [0, int(4*height/5)]
    limit2_point2 = [width, int(4*height/5)]


    # add limits to image
    cv2.line(image, limit1_point1, limit1_point2, blue, 2)
    cv2.putText(image, 'limit1', limit1_point1, font,  
                   fontScale, blue, thickness, cv2.LINE_AA)
    

    cv2.line(image, limit2_point1, limit2_point2, blue, 2)
    cv2.putText(image, 'limit2', limit2_point1, font,  
                   fontScale, blue, thickness, cv2.LINE_AA)



    # draw robot trajectory which is assumed to be normal to the camera sensor
    cv2.line(image, (int(width/2), height), (int(width/2), 0), red, 3)

    # Define ROI (Region of Interest) mask
    roi_vertices = [(0, 0), (width, 0), (width, height), (0, height)]
    roi_vertices = [limit2_point1, limit2_point2, limit1_point2, limit1_point1]
    cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=1/5*height, maxLineGap=50)


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
            if arr > 0.7:
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
    cv2.putText(image, 'ROI', (limit1_point1[0], limit1_point1[1]-30), font,  
                   fontScale, green, thickness, cv2.LINE_AA)
    cv2.polylines(image, [np.int32([roi_vertices])], True, green, 2)


    #cv2.imwrite("./output/frame"+str(frame_count)+".jpg", image)

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
        
        
            

        try:
            processed_image = process_image(image, frame_count)
            cv2.imshow("Frame", processed_image)
            cv2.waitKey(1000)
            frame_count = frame_count + 1
            
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    
    vidcap.release()
    return frame_count




frames = extract_images_from_video(video, 1000, True)


#processed_image = process_image(cv2.imread("frame1.jpg"))
#cv2.imshow("Frame", processed_image)
#cv2.moveWindow("Frame", 1000, 0)
#cv2.waitKey(0)