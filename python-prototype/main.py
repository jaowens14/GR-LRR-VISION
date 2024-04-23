
import cv2
import numpy as np
import traceback

video = './VID_20240405_120134.mp4'
video = 0

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)

font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
  
# Line thickness of 2 px 
thickness = 1

left_percent = 0
right_percent = 0

def get_x_vector_midpoint(points):
    '''expects points [(x1, y   1) (x2, y2)]'''
    x1, y1, x2, y2 = points
    midpoint_x = (x1+x2)//2
    midpoint_y = (y1+y2)//2

    return midpoint_x

def sort_by_distance_to_center(edges, width):
    # this is a really complicated line that sorts the edges found by x1
    # the purpose is to get the edges are that closest to the middle of the image
    
    edges = np.array(sorted(edges, key=lambda x: abs(get_x_vector_midpoint(x) - int(width/2))))

    return edges


def draw_line_with_end_points(image, points, label):
    x1, y1, x2, y2 = points
    cv2.line(image, (x1,y1),(x2, y2), green, 1)
    cv2.circle(image, (x1,y1), radius=3, color=red, thickness=3)
    cv2.circle(image, (x2,y2), radius=3, color=red, thickness=3)
    cv2.putText(image, label + str(points), (x1,x2), font,  
                   fontScale, blue, thickness, cv2.LINE_AA)
    

def find_vector_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    print(a1, a2, b1, b2)
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (int(x//z), int(y//z))

def process_image(image, frame_count):
    image = cv2.resize(image, (0, 0), fx = 1, fy = 1)

    right_edges = [[0,0,0,0]]
    left_edges = [[0,0,0,0]]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # create mask
    mask = np.zeros_like(edges)
    height, width = mask.shape
    # {x, y}
    robot_reference_vector = [0, 1]

    limit1_point1 = [0, int(height/5)]
    limit1_point2 = [width, int(height/5)]

    limit2_point1 = [0, int(4*height/5)]
    limit2_point2 = [width, int(4*height//5)]

    limit_reference_point = [width//2, 4*height//5]

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
    detected_edges = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=1/5*height, maxLineGap=50)


    for n, v in enumerate(detected_edges):
        x1, y1, x2, y2 = v[0]
        vector = [x2-x1, y2-y1]
        unit_vector = vector/np.linalg.norm(vector)

        try: 
            slope = round(np.dot(robot_reference_vector, unit_vector), 4)
            
            # if the slope is positive its probably a right edge
            if slope > 0: 
                right_edges.append(v[0])

            elif slope == 0:
                print("slope is zero")
            
            # else assume it is a left edge
            else:
                left_edges.append(v[0])


            #cv2.putText(image, str(slope), (x1,y1), 1, 1, red)
            #cv2.putText(image, str(x1), (x1,y1), 1, 1, red)
            #cv2.putText(image, str(x2), (x2,y2), 1, 1, red)
        except:
            print("error in processing")
            traceback.print_exc() 

    right_edge = sort_by_distance_to_center(right_edges, width)[0]
    left_edge = sort_by_distance_to_center(left_edges, width)[0]

    draw_line_with_end_points(image, right_edge, "right edge")
    draw_line_with_end_points(image, left_edge, "left edge")
    # find the intersection

    beam_point = find_vector_intersect(right_edge[0:2], right_edge[2:4], left_edge[0:2], left_edge[2:4])
    left_offset_point =  find_vector_intersect(left_edge[0:2], left_edge[2:4], limit2_point1, limit2_point2)
    right_offset_point = find_vector_intersect(right_edge[0:2], right_edge[2:4], limit2_point1, limit2_point2)


    l_x = width//2 - left_offset_point[0]
    
    r_x = right_offset_point[0] - width//2

    distance_between_offset_points = r_x+l_x
    
    left_percent = round((l_x/(distance_between_offset_points)) * 100.0,2)
    right_percent = round((r_x/(distance_between_offset_points)) * 100.0,2)

    web_trajectory = [limit_reference_point[0]-beam_point[0], limit_reference_point[1]-beam_point[1]]
    print("web")
    print(web_trajectory)

    robot_angle = round(np.degrees(np.arccos((np.dot(robot_reference_vector, web_trajectory))/(np.linalg.norm(robot_reference_vector)*np.linalg.norm(web_trajectory)))),2)

    print(left_offset_point)
    print(right_offset_point)
    print(beam_point)
    cv2.circle(image, (beam_point), radius=3, color=red, thickness=3)
    cv2.putText(image, 'beam point', beam_point, font,  
                   fontScale, blue, thickness, cv2.LINE_AA)
    
    cv2.putText(image, str(robot_angle)+' deg', limit_reference_point, font,  
                   fontScale, blue, thickness, cv2.LINE_AA)

    cv2.circle(image, (left_offset_point), radius=3, color=red, thickness=3)
    cv2.circle(image, (right_offset_point), radius=3, color=red, thickness=3)

    cv2.putText(image, 'left % right %', (20,height-60), font,  
                   fontScale, blue, thickness, cv2.LINE_AA)
    cv2.putText(image, str(left_percent)+' '+str(right_percent), (20,height-20), font,  
                   fontScale, blue, thickness, cv2.LINE_AA)
           

    
    cv2.line(image, limit_reference_point, beam_point, blue, 2)


    print("right edge")
    print(right_edge)
    print("left edge")
    print(left_edge)
    # Draw ROI on the image
    cv2.putText(image, 'ROI', (limit1_point1[0], limit1_point1[1]-30), font,  
                   fontScale, green, thickness, cv2.LINE_AA)
    cv2.polylines(image, [np.int32([roi_vertices])], True, green, 2)


    #cv2.imwrite("./output/frame"+str(frame_count)+".jpg", image)

    return image


def extract_images_from_video(path_in=0, subsample_rate=1000, debug=False, save_images=False):
    print("Video Capture Started")
    #vidcap = cv2.VideoCapture(path_in)
    vidcap = cv2.VideoCapture(path_in, cv2.CAP_DSHOW)


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
            if save_images:
                cv2.imwrite("frame"+str(frame_count), processed_image)
            cv2.waitKey(100)
            frame_count = frame_count + 1
            
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    
    vidcap.release()
    return frame_count


extract_images_from_video()