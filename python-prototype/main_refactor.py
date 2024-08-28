
import cv2
import numpy as np
import asyncio

images = asyncio.Queue(10)

video = './vid.mp4'

# visual constants
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
thickness = 1



def get_x_vector_midpoint(points):
    '''expects points [(x1, y1) (x2, y2)]'''
    x1, y1, x2, y2 = points
    midpoint_x = (x1+x2)//2
    #midpoint_y = (y1+y2)//2
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


#def low_pass_filter(left_edge, right_edge):
#    # following the pattern: new_value = current_value + (new_measurement - current_value) * gain
#    # y = x+ (m - x) * 0.1
#    new_left_edge = last_left_edge + (left_edge - last_left_edge) * 0.1



def resize_image(image, scale = 1):
    image = cv2.resize(image, (0, 0), fx = scale, fy = scale)
    return image


def gray_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def blur_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def process_image(image, last_image, frame_count, scale):
    try: 

        right_edges = [[0,0,0,0]]
        left_edges = [[0,0,0,0]]


        # Apply Gaussian Blur

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

        #y = x+ (m - x) * 0.1

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
    except Exception as e:
        return gray



def start_video_capture(video):
    vidcap = cv2.VideoCapture(video)
    return vidcap

def read_images(vidcap):
    success, image = vidcap.read()
    images.put(image)
    return success



import base64
import cv2
import asyncio
import numpy as np

image_queue = asyncio.Queue(10)
angle_queue = asyncio.Queue(10)


def setup_camera():
    vidcap = cv2.VideoCapture(0)
    #vidcap = cv2.VideoCapture('VID_20240405_120134.mp4')
    if vidcap.isOpened():
        print("Video Capture Started")
        return vidcap
    else:
        exit()


async def run_camera_server():

    vid = setup_camera()
    while vid.isOpened():
        try:
            success, image = vid.read()
            processed_image, left_percent, right_percent, robot_angle = process_image(image)

            encoded = cv2.imencode('.jpg', processed_image)[1]

            data = str(base64.b64encode(encoded))
            await image_queue.put(data[2:len(data)-1])
            #print("image queue fired")
            #await left_offset_queue.put(left_percent)
            #await right_offset_queue.put(right_percent)
            await angle_queue.put(robot_angle)
            #print("angle queue fired")
            
            await asyncio.sleep(0.1) # should run about every 1/10 a second

        except Exception as e:
            print(e)


asyncio.run(run_camera_server())