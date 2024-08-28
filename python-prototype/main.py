import cv2
import numpy as np

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)


def draw_line_with_end_points(image, points):
    x1, y1, x2, y2 = points
    cv2.line(image, (x1,y1),(x2, y2), blue, 1)
    cv2.circle(image, (x1,y1), radius=3, color=red, thickness=3)
    cv2.circle(image, (x2,y2), radius=3, color=red, thickness=3)



def get_edges(roi_image):
    # h = 768/256 = 3
    h, w = roi_image.shape
    section_width = 2 ** 7
    final_image = np.zeros_like(roi_image) # create zeros mat to add up in the end
    for y in range(0, h, section_width):
        y1 = y
        y2 = y+section_width
        x1 = 0
        x2 = w

        roi_vertices = [(0,   y),
                        (w,   y),
                        (w,   y+section_width),
                        (0,   y+section_width), 
                        ]

        empty_array = np.zeros_like(roi_image) # full size

        section_mask = cv2.fillPoly(empty_array, np.int32([roi_vertices]), 255) # draw section on empty

        roi = cv2.bitwise_and(roi_image, section_mask) # get the section from the original image # full

        _ , bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour (ROI)
            contour = max(contours, key=cv2.contourArea)

            # Or, get the polygon approximation of the contour (for more accurate vertices)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx_verts = cv2.approxPolyDP(contour, epsilon, True)
            polygon_vertices = approx_verts.reshape(-1, 2)  # Flatten to get (x, y) pairs



        #final_image = cv2.add(final_image, roi)


        return roi







def get_image(vidcap):
    return vidcap.read()

def make_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def define_vertices(image):
    offset = 2
    vertices = []
    offset_vertices = []
    h, w = image.shape
    u = w//8 # 128 # section unit width
    sh = 2 ** 7 # section height
    # define vertices
    for y in range(0, h, sh): 
        y1 = y
        y2 = y+sh

        x1 = (3 * u) - y1 // 2
        x2 = (5 * u) + y1 // 2

        x3 = (3 * u) - y2 // 2
        x4 = (5 * u) + y2 // 2

        roi_vertices = [(x1,   y1), (x2,   y1), (x4,   y2), (x3,   y2),]
        
        offset_roi_vertices = [(x1+offset,   y1+offset), (x2-offset,   y1+1), (x4-offset,   y2-offset), (x3+offset,   y2-offset),]

        vertices.append(roi_vertices)
        offset_vertices.append(offset_roi_vertices)
    return vertices, offset_vertices


def image_to_sections(image, vertices):

    sections = []

    for vertex in vertices:
        mask = np.zeros_like(image) # create a empty matrix

        cv2.fillPoly(mask, np.int32([vertex]), 255) # draw roi on empty

        roi = cv2.bitwise_and(image, mask) # union with input image and mask

        sections.append(roi)

    return sections



def detect_edges_in_section(section, vertex):
    h, w = section.shape
    print
    offset_mask = np.zeros_like(section) # create a empty matrix
    blurred = cv2.GaussianBlur(section, (5, 5), 0) # blur the section
    edges = cv2.Canny(blurred, 50, 150) # detect edges

    cv2.fillPoly(offset_mask, np.int32([vertex]), 255) # draw offset roi on mask

    roi = cv2.bitwise_and(edges, offset_mask) # union with offset mask
    
    # detect lines in the real roi
    detected_edges = cv2.HoughLinesP(image=roi, rho=1, theta=np.pi / 180, threshold=25, minLineLength=20, maxLineGap=10) 

    # can return none, check for this in the next step
    return detected_edges if detected_edges is not None else [[[0,0,0,0]]]


def gather_all_edges(sections, offsets):
    # this is a vstack of a list of results. 
    # each result is the return from the detect edges function 
    # which was passed the image section and the offsets
    return np.vstack([detect_edges_in_section(section, offsets[i]) for i, section in enumerate(sections)])

def append_slope_to_edge(edges):
    # get the vectors
    x1 = edges[:, :, 0]
    y1 = edges[:, :, 1]
    x2 = edges[:, :, 2]
    y2 = edges[:, :, 3]

    v = np.hstack([x2-x1, y2-y1])

    unit_vectors = (v.T / np.linalg.norm(v, axis=1)).T # haad to rework this for vector math

    slopes = unit_vectors[:,1] / unit_vectors[:,0]

    return edges


def sort_edges_by_slope(edges):
    left_edges = []
    right_edges = []
    robot_reference_vector = [0, 1]
    for n, v in enumerate(edges):
        x1, y1, x2, y2 = v[0]
        vector = [x2-x1, y2-y1]
        unit_vector = vector/np.linalg.norm(vector)
        slope = round(np.dot(robot_reference_vector, unit_vector), 4)
        # if the slope is positive its probably a right edge
        if slope > 0: 
            right_edges.append(v[0])
        elif slope < 0:
            left_edges.append(v[0])
    return left_edges, right_edges


def main():
    video_path = "vid.mp4"
    subsample_rate=100
    vidcap = cv2.VideoCapture(video_path)

    success, image = vidcap.read() #extract first frame.
    frame_count = 0
    while success:

        vidcap.set(cv2.CAP_PROP_POS_MSEC, (frame_count*subsample_rate))
        
        success, initial_image = get_image(vidcap)

        gray_image = make_gray_image(initial_image)

        vertices, offsets = define_vertices(gray_image)

        sections = image_to_sections(gray_image, vertices)

        all_edges = gather_all_edges(sections, offsets)

        left_edges, right_edges = sort_edges_by_slope(all_edges)

        all_edges = append_slope_to_edge(all_edges)

        # gather all the edges
        
        

        for n, v in enumerate(all_edges):
            draw_line_with_end_points(initial_image, v[0])
        cv2.imshow("processed_section", initial_image)
        cv2.waitKey(10)
    

        #roi_image = apply_roi(gray_image)
        #cv2.imshow("roi", roi_image)
        #edges_image = get_edges(roi_image)
        #cv2.imshow("edges_image", edges_image)
        


        frame_count += 1

    vidcap.release()
    return frame_count




if __name__ == "__main__":
    main()