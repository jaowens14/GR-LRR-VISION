import cv2
import numpy as np

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)


def draw_line_with_end_points(image, points, color):
    x1, y1, x2, y2 = points
    cv2.line(image, (x1,y1),(x2, y2), color, 1)
    cv2.circle(image, (x1,y1), radius=3, color=red, thickness=3)
    cv2.circle(image, (x2,y2), radius=3, color=red, thickness=3)

def draw_a_lot_of_points(image, xs, ys, color):
    for i in range(len(xs)):
        cv2.circle(image, (xs[i], ys[i]), radius=3, color=color, thickness=3)



#def get_edges(roi_image):
#    # h = 768/256 = 3
#    h, w = roi_image.shape
#    section_height = 2 ** 7 # this happens to be the section height
#    final_image = np.zeros_like(roi_image) # create zeros mat to add up in the end
#    for y in range(0, h, section_height):
#        y1 = y
#        y2 = y+section_height
#        x1 = 0
#        x2 = w
#
#        roi_vertices = [(0,   y),
#                        (w,   y),
#                        (w,   y+section_height),
#                        (0,   y+section_height), 
#                        ]
#
#        empty_array = np.zeros_like(roi_image) # full size
#
#        section_mask = cv2.fillPoly(empty_array, np.int32([roi_vertices]), 255) # draw section on empty
#
#        roi = cv2.bitwise_and(roi_image, section_mask) # get the section from the original image # full
#
#        _ , bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)
#        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        if contours:
#            # Get the largest contour (ROI)
#            contour = max(contours, key=cv2.contourArea)
#
#            # Or, get the polygon approximation of the contour (for more accurate vertices)
#            epsilon = 0.01 * cv2.arcLength(contour, True)
#            approx_verts = cv2.approxPolyDP(contour, epsilon, True)
#            polygon_vertices = approx_verts.reshape(-1, 2)  # Flatten to get (x, y) pairs
#
#
#
#        #final_image = cv2.add(final_image, roi)
#
#
#        return roi







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
    sh = 2 ** 6 # section height
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



def get_edge_closest_to_center(lefts, rights):
    if lefts.any() and rights.any():
        return lefts[np.argsort(get_location(lefts))][0], rights[np.argsort(get_location(rights))][0]
    else: 
        return [[0,0,0,0]], [[0,0,0,0]]


def detect_edges_in_section(section, vertex):
    h, w = section.shape
    offset_mask = np.zeros_like(section) # create a empty matrix
    blurred = cv2.GaussianBlur(section, (5, 5), 0) # blur the section
    edges = cv2.Canny(blurred, 50, 150) # detect edges

    cv2.fillPoly(offset_mask, np.int32([vertex]), 255) # draw offset roi on mask

    roi = cv2.bitwise_and(edges, offset_mask) # union with offset mask
    
    # detect lines in the real roi
    detected_edges = cv2.HoughLinesP(image=roi, rho=1, theta=np.pi / 180, threshold=25, minLineLength= 2 ** 5, maxLineGap=10) # these parameters need to be updated based on the size of the section
    
    if detected_edges is not None:
        #for edge in detected_edges:
        #    draw_line_with_end_points(section, edge[0], blue)
        #cv2.imshow("section", section)
        #cv2.waitKey(0)
        left_edges, right_edges = split_edges_into_left_and_right(detected_edges, section)
        left_edges, right_edges = filter_edges_by_slope(left_edges, right_edges)
        left_edge, right_edge = get_edge_closest_to_center(left_edges, right_edges)
        return left_edge, right_edge # detected_edges # vstack the edges to get them all
    else:
        return [[[0,0,0,0]]]

def gather_all_edges(sections, offsets):
    # this is a vstack of a list of results. 
    # each result is the return from the detect edges function 
    # which was passed the image section and the offsets
    return np.vstack([detect_edges_in_section(section, offsets[i]) for i, section in enumerate(sections)])



def get_slopes(edges):
    # get the vectors    
    x1 = edges[:, :, 0]
    y1 = edges[:, :, 1]
    x2 = edges[:, :, 2]
    y2 = edges[:, :, 3]
    v = np.hstack([x2-x1, y2-y1])
    unit_vectors = (v.T / np.linalg.norm(v, axis=1)).T # had to rework this for vector math
    return unit_vectors[:,1] / unit_vectors[:,0]

def get_location(edges):
    x1 = edges[:, :, 0]
    y1 = edges[:, :, 1]
    x2 = edges[:, :, 2]
    y2 = edges[:, :, 3]
    v = np.hstack([x2+x1, y2+y1]) * 0.5 # vector mid point
    return v[:, 0].astype(int) # return xs and compare to width of image outside

def get_z_score(edges):
    slopes = get_slopes(edges)
    return (slopes - np.mean(slopes)) / np.std(slopes)


def split_edges_into_left_and_right(edges, image):
    h, w = image.shape
    rights = edges[get_location(edges) > w//2] # right edges on the right side of the image
    lefts = edges[get_location(edges) < w//2] # left edges on the left side of the image
    return lefts, rights


def filter_edges_by_slope(left_edges, right_edges):
    # the ROI slope is about 2.0. 
    # this comes from the define_vertices function. 
    # these lines are kind of confusing because when you look at the edges you may think that the slope sign is wrong.
    # this is the sign of the slope in image space. So (0,0) is in the upper left instead of lower right.
    roi_s = 2.0
    tol = 1.5
    right_edges = right_edges[(get_slopes(right_edges) < roi_s+tol) & (get_slopes(right_edges) >= roi_s-tol)] 
    # needs to be less then the roi_slope plus tol and greater than toi_slope minus the tol
    left_edges = left_edges[(get_slopes(left_edges) > -(roi_s+tol)) & (get_slopes(left_edges) <= -(roi_s-tol))] 
    return left_edges, right_edges


def check_fit(xs, ys):
    xs = np.reshape(xs, (xs.shape[0],))
    ys = np.reshape(ys, (ys.shape[0],))
    print(xs.shape)
    coefficients, err, _, _,_ = np.polyfit(xs, ys, deg=1, full=True) # highest power first
    print("coef", coefficients)
    y = np.polyval(coefficients, xs)
    return np.abs(y-ys)

def estimate_edge(edges, image):
    edges = edges[~np.all(edges == 0, axis=(1, 2))] # this removes any zero edges

    x1 = edges[:, :, 0]
    y1 = edges[:, :, 1]
    x2 = edges[:, :, 2]
    y2 = edges[:, :, 3]
    x = x2 - x1
    y = y2 - y1
    v = np.hstack([x, y])
    m = np.matmul(v, v.T)
    # this selects all the entries in edges where the multiplcation result m is greater than the mean.
    edges = edges[np.all(m >= m.mean(axis=0), axis=1)]
    return edges

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
    print(int(x//z), int(y//z))
    if z == 0:                          # lines are parallel
        return [512, 0, 512, 768]       # return default
    return [int(x//z), int(y//z), 512, 768]



def filter_edges_by_mean_x(lefts, rights):
    right_x1 = rights[:, :, 0]
    right_edges = rights[np.all(right_x1 <= right_x1.mean(axis=0), axis=1)]

    left_x1 = lefts[:, :, 0]
    left_edges = lefts[np.all(left_x1 >= left_x1.mean(axis=0), axis=1)]
    return left_edges, right_edges

def remove_null_edges(all_edges):
    return all_edges[~np.all(all_edges == 0, axis=(1, 2))] # this removes any zero edges


def main():
    video_path = "vid.mp4"
    subsample_rate=100
    vidcap = cv2.VideoCapture(video_path)

    success, image = vidcap.read() #extract first frame.
    h, w, _ = image.shape

    frame_count = 0
    while success:

        vidcap.set(cv2.CAP_PROP_POS_MSEC, (frame_count*subsample_rate))

        success, initial_image = get_image(vidcap)

        gray_image = make_gray_image(initial_image)

        vertices, offsets = define_vertices(gray_image)

        sections = image_to_sections(gray_image, vertices)

        all_edges = gather_all_edges(sections, offsets)

        all_edges = remove_null_edges(all_edges)

        left_edges, right_edges = split_edges_into_left_and_right(all_edges, gray_image)
        left_edges, right_edges = filter_edges_by_mean_x(left_edges, right_edges)
        for n, v in enumerate(left_edges):
            draw_line_with_end_points(initial_image, v[0], green)

        for n, v in enumerate(right_edges):
            draw_line_with_end_points(initial_image, v[0], red)


        for i in range(min(len(left_edges), len(right_edges))):
            vector = find_vector_intersect(right_edges[i, :, 0:2], right_edges[i, :, 2:4], left_edges[i, :, 0:2], left_edges[i, :, 2:4])
            draw_line_with_end_points(initial_image, [w//2, h, w//2, 0], green)

            draw_line_with_end_points(initial_image, vector, blue)
            print(vector)
            print(vector[0:2])

        #    points.append(vector[0:2])
        #    print("mean")
        #    print(points)
        #print(np.mean(points, axis = 0))
        ## web_trajectory = update_web_trajectory(cleaned_edges)
# 
        # robot_angle = estimate_robot_angle(web_trajectory)
# 
        # robot_offset = estimate_robot_offset(web_trajectory)
                
        



        #draw_a_lot_of_points(initial_image, right_xs, right_ys, blue)
        #draw_a_lot_of_points(initial_image, left_xs, left_ys, red)

        
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