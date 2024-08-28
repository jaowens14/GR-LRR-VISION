

def preprocess_image(og_image):
    # apply a set of gradients to the image to blur more away from the web
    h, w, _ = og_image.shape
    gray = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    lst = []
    for y in range(h):
        row = []
        for x in range(w):
            e = (x - 384 + y//3) * (x - 640 - y//3) 
            if e <= 0:
                e = 0
            row.append(e)

        lst.append(row)

    image = np.asarray(lst, dtype=np.float64)

    # Normalize the image matrix to the range [0, 1]
    normalized_image = image / np.max(image)

    # Scale the normalized image to the range [0, 255]
    scaled_image = normalized_image * 255.0

    # Optionally, convert to an integer data type like np.uint8
    gray_scale_mask = cv2.bitwise_not(scaled_image.astype(np.uint8))



    gradient_map = gray_scale_mask

    blurred_image = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # Get the kernel size from the gradient map, ensure it's odd
            kernel_size = int(gradient_map[i, j])
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Apply Gaussian blur with the corresponding kernel size
            blurred_pixel = cv2.GaussianBlur(gray[i:i+1, j:j+1], (kernel_size*kernel_size, kernel_size*kernel_size), 0)
            blurred_image[i, j] = blurred_pixel[0, 0]

    return blurred_image



red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)


def draw_line_with_end_points(image, points, label):
    x1, y1, x2, y2 = points
    cv2.line(image, (x1,y1),(x2, y2), blue, 1)
    cv2.circle(image, (x1,y1), radius=3, color=red, thickness=3)
    cv2.circle(image, (x2,y2), radius=3, color=red, thickness=3)
    #cv2.putText(image, label + str(points), (x1,x2), font,  
     #              fontScale, blue, thickness, cv2.LINE_AA)


def process_section(image):
    # Blur
    # Canny
    edges = cv2.Canny(image, 100, 250)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 15, None, 0, 0)
    print(lines)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    return image





