import cv2
import numpy as np
import colors

def start_stream(path_or_camera_index):
    video_capture = cv2.VideoCapture(path_or_camera_index, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        raise IOError

    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video_capture.get(cv2.CAP_PROP_FPS)
    print ("Length: %.2f | Width: %.2f | Height: %.2f | Fps: %.2f" % (length, width, height, fps))
    return video_capture