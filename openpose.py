import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='D:\\Projects\\4 sem-AI MINI project-HUMAN POSE ESTIMATION\\image1.jpg', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
# cap = cv.VideoCapture(args.input if args.input else 0)  # Comment out or remove this line

# Read input image
input_image = cv.imread("D:\\Projects\\4 sem-AI MINI project-HUMAN POSE ESTIMATION\\image.jpg")

# Check if the input image exists
if input_image is None:
    print("Error: Unable to read the input image.")
    exit()

# Use the input image dimensions
frameWidth = input_image.shape[1]
frameHeight = input_image.shape[0]

# Pass the input image to the network
net.setInput(cv.dnn.blobFromImage(input_image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
out = net.forward()
out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

assert(len(BODY_PARTS) == out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponding body's part.
    heatMap = out[0, i, :, :]

    # Originally, we try to find all the local maximums. To simplify a sample
    # we just find a global one. However only a single pose at the same time
    # could be detected this way.
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]
    # Add a point if its confidence is higher than threshold.
    points.append((int(x), int(y)) if conf > args.thr else None)

# Create a new image for visualization
output_image = input_image.copy()

# Draw the detected poses on the input image
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        cv.line(output_image, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.ellipse(output_image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(output_image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

# Display the result
cv.imshow('Pose Estimation Result', output_image)
cv.waitKey(0)  # Add this line to keep the window open until a key is pressed
cv.destroyAllWindows()  # Add this line to close all OpenCV windows after the key is pressed