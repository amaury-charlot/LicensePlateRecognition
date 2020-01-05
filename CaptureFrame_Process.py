import cv2
import numpy as np
import os
import pandas as pd
import Localization
import Recognize
import matplotlib.pyplot as plt

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""

"""FUNCTIONS"""
# returns index of the selected contour
def select_contours(contours):
    int_contour = []
    for ind, c in enumerate(contours):
        perimeter = cv2.arcLength(c, True)
        if (perimeter > 300) & (perimeter < 600):
            int_contour.append(ind)

    #final_contour = np.asarray(int_contour)
    #print(final_contour)
    print(int_contour)
    return int_contour


# ISODATA threshold finding method
# Takes gray_scale image as input
def isodata_threshold(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    hist = hist[:120]
    t = [60]
    i = 0
    epsilon = 0.5
    # first iteration
    ginf = np.arange(0, t[0])
    gsup = np.arange(t[0], 120)
    m1 = np.average(ginf, weights=hist[:t[i]])
    m2 = np.average(gsup, weights=hist[t[i]:])
    t.append(int(np.average([m1, m2])))
    i = 1

    while np.abs(t[i-1]-t[i]) > epsilon:
        ginf = np.arange(0, t[i])
        gsup = np.arange(t[i], 120)
        m1 = np.average(ginf, weights=hist[:t[i]])
        m2 = np.average(gsup, weights=hist[t[i]:])
        t.append(int(np.average([m1, m2])))
        i += 1

    print("threshold is : ", t[i])
    return t[i]

# Function to determine the optimal thresholds for edge detection
# Still not used
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def yellow_mode(frame):
    # Blur the image to uniformize color of the plate
    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    # Convert to HSV color model
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Keep only "yellow" parts of the image
    light_orange = (15, 60, 50)
    dark_orange = (35, 255, 220)
    mask = cv2.inRange(hsv_img, light_orange, dark_orange)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # HSV to gray scale conversion
    #gray = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Binarization
    (thresh, binary) = cv2.threshold(gray, 62, 255, cv2.THRESH_BINARY)

    # edge detection
    edged = cv2.Canny(binary, 50, 100)

    # retrieve contours of the plate
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Put original image in gray scale
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plate_image = Localization.plate_detection(gray_original, contours)
    cv2.imshow('Plate image', plate_image)
    cv2.waitKey(0)

    #plate_image = cv2.GaussianBlur(plate_image, (5, 5), 0)
    resize_factor = 85 / plate_image.shape[0]
    dim = (int(plate_image.shape[1] * resize_factor), 85)
    plate_image = cv2.resize(plate_image, dim, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Resized plate', plate_image)
    cv2.waitKey(0)

    # Find histogram of image intensity
    """hist = cv2.calcHist(plate_image, [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()"""
    """plt.hist(plate_image.ravel(), 256, [0, 256])
    plt.show()"""
    T = isodata_threshold(plate_image)
    bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
    plate_number = Recognize.segment_and_recognize(bin_plate)
    return plate_number


def random_plate_mode(frame):
    # RGB to Gray Scale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # initial color model is always BGR

    # Noise removal
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Gray to binary conversion
    (thresh, binary) = cv2.threshold(blur, 62, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Binary Conversion', binary)
    cv2.waitKey(0)

    """kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)
    cv2.imshow('Binary Conversion', binary)
    cv2.waitKey(0)"""
    # Find edges of the Gray Scale image
    # Explore Sobel edge detector for horizontal and vertical edges

    edged = cv2.Canny(binary, 50, 100)  # 2nd and 3rd inputs are minVal and maxVal thresholds for edge detection (find ways to set them automatically = auto_Canny)
    cv2.imshow('Canny edges', edged)
    cv2.waitKey(0)

    # retrieve contours of the plate
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    plate_image = Localization.plate_detection(binary, contours)
    cv2.imshow('Plate image', plate_image)
    cv2.waitKey(0)

    Recognize.segment_and_recognize(plate_image)

    """# create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], True))
        print(hull[i])"""

    # create an empty black image
    # drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    """for i in range(len(contours)):
        #color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        #cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(edged, hull, i, color, 1, 8)"""

    # approximate contours
    """for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)

        cv2.drawContours(edged, cnt, -1, (200, 0, 255), 2)"""


"""CODE"""
# def CaptureFrame_Process(file_path, sample_frequency, save_path):
capture = cv2.VideoCapture("TrainingSet\Categorie I\Video3_2.avi")

# parameters
act_frame = 0
fps = 12
sample_frequency = 0.5  # frequency for choosing the frames to analyze

# initialization
ret, frame = capture.read()
recognized_plates = []

# display image to analyze (each 24 frames)
while ret:
    # Show actual frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)  # gives enough time for image to be displayed
    mode = 0
    if ~mode:  # yellow plates
        recognized_plates.append([yellow_mode(frame), act_frame, act_frame/fps])
    else:  # other plate colours
        random_plate_mode(frame)

    # Write csv file (using pandas) to keep a record of plate number
    df = pd.DataFrame(recognized_plates, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])
    df.to_csv('record.csv', index=None)

    # Pass to next frame
    act_frame += 24
    capture.set(cv2.CAP_PROP_POS_FRAMES, act_frame)
    ret, frame = capture.read()

# release pointer in memory
capture.release()

# Destroy all windows
cv2.destroyAllWindows()




