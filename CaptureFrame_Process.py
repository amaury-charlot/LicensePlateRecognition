import cv2
import numpy as np
import os
import pandas as pd
import Localization
import Recognize
import matplotlib.pyplot as plt


def majority_vote(recognized_plates):
	N = len(recognized_plates)
	label_count = 0
	labels = np.zeros((N, 1))
	for i in range(N-1):
		ref_number = recognized_plates[i][0]
		test_number = recognized_plates[i+1][0]
		differences = sum([cr != ct for cr, ct in zip(str(ref_number), str(test_number))])  # Count number of different characters
		if differences > 2:
			label_count += 1
		labels[i+1] = label_count

	max_label = label_count

	j = 0
	final_recognized_plates = []
	label_i = []
	current_frame = 0
	fps = 12

	for i in range(max_label):
		current_frame = recognized_plates[j][1]
		while (j < N) and (labels[j] == i):
			label_i.append(recognized_plates[j][0])
			j += 1

		label_set = set(label_i)
		label_count = [label_i.count(x) for x in label_set]
		label_name = [x for x in label_set]
		max_count = max(label_count)
		second_max_count = 0

		for k, cnt in enumerate(label_count):
			if (cnt != max(label_count)) and (cnt > second_max_count):
				second_max_count = cnt

		# Measures the degree of difference between first and second maximum
		if second_max_count != 0:
			significancy = (float(max_count) / second_max_count) - 1
		else:
			significancy = 1

		for k, cnt in enumerate(label_count):
			if significancy >= 1:
				if cnt == max_count:
					final_recognized_plates.append([label_name[k], current_frame, current_frame / fps])
			else:
				final_recognized_plates.append([label_name[k], current_frame, current_frame / fps])

		"""for k, cnt in enumerate(label_count):
			if cnt == max(label_count):
				final_recognized_plates.append([label_name[k], current_frame, current_frame / fps])"""

		# print("label_i is : ", label_i, "and final label_i is : ", final_label_i)
		label_i.clear()

	return final_recognized_plates


def isodata_threshold(img):
	"""
	Finds optimal threshold using ISODATA algorithm to binarize the license plate image.
	:param img: image of the license plate. Should be in gray scale
	:return: optimal threshold
	"""
	hist, bins = np.histogram(img.ravel(), 256, [0, 256])  # Computes image's intensity histogram
	h = 1/8 * np.ones(8)
	hist = np.convolve(h, hist)[:256]	# Filter histogram using 8-point averager

	# Find optimal boundaries tmax and tmin based on threshold T
	# Set threshold
	N = len(hist)
	T = 100

	# Find lower bound tmin
	s = 0
	while ~((hist[s] <= T) & (hist[s + 1] > T)) & (s < N-2):
		s += 1
	tmin = s

	# Find upper bound tmax
	s = N - 1
	while ~((hist[s - 1] > T) & (hist[s] <= T)) & (s > 1):
		s -= 1
	tmax = s
	
	# ISODATA threshold algorithm
	# Initialization
	t = [int(np.average((tmin, tmax)))]
	epsilon = 0.5
	
	# first iteration
	ginf = np.arange(tmin, t[0])
	gsup = np.arange(t[0], tmax)

	if np.sum(hist[tmin:t[0]]) and np.sum(hist[t[0]:tmax]):
		m1 = np.average(ginf, weights=hist[tmin:t[0]])
		m2 = np.average(gsup, weights=hist[t[0]:tmax])
	else:
		plt.plot(np.arange(len(hist)), hist)
		plt.show()

	t.append(int(np.average([m1, m2])))
	i = 1

	# Loop until convergence of the threshold
	while np.abs(t[i-1]-t[i]) > epsilon:
		ginf = np.arange(tmin, t[i])
		gsup = np.arange(t[i], tmax)
		m1 = np.average(ginf, weights=hist[tmin:t[i]])
		m2 = np.average(gsup, weights=hist[t[i]:tmax])
		t.append(int(np.average([m1, m2])))
		i += 1
	
	# Return final threshold
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
	"""
	Localize Dutch yellow license plates and recognize them.
	:param frame: Actual frame extracted from the video.
	:return: list containing all plates recognized
	"""

	# Blur the image to uniformize color of the plate
	blur = cv2.GaussianBlur(frame, (9, 9), 0)

	# Keep record of gray_scales frame for window detection

	
	# Convert to HSV color model
	hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	
	# Yellow parts extraction
	light_orange = (15, 60, 70)
	dark_orange = (37, 255, 220)
	mask = cv2.inRange(hsv_img, light_orange, dark_orange)
	masked = cv2.bitwise_and(frame, frame, mask=mask)
	cv2.imshow("Masked", masked)
	cv2.waitKey(25)
	
	# BGR to gray scale conversion
	gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
	
	# Binarize frame with very low threshold to ease edge detection
	(thresh, binary) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
	
	# Perform canny edge detection
	edged = cv2.Canny(binary, 50, 100)

	# retrieve contours of the plate
	contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	# Change original image to gray scale
	gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Localize all plates in the frame and return them in a single list
	plates = Localization.plate_detection(gray_original, contours)
	
	if plates is not None: # Check for empty list
		plate_number = []
		for plate_image in plates: # Iterate through all plates
			# Extend image using linear interpolation to ease character recognition
			resize_factor = 85 / plate_image.shape[0]
			dim = (int(plate_image.shape[1] * resize_factor), 85)
			plate_image = cv2.resize(plate_image, dim, interpolation=cv2.INTER_LINEAR)

			# Crop edges of the image with a thickness of epsilon to keep only characters zone and remove plate boundaries
			epsilon = 10
			plate_image = plate_image[epsilon:plate_image.shape[0] - epsilon, epsilon:plate_image.shape[1] - epsilon]
			plate_image = cv2.GaussianBlur(plate_image, (5, 5), 0)

			# Show plate image
			# cv2.imshow("plate_image", plate_image)
			# cv2.waitKey(25)

			# Lowers threshold T by steps of 20 until plate is well recognized (!= None)
			# Reason is that ISODATA algorithm's output tends sometimes to be too high for correct segmentation
			first_time = 0
			intermediate_plate_number = None
			while (first_time < 5) and (intermediate_plate_number is None):
				# First iteration
				if first_time == 0:
					T = isodata_threshold(plate_image)  # Find threshold using ISODATA algorithm
					bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
					first_time += 1
				# Next iterations
				else:
					T -= 20 # Lowers initial threshold
					bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
					first_time += 1

				# Show binarized plate
				# cv2.imshow("bin_plate", bin_plate)
				# cv2.waitKey(10)

				# Record each recognized plate number in the plate_number list
				intermediate_plate_number = Recognize.segment_and_recognize(bin_plate)
			plate_number.append(intermediate_plate_number)
	else:
		plate_number = None # If no localized plates -> set plate_number to none

	# Return all plates recognized in the frame
	return plate_number


#def CaptureFrame_Process(file_path, sample_frequency, save_path):
"""
Captures frames from the specified file_path video with frequency of sample_frequency and return the output in save_path
:param file_path: Should be a video
:param sample_frequency: Should be in seconds
:param save_path: Should be a .csv file
:return:
"""
file_path = "test.avi"  # "TrainingSet/Categorie_I/Video4_2.avi"

# Retrieves the video at the specified file_path
capture = cv2.VideoCapture(file_path)

# Initialization
act_frame = 0
fps = 12
sample_frequency = 0.5  # frequency for choosing the frames to analyze

# initialization
ret, frame = capture.read()
recognized_plates = []

# Extract frame each 24 frames until the end of the video
while ret:
	# Show current frame
	cv2.imshow('Frame', frame)
	cv2.waitKey(0)

	# Choose mode in which to operate : 0 = Yellow mode, 1 = Random Color mode
	plates = yellow_mode(frame)
	if plates is not None:
		plates = np.unique(list(filter(None, plates)))
		for plate_number in plates:
			recognized_plates.append([plate_number, act_frame, act_frame/fps])

	# Pass to next frame
	act_frame += 4
	capture.set(cv2.CAP_PROP_POS_FRAMES, act_frame)
	ret, frame = capture.read()

final_recognized_plates = majority_vote(recognized_plates)

# Write csv file (using pandas) to keep a record of plate number
df = pd.DataFrame(recognized_plates, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])
save_path = 'record.csv'
df.to_csv(save_path, index=None)  # 'record.csv'

# release pointer in memory
capture.release()

# Destroy all windows
cv2.destroyAllWindows()

# Execute evaluation.py
command = "python evaluation.py --file_path record.csv --ground_truth_path test.csv"
os.system(command)
