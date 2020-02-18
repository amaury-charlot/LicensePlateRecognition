import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def find_corners2(contour):
	hull = cv2.convexHull(contour)
	rect = np.zeros((4, 2))
	pts = []
	for pt in hull:
		pts.append(pt[0])

	s = np.sum(pts, axis=1)
	# Top-left
	rect[0] = pts[np.argmin(s)]
	# Bottom-right
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def find_corners(contour):
	coefficient = .01
	while True:
		epsilon = coefficient * cv2.arcLength(contour, True)
		poly_approx = cv2.approxPolyDP(contour, epsilon, True)
		hull = cv2.convexHull(poly_approx)
		if len(hull) == 4:
			return hull
		else:
			if len(hull) > 4:
				coefficient += .01
			else:
				coefficient -= .01


def hough_transform(img, contour):
	# Retrieve image shape
	img_shape = img.shape
	x_max = img_shape[1]
	y_max = img_shape[0]

	# Initialize theta bounds
	theta_max = 3.0 * np.pi / 4
	theta_min = - 1.0 * np.pi / 4

	# Initialize r bounds
	r_min = 0.0
	r_max = math.hypot(x_max, y_max)

	# Segment theta and r into dim "bins"
	r_dim = 150
	theta_dim = 150

	# Initialize accumulator hough_space
	hough_space = np.zeros((r_dim, theta_dim))

	# Hough transform algorithm loop
	for point in contour:
		x = point[0, 0]
		y = point[0, 1]
		for theta_i in range(theta_dim):
			theta = theta_min + 1.0 * theta_i * (theta_max-theta_min) / theta_dim
			r = x * math.cos(theta) + y * math.sin(theta)
			r_i = int(r_dim * r / r_max)
			hough_space[r_i, theta_i] = hough_space[r_i, theta_i] + 1

	"""
	plt.imshow(hough_space)
	plt.show()
	plt.xlim(0, theta_dim)
	plt.ylim(0, r_dim)
	"""

	max_table = np.zeros((4, 1))
	max_indexes = np.zeros((4, 2))
	"""
	for i in range(theta_dim):
		for j in range(r_dim):
			if hough_space[i, j] > np.min(max_table):
				min_ind = np.argmin(max_table)
				max_table[min_ind] = hough_space[i, j]
				max_indexes[min_ind] = [i, j]"""

	# Find 4 maximums within the hough_space accumulator with a window size of k
	k = 15
	for pnt_nb in range(4):
		for i in range(theta_dim):
			for j in range(r_dim):
				if hough_space[i,j] > max_table[pnt_nb]:
					max_table[pnt_nb] = hough_space[i, j]
					max_indexes[pnt_nb] = [i, j]
		ind = max_indexes[pnt_nb].astype(int)
		hough_space[max(ind[0]-int(k/2), 0):min(ind[0]+int(k/2)+1, theta_dim), max(ind[1]-int(k/2), 0):min(ind[1]+int(k/2)+1, r_dim)] = 0
		plt.imshow(hough_space)
		plt.show()
		plt.xlim(0, theta_dim)
		plt.ylim(0, r_dim)

	# Compute final r and theta points
	r_f = 1.0 * max_indexes[:, 0] * r_max / r_dim
	theta_f = theta_min + 1.0 * max_indexes[:, 1] * (theta_max-theta_min) / theta_dim

	# transform from hough parameter space (r,theta) to euclidean space (a,b)
	line_coefs = np.zeros((4, 2)) # Col 1 : a, Col 2: b, line equation y = ax+b
	if np.min(np.abs(np.sin(theta_f))) > 0:
		line_coefs[:, 0] = - np.divide(np.cos(theta_f), np.sin(theta_f)) # Compute slope a
		line_coefs[:, 1] = np.divide(r_f, np.sin(theta_f)) # Compute intercept b

	x_init = 0
	y_init = line_coefs.dot([x_init, 1]).astype(int)
	x_end = x_max - 1
	y_end = line_coefs.dot([x_end, 1]).astype(int)

	# draw hough lines
	houghed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	for i in range(4):
		cv2.line(houghed, (x_init, y_init[i]), (x_end, y_end[i]), (0, 255, 0), 2)
	cv2.imshow("hough", houghed)
	cv2.waitKey(25)

	# Find crossing  points between lines
	# two first lines are horizontal lines (more points), 2 last are vertical ones
	cross_points = np.zeros((4, 2))
	i = 0
	for horizontal_i in range(2):
		for vertical_i in range(2, 4):
			# Retrieve coefficients from two lines
			a1 = line_coefs[horizontal_i, 0]
			b1 = line_coefs[horizontal_i, 1]
			a2 = line_coefs[vertical_i, 0]
			b2 = line_coefs[vertical_i, 1]

			# Calculate crossing points using line coefficients
			x = (b2 - b1) / (a1 - a2)
			y = a1*x + b1
			cross_points[i] = [int(round(x)), int(round(y))]
			i += 1

	# Print crossing points on image
	"""for pnt in cross_points:
		cv2.circle(houghed, pnt, 5, (0, 0, 255), -1)
	cv2.imshow("hough", houghed)
	cv2.waitKey(0)"""

	# return crossing points coordinates
	return cross_points


def verify_plate(box):
	"""
	Verifies that a plate correspond to basic standards and has appropriate properties
	:param box: coordinates of the four vertices of the bounding rectangle
	:return: boolean value: True if plate is acceptable, False otherwise
	"""
	rect = order_points(box)
	(tl, tr, br, bl) = rect

	# Computes the width of the plate
	lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	Width = max(int(lower_width), int(upper_width))

	# Computes the height of the plate
	right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	Height = max(int(right_height), int(left_height))

	# Calculate aspect_ratio of the plate
	if Width and Height:
		aspect_ratio = Height/Width
	else:
		aspect_ratio = 1

	# Calculate Area of the plate
	area = cv2.contourArea(box)

	# Set conditions for an acceptable plate
	return (Width > 100) and (aspect_ratio < 0.3) and (area > 2600)


def order_points(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect


def plate_transform(image, box):
	"""
	Transforms a inclined plate into a straight plate
	:param image: plate image
	:param box: list of the four vertices' coordinates of the plate's bounding rectangle
	:return: straightened image
	"""

	# obtain the bounding rectangle's vertices and order them
	rect = order_points(box)
	(tl, tr, br, bl) = rect

	# Computes the width of the plate
	lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	width = max(int(lower_width), int(upper_width))

	# Computes the height of the plate
	right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	height = max(int(right_height), int(left_height))

	# Construct the set of destination points to obtain a "birds eye view" of the plate
	dst = np.array([
		[0, 0],
		[width - 1, 0],
		[width - 1, height - 1],
		[0, height - 1]], dtype="float32")

	# compute the perspective transform matrix
	transform_matrix = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, transform_matrix, (width, height))

	# return the warped image
	return warped

def plate_detection(image, contours):
	"""
	Detects the plate on the frame
	:param image: frame to be analyzed
	:param contours: contours retrieved of the pre-processed frame
	:return: list containing images of all plates detected
	"""
	final_contours = []
	i = 0
	corner_table = np.zeros((4, 2))
	# hull_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)

	for cnt in contours: # Loops and verify all contours for acceptable plates
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		if verify_plate(box):
			corners = find_corners2(cnt)

			"""corners = find_corners(cnt)
			for i in range(4):
				corner_table[i, 0] = int(corners[i, 0, 0])
				corner_table[i, 1] = int(corners[i, 0, 1])
			print(corner_table)
			for pnt in cross_points:
				cv2.circle(houghed, pnt, 5, (0,0,255), -1)
			cv2.imshow("hough",houghed)
			cv2.waitKey(0)"""
			
			final_contours.append(box)
		i += 1

	if not final_contours: # Returns None if no acceptable plate found
		return None

	# localized = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	# cv2.drawContours(localized, final_contours, 0, (0, 255, 0), 3)

	# Transforms and straighten each acceptable contours
	plate_img = []
	for cnt in final_contours:
		plate_img.append(plate_transform(image, cnt))

		# Show each localized plates
		cv2.imshow('Localized', plate_img[len(plate_img)-1])
		cv2.waitKey(25)
	return plate_img
