import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""

def find_vertical_bounds(horizontal_project):
	N = len(horizontal_project)
	inf_array = horizontal_project[:int(N/2)]
	inf_bound = np.argmin(inf_array)
	sup_array = horizontal_project[int(N/2):N-1]
	sup_bound = int(N/2) + np.argmin(sup_array)
	return [inf_bound, sup_bound]

def find_horizontal_bounds(vertical_projection):
	N = len(vertical_projection)
	bool_bounds = (vertical_projection < 255)
	start_ind = 0
	end_ind = 1
	bounds = []
	for b in range(N-1):
		if bool_bounds[end_ind] & ~bool_bounds[start_ind]: # upwards transition
			bounds.append(end_ind)
		elif ~bool_bounds[end_ind] & bool_bounds[start_ind]:
			bounds.append(start_ind)
		start_ind += 1
		end_ind += 1

	return bounds


def divide_characters(image, bounds):
	N = len(bounds)
	for i in range(N-1):
		filename = "Characters/character_" + str(i) + ".jpg"
		cv2.imwrite(filename, image[:, bounds[i]:bounds[i+1]])


def segment_and_recognize(plate_img):
	# Maybe implement erosion or dilatation techniques to improve recognition
	# Explore connected components analysis

	# dilatation (to close gaps in the letters)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	plate_img = cv2.erode(plate_img, kernel, iterations=1)

	# Perform projections
	horizontal_project = np.sum(plate_img, axis=1)

	vertical_bounds = find_vertical_bounds(horizontal_project)
	new_plate = plate_img[vertical_bounds[0]+1:vertical_bounds[1]][:]

	vertical_project = np.sum(new_plate, axis=0)
	horizontal_bounds = find_horizontal_bounds(vertical_project)

	"""for bnd in vertical_bounds:
		plate_img = cv2.line(plate_img, (0, img_height-bnd), (img_width, img_height-bnd), (160, 0, 0), 1)"""

	# find vertical bounds
	img_width = new_plate.shape[1]
	img_height = new_plate.shape[0]

	# turn grayscale to BGR
	new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

	# draw bounding lines
	for bnd in horizontal_bounds:
		new_plate = cv2.line(new_plate, (bnd, 0), (bnd, img_height), (0, 255, 0), 1)

	cv2.imshow('Plate image', new_plate)
	cv2.waitKey(0)

	divide_characters(new_plate, horizontal_bounds)

	"""
	# afficher les projections horizontales et verticales
	print(horizontal_project)
	xx = np.arange(len(horizontal_project))
	plt.plot(xx, horizontal_project)
	plt.show()"""

	recognized_plates = 0
	return recognized_plates
